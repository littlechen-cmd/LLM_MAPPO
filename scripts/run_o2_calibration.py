"""Owner-run entry point for one receipted O2 calibration run."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Sequence

from llm_mappo.o2_contract import (
    O2ExperimentConfig,
    O2RunSpec,
    expand_o2_matrix,
    verify_o1_authorization,
)
from llm_mappo.o2_evidence import (
    O2EvidenceWriter,
    compute_throughput_grid,
    load_o2_checkpoint,
    save_o2_checkpoint,
)
from llm_mappo.o2_training import O2Trainer


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen O2 YAML contract.")
    parser.add_argument("--o1-run", required=True, help="Completed receipted O1 run.")
    parser.add_argument("--output-root", required=True, help="O2 artifact root.")
    parser.add_argument("--run", required=True, help="One frozen GROUP:SEED member.")
    parser.add_argument("--device", default="cuda:0", help="Torch logical device.")
    parser.add_argument("--resume", help="Interrupted O2 run directory to resume.")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="Diagnostic-only bounded prefix; cannot create a formal O2 result.",
    )
    return parser.parse_args(argv)


def _selected_run(value: str, config: O2ExperimentConfig) -> O2RunSpec:
    try:
        group, raw_seed = value.split(":", 1)
        candidate = O2RunSpec(group, int(raw_seed), config.real_env_steps)
    except ValueError as error:
        raise ValueError("--run must be an exact frozen GROUP:SEED pair.") from error
    if candidate not in expand_o2_matrix(config):
        raise ValueError("--run is not a member of the frozen O2 matrix.")
    return candidate


def _code_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def run(arguments: argparse.Namespace) -> dict:  # noqa: C901
    config = O2ExperimentConfig.from_yaml(arguments.config)
    o1 = verify_o1_authorization(arguments.o1_run)
    selected = _selected_run(arguments.run, config)
    if arguments.smoke_steps is not None and arguments.smoke_steps < 1:
        raise ValueError("--smoke-steps must be positive.")
    diagnostic_only = arguments.smoke_steps is not None
    identity = {
        "code_commit": _code_commit(),
        "config_sha256": config.sha256(),
        "seed": selected.seed,
        "group": selected.group,
        "o1_code_commit": o1["code_commit"],
        "o1_summary_sha256": o1["summary_sha256"],
    }
    if arguments.resume:
        directory = Path(arguments.resume)
        manifest = json.loads(
            (directory / "run_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("identity") != identity:
            raise ValueError("O2 resume identity does not exactly match this run.")
        if manifest.get("diagnostic_only") != diagnostic_only:
            raise ValueError("O2 resume diagnostic mode does not match the artifact.")
        writer = O2EvidenceWriter.open_existing(directory)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{selected.group.lower()}_seed{selected.seed}_{timestamp}"
        directory = Path(arguments.output_root) / name
        writer = O2EvidenceWriter.create(
            directory,
            {
                "schema": "o2-run-manifest-v1",
                "identity": identity,
                "diagnostic_only": diagnostic_only,
                "real_env_steps_budget": selected.real_env_steps,
                "llm_kd": False,
            },
        )
    trainer = O2Trainer(experiment=config, run=selected, device=arguments.device)
    if arguments.resume:
        restored = load_o2_checkpoint(
            directory / "checkpoint_latest.pt",
            expected_identity=identity,
            actor=trainer.actor,
            critic=trainer.critic,
            optimizer=trainer.updater.optimizer,
        )
        trainer.restore_runtime_state(restored["trainer_state"])
        trainer.restore_calibration_state(restored["calibration_state"])
    next_checkpoint = [
        ((trainer.schedule.global_env_steps // 10000) + 1) * 10000
    ]

    def on_step(row):
        writer.write_teacher_step_count(
            {name: row[name] for name in (
                "real_env_steps", "teacher_queries", "shadow_calls", "ema_updates"
            )}
        )

    def on_update(row):
        writer.write_update(
            {name: row[name] for name in (
                "real_env_steps", "policy_loss", "value_loss", "astar_loss"
            )}
        )

    def on_checkpoint(runtime):
        if trainer.schedule.global_env_steps < next_checkpoint[0]:
            return
        save_o2_checkpoint(
            directory / "checkpoint_latest.pt",
            identity=identity,
            actor=trainer.actor,
            critic=trainer.critic,
            optimizer=trainer.updater.optimizer,
            schedule_state=trainer.schedule.state_dict(),
            calibration_state=(
                None
                if trainer.calibrator is None
                else trainer.calibrator.ema.state_dict()
            ),
            trainer_state=runtime,
            rollout_empty=True,
        )
        while trainer.schedule.global_env_steps >= next_checkpoint[0]:
            next_checkpoint[0] += 10000

    try:
        summary = trainer.run(
            max_steps=arguments.smoke_steps,
            on_step=on_step,
            on_update=on_update,
            on_episode=writer.write_episode,
            on_checkpoint=on_checkpoint,
        )
        save_o2_checkpoint(
            directory / "checkpoint_final.pt",
            identity=identity,
            actor=trainer.actor,
            critic=trainer.critic,
            optimizer=trainer.updater.optimizer,
            schedule_state=trainer.schedule.state_dict(),
            calibration_state=(
                None
                if trainer.calibrator is None
                else trainer.calibrator.ema.state_dict()
            ),
            trainer_state=trainer.runtime_state(),
            rollout_empty=True,
        )
        _write_throughput_grid(directory)
        writer.close(summary={**summary, "diagnostic_only": diagnostic_only})
        return {"run_directory": str(directory), **summary}
    except Exception as error:
        writer.fail(reason=type(error).__name__)
        raise
    finally:
        trainer.environment.close()
        trainer.student_shadow.close()
        trainer.teacher_shadow.close()


def _write_throughput_grid(directory: Path) -> None:
    with (directory / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    rows = compute_throughput_grid(episodes, list(range(0, 150001, 10000)))
    grid_path = directory / "throughput_grid.csv"
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=("real_env_steps", "throughput")).writeheader()
        csv.DictWriter(
            handle, fieldnames=("real_env_steps", "throughput")
        ).writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    print(run(arguments))


if __name__ == "__main__":
    main()
