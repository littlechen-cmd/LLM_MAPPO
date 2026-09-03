"""Run one preregistered R1-C diagnostic arm and its fixed evaluation suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from llm_mappo.e1_evidence import (
    E1EvidenceWriter,
    E1TensorBoardWriter,
    load_e1_checkpoint,
    save_e1_checkpoint,
)
from llm_mappo.e1_training import E1Trainer, load_e1_raw_semantic_evidence
from llm_mappo.e1_vector_env import _new_environment
from llm_mappo.o2_device import device_provenance
from llm_mappo.r1_diagnostics import (
    load_r1c_diagnostic,
    r1c_identity,
    r1c_trend,
)
from llm_mappo.r1_evaluation import evaluate_r1c_checkpoint


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/optimization/r1_4agv_lowload.yaml"
    )
    parser.add_argument("--records", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", help="Existing running R1-C arm directory only.")
    return parser.parse_args()


def _commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _parameter_hash(trainer) -> str:
    digest = sha256()
    for module in (trainer.updater.actor, trainer.updater.critic):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _layout_hash(values, run) -> str:
    environment = _new_environment(values, run)
    try:
        environment.reset(seed=int(run.seed))
        return str(environment.env.shadow_layout_hash())
    finally:
        environment.close()


def _episodes(directory: Path) -> list[dict[str, Any]]:
    with (directory / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def main():  # noqa: C901
    args = _arguments()
    diagnostic = load_r1c_diagnostic(args.config, args.arm)
    labels = load_e1_raw_semantic_evidence(args.records)
    trainer = E1Trainer(
        run=diagnostic.run, environment=diagnostic.environment,
        training=diagnostic.training, labels=labels, device=args.device,
    )
    identity = r1c_identity(
        code_commit=_commit(), diagnostic=diagnostic,
        raw_records_sha256=labels.records_sha256,
        layout_hash=_layout_hash(diagnostic.environment, diagnostic.run),
        initial_parameter_sha256=_parameter_hash(trainer),
    )
    output_root = Path(args.output_root or diagnostic.artifact_root)
    if args.resume:
        directory = Path(args.resume)
        writer = E1EvidenceWriter.open_existing(directory)
        manifest = json.loads(
            (directory / "run_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("identity") != identity:
            raise ValueError("R1-C resume identity mismatch.")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = output_root / args.arm / (
            f"seed_{diagnostic.run.seed:04d}_{timestamp}"
        )
        writer = E1EvidenceWriter.create(directory, {
            "schema": "r1c-run-manifest-v1",
            "identity": identity,
            "device": device_provenance(args.device),
            "profile": dict(diagnostic.environment),
            "training": dict(diagnostic.training),
            "real_env_steps_budget": diagnostic.run.real_environment_steps,
            "requires_completed_episodes": True,
            "diagnostic_only": True,
            "label_provenance": labels.provenance(),
            "rollout_execution": {
                "num_env_workers": diagnostic.training["num_env_workers"],
                "rollout_length": diagnostic.training["rollout_length"],
                "global_step_unit": "joint_environment_transition",
            },
        })
    board = E1TensorBoardWriter(directory / "tensorboard")
    try:
        if args.resume:
            checkpoint = load_e1_checkpoint(
                directory / "checkpoint_latest.pt", expected_identity=identity,
                actor=trainer.updater.actor, critic=trainer.updater.critic,
                optimizer=trainer.updater.optimizer,
            )
            trainer.restore_calibration_state(checkpoint["calibration_state"])
            trainer.restore_runtime_state(checkpoint["trainer_state"])

        def update(
            metrics: Mapping[str, Any], runtime: Mapping[str, Any], episode_rows
        ):
            state = trainer.schedule.state_dict()
            step = state["global_env_steps"]
            names = (
                "policy_loss", "value_loss", "entropy", "total_loss",
                "approx_kl", "clip_fraction", "explained_variance", "grad_norm",
                "learning_rate", "astar_loss", "astar_valid_rate",
                "calibration_sample_rate", "delta_g_mean", "delta_g_positive_rate",
                "rc_confidence_mean", "semantic_loss", "semantic_valid_rate",
                "semantic_reliability_mean", "semantic_valid_denominator",
                "num_env_workers",
                "rollout_length", "global_environment_steps",
                "environment_steps_per_second", "rollout_wall_time",
                "policy_inference_time", "ppo_update_time", "total_elapsed_time",
                "peak_cuda_memory_allocated", "peak_cuda_memory_reserved",
            )
            update_row = {
                "real_env_steps": step, "lambda_a": state["lambda_a"],
                "lambda_l": state["lambda_l"],
                **{key: metrics[key] for key in names},
            }
            evidence = {
                "update_row": update_row,
                "episode_rows": [dict(row) for row in episode_rows],
            }
            save_e1_checkpoint(
                directory / "checkpoint_latest.pt", identity=identity,
                actor=trainer.updater.actor, critic=trainer.updater.critic,
                optimizer=trainer.updater.optimizer, schedule_state=state,
                calibration_state=None, trainer_state=runtime,
                evidence_state=evidence,
            )
            reconciled = writer.reconcile_checkpoint_evidence(evidence)
            if reconciled["update_appended"]:
                board.add_update(update_row)
            for row in episode_rows[-int(reconciled["episodes_appended"]):]:
                board.add_episode(row)

        summary = trainer.run_prefix(
            diagnostic.run.real_environment_steps, on_update=update
        )
        counts = (
            "real_env_steps", "teacher_queries", "shadow_calls", "ema_updates",
            "semantic_valid_slots", "semantic_total_slots", "planner_query_count",
        )
        writer.append("teacher_step_counts.csv", {key: summary[key] for key in counts})
        save_e1_checkpoint(
            directory / "checkpoint_final.pt", identity=identity,
            actor=trainer.updater.actor, critic=trainer.updater.critic,
            optimizer=trainer.updater.optimizer,
            schedule_state=trainer.schedule.state_dict(), calibration_state=None,
            trainer_state=trainer.runtime_state(), evidence_state=None,
        )
        completed = writer.complete(summary)
        trend = r1c_trend(_episodes(directory))
        (directory / "r1c_trend.json").write_text(
            json.dumps(trend, indent=2, sort_keys=True), encoding="utf-8"
        )
        evaluation = evaluate_r1c_checkpoint(
            directory=directory, environment=diagnostic.environment,
            run=diagnostic.run, dataset=labels.dataset, identity=identity,
            seeds=diagnostic.evaluation_seeds, device=args.device,
        )
        receipt = {"training": completed, "trend": trend, "evaluation": evaluation}
        (directory / "r1c_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps({"run_directory": str(directory), **receipt}, sort_keys=True))
    except Exception as error:
        writer.fail(type(error).__name__)
        raise
    finally:
        board.close()
        trainer.close()


if __name__ == "__main__":
    main()
