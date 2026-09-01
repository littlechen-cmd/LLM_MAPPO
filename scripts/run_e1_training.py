"""Owner-run entry point for one E1 formal or diagnostic training member."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import torch
import yaml

from llm_mappo.e1_evidence import E1EvidenceWriter, load_e1_checkpoint, save_e1_checkpoint
from llm_mappo.e1_protocol import (
    expand_e1_formal_matrix,
    load_e1_governance_manifest,
    resolve_e1_run,
)
from llm_mappo.e1_training import E1Trainer, load_e1_raw_semantic_evidence
from llm_mappo.e1_qmix import E1QMIXDGTrainer
from llm_mappo.o2_device import device_provenance


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governance", default="configs/g3_experiment_manifest.yaml")
    parser.add_argument("--records", required=True)
    parser.add_argument("--run", required=True, help="Exact GROUP:SEED identity.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    parser.add_argument("--stop-at", type=int, help="Diagnostic interruption boundary.")
    parser.add_argument("--diagnostic-rollout-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _commit(): return subprocess.run(["git", "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()


def main():  # noqa: C901
    args = _arguments(); manifest = load_e1_governance_manifest(args.governance)
    runs = expand_e1_formal_matrix(manifest)
    run = resolve_e1_run(runs, args.run, smoke=args.smoke)
    seed = run.seed
    if args.smoke and seed not in {9001, 9002, 9003, 9004}:
        raise ValueError("E1 smoke seeds must be in 9001..9004.")
    labels = load_e1_raw_semantic_evidence(args.records)
    identity = {"code_commit": _commit(), "governance_sha256": sha256(Path(args.governance).read_bytes()).hexdigest(),
                "group": run.group, "seed": run.seed, "raw_records_sha256": labels.records_sha256,
                "exploratory_noisy_teacher": True}
    environment = {"environment_id": "llm-mappo-medium-3ag-v1", "n_agents": 5,
        "dynamic_ingress_interval": 40, "batch_size_range": [4, 8], "queue_size": 8,
        "task_target": 50, "max_steps": 1000, "deadlock_steps": 180,
        **manifest["route_profiles"]["optimization"]["energy"]}
    if args.diagnostic_rollout_steps is not None and args.stop_at is None and not args.resume:
        raise ValueError("Diagnostic rollout override requires --stop-at.")
    training = {"rollout_steps": args.diagnostic_rollout_steps or 128,
                "rollout_length": args.diagnostic_rollout_steps or 128,
                "num_env_workers": 16, "update_epochs": 4, "minibatch_steps": 64}
    target = run.real_environment_steps if args.stop_at is None else args.stop_at
    if not 1 <= target <= run.real_environment_steps: raise ValueError("--stop-at is out of range.")
    if args.resume:
        directory = Path(args.resume); writer = E1EvidenceWriter.open_existing(directory)
        if json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))["identity"] != identity: raise ValueError("E1 resume identity mismatch.")
    else:
        directory = Path(args.output_root) / run.group.lower().replace("+", "-plus-") / f"seed_{run.seed:03d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        writer = E1EvidenceWriter.create(directory, {"schema": "e1-run-manifest-v1", "identity": identity,
            "device": device_provenance(args.device), "real_env_steps_budget": run.real_environment_steps,
            "diagnostic_only": args.stop_at is not None, "label_provenance": labels.provenance(),
            "rollout_execution": {"num_env_workers": 16, "rollout_length": 128,
                                  "global_step_unit": "joint_environment_transition"}})
    if run.algorithm == "qmix":
        trainer = E1QMIXDGTrainer(run=run, environment={**environment,
            "observation_schema": run.observation_schema}, device=args.device)
        try:
            summary = trainer.run_prefix(target)
            torch.save(trainer.runtime_state(), directory / ("checkpoint_final.pt" if target == run.real_environment_steps else "checkpoint_latest.pt"))
            writer.append("teacher_step_counts.csv", {"real_env_steps": summary["real_env_steps"],
                "teacher_queries": 0, "shadow_calls": 0, "ema_updates": 0,
                "semantic_valid_slots": 0, "semantic_total_slots": 0,
                "planner_query_count": summary["planner_query_count"]})
            if target == run.real_environment_steps: writer.complete(summary)
            print(json.dumps({"run_directory": str(directory), **summary}, sort_keys=True))
        except Exception as error:
            writer.fail(type(error).__name__); raise
        finally:
            trainer.close()
        return
    trainer = E1Trainer(run=run, environment=environment, training=training, labels=labels, device=args.device)
    try:
        if args.resume:
            checkpoint = load_e1_checkpoint(directory / "checkpoint_latest.pt", expected_identity=identity, actor=trainer.updater.actor, critic=trainer.updater.critic, optimizer=trainer.updater.optimizer)
            trainer.restore_runtime_state(checkpoint["trainer_state"])
        def update(metrics, runtime):
            state = trainer.schedule.state_dict(); step = state["global_env_steps"]
            writer.append("updates.csv", {"real_env_steps": step,
                **{key: metrics[key] for key in ("policy_loss", "value_loss", "astar_loss", "semantic_loss", "semantic_valid_denominator", "num_env_workers", "rollout_length", "global_environment_steps", "environment_steps_per_second", "rollout_wall_time", "policy_inference_time", "ppo_update_time", "total_elapsed_time", "peak_cuda_memory_allocated", "peak_cuda_memory_reserved")},
                "lambda_a": state["lambda_a"], "lambda_l": state["lambda_l"]})
            save_e1_checkpoint(directory / "checkpoint_latest.pt", identity=identity, actor=trainer.updater.actor, critic=trainer.updater.critic, optimizer=trainer.updater.optimizer, schedule_state=state, calibration_state=None if trainer.calibrator is None else trainer.calibrator.ema.state_dict(), trainer_state=runtime)
        summary = trainer.run_prefix(target, on_update=update)
        writer.append("teacher_step_counts.csv", {key: summary[key] for key in ("real_env_steps", "teacher_queries", "shadow_calls", "ema_updates", "semantic_valid_slots", "semantic_total_slots", "planner_query_count")})
        save_e1_checkpoint(directory / ("checkpoint_final.pt" if target == run.real_environment_steps else "checkpoint_latest.pt"), identity=identity, actor=trainer.updater.actor, critic=trainer.updater.critic, optimizer=trainer.updater.optimizer, schedule_state=trainer.schedule.state_dict(), calibration_state=None if trainer.calibrator is None else trainer.calibrator.ema.state_dict(), trainer_state=trainer.runtime_state())
        (writer.complete if target == run.real_environment_steps else lambda value: None)(summary)
        print(json.dumps({"run_directory": str(directory), **summary}, sort_keys=True))
    except Exception as error:
        writer.fail(type(error).__name__); raise
    finally: trainer.close()


if __name__ == "__main__": main()
