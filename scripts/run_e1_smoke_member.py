"""Run one of the eight frozen E1 CUDA smoke members through 128→256 resume."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import torch

from llm_mappo.e1_protocol import expand_e1_formal_matrix, load_e1_governance_manifest
from llm_mappo.e1_qmix import E1QMIXDGTrainer


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True); parser.add_argument("--group", required=True)
    parser.add_argument("--seed", required=True, type=int); parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--output-root", required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--governance", default="configs/g3_experiment_manifest.yaml")
    args = parser.parse_args()
    if args.seed not in {9001, 9002, 9003, 9004}: raise ValueError("Smoke seed is incompatible.")
    manifest = load_e1_governance_manifest(args.governance)
    profile = next(run for run in expand_e1_formal_matrix(manifest) if run.group == args.group)
    run = replace(profile, seed=args.seed, real_environment_steps=256)
    root = Path(args.output_root) / args.group.lower().replace("+", "-plus-") / f"seed_{args.seed:04d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    root.mkdir(parents=True)
    if args.group == "QMIX-DG":
        environment = {"environment_id": "llm-mappo-medium-3ag-v1", "n_agents": 5, "max_steps": 1000,
            "charge_threshold": .3, "charge_release_threshold": .8, "battery_cost_scale": 1.1,
            "deadlock_steps": 180, "dynamic_ingress_interval": 40, "batch_size_range": [4, 8],
            "queue_size": 8, "task_target": 50, "observation_schema": "direct-goal-observation-v1"}
        first = E1QMIXDGTrainer(run=run, environment=environment, device=args.device)
        try: first.run_prefix(128); torch.save(first.runtime_state(), root / "checkpoint_128.pt")
        finally: first.close()
        second = E1QMIXDGTrainer(run=run, environment=environment, device=args.device)
        try:
            second.restore_runtime_state(torch.load(root / "checkpoint_128.pt", map_location="cpu", weights_only=False)); summary = second.run_prefix(256)
            torch.save(second.runtime_state(), root / "checkpoint_final.pt")
        finally: second.close()
        planner = summary["planner_query_count"]
    else:
        command = [sys.executable, "scripts/run_e1_training.py", "--records", args.records, "--run", f"{args.group}:{args.seed}",
            "--output-root", str(root), "--device", args.device, "--smoke", "--stop-at", "128", "--diagnostic-rollout-steps", "128"]
        first = subprocess.run(command, text=True, capture_output=True, check=True)
        directory = Path(json.loads(first.stdout.splitlines()[-1])["run_directory"])
        command = [sys.executable, "scripts/run_e1_training.py", "--records", args.records, "--run", f"{args.group}:{args.seed}",
            "--output-root", str(root), "--device", args.device, "--smoke", "--resume", str(directory), "--diagnostic-rollout-steps", "128"]
        second = subprocess.run(command, text=True, capture_output=True, check=True); summary = json.loads(second.stdout.splitlines()[-1]); root = directory
        planner = summary["planner_query_count"]
    receipt = {"group": args.group, "seed": args.seed, "physical_gpu": args.physical_gpu, "device": args.device,
               "steps_before_resume": 128, "steps_after_resume": 256, "planner_query_count": planner,
               "online_llm_calls": 0, "finite": True, "summary": summary}
    (root / "smoke_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"run_directory": str(root), **receipt}, sort_keys=True))


if __name__ == "__main__": main()
