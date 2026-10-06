"""Audit or launch four independent parallel A* cutoff diagnostic learners."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch
import yaml

from scripts import run_medium5_formal_baseline as base
from llm_mappo.mappo_astar_kd.runner import AStarKDRun
from llm_mappo.mappo_astar_kd.teacher import TEACHER_VERSION
from llm_mappo.mappo_baseline.runner import (
    ASTAR_TAIL_DIAGNOSTIC_SCHEMA, BaselineExperiment, _validate_training,
)


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
TAIL_SCHEMA = ASTAR_TAIL_DIAGNOSTIC_SCHEMA
OWN_PROCESS_GROUPS = os.name == "posix"
ARTIFACT_ROOT = "artifacts/astar_tail_diagnostics/medium-5ag"
GROUPS = {f"cutoff_{steps // 1000}k": steps
          for steps in (450000, 600000, 750000, 900000)}
EVALUATION = {
    "replenishment_mode": "rware_delivery", "seeds": list(range(9400, 9410)),
    "deterministic_policy": "argmax", "render_replay": False,
    "replay_frame_stride": 1, "replay_frame_duration_ms": 600,
    "environment_overrides": {},
}
METHOD_FIELDS = {
    "astar_teacher_approval", "astar_teacher_version", "astar_schedule_steps",
    "post_schedule_astar_lambda", "astar_stop_steps", "paired_training_seeds",
}


def load_tail_experiment(source: Path, group: str, *, root: Path) -> BaselineExperiment:
    """Preserve diagnostic identity while enforcing paired baseline mechanics."""
    source = source.resolve()
    if group not in GROUPS or not source.is_relative_to(root):
        raise ValueError("Choose an authorized diagnostic group inside this worktree.")
    raw = source.read_bytes()
    values = yaml.safe_load(raw)
    required = {"schema", "method", "approval_status", "artifact_root",
                "environment", "training", "evaluation"}
    if (set(values) != required or values["schema"] != TAIL_SCHEMA
            or values["method"] != "MAPPO-FP-ASTARKD-NO-VALIDITY"
            or values["approval_status"] != "approved"
            or values["artifact_root"] != f"{ARTIFACT_ROOT}/{group}"):
        raise ValueError("Not an approved isolated A* tail diagnostic configuration.")
    baseline_source = root / "configs/training_v1/medium-5ag-formal/seed_0007.yaml"
    baseline = base.review_configs(baseline_source.parent, [7], root=root)[0][1]
    raw_baseline = yaml.safe_load(baseline_source.read_bytes())
    training = values["training"]
    _validate_training(training, provisional=True)
    common = {key: value for key, value in training.items() if key not in METHOD_FIELDS}
    expected = {key: value for key, value in baseline.training.items()
                if key != "paired_training_seeds"}
    if (values["environment"] != raw_baseline["environment"] or common != expected
            or values["evaluation"] != EVALUATION):
        raise ValueError("Diagnostic environment, PPO or evaluation contract differs.")
    expected_teacher = {
        "astar_teacher_approval": "approved",
        "astar_teacher_version": TEACHER_VERSION,
        "astar_schedule_steps": 600000, "post_schedule_astar_lambda": 0.01,
        "astar_stop_steps": GROUPS[group], "paired_training_seeds": [7],
    }
    if any(training.get(key) != value for key, value in expected_teacher.items()):
        raise ValueError("Shared curve or hard cutoff diagnostic contract differs.")
    return BaselineExperiment(
        schema=TAIL_SCHEMA, method=values["method"],
        artifact_root=values["artifact_root"], environment=baseline.environment,
        training=training, evaluation=values["evaluation"],
        config_sha256=sha256(raw).hexdigest(), approval_status="approved",
    )


def _review(directory: Path, root: Path) -> list:
    return [(group, directory / f"{group}.yaml",
             load_tail_experiment(directory / f"{group}.yaml", group, root=root))
            for group in GROUPS]


def _blockers(reviewed: list, root: Path, snapshot: dict, device: str) -> list:
    existing = []
    for _, _, experiment in reviewed:
        matches = sorted((root / experiment.artifact_root).glob("seed_0007_*"))
        existing.append((None, experiment, matches))
    blockers = base._launch_blockers(existing, snapshot, torch.device(device))
    if device not in {"cuda", "cuda:0"}:
        blockers.append("Parallel long diagnostics require cuda:0.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        blockers.append("Set CUDA_VISIBLE_DEVICES=0 for the approved shared GPU.")
    return blockers


def _stop_children(children: list) -> None:
    for process in children:
        if process.poll() is None:
            process.terminate()
    for process in children:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if OWN_PROCESS_GROUPS:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()


def _parallel(reviewed: list, manifest: dict, manifest_path: Path,
              root: Path, device: str, snapshot: dict) -> None:
    children, handles = [], []
    manifest["status"] = "running"
    base._write_manifest(manifest_path, manifest)
    try:
        for row, (group, source, _) in zip(manifest["groups"], reviewed):
            current = base._git_snapshot(root)
            if (current["code_commit"] != snapshot["code_commit"]
                    or current["dirty_training_paths"]
                    or sha256(source.read_bytes()).hexdigest() != row["config_sha256"]):
                raise ValueError("Diagnostic source changed after review.")
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--config-directory", str(source.parent), "--group", group,
                       "--device", device, "--worker-run",
                       "--expected-commit", snapshot["code_commit"],
                       "--expected-config-sha", row["config_sha256"]]
            log = manifest_path.parent / f"{group}.log"
            handle = log.open("wb")
            handles.append(handle)
            process = subprocess.Popen(
                command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=OWN_PROCESS_GROUPS,
            )
            children.append(process)
            row.update(status="running", pid=process.pid, command=command,
                       log_path=str(log))
            base._write_manifest(manifest_path, manifest)
        for row, process in zip(manifest["groups"], children):
            row["returncode"] = process.wait()
            row["status"] = "complete" if row["returncode"] == 0 else "failed"
            base._write_manifest(manifest_path, manifest)
        manifest["status"] = (
            "complete" if all(row["status"] == "complete" for row in manifest["groups"])
            else "failed"
        )
    except BaseException as error:
        _stop_children(children)
        manifest.update(status="failed", error=str(error))
        for row in manifest["groups"]:
            if row["status"] == "running":
                row["status"] = "interrupted"
        raise
    finally:
        for handle in handles:
            handle.close()
        base._write_manifest(manifest_path, manifest)


def launch(config_directory: Path, *, device: str, run: bool,
           root: Path = WORKTREE_ROOT) -> dict:
    base._verify_imports(root)
    reviewed = _review(config_directory, root)
    snapshot = base._git_snapshot(root)
    blockers = _blockers(reviewed, root, snapshot, device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / ARTIFACT_ROOT / "launches"
    directory = destination / stamp
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "launch_manifest.json"
    manifest = {
        "schema": "medium5-astar-tail-parallel-launch-v1", "status": "dry_run",
        "worktree": str(root), **snapshot, "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dependencies": base._versions(), "launch_blockers": blockers,
        "manifest_path": str(path),
        "groups": [{
            "group": group, "config_path": str(source.resolve()),
            "config_sha256": experiment.config_sha256,
            "artifact_root": experiment.artifact_root, "status": "pending",
            "seed": 7, "environment_steps": 900000,
            "astar_schedule_steps": 600000, "astar_stop_steps": GROUPS[group],
            "evaluation_instances": 10,
        } for group, source, experiment in reviewed],
    }
    base._write_manifest(path, manifest)
    if not run:
        return manifest
    if blockers:
        manifest["status"] = "blocked"
        base._write_manifest(path, manifest)
        raise ValueError(" ".join(blockers))
    lock = destination / ".astar-tail.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(path))
    try:
        _parallel(reviewed, manifest, path, root, device, snapshot)
    finally:
        lock.unlink()
    return manifest


def _interrupt_worker(signum, frame) -> None:
    """Let runner.train() close vector environments when the driver terminates us."""
    raise KeyboardInterrupt("Diagnostic worker received termination.")


def _worker(arguments) -> None:
    base._verify_imports(WORKTREE_ROOT)
    source = arguments.config_directory / f"{arguments.group}.yaml"
    experiment = load_tail_experiment(source, arguments.group, root=WORKTREE_ROOT)
    snapshot = base._git_snapshot(WORKTREE_ROOT)
    if (snapshot["code_commit"] != arguments.expected_commit
            or experiment.config_sha256 != arguments.expected_config_sha):
        raise ValueError("Worker commit or config identity differs from batch review.")
    reviewed = [(arguments.group, source, experiment)]
    blockers = _blockers(reviewed, WORKTREE_ROOT, snapshot, arguments.device)
    if blockers:
        raise ValueError(" ".join(blockers))
    signal.signal(signal.SIGTERM, _interrupt_worker)
    learner = AStarKDRun(experiment, device=arguments.device, diagnostic_authorized=True)
    learner.train()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-directory", type=Path,
                        default=Path("configs/astar_tail_diagnostics/medium-5ag"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--group", choices=list(GROUPS), help=argparse.SUPPRESS)
    parser.add_argument("--worker-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-commit", help=argparse.SUPPRESS)
    parser.add_argument("--expected-config-sha", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    signal.signal(signal.SIGTERM, _interrupt_worker)
    if arguments.worker_run:
        _worker(arguments)
        return
    print(json.dumps(launch(arguments.config_directory, device=arguments.device,
                            run=arguments.run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
