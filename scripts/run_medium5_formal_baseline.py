"""Audit the approved medium baseline batch; use --run to train serially."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import llm_mappo
import rware
from llm_mappo.formal_training import EVALUATION_SEEDS, SCHEMA, TRAIN_SEEDS
from llm_mappo.mappo_baseline import runner
from llm_mappo.mappo_baseline.fp_state import FPStateSchema
from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment
from llm_mappo.mappo_baseline.structured_observation import StructuredObservationSchema


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = "artifacts/formal_environment_training/medium-5ag/baseline"
CODE_PREFIXES = ("llm_mappo/", "rware/", "train/", "eval/", "scripts/", "configs/")
CODE_FILES = {"pyproject.toml", "setup.py", "setup.cfg", ".gitignore"}


def _git_snapshot(root: Path) -> dict:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    entries = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
    ).decode("utf-8").split("\0")
    paths = []
    index = 0
    while index < len(entries) and entries[index]:
        entry = entries[index]
        paths.append(entry[3:])
        index += 1
        if "R" in entry[:2] or "C" in entry[:2]:
            paths.append(entries[index])
            index += 1
    code_paths = [
        path for path in paths if path.startswith(CODE_PREFIXES)
        or path in CODE_FILES or path.startswith("requirements")
    ]
    return {"code_commit": commit, "dirty": bool(paths), "dirty_paths": paths,
            "dirty_training_paths": code_paths}


def _verify_imports(root: Path) -> None:
    if Path.cwd().resolve() != root:
        raise ValueError("Run from this worktree root with python -m scripts.")
    for module in (llm_mappo, rware, runner):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError(f"Cross-worktree import detected: {module.__name__}")


def review_configs(config_directory: Path, seeds: list[int], *, root: Path) -> list:
    """Validate all requested configs before opening any training run."""
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(TRAIN_SEEDS):
        raise ValueError("Use a unique nonempty subset of seeds 7,17,...,77.")
    directory = config_directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Formal configs must be inside the current worktree.")
    reviewed = []
    contract = None
    for seed in seeds:
        source = directory / f"seed_{seed:04d}.yaml"
        experiment = load_baseline_experiment(source)
        if (experiment.schema != SCHEMA or experiment.approval_status != "approved"
                or experiment.method != "MAPPO-FP-DG"
                or experiment.environment.get("scene_id") != "medium-5ag"
                or experiment.artifact_root != ARTIFACT_ROOT):
            raise ValueError(f"Not an approved formal medium baseline: {source}")
        training = experiment.training
        if (training["seed"] != seed or training["environment_steps"] != 900_000
                or training["optimization_schedule_steps"] != 900_000
                or training["learner_torch_threads"] != 4
                or training["worker_numerical_threads"] != 1):
            raise ValueError(f"Formal medium training contract differs: {source}")
        evaluation = experiment.evaluation
        if (tuple(evaluation["seeds"]) != EVALUATION_SEEDS
                or evaluation["replenishment_mode"] != "rware_delivery"
                or evaluation["deterministic_policy"] != "argmax"
                or evaluation.get("environment_overrides") != {}
                or evaluation.get("render_replay") is not False):
            raise ValueError(f"Formal medium evaluation contract differs: {source}")
        seed_training = {key: value for key, value in training.items() if key != "seed"}
        signature = json.dumps({
            "environment": dict(experiment.environment), "training": seed_training,
            "evaluation": dict(evaluation),
        }, sort_keys=True)
        if contract is not None and signature != contract:
            raise ValueError("Requested seed configs differ beyond the training seed.")
        contract = signature
        existing = sorted((root / ARTIFACT_ROOT).glob(f"seed_{seed:04d}_*"))
        reviewed.append((source, experiment, existing))
    return reviewed


def _write_manifest(path: Path, values: dict) -> None:
    path.write_text(
        json.dumps(values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )


def _versions() -> dict:
    versions = {"python": sys.version, "torch": torch.__version__}
    for package in ("numpy", "PyYAML", "gymnasium", "tensorboard"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _launch_blockers(reviewed: list, snapshot: dict, device: torch.device) -> list:
    blockers = []
    if snapshot["dirty_training_paths"]:
        blockers.append("Commit training code/config changes before --run.")
    for _, experiment, existing in reviewed:
        if existing:
            blockers.append(
                f"Seed {experiment.training['seed']} already has a run: "
                + ", ".join(str(path) for path in existing)
                + "; select only remaining seeds."
            )
    cuda_available = torch.cuda.is_available()
    if device.type == "cuda" and (
        not cuda_available or (device.index is not None
                               and device.index >= torch.cuda.device_count())
    ):
        blockers.append("Requested CUDA device is unavailable; no training started.")
    return blockers


def _run_reviewed_batch(reviewed: list, manifest: dict, manifest_path: Path,
                        root: Path, device: str, snapshot: dict) -> None:
    manifest["status"] = "running"
    _write_manifest(manifest_path, manifest)
    for row, (source, experiment, _) in zip(manifest["runs"], reviewed):
        current = _git_snapshot(root)
        if (current["code_commit"] != snapshot["code_commit"]
                or current["dirty_training_paths"]
                or sha256(source.read_bytes()).hexdigest() != row["config_sha256"]):
            raise ValueError("Training code/config changed after batch review.")
        if list((root / ARTIFACT_ROOT).glob(f"seed_{row['seed']:04d}_*")):
            raise FileExistsError(f"Seed {row['seed']} acquired another run.")
        baseline = BaselineRun(experiment, device=device)
        row.update({"status": "running",
                    "run_directory": str(baseline.directory.resolve())})
        _write_manifest(manifest_path, manifest)
        baseline.train()
        row["status"] = "complete"
        _write_manifest(manifest_path, manifest)
    manifest["status"] = "complete"
    _write_manifest(manifest_path, manifest)


def launch(config_directory: Path, seeds: list[int], *, device: str,
           run: bool, root: Path = WORKTREE_ROOT) -> dict:
    """Dry-run never constructs a learner or consumes reserved evaluation instances."""
    _verify_imports(root)
    reviewed = review_configs(config_directory, seeds, root=root)
    snapshot = _git_snapshot(root)
    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda for this training batch.")
    blockers = _launch_blockers(reviewed, snapshot, torch_device)
    actor_dimension = StructuredObservationSchema(
        observation_schema="direct-goal-observation-v2"
    ).dimension
    fp_dimension = FPStateSchema(
        5, n_shelves=144, n_chargers=5, n_picking_stations=2,
    ).dimension
    manifest = {
        "schema": "medium5-formal-baseline-launch-v1",
        "status": "dry_run", "worktree": str(root), **snapshot,
        "device": device, "cuda_available": torch.cuda.is_available(),
        "actor_observation_dimension": actor_dimension,
        "fp_state_dimension": fp_dimension,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dependencies": _versions(), "launch_blockers": blockers,
        "runs": [{
            "seed": experiment.training["seed"],
            "config_path": str(source), "config_sha256": experiment.config_sha256,
            "environment_steps": 900_000, "evaluation_instances": 200,
            "status": "pending",
        } for source, experiment, _ in reviewed],
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "artifacts/formal_environment_training/medium-5ag/launches"
    directory = destination / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory / "launch_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    _write_manifest(manifest_path, manifest)
    if not run:
        return manifest
    if blockers:
        manifest["status"] = "blocked"
        _write_manifest(manifest_path, manifest)
        raise ValueError(" ".join(blockers) + f" Manifest: {manifest_path}")
    lock = destination / ".medium5-baseline.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(manifest_path))
    except FileExistsError as error:
        manifest.update({"status": "blocked", "error": "Batch lock already exists."})
        _write_manifest(manifest_path, manifest)
        raise FileExistsError(
            f"Another medium baseline batch owns {lock}; inspect it before retrying."
        ) from error
    try:
        _run_reviewed_batch(reviewed, manifest, manifest_path, root, device, snapshot)
    except BaseException as error:
        manifest.update({"status": "failed", "error": str(error)})
        for row in manifest["runs"]:
            if row["status"] == "running":
                row["status"] = "failed"
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        lock.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-directory", type=Path,
                        default=Path("configs/training_v1/medium-5ag-formal"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(launch(
        arguments.config_directory, arguments.seeds,
        device=arguments.device, run=arguments.run,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
