"""Launch A*KD seed 7 first; remaining seeds require explicit completed-run review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import torch

from scripts import run_medium5_formal_baseline as base
from llm_mappo.formal_training import EVALUATION_SEEDS, SCHEMA, TRAIN_SEEDS
from llm_mappo.mappo_astar_kd.runner import AStarKDRun, load_astar_kd_experiment
from llm_mappo.mappo_astar_kd.teacher import TEACHER_VERSION
from llm_mappo.mappo_baseline.fp_state import FPStateSchema
from llm_mappo.mappo_baseline.structured_observation import StructuredObservationSchema


WORKTREE_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = "artifacts/formal_environment_training/medium-5ag/astar-kd"
def _validate_seed_batch(seeds: list[int]) -> None:
    if 7 in seeds and seeds != [7]:
        raise ValueError("Run seed 7 separately before the remaining seeds.")


def review_configs(config_directory: Path, seeds: list[int], *, root: Path) -> list:
    """Validate all requested configs before opening any training run."""
    _validate_seed_batch(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(TRAIN_SEEDS):
        raise ValueError("Use a unique nonempty subset of seeds 7,17,...,77.")
    directory = config_directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError("Formal configs must be inside the current worktree.")
    reviewed = []
    contract = None
    for seed in seeds:
        source = directory / f"seed_{seed:04d}.yaml"
        experiment = load_astar_kd_experiment(source)
        if (experiment.schema != SCHEMA or experiment.approval_status != "approved"
                or experiment.method != "MAPPO-FP-ASTARKD-NO-VALIDITY"
                or experiment.environment.get("scene_id") != "medium-5ag"
                or experiment.artifact_root != ARTIFACT_ROOT):
            raise ValueError(f"Not an approved formal medium A*KD: {source}")
        training = experiment.training
        if (training["seed"] != seed or training["environment_steps"] != 900_000
                or training["optimization_schedule_steps"] != 900_000
                or training["learner_torch_threads"] != 4
                or training["worker_numerical_threads"] != 1):
            raise ValueError(f"Formal medium training contract differs: {source}")
        baseline_source = root / "configs/training_v1/medium-5ag-formal" / source.name
        baseline = base.review_configs(baseline_source.parent, [seed], root=root)[0][1]
        method_fields = {
            "astar_teacher_approval", "astar_teacher_version",
            "astar_schedule_steps", "post_schedule_astar_lambda",
        }
        common_training = {
            key: value for key, value in training.items() if key not in method_fields
        }
        if (dict(experiment.environment) != dict(baseline.environment)
                or dict(experiment.evaluation) != dict(baseline.evaluation)
                or common_training != dict(baseline.training)):
            raise ValueError("A*KD differs from paired baseline common configuration.")
        if (training.get("astar_teacher_approval") != "approved"
                or training.get("astar_teacher_version") != TEACHER_VERSION
                or training.get("astar_schedule_steps") != 600_000
                or training.get("post_schedule_astar_lambda") != 0.0):
            raise ValueError("Require approved A*KD 600k followed by 300k pure PPO.")
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


def _finite_json(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return True


def _seed7_review(directory: Path, root: Path, commit: str,
                  reviewed_run: Path | None, confirmed: bool) -> dict:
    if not confirmed or reviewed_run is None:
        raise ValueError("Review completed seed 7 before remaining seeds; provide "
                         "--seed7-reviewed-run and --confirm-seed7-review.")
    path = reviewed_run.resolve()
    expected_root = (root / ARTIFACT_ROOT).resolve()
    if path.parent != expected_root or not path.name.startswith("seed_0007_"):
        raise ValueError("Reviewed seed 7 run must belong to this A*KD artifact root.")
    experiment = review_configs(directory, [7], root=root)[0][1]
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (path / "evaluation/evaluation.json").read_text(encoding="utf-8")
    )
    identity = manifest.get("identity", {})
    if (state.get("status") != "complete"
            or identity.get("seed") != 7
            or identity.get("schema") != SCHEMA
            or identity.get("method") != "MAPPO-FP-ASTARKD-NO-VALIDITY"
            or identity.get("config_sha256") != experiment.config_sha256
            or identity.get("code_commit") != commit
            or summary.get("identity") != identity
            or summary.get("global_environment_steps") != 900_000
            or summary.get("updates") != 879
            or summary.get("evaluation") != evaluation
            or tuple(row.get("seed") for row in evaluation.get("episodes", []))
            != EVALUATION_SEEDS
            or not _finite_json(summary)
            or manifest.get("environment") != dict(experiment.environment)
            or manifest.get("training") != dict(experiment.training)
            or manifest.get("evaluation") != dict(experiment.evaluation)):
        raise ValueError("Reviewed seed 7 evidence does not match completed contract.")
    checkpoint = path / "checkpoint_final.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError("Reviewed seed 7 final checkpoint is missing.")
    return {
        "owner_confirmed_review": True, "run_directory": str(path),
        "identity": identity,
        "final_checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "summary_sha256": sha256((path / "summary.json").read_bytes()).hexdigest(),
    }


def _run_reviewed_batch(reviewed: list, manifest: dict, manifest_path: Path,
                        root: Path, device: str, snapshot: dict) -> None:
    manifest["status"] = "running"
    base._write_manifest(manifest_path, manifest)
    for row, (source, experiment, _) in zip(manifest["runs"], reviewed):
        current = base._git_snapshot(root)
        if (current["code_commit"] != snapshot["code_commit"]
                or current["dirty_training_paths"]
                or sha256(source.read_bytes()).hexdigest() != row["config_sha256"]):
            raise ValueError("Training code/config changed after batch review.")
        if list((root / ARTIFACT_ROOT).glob(f"seed_{row['seed']:04d}_*")):
            raise FileExistsError(f"Seed {row['seed']} acquired another run.")
        baseline = AStarKDRun(experiment, device=device)
        row.update({"status": "running",
                    "run_directory": str(baseline.directory.resolve())})
        base._write_manifest(manifest_path, manifest)
        baseline.train()
        row["status"] = "complete"
        base._write_manifest(manifest_path, manifest)
    manifest["status"] = "complete"
    base._write_manifest(manifest_path, manifest)


def _review_gate(seeds: list[int], config_directory: Path, root: Path,
                 commit: str, reviewed_run: Path | None,
                 confirmed: bool) -> tuple[dict | None, list[str]]:
    if seeds == [7]:
        return None, []
    try:
        return _seed7_review(
            config_directory, root, commit, reviewed_run, confirmed,
        ), []
    except (ValueError, OSError, KeyError, TypeError) as error:
        return None, [str(error)]


def launch(config_directory: Path, seeds: list[int], *, device: str,
           run: bool, root: Path = WORKTREE_ROOT,
           seed7_reviewed_run: Path | None = None,
           confirm_seed7_review: bool = False) -> dict:
    """Dry-run never constructs a learner or consumes reserved evaluation instances."""
    base._verify_imports(root)
    reviewed = review_configs(config_directory, seeds, root=root)
    snapshot = base._git_snapshot(root)
    torch_device = torch.device(device)
    if torch_device.type not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda for this training batch.")
    blockers = base._launch_blockers(reviewed, snapshot, torch_device)
    review_attestation, review_blockers = _review_gate(
        seeds, config_directory, root, snapshot["code_commit"],
        seed7_reviewed_run, confirm_seed7_review,
    )
    blockers.extend(review_blockers)
    actor_dimension = StructuredObservationSchema(
        observation_schema="direct-goal-observation-v2"
    ).dimension
    fp_dimension = FPStateSchema(
        5, n_shelves=144, n_chargers=5, n_picking_stations=2,
    ).dimension
    manifest = {
        "schema": "medium5-formal-astar-kd-launch-v1",
        "status": "dry_run", "worktree": str(root), **snapshot,
        "device": device, "cuda_available": torch.cuda.is_available(),
        "actor_observation_dimension": actor_dimension,
        "fp_state_dimension": fp_dimension,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dependencies": base._versions(), "launch_blockers": blockers,
        "astar_schedule_steps": 600_000, "pure_ppo_tail_steps": 300_000,
        "seed7_review_attestation": review_attestation,
        "runs": [{
            "seed": experiment.training["seed"],
            "config_path": str(source), "config_sha256": experiment.config_sha256,
            "environment_steps": 900_000, "evaluation_instances": 200,
            "status": "pending",
        } for source, experiment, _ in reviewed],
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (root / "artifacts/formal_environment_training/medium-5ag"
                   / "astar-kd-launches")
    directory = destination / timestamp
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory / "launch_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    base._write_manifest(manifest_path, manifest)
    if not run:
        return manifest
    if blockers:
        manifest["status"] = "blocked"
        base._write_manifest(manifest_path, manifest)
        raise ValueError(" ".join(blockers) + f" Manifest: {manifest_path}")
    lock = destination / ".medium5-astar-kd.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(str(manifest_path))
    except FileExistsError as error:
        manifest.update({"status": "blocked", "error": "Batch lock already exists."})
        base._write_manifest(manifest_path, manifest)
        raise FileExistsError(
            f"Another medium A*KD batch owns {lock}; inspect it before retrying."
        ) from error
    try:
        _run_reviewed_batch(reviewed, manifest, manifest_path, root, device, snapshot)
    except BaseException as error:
        manifest.update({"status": "failed", "error": str(error)})
        for row in manifest["runs"]:
            if row["status"] == "running":
                row["status"] = "failed"
        base._write_manifest(manifest_path, manifest)
        raise
    finally:
        lock.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-directory", type=Path,
                        default=Path("configs/training_v1/medium-5ag-formal-astar-kd"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--seed7-reviewed-run", type=Path)
    parser.add_argument("--confirm-seed7-review", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(launch(
        arguments.config_directory, arguments.seeds,
        device=arguments.device, run=arguments.run,
        seed7_reviewed_run=arguments.seed7_reviewed_run,
        confirm_seed7_review=arguments.confirm_seed7_review,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
