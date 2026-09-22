"""Runnable, resumable construction of the main native-209D dataset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from llm_mappo.mappo_baseline.runner import _validate_environment
from llm_mappo.mappo_baseline.vector_env import new_baseline_environment

from .collector import collect_complete_episode, load_phase3_collection_policy
from .dataset import (
    MAIN_BASELINE_DISTRIBUTION,
    DistributionTrajectory,
    build_partition_windows,
    split_complete_episodes,
)
from .trajectory import ActorObservationTrajectory


DATASET_BUILD_RUN_V1 = "llm-dataset-build-run-v1"


@dataclass(frozen=True)
class DatasetBuildConfig:
    """Fully resolved contract for one main-distribution collection run."""

    schema: str
    artifact_root: str
    source: Mapping[str, object]
    environment: Mapping[str, object]
    collection: Mapping[str, object]
    partition: Mapping[str, object]
    claims: Mapping[str, object]
    config_sha256: str


def load_dataset_build_config(path: str | Path) -> DatasetBuildConfig:
    """Load one explicit, hash-bound native-209D dataset-build contract."""

    source = Path(path)
    raw = source.read_bytes()
    values = yaml.safe_load(raw)
    expected = {
        "schema", "artifact_root", "source", "environment", "collection",
        "partition", "claims",
    }
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("Dataset build config has unexpected top-level keys.")
    config = DatasetBuildConfig(
        schema=values["schema"],
        artifact_root=values["artifact_root"],
        source=values["source"],
        environment=values["environment"],
        collection=values["collection"],
        partition=values["partition"],
        claims=values["claims"],
        config_sha256=sha256(raw).hexdigest(),
    )
    _validate_config(config)
    _validate_environment(config.environment)
    return config


def run_dataset_build(
    config: DatasetBuildConfig,
    *,
    source_checkpoint: str | Path,
    device: str,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    resume: str | Path | None = None,
    policy_loader=load_phase3_collection_policy,
    environment_factory=new_baseline_environment,
) -> dict[str, object]:
    """Collect only complete native-209D episodes and persist split artifacts."""

    _validate_config(config)
    directory = _prepare_run_directory(config, output_root, run_id, resume)
    manifest_path = directory / "manifest.json"
    existing_records = _load_records(directory / "trajectories.jsonl")
    _validate_existing_records(existing_records)
    policy = policy_loader(
        source_checkpoint,
        device=device,
        expected_artifact_sha256=str(config.source["checkpoint_sha256"]),
    )
    environment_digest = _canonical_sha256(dict(config.environment))
    planned = _planned_episode_ids(config.collection)
    completed = {item.trajectory.episode_id for item in existing_records}
    records = list(existing_records)
    try:
        for episode_id, seed in planned:
            if episode_id in completed:
                continue
            environment = environment_factory(
                config.environment,
                replenishment_mode=config.environment.get(
                    "training_replenishment_mode", "dynamic_ingress"
                ),
            )
            try:
                records.extend(
                    collect_complete_episode(
                        environment,
                        policy,
                        seed=seed,
                        episode_id=episode_id,
                        environment_sha256=environment_digest,
                        config_sha256=config.config_sha256,
                        max_steps=int(config.collection["max_steps"]),
                    )
                )
                _atomic_text(directory / "trajectories.jsonl", _records_text(records))
                completed.add(episode_id)
            finally:
                close = getattr(environment, "close", None)
                if close is not None:
                    close()
        partitions = split_complete_episodes(
            records, config.partition["split_ratios"], salt=str(config.partition["salt"])
        )
        windows = build_partition_windows(
            partitions, history_length=int(config.collection["history_length"])
        )
        partition_payload = {
            name: sorted({item.trajectory.episode_id for item in values})
            for name, values in partitions.items()
        }
        _atomic_json(directory / "episode_partitions.json", partition_payload)
        window_text = _windows_text(windows)
        _atomic_text(directory / "windows.jsonl", window_text)
        trajectory_text = _records_text(records)
        manifest = {
            "schema": DATASET_BUILD_RUN_V1,
            "distribution": MAIN_BASELINE_DISTRIBUTION,
            "config_sha256": config.config_sha256,
            "environment_sha256": environment_digest,
            "source_checkpoint_sha256": config.source["checkpoint_sha256"],
            "behavior_policy_id": policy.behavior_policy_id,
            "observation_schema": "direct-goal-observation-v2",
            "observation_dimension": 209,
            "episode_count": len(completed),
            "trajectory_count": len(records),
            "window_counts": {name: len(values) for name, values in windows.items()},
            "trajectories_sha256": sha256(trajectory_text.encode("utf-8")).hexdigest(),
            "windows_sha256": sha256(window_text.encode("utf-8")).hexdigest(),
            "main_pretraining_eligible": True,
            "formal_performance_evidence": False,
        }
        _atomic_json(manifest_path, manifest)
        _atomic_json(directory / "state.json", {"status": "completed"})
    except Exception as error:
        _atomic_json(
            directory / "state.json",
            {"status": "failed", "error_type": type(error).__name__},
        )
        raise
    return {"run_directory": str(directory.resolve()), "manifest": manifest}


def _validate_config(config: DatasetBuildConfig) -> None:
    if (
        not isinstance(config, DatasetBuildConfig)
        or config.schema != DATASET_BUILD_RUN_V1
    ):
        raise ValueError("Dataset build schema differs.")
    if not isinstance(config.artifact_root, str) or not config.artifact_root:
        raise ValueError("Dataset artifact root is invalid.")
    _validate_digest(config.config_sha256, "Dataset config SHA-256")
    expected_source = {
        "checkpoint_sha256",
        "policy_observation_schema",
        "policy_observation_dimension",
    }
    if set(config.source) != expected_source:
        raise ValueError("Dataset source contract differs.")
    _validate_digest(
        str(config.source["checkpoint_sha256"]), "Dataset source SHA-256"
    )
    if (
        config.source["policy_observation_schema"] != "direct-goal-observation-v2"
        or config.source["policy_observation_dimension"] != 209
    ):
        raise ValueError("Dataset source must be native 209D v2.")
    if dict(config.environment).get("n_agents") != 5:
        raise ValueError("Dataset collection requires exactly five AGVs.")
    _validate_collection(config.collection)
    _validate_partition(config.partition)
    expected_claims = {
        "distribution_role": "main_native_209d_collection",
        "main_pretraining_eligible": True,
        "formal_performance_evidence": False,
    }
    if dict(config.claims) != expected_claims:
        raise ValueError("Dataset claim boundary differs.")


def _validate_collection(collection: Mapping[str, object]) -> None:
    expected = {"seeds", "episodes_per_seed", "max_steps", "history_length"}
    if set(collection) != expected:
        raise ValueError("Dataset collection contract differs.")
    seeds = collection["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Dataset collection seeds are invalid.")
    for field in ("episodes_per_seed", "max_steps", "history_length"):
        if type(collection[field]) is not int or int(collection[field]) < 1:
            raise ValueError(f"Dataset collection {field} is invalid.")


def _validate_partition(partition: Mapping[str, object]) -> None:
    if set(partition) != {"split_ratios", "salt"}:
        raise ValueError("Dataset partition contract differs.")
    ratios = partition["split_ratios"]
    if (
        not isinstance(ratios, dict)
        or set(ratios) != {"train", "validation", "test"}
        or any(type(value) is not float or value <= 0 for value in ratios.values())
        or not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("Dataset partition ratios are invalid.")
    if not isinstance(partition["salt"], str) or not partition["salt"]:
        raise ValueError("Dataset partition salt is invalid.")


def _prepare_run_directory(config, output_root, run_id, resume) -> Path:
    if resume is not None:
        directory = Path(resume)
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("schema") != DATASET_BUILD_RUN_V1
            or manifest.get("config_sha256") != config.config_sha256
            or manifest.get("source_checkpoint_sha256")
            != config.source["checkpoint_sha256"]
        ):
            raise ValueError("Dataset build resume identity differs.")
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        if state.get("status") not in {"running", "failed"}:
            raise ValueError("Only an interrupted dataset build can resume.")
        _atomic_json(directory / "state.json", {"status": "running"})
        return directory
    identifier = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not isinstance(identifier, str) or not identifier or any(
        character not in (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        )
        for character in identifier
    ):
        raise ValueError("Dataset run_id contains unsafe characters.")
    directory = Path(output_root or config.artifact_root) / identifier
    directory.mkdir(parents=True, exist_ok=False)
    _atomic_json(directory / "manifest.json", {
        "schema": DATASET_BUILD_RUN_V1,
        "config_sha256": config.config_sha256,
        "source_checkpoint_sha256": config.source["checkpoint_sha256"],
    })
    _atomic_json(directory / "state.json", {"status": "running"})
    return directory


def _planned_episode_ids(collection: Mapping[str, object]) -> list[tuple[str, int]]:
    planned = []
    for seed in collection["seeds"]:
        for index in range(int(collection["episodes_per_seed"])):
            planned.append(
                (f"collection-seed-{seed}-episode-{index}", int(seed) + index)
            )
    return planned


def _load_records(path: Path) -> list[DistributionTrajectory]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict) or set(value) != {
            "distribution", "trajectory"
        }:
            raise ValueError("Dataset trajectory record is malformed.")
        trajectory = value["trajectory"]
        if not isinstance(trajectory, dict):
            raise ValueError("Dataset trajectory payload is malformed.")
        records.append(
            DistributionTrajectory(
                value["distribution"],
                ActorObservationTrajectory(
                    episode_id=trajectory["episode_id"],
                    agent_id=trajectory["agent_id"],
                    timesteps=np.asarray(trajectory["timesteps"], dtype=np.int64),
                    observations=np.asarray(
                        trajectory["observations"], dtype=np.float32
                    ),
                    reset_masks=np.asarray(
                        trajectory["reset_masks"], dtype=np.bool_
                    ),
                    behavior_policy_id=trajectory["behavior_policy_id"],
                    environment_sha256=trajectory["environment_sha256"],
                    config_sha256=trajectory["config_sha256"],
                    observation_schema=trajectory["observation_schema"],
                    schema_version=trajectory["schema_version"],
                ),
            )
        )
    return records


def _validate_existing_records(records: list[DistributionTrajectory]) -> None:
    if not records:
        return
    split_complete_episodes(
        records,
        {"train": 0.34, "validation": 0.33, "test": 0.33},
        salt="validation",
    )


def _records_text(records: list[DistributionTrajectory]) -> str:
    return "".join(
        json.dumps(
            {
                "distribution": item.distribution,
                "trajectory": item.trajectory.to_record(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        for item in records
    )


def _windows_text(windows) -> str:
    return "".join(
        json.dumps(
            {
                "partition": partition,
                "episode_id": window.episode_id,
                "agent_id": window.agent_id,
                "start_timestep": window.start_timestep,
                "end_timestep": window.end_timestep,
                "observations": window.observations.tolist(),
                "reset_masks": window.reset_masks.tolist(),
                "stratum": window.stratum,
            },
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        for partition in ("train", "validation", "test")
        for window in windows[partition]
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is invalid.")


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
