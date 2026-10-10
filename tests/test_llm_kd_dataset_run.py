"""Runnable main-distribution dataset-build contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import yaml

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_baseline.runner import load_baseline_experiment
from llm_mappo.mappo_llm_kd.dataset_run import (
    DATASET_BUILD_RUN_V1,
    DatasetBuildConfig,
    load_dataset_build_config,
    run_dataset_build,
)


class _Policy:
    hidden_dim = 4
    behavior_policy_id = "MAPPO-FP-DG:" + "c" * 64

    def start_episode(self, seed):
        self.seed = seed

    def act(self, observations, action_masks, hidden):
        return np.zeros(5, dtype=np.int64), hidden


class _Environment:
    n_agents = 5

    def __init__(self):
        self.steps = 0

    def reset(self, *, seed):
        self.seed = seed
        self.steps = 0

    def action_masks(self):
        return np.ones((5, 5), dtype=np.bool_)

    def step(self, actions):
        self.steps += 1
        return SimpleNamespace(
            terminated=False,
            truncated=self.steps == 3,
            metrics=SimpleNamespace(deadlocked=False),
        )

    def close(self):
        pass


def _config(tmp_path):
    return DatasetBuildConfig(
        schema=DATASET_BUILD_RUN_V1,
        artifact_root=str(tmp_path),
        source={
            "checkpoint_sha256": "a" * 64,
            "policy_observation_schema": "direct-goal-observation-v2",
            "policy_observation_dimension": 209,
        },
        environment={"n_agents": 5},
        collection={
            "seeds": [11, 12, 13],
            "episodes_per_seed": 1,
            "max_steps": 3,
            "history_length": 2,
        },
        partition={
            "split_ratios": {"train": 0.34, "validation": 0.33, "test": 0.33},
            "salt": "core-integration-test",
        },
        claims={
            "distribution_role": "main_native_209d_collection",
            "main_pretraining_eligible": True,
            "formal_performance_evidence": False,
        },
        config_sha256="b" * 64,
    )


def test_dataset_build_writes_complete_native_209d_episode_partitions(
    monkeypatch, tmp_path
):
    schema = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    )
    index = {name: position for position, name in enumerate(schema.feature_names)}

    def observations(environment, schema, **kwargs):
        assert schema.dimension == 209
        rows = np.zeros((5, 209), dtype=np.float32)
        rows[:, index["self_orientation_up"]] = 1.0
        rows[:, index["self_phase_task"]] = 1.0
        rows[:, index["self_previous_action_0"]] = 1.0
        rows[:, index["goal_type_task"]] = 1.0
        rows[:, index["self_priority_weight"]] = 0.5
        rows[:, index["goal_normalized_distance"]] = 0.5
        return rows

    monkeypatch.setattr(
        "llm_mappo.mappo_llm_kd.collector.build_structured_observations",
        observations,
    )

    result = run_dataset_build(
        _config(tmp_path),
        source_checkpoint=tmp_path / "collection_actor_final.pt",
        device="cpu",
        run_id="dataset-smoke",
        policy_loader=lambda *_args, **_kwargs: _Policy(),
        environment_factory=lambda _values, **_kwargs: _Environment(),
    )

    directory = tmp_path / "dataset-smoke"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (directory / "trajectories.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    partitions = json.loads(
        (directory / "episode_partitions.json").read_text(encoding="utf-8")
    )

    assert result["run_directory"] == str(directory.resolve())
    assert json.loads((directory / "state.json").read_text(encoding="utf-8")) == {
        "status": "completed"
    }
    assert manifest["distribution"] == "native-209d-collection-main-v1"
    assert manifest["trajectory_count"] == 15
    assert manifest["episode_count"] == 3
    assert len(records) == 15
    assert {record["distribution"] for record in records} == {
        "native-209d-collection-main-v1"
    }
    assert {len(record["trajectory"]["observations"][0]) for record in records} == {209}
    assert sorted(partitions) == ["test", "train", "validation"]
    assert all(len(values) == 1 for values in partitions.values())


def test_dataset_build_config_is_hash_bound_and_cli_exposes_the_run_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        load_baseline_experiment(
            root / "configs" / "llm_kd"
            / "medium-5ag-low-workload-collection-baseline.yaml"
        ).environment
    )
    values = {
        "schema": DATASET_BUILD_RUN_V1,
        "artifact_root": str(tmp_path / "artifacts"),
        "source": {
            "checkpoint_sha256": "a" * 64,
            "policy_observation_schema": "direct-goal-observation-v2",
            "policy_observation_dimension": 209,
        },
        "environment": environment,
        "collection": {
            "seeds": [11, 12, 13],
            "episodes_per_seed": 1,
            "max_steps": 3,
            "history_length": 2,
        },
        "partition": {
            "split_ratios": {"train": 0.34, "validation": 0.33, "test": 0.33},
            "salt": "core-integration-test",
        },
        "claims": {
            "distribution_role": "main_native_209d_collection",
            "main_pretraining_eligible": True,
            "formal_performance_evidence": False,
        },
    }
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    loaded = load_dataset_build_config(path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_llm_kd_dataset_build",
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert loaded.schema == DATASET_BUILD_RUN_V1
    assert loaded.config_sha256
    assert "--source-checkpoint" in completed.stdout
    assert completed.returncode == 0
