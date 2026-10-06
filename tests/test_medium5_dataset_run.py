"""Bounded medium C1 dataset selection from complete episode artifacts."""

from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.medium_dataset_run import (
    _json_sha256, run_medium5_episode_collection, select_medium5_windows,
)


def test_medium5_selection_is_bounded_disjoint_and_reproducible(tmp_path):
    identity = {
        "schema": "medium5-c1-dataset-collection-v1",
        "environment_sha256": "a" * 64,
        "source_checkpoint_sha256": "b" * 64,
        "actor_sha256": "c" * 64,
        "baseline_config_sha256": "d" * 64,
        "actor_identity": {
            "code_commit": "e" * 40,
            "reward_definition_sha256": "f" * 64,
        },
        "behavior_policy_id": "MAPPO-FP-DG:" + "c" * 64,
    }
    (tmp_path / "collection_identity.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    schema = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    )
    index = {name: number for number, name in enumerate(schema.feature_names)}
    for seed in range(10000, 10007):
        observations = np.zeros((5, 16, 209), dtype=np.float32)
        observations[:, :, index["self_orientation_up"]] = 1
        observations[:, :, index["self_phase_task"]] = 1
        observations[:, :, index["self_previous_action_0"]] = 1
        observations[:, :, index["goal_type_task"]] = 1
        observations[:, :, index["self_priority_weight"]] = 0.5
        for agent in range(5):
            observations[agent, :, index["goal_normalized_distance"]] = (
                (seed - 10000) * 0.1 + agent * 0.01
                + np.arange(16) * 0.0001
            )
        reset_masks = np.zeros((5, 16), dtype=np.bool_)
        reset_masks[:, 0] = True
        path = episodes / f"seed_{seed:05d}.npz"
        np.savez_compressed(
            path, observations=observations, reset_masks=reset_masks
        )
        receipt = {
            "schema": "medium5-c1-episode-v1", "seed": seed,
            "episode_id": f"collection-seed-{seed}-episode-0",
            "steps": 16,
            "npz_sha256": sha256(path.read_bytes()).hexdigest(),
            "collection_identity_sha256": _json_sha256(identity),
        }
        (episodes / f"seed_{seed:05d}.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
    result = select_medium5_windows(
        tmp_path, minimum_episodes=7,
        budgets={"train": 20, "validation": 5, "test": 5},
        minimum_coverage={"train": 0, "validation": 0, "test": 0},
    )
    manifest = result["manifest"]
    windows = [json.loads(line) for line in (
        tmp_path / "windows.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert manifest["window_counts"] == {
        "train": 20, "validation": 5, "test": 5,
    }
    assert manifest["main_pretraining_eligible"] is False
    assert len(windows) == 30
    assert all(len(item["observations"]) == 8 for item in windows)
    assert len({
        (item["episode_id"], item["agent_id"], item["end_timestep"])
        for item in windows
    }) == 30
    partitions = json.loads((
        tmp_path / "episode_partitions.json"
    ).read_text(encoding="utf-8"))
    assert len(set().union(*map(set, partitions.values()))) == 7
    assert not set(partitions["train"]) & set(partitions["test"])
    first_sha = manifest["windows_sha256"]
    again = select_medium5_windows(
        tmp_path, minimum_episodes=7,
        budgets={"train": 20, "validation": 5, "test": 5},
        minimum_coverage={"train": 0, "validation": 0, "test": 0},
    )
    assert again["manifest"]["windows_sha256"] == first_sha
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be reselected"):
        select_medium5_windows(
            tmp_path, minimum_episodes=7,
            budgets={"train": 20, "validation": 5, "test": 5},
            minimum_coverage={"train": 0, "validation": 0, "test": 0},
        )


def test_labeled_dataset_refuses_episode_extension(tmp_path):
    directory = tmp_path / ("main-" + "a" * 12) / "labels"
    directory.mkdir(parents=True)
    (directory / "attempt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be extended"):
        run_medium5_episode_collection(
            baseline_config=tmp_path / "missing.yaml",
            source_checkpoint=tmp_path / "missing.pt",
            actor_artifact=tmp_path / "missing-actor.pt",
            actor_sha256="a" * 64,
            output_root=tmp_path,
            resume=True,
            episode_count=121,
        )
