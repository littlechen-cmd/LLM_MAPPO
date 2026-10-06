"""Fail-closed identity checks for the medium C1 behavior Actor."""

from __future__ import annotations

from hashlib import sha256

import pytest
import torch

from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_llm_kd.collector import (
    MEDIUM5_ACTOR_SCHEMA, load_medium5_collection_policy,
)


def test_medium5_actor_loader_requires_exact_pinned_identity(tmp_path):
    identity = {
        "schema": "core-mappo-209d-delivery-c1-v3",
        "method": "MAPPO-FP-DG",
        "observation_schema": "direct-goal-observation-v2",
        "actor_observation_schema": "structured-direct_goal-v2",
        "actor_observation_dimension": 209,
        "n_agents": 5,
        "goal_interface": "direct_goal",
        "global_environment_steps": 600000,
        "config_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "source_checkpoint_sha256": "c" * 64,
        "reward_definition_sha256": "d" * 64,
        "code_commit": "e" * 40,
    }
    path = tmp_path / "actor.pt"
    torch.save({
        "schema": MEDIUM5_ACTOR_SCHEMA,
        "identity": identity,
        "actor_config": {
            "observation_dim": 209, "action_dim": 5, "hidden_dim": 64,
        },
        "actor_state_dict": MAPPOActor(209, 5, 64).state_dict(),
    }, path)
    digest = sha256(path.read_bytes()).hexdigest()
    policy = load_medium5_collection_policy(
        path, device="cpu", expected_artifact_sha256=digest,
        expected_identity=identity,
    )
    assert policy.behavior_policy_id == f"MAPPO-FP-DG:{digest}"
    with pytest.raises(ValueError, match="SHA-256"):
        load_medium5_collection_policy(
            path, device="cpu", expected_artifact_sha256="0" * 64,
            expected_identity=identity,
        )
    with pytest.raises(ValueError, match="identity"):
        load_medium5_collection_policy(
            path, device="cpu", expected_artifact_sha256=digest,
            expected_identity={**identity, "config_sha256": "0" * 64},
        )


def test_medium5_actor_loader_rejects_diagnostic_schema(tmp_path):
    path = tmp_path / "diagnostic.pt"
    torch.save({"schema": "mappo-collection-actor-209d-v1"}, path)
    with pytest.raises(ValueError, match="schema"):
        load_medium5_collection_policy(
            path, device="cpu",
            expected_artifact_sha256=sha256(path.read_bytes()).hexdigest(),
            expected_identity={},
        )
