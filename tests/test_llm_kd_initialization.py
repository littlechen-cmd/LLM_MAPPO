"""Narrow Phase 3 representation initialization integration contracts."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from llm_mappo.mappo_astar_kd.networks import AStarKDActor
from llm_mappo.mappo_astar_kd.teacher import AStarKDTeacher
from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment
from llm_mappo.mappo_llm_kd.provider import OpenAICompatibleProvider
from llm_mappo.mappo_llm_kd.pretraining import (
    ACTOR_ARCHITECTURE_V1,
    HIDDEN_DIM,
    OBSERVATION_DIM,
    SEMANTIC_DIMENSIONS,
    SEMANTIC_ONTOLOGY_V1,
)
from llm_mappo.mappo_llm_kd.representation_export import (
    export_actor_representation,
)


ROOT = Path(__file__).resolve().parents[1]
COLLECTION_CONFIG = (
    ROOT / "configs" / "llm_kd" / "medium-5ag-low-workload-collection-baseline.yaml"
)
ASTAR_CONFIG = (
    ROOT / "configs" / "core_integration" / "medium-5ag-astar-kd-no-validity.yaml"
)
CORE_CONFIG_ROOT = ROOT / "configs" / "core_integration"


def _identity():
    return {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": OBSERVATION_DIM,
        "ontology_schema": SEMANTIC_ONTOLOGY_V1,
        "semantic_dimensions": SEMANTIC_DIMENSIONS,
        "actor_architecture": ACTOR_ARCHITECTURE_V1,
        "hidden_dim": HIDDEN_DIM,
        "dataset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }


def _initialization(tmp_path):
    source = MAPPOActor(209, 5, 64)
    with torch.no_grad():
        for parameter in list(source.encoder.parameters()) + list(source.gru.parameters()):
            parameter.fill_(0.125)
    artifact = tmp_path / "representation.pt"
    export_actor_representation(artifact, source, identity=_identity())
    return {
        "schema": "mappo-representation-initialization-v1",
        "artifact_path": str(artifact),
        "artifact_sha256": sha256(artifact.read_bytes()).hexdigest(),
        "identity": _identity(),
    }, source


def _with_representation(experiment, initialization):
    return replace(
        experiment,
        training={
            **experiment.training,
            "representation_initialization": initialization,
        },
    )


def _assert_same_state(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same_state(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_same_state(first, second)
    else:
        assert left == right


def test_plain_mappo_imports_only_encoder_and_gru_after_fresh_construction(
    tmp_path,
):
    initialization, source = _initialization(tmp_path)
    original = load_baseline_experiment(COLLECTION_CONFIG)
    reference = BaselineRun(
        original,
        device="cpu",
        output_root=tmp_path / "reference",
        evaluation_only=True,
    )
    initialized = BaselineRun(
        _with_representation(original, initialization),
        device="cpu",
        output_root=tmp_path / "initialized",
        evaluation_only=True,
    )
    try:
        assert initialized.representation_initialization_receipt["file_sha256"] == (
            initialization["artifact_sha256"]
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                initialized.trainer.actor.encoder.parameters(),
                source.encoder.parameters(),
            )
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                initialized.trainer.actor.gru.parameters(), source.gru.parameters()
            )
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                initialized.trainer.actor.action_head.parameters(),
                reference.trainer.actor.action_head.parameters(),
            )
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                initialized.trainer.critic.parameters(), reference.trainer.critic.parameters()
            )
        )
        _assert_same_state(
            initialized.trainer.value_normalizer.state_dict(),
            reference.trainer.value_normalizer.state_dict(),
        )
        _assert_same_state(
            initialized.trainer.actor_optimizer.state_dict(),
            reference.trainer.actor_optimizer.state_dict(),
        )
        _assert_same_state(
            initialized.trainer.critic_optimizer.state_dict(),
            reference.trainer.critic_optimizer.state_dict(),
        )
        assert torch.equal(
            torch.as_tensor(initialized.actor_hidden_states),
            torch.as_tensor(reference.actor_hidden_states),
        )
        assert torch.equal(
            torch.as_tensor(initialized.critic_hidden_states),
            torch.as_tensor(reference.critic_hidden_states),
        )
    finally:
        initialized.close()
        reference.close()


def test_astar_kd_import_happens_after_final_astar_actor_replacement(tmp_path):
    initialization, source = _initialization(tmp_path)
    original = load_baseline_experiment(ASTAR_CONFIG)
    environment = {
        **original.environment,
        "observation_schema": "direct-goal-observation-v2",
    }
    experiment = replace(original, environment=environment)
    reference = BaselineRun(
        experiment,
        device="cpu",
        output_root=tmp_path / "reference",
        evaluation_only=True,
    )
    run = BaselineRun(
        _with_representation(experiment, initialization),
        device="cpu",
        output_root=tmp_path,
        evaluation_only=True,
    )
    try:
        assert isinstance(run.trainer.actor, AStarKDActor)
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                run.trainer.actor.encoder.parameters(), source.encoder.parameters()
            )
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(
                run.trainer.actor.motion_prior_head.parameters(),
                reference.trainer.actor.motion_prior_head.parameters(),
            )
        )
        _assert_same_state(
            run.trainer.critic.state_dict(), reference.trainer.critic.state_dict()
        )
        _assert_same_state(
            run.trainer.value_normalizer.state_dict(),
            reference.trainer.value_normalizer.state_dict(),
        )
        _assert_same_state(
            run.trainer.actor_optimizer.state_dict(),
            reference.trainer.actor_optimizer.state_dict(),
        )
        _assert_same_state(
            run.trainer.critic_optimizer.state_dict(),
            reference.trainer.critic_optimizer.state_dict(),
        )
        assert torch.equal(
            torch.as_tensor(run.actor_hidden_states),
            torch.as_tensor(reference.actor_hidden_states),
        )
        assert torch.equal(
            torch.as_tensor(run.critic_hidden_states),
            torch.as_tensor(reference.critic_hidden_states),
        )
    finally:
        run.close()
        reference.close()


@pytest.mark.parametrize(
    "config_name",
    (
        "medium-5ag-baseline.yaml",
        "medium-5ag-llm-kd.yaml",
        "medium-5ag-astar-kd-no-validity.yaml",
        "medium-5ag-astar-llm-kd-no-validity.yaml",
    ),
)
def test_core_evaluation_calls_neither_astar_nor_online_llm(
    tmp_path, monkeypatch, config_name
):
    initialization, _ = _initialization(tmp_path)
    experiment = load_baseline_experiment(CORE_CONFIG_ROOT / config_name)
    if experiment.method.endswith("LLM-INIT"):
        experiment = _with_representation(experiment, initialization)
    experiment = replace(
        experiment,
        evaluation={
            **experiment.evaluation,
            "environment_overrides": {
                **experiment.evaluation["environment_overrides"],
                "max_steps": 2,
            },
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation must not invoke a teacher or provider")

    monkeypatch.setattr(AStarKDTeacher, "query", forbidden)
    monkeypatch.setattr(OpenAICompatibleProvider, "complete", forbidden)
    run = BaselineRun(
        experiment,
        device="cpu",
        output_root=tmp_path / config_name,
        evaluation_only=True,
    )
    try:
        result = run._evaluation_episode(9400)
        assert result["seed"] == 9400
        assert run._evaluation_planner is None
    finally:
        run.close()
