"""Short CPU lifecycle smoke coverage for all core-integration contracts."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment
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
CONFIG_ROOT = ROOT / "configs" / "core_integration"
CONFIG_NAMES = (
    "medium-5ag-baseline.yaml",
    "medium-5ag-llm-kd.yaml",
    "medium-5ag-astar-kd-no-validity.yaml",
    "medium-5ag-astar-llm-kd-no-validity.yaml",
)


def _initialization(tmp_path: Path) -> dict[str, object]:
    actor = MAPPOActor(OBSERVATION_DIM, 5, HIDDEN_DIM)
    identity = {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": OBSERVATION_DIM,
        "ontology_schema": SEMANTIC_ONTOLOGY_V1,
        "semantic_dimensions": SEMANTIC_DIMENSIONS,
        "actor_architecture": ACTOR_ARCHITECTURE_V1,
        "hidden_dim": HIDDEN_DIM,
        "dataset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    artifact = tmp_path / "representation.pt"
    export_actor_representation(artifact, actor, identity=identity)
    return {
        "schema": "mappo-representation-initialization-v1",
        "artifact_path": str(artifact),
        "artifact_sha256": sha256(artifact.read_bytes()).hexdigest(),
        "identity": identity,
    }


def _smoke_experiment(config_name: str, tmp_path: Path):
    experiment = load_baseline_experiment(CONFIG_ROOT / config_name)
    training = {
        **experiment.training,
        "num_env_workers": 1,
        "environment_steps": 4,
        "optimization_schedule_steps": 4,
        "rollout_length": 2,
        "ppo_epochs": 1,
        "checkpoint_interval_steps": 2,
        "milestone_checkpoint_steps": [],
    }
    if experiment.method.endswith("LLM-INIT"):
        training["representation_initialization"] = _initialization(tmp_path)
    evaluation = {
        **experiment.evaluation,
        "environment_overrides": {
            **experiment.evaluation["environment_overrides"],
            "max_steps": 2,
        },
    }
    return replace(
        experiment,
        approval_status="approved",
        training=training,
        evaluation=evaluation,
    )


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_core_contract_cpu_smoke(config_name, tmp_path):
    experiment = _smoke_experiment(config_name, tmp_path)
    run = BaselineRun(experiment, device="cpu", output_root=tmp_path / "run")
    checkpoint = run.directory / "checkpoint_latest.pt"
    try:
        run._checkpoint(checkpoint)
    finally:
        run.close()

    resumed = BaselineRun(experiment, device="cpu", resume=run.directory)
    summary = resumed.train()
    final_checkpoint = run.directory / "checkpoint_final.pt"
    assert summary["global_environment_steps"] == 4
    assert summary["updates"] >= 1
    assert final_checkpoint.exists()

    evaluation = BaselineRun(
        experiment,
        device="cpu",
        output_root=tmp_path / "evaluation",
        evaluation_only=True,
    )
    result = evaluation.evaluate_checkpoint(final_checkpoint)
    assert result["status"] == "complete"
