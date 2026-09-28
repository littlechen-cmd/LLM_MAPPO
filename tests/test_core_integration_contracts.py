"""Frozen contracts for the four core-integration methods."""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

import llm_mappo.mappo_baseline.runner as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "core_integration"
CONTRACTS = {
    "medium-5ag-baseline.yaml": (
        "MAPPO-FP-DG",
        "mappo-fp-direct-goal-209d-delivery-c1-checkpoint-v3",
        "artifacts/core_integration/medium5_c1_v3/baseline",
    ),
    "medium-5ag-llm-kd.yaml": (
        "MAPPO-FP-DG-LLM-INIT",
        "mappo-fp-llm-init-209d-delivery-c1-checkpoint-v3",
        "artifacts/core_integration/medium5_c1_v3/llm_kd",
    ),
    "medium-5ag-astar-kd-no-validity.yaml": (
        "MAPPO-FP-ASTARKD-NO-VALIDITY",
        "mappo-fp-astar-kd-209d-delivery-c1-checkpoint-v3",
        "artifacts/core_integration/medium5_c1_v3/astar_kd",
    ),
    "medium-5ag-astar-llm-kd-no-validity.yaml": (
        "MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT",
        "mappo-fp-astar-kd-llm-init-209d-delivery-c1-checkpoint-v3",
        "artifacts/core_integration/medium5_c1_v3/astar_llm_kd",
    ),
}


def test_four_configs_freeze_exact_identity_and_shared_contract():
    experiments = {
        name: runner.load_baseline_experiment(CONFIG_ROOT / name)
        for name in CONTRACTS
    }
    baseline = experiments["medium-5ag-baseline.yaml"]

    for name, (method, checkpoint_schema, artifact_root) in CONTRACTS.items():
        experiment = experiments[name]
        assert experiment.method == method
        assert runner.checkpoint_schema_for_experiment(experiment) == checkpoint_schema
        assert runner.checkpoint_schema_for_method(method) != checkpoint_schema
        assert experiment.artifact_root == artifact_root
        expected_status = "provisional" if method.endswith("LLM-INIT") else "approved"
        assert experiment.approval_status == expected_status
        assert experiment.environment == baseline.environment
        comparable_training = dict(experiment.training)
        comparable_training.pop("representation_initialization", None)
        comparable_training.pop("post_schedule_astar_lambda", None)
        assert comparable_training == baseline.training
        assert experiment.evaluation == baseline.evaluation
        assert experiment.environment["observation_schema"] == (
            "direct-goal-observation-v2"
        )


@pytest.mark.parametrize("name", [
    name for name, (method, _, _) in CONTRACTS.items()
    if method.endswith("LLM-INIT")
])
def test_provisional_config_refuses_training_before_creating_artifacts(
    name, tmp_path
):
    experiment = runner.load_baseline_experiment(CONFIG_ROOT / name)
    output_root = tmp_path / name

    with pytest.raises(ValueError, match="provisional"):
        runner.BaselineRun(experiment, device="cpu", output_root=output_root)

    assert not output_root.exists()


def _assert_state_unchanged(before, after):
    if isinstance(before, torch.Tensor):
        assert torch.equal(before, after)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            _assert_state_unchanged(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        for left, right in zip(before, after):
            _assert_state_unchanged(left, right)
    elif isinstance(before, np.ndarray):
        np.testing.assert_array_equal(before, after)
    else:
        assert before == after


def test_invalid_checkpoint_is_rejected_before_mutating_runtime_state(tmp_path):
    experiment = runner.load_baseline_experiment(
        CONFIG_ROOT / "medium-5ag-baseline.yaml"
    )
    run = runner.BaselineRun(
        experiment, device="cpu", output_root=tmp_path, evaluation_only=True
    )
    payload = {
        "schema": runner.checkpoint_schema_for_experiment(experiment),
        "identity": dict(run.identity),
        "trainer": deepcopy(run.trainer.state_dict()),
    }
    payload["trainer"]["actor"]["encoder.0.weight"] = torch.full_like(
        payload["trainer"]["actor"]["encoder.0.weight"], 0.314
    )
    payload["trainer"]["actor"]["action_head.weight"] = torch.zeros(
        (1, 1), dtype=torch.float32
    )
    checkpoint = tmp_path / "invalid-checkpoint.pt"
    torch.save(payload, checkpoint)
    trainer_before = deepcopy(run.trainer.state_dict())
    rng_before = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }

    with pytest.raises(RuntimeError, match="size mismatch"):
        run.evaluate_checkpoint(checkpoint)

    _assert_state_unchanged(trainer_before, run.trainer.state_dict())
    _assert_state_unchanged(rng_before["python"], random.getstate())
    _assert_state_unchanged(rng_before["numpy"], np.random.get_state())
    assert torch.equal(rng_before["torch"], torch.get_rng_state())
