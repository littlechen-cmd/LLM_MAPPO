"""A* guidance must end without ending the matched PPO learning schedule."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from llm_mappo.mappo_baseline.runner import BaselineRun, _validate_training
from llm_mappo.mappo_astar_kd.runner import load_astar_kd_experiment


def _profile():
    return load_astar_kd_experiment(
        Path("configs/core_integration/medium-5ag-astar-kd-no-validity.yaml")
    )


@pytest.mark.parametrize("bad", [0, -1, 450_001, 1.5, True, "300000", None,
                                  float("nan"), float("inf")])
def test_independent_kd_horizon_rejects_invalid_steps(bad):
    training = {**_profile().training, "astar_schedule_steps": bad}
    with pytest.raises(ValueError, match="A\\*KD.*horizon"):
        _validate_training(training)


def test_independent_kd_cutoff_keeps_ppo_learning_and_zero_shared_kd_gradient(tmp_path):
    profile = _profile()
    profile = replace(profile, approval_status="approved", training={
        **profile.training,
        "num_env_workers": 1,
        "environment_steps": 900_000,
        "optimization_schedule_steps": 900_000,
        "astar_schedule_steps": 600_000,
        "post_schedule_learning_rate": 0.0,
        "post_schedule_astar_lambda": 0.0,
    })
    run = BaselineRun(profile, device="cpu", output_root=tmp_path,
                      evaluation_only=True)
    try:
        for steps, expected in [(0, .05), (420_000, .01), (599_999, .01),
                                (600_000, 0.0), (899_999, 0.0)]:
            run.global_steps = steps
            run.trainer.set_global_environment_steps(steps)
            assert run._current_astar_lambda() == pytest.approx(expected)
        run.global_steps = 600_000
        run.trainer.set_global_environment_steps(600_000)
        run._apply_learning_rate_schedule()
        assert run.trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert run.trainer.critic_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        observations = torch.ones((1, 1, run.actor_schema.dimension))
        hidden = torch.zeros((1, 64))
        continuation = torch.ones((1, 1))
        _, auxiliary = run.trainer._actor_sequence(observations, hidden, continuation)
        selected = {
            "astar_soft_targets": torch.tensor([[[1.0, 0.0, 0.0]]]),
            "astar_validity": torch.ones((1, 1), dtype=torch.bool),
            "astar_supervision_scope": torch.ones((1, 1), dtype=torch.bool),
            "astar_attenuation": torch.ones((1, 1)),
        }
        loss = run.trainer._actor_auxiliary_loss(auxiliary, selected)
        assert loss.item() == 0.0
        parameters = tuple(run.trainer.actor.encoder.parameters()) + tuple(
            run.trainer.actor.gru.parameters()
        )
        gradients = torch.autograd.grad(loss, parameters)
        assert all(torch.count_nonzero(gradient).item() == 0 for gradient in gradients)
    finally:
        run.close()


def test_cpu_training_logs_independent_kd_tail_with_positive_ppo_lr(tmp_path):
    profile = _profile()
    profile = replace(profile, approval_status="approved", training={
        **profile.training,
        "num_env_workers": 1,
        "environment_steps": 8,
        "optimization_schedule_steps": 8,
        "astar_schedule_steps": 4,
        "post_schedule_learning_rate": 0.0,
        "post_schedule_astar_lambda": 0.0,
        "rollout_length": 2,
        "ppo_epochs": 1,
        "checkpoint_interval_steps": 8,
        "milestone_checkpoint_steps": [],
    })
    run = BaselineRun(profile, device="cpu", output_root=tmp_path)
    summary = run.train()
    assert summary["global_environment_steps"] == 8
    with (run.directory / "updates.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    cutoff = next(row for row in rows if int(row["global_environment_steps"]) == 4)
    assert float(cutoff["astar_lambda"]) == 0.0
    assert float(cutoff["actor_learning_rate"]) == pytest.approx(1.5e-4)
    assert float(cutoff["astar_shared_grad_norm"]) == 0.0
