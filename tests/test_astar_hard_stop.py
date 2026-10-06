"""Hard A* cutoffs preserve a shared guidance curve and diagnostic isolation."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from llm_mappo.mappo_astar_kd.trainer import AStarKDTrainer, astar_kd_lambda
from llm_mappo.mappo_baseline.runner import (
    BaselineRun, _validate_training, checkpoint_schema_for_experiment,
    load_baseline_experiment,
)
from llm_mappo.mappo_baseline.trainer import MAPPOConfig


def _profile():
    return load_baseline_experiment(Path(
        "configs/training_v1/medium-5ag-formal-astar-kd/seed_0007.yaml"
    ))


@pytest.mark.parametrize("stop", [450_000, 600_000, 750_000, 900_000])
def test_hard_stop_preserves_common_prefix_and_cuts_at_boundary(stop):
    for steps, expected in [(0, .05), (210_000, .03), (420_000, .01),
                            (449_999, .01)]:
        assert astar_kd_lambda(steps, 600_000, stop_steps=stop) == pytest.approx(expected)
    assert astar_kd_lambda(stop - 1, 600_000, stop_steps=stop) == .01
    assert astar_kd_lambda(stop, 600_000, stop_steps=stop) == 0.0
    assert astar_kd_lambda(900_000, 600_000, stop_steps=stop) == 0.0


@pytest.mark.parametrize("bad", [0, -1, 900_001, 1.5, True, "450000",
                                  float("nan"), float("inf")])
def test_runner_rejects_invalid_hard_stop(bad):
    with pytest.raises(ValueError, match="A\\*KD.*stop"):
        _validate_training({**_profile().training, "astar_stop_steps": bad},
                           provisional=True)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "450000",
                                  float("nan"), float("inf")])
def test_trainer_rejects_invalid_hard_stop(bad):
    with pytest.raises(ValueError, match="A\\*KD.*stop"):
        AStarKDTrainer(MAPPOConfig(observation_dim=3, state_dim=3, action_dim=5),
                       "cpu", total_environment_steps=600_000,
                       astar_stop_steps=bad)


@pytest.mark.parametrize("stop", [750_000, 900_000])
def test_tail_diagnostic_requires_authorization_and_distinct_checkpoint(tmp_path, stop):
    profile = replace(_profile(), schema="astar-tail-diagnostic-v1")
    with pytest.raises(ValueError, match="diagnostic authorization"):
        BaselineRun(profile, device="cpu", output_root=tmp_path)
    with pytest.raises(ValueError, match="diagnostic runner"):
        source = tmp_path / "diagnostic.yaml"
        source.write_text("schema: astar-tail-diagnostic-v1\n", encoding="utf-8")
        load_baseline_experiment(source)
    profile = replace(profile, training={**profile.training,
                      "num_env_workers": 1, "astar_stop_steps": stop,
                      "post_schedule_astar_lambda": .01})
    run = BaselineRun(profile, device="cpu", output_root=tmp_path,
                      diagnostic_authorized=True, evaluation_only=True)
    try:
        assert checkpoint_schema_for_experiment(profile) == (
            "mappo-fp-astar-tail-diagnostic-checkpoint-v1"
        )
        assert run.identity["scene_id"] == "medium-5ag"
        assert run.identity["fp_state_dimension"] == 689
        for steps, expected in [(600_000, .01), (stop - 1, .01), (stop, 0.0)]:
            run.global_steps = steps
            run.trainer.set_global_environment_steps(steps)
            assert run._current_astar_lambda() == expected
            _, auxiliary = run.trainer._actor_sequence(
                torch.randn((1, 1, 209)), torch.zeros((1, 64)), torch.ones((1, 1))
            )
            loss = run.trainer._actor_auxiliary_loss(auxiliary, {
                "astar_soft_targets": torch.tensor([[[1., 0., 0.]]]),
                "astar_validity": torch.ones((1, 1), dtype=torch.bool),
                "astar_supervision_scope": torch.ones((1, 1), dtype=torch.bool),
                "astar_attenuation": torch.ones((1, 1)),
            })
            assert (loss.item() == 0.0) == (steps == stop)
            parameters = tuple(run.trainer.actor.encoder.parameters()) + tuple(
                run.trainer.actor.gru.parameters()
            )
            gradients = torch.autograd.grad(loss, parameters)
            gradient_norm = sum(gradient.abs().sum().item() for gradient in gradients)
            assert (gradient_norm == 0.0) == (steps == stop)
    finally:
        run.close()
