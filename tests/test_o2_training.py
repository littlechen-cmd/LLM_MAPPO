"""Focused PPO and O2-group isolation tests for the experiment layer."""

import numpy as np
import torch

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _rollout_records(count=4):
    from llm_mappo.o2_training import O2Rollout

    rollout = O2Rollout(n_agents=2)
    for index in range(count):
        physical = np.full((2, 613), index / 10.0, dtype=np.float32)
        semantic = np.zeros((2, 61), dtype=np.float32)
        masks = np.ones((2, 5), dtype=bool)
        rollout.add(
            physical_observations=physical,
            semantic_observations=semantic,
            actions=np.asarray([1, 2], dtype=np.int64),
            log_probs=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_masks=masks,
            astar_preferences=np.asarray([[1.0, 0.0, 0.0]] * 2),
            astar_valid=np.asarray([True, True]),
            calibration_selected=True,
            reward_confidence=1.0,
            reward=float(index + 1),
            done=index == count - 1,
            value=0.0,
        )
    return rollout


def test_o2_rollout_gae_respects_terminal_boundary_and_old_log_probs():
    from llm_mappo.o2_training import O2PPOHyperparameters

    rollout = _rollout_records()
    data = rollout.tensors(
        last_value=100.0,
        hyperparameters=O2PPOHyperparameters(),
        device="cpu",
    )

    assert data.old_log_probs.shape == (4, 2)
    assert torch.allclose(data.old_log_probs[0], torch.tensor([-1.0, -1.0]))
    assert data.returns[-1].item() == 4.0
    assert data.advantages[-1].item() == 4.0


def test_mappo_dg_cannot_acquire_astar_or_semantic_loss():
    from llm_mappo.o2_training import O2PPOHyperparameters, O2PPOUpdater
    from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor

    torch.manual_seed(7)
    updater = O2PPOUpdater(
        actor=O0StudentActor(),
        critic=O0CentralizedCritic(),
        hyperparameters=O2PPOHyperparameters(update_epochs=1, minibatch_steps=4),
        method="MAPPO-DG",
        device="cpu",
    )
    metrics = updater.update(_rollout_records(), last_value=0.0, lambda_a=0.05)

    assert metrics["astar_loss"] == 0.0
    assert metrics["semantic_loss"] == 0.0
    assert metrics["finite"] is True


def test_rc_astar_kd_uses_selected_valid_weight_and_remains_finite():
    from llm_mappo.o2_training import O2PPOHyperparameters, O2PPOUpdater
    from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor

    torch.manual_seed(11)
    updater = O2PPOUpdater(
        actor=O0StudentActor(),
        critic=O0CentralizedCritic(),
        hyperparameters=O2PPOHyperparameters(update_epochs=1, minibatch_steps=4),
        method="RC-AStarKD",
        device="cpu",
    )
    metrics = updater.update(_rollout_records(), last_value=0.0, lambda_a=0.05)

    assert metrics["astar_loss"] > 0.0
    assert metrics["semantic_loss"] == 0.0
    assert metrics["finite"] is True


def test_o2_mappo_dg_short_prefix_has_no_teacher_side_effects():
    from llm_mappo.o2_contract import O2ExperimentConfig, O2RunSpec
    from llm_mappo.o2_training import O2Trainer

    experiment = O2ExperimentConfig.from_yaml(
        ROOT / "configs/optimization/o2_calibration.yaml"
    )
    trainer = O2Trainer(
        experiment=experiment,
        run=O2RunSpec("MAPPO-DG", 107, 150000),
        device="cpu",
    )
    result = trainer.run(max_steps=1)
    trainer.environment.close()
    trainer.student_shadow.close()
    trainer.teacher_shadow.close()

    assert result["real_env_steps"] == 1
    assert result["teacher_queries"] == 0
    assert result["shadow_calls"] == 0
    assert result["ema_updates"] == 0
    assert result["semantic_loss"] == 0.0


def test_o2_runtime_snapshot_resumes_from_an_empty_update_boundary():
    from llm_mappo.o2_contract import O2ExperimentConfig, O2RunSpec
    from llm_mappo.o2_training import O2Trainer

    experiment = O2ExperimentConfig.from_yaml(
        ROOT / "configs/optimization/o2_calibration.yaml"
    )
    first = O2Trainer(
        experiment=experiment,
        run=O2RunSpec("MAPPO-DG", 107, 150000),
        device="cpu",
    )
    first.run(max_steps=2)
    runtime = first.runtime_state()
    first.environment.close()
    first.student_shadow.close()
    first.teacher_shadow.close()

    resumed = O2Trainer(
        experiment=experiment,
        run=O2RunSpec("MAPPO-DG", 107, 150000),
        device="cpu",
    )
    resumed.restore_runtime_state(runtime)
    result = resumed.run(max_steps=4)
    resumed.environment.close()
    resumed.student_shadow.close()
    resumed.teacher_shadow.close()

    assert result["real_env_steps"] == 4
    assert result["teacher_queries"] == 0
