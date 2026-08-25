import numpy as np
import torch

from llm_mappo.optimization_buffer import (
    LinearEnvStepSchedule,
    OptimizationRolloutBuffer,
)


def test_optimization_buffer_preserves_masks_and_zero_kd_losses():
    buffer = OptimizationRolloutBuffer(n_agents=2)
    buffer.add(
        physical_observations=np.zeros((2, 613), dtype=np.float32),
        semantic_observations=np.zeros((2, 61), dtype=np.float32),
        actions=np.asarray([0, 1]),
        log_probs=np.zeros(2, dtype=np.float32),
        action_masks=np.ones((2, 5), dtype=bool),
        astar_preferences=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        astar_valid=np.asarray([True, False]),
        calibration_selected=True,
        reward_confidence=0.5,
        semantic_targets=np.zeros((2, 3), dtype=np.float32),
        semantic_validity=np.asarray([1.0, 0.0], dtype=np.float32),
        ood_reliability=0.75,
    )
    batch = buffer.tensors("cpu")

    assert batch.physical_observations.shape == (1, 2, 613)
    assert batch.semantic_observations.shape == (1, 2, 61)
    assert batch.astar_valid.tolist() == [[True, False]]
    assert batch.calibration_selected.tolist() == [True]
    assert batch.ood_reliability.tolist() == [0.75]

    motion_logits = torch.zeros((1, 2, 3), requires_grad=True)
    semantic_scores = torch.zeros((1, 2, 3), requires_grad=True)
    assert batch.astar_kl_loss(motion_logits, lambda_a=0.05).item() > 0.0
    assert batch.semantic_mse_loss(semantic_scores, lambda_l=0.10).item() == 0.0

    all_invalid = OptimizationRolloutBuffer(n_agents=2)
    all_invalid.add(
        physical_observations=np.zeros((2, 613), dtype=np.float32),
        semantic_observations=np.zeros((2, 61), dtype=np.float32),
        actions=np.asarray([0, 1]),
        log_probs=np.zeros(2, dtype=np.float32),
        action_masks=np.ones((2, 5), dtype=bool),
        astar_preferences=np.zeros((2, 3), dtype=np.float32),
        astar_valid=np.zeros(2, dtype=bool),
        calibration_selected=False,
        reward_confidence=0.0,
        semantic_targets=np.zeros((2, 3), dtype=np.float32),
        semantic_validity=np.zeros(2, dtype=np.float32),
        ood_reliability=0.0,
    )
    invalid_batch = all_invalid.tensors("cpu")
    assert invalid_batch.astar_kl_loss(motion_logits, lambda_a=0.05).requires_grad
    assert invalid_batch.semantic_mse_loss(semantic_scores, lambda_l=0.10).requires_grad


def test_linear_env_step_schedule_counts_only_real_transitions():
    schedule = LinearEnvStepSchedule(total_env_steps=150_000)

    assert schedule.weights() == (0.05, 0.10)
    schedule.advance_real_env_steps(75_000)
    assert schedule.weights() == (0.025, 0.05)
    schedule.advance_real_env_steps(75_000)
    assert schedule.weights() == (0.0, 0.0)
