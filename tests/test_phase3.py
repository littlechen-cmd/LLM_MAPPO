import csv

import numpy as np
import pytest
import torch

from llm_mappo.mappo import DualHeadMAPPOPolicy
from llm_mappo.phase2 import ACTION_COUNT, Phase2Warehouse
from llm_mappo.phase3_training import (
    Phase3TrainingConfig,
    _engagement_targets,
    _training_episode_seed,
    _validate_training_seed_groups,
    _write_csv,
    evaluate_phase3,
)


def test_phase3_priority_schedule_and_observation_features():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        priority_schedule=("A", "B", "C"),
    )
    try:
        observations = env.reset(seed=3)
        assert observations.shape == (3, env.actor_observation_dim)
        assert env.actor_observation_dim == 615
        assert [task.label for task in env.env.task_queue.tasks] == ["A1", "B1", "C1"]
    finally:
        env.close()


def test_phase3_dual_head_outputs_engagement_and_motion_distribution():
    policy = DualHeadMAPPOPolicy(observation_dim=615, action_dim=ACTION_COUNT)
    observations = np.zeros((3, 615), dtype=np.float32)
    actions, log_probs, value, engagement = policy.act(observations)
    assert actions.shape == (3,)
    assert log_probs.shape == (3,)
    assert isinstance(value, float)
    assert engagement.shape == (3,)
    assert np.all((engagement >= 0.0) & (engagement <= 1.0))


def test_phase3_r2_motion_loss_does_not_update_engagement_branch():
    policy = DualHeadMAPPOPolicy(observation_dim=615, action_dim=ACTION_COUNT)
    observations = torch.randn(3, 615)
    policy.actor(observations).sum().backward()
    engagement_gradients = [
        parameter.grad
        for parameter in policy.actor.engagement_encoder.parameters()
    ]
    engagement_gradients.extend(
        parameter.grad for parameter in policy.actor.engagement_head.parameters()
    )
    assert all(gradient is None for gradient in engagement_gradients)
    assert any(
        parameter.grad is not None
        for parameter in policy.actor.motion_encoder.parameters()
    )


def test_phase3_r2_engagement_loss_does_not_update_motion_branch():
    policy = DualHeadMAPPOPolicy(observation_dim=615, action_dim=ACTION_COUNT)
    observations = torch.randn(3, 615)
    policy.engagement(observations).sum().backward()
    assert all(
        parameter.grad is None
        for parameter in policy.actor.motion_encoder.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in policy.actor.engagement_encoder.parameters()
    )


def test_phase3_r2_uses_lower_label_for_idle_and_locked_agvs():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        priority_schedule=("A", "B", "C"),
    )
    try:
        env.reset(seed=3)
        env.env.task_queue.release_agent(1)
        env.env.agents[0].task_id = None
        env.env.agents[1].picking_lock_steps = 1
        assert np.allclose(_engagement_targets(env), [0.1, 0.1, 0.3])
    finally:
        env.close()


def test_phase3_dynamic_ingress_rotates_batches_and_rank_labels():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        priority_schedule=None,
        batch_interval=2,
        batch_size_range=(1, 1),
        initial_priority_label="A",
        request_queue_size=1,
        task_completion_target=4,
        include_priority_features=True,
    )
    try:
        observations = env.reset(seed=3)
        assert observations.shape == (3, 615)
        assert [task.label for task in env.env.task_queue.active_tasks] == [
            "A1",
            "B1",
        ]
        for _ in range(4):
            env.step([0, 0, 0])
        assert [task.label for task in env.env.task_queue.active_tasks] == [
            "A1",
            "B1",
            "C1",
            "D1",
        ]
        assert np.allclose(
            _engagement_targets(env),
            [0.1 + 0.7 * 3 / 4, 0.1 + 0.7 * 2 / 4, 0.1 + 0.7 / 4],
        )
        assert env.metrics.created_tasks == 4
        assert env.metrics.task_completion_target == 4
    finally:
        env.close()


def test_phase3_dynamic_ingress_does_not_arrive_after_the_time_limit():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=2,
        priority_schedule=None,
        batch_interval=2,
        batch_size_range=(1, 1),
        initial_priority_label="A",
        request_queue_size=1,
        task_completion_target=4,
        include_priority_features=True,
    )
    try:
        env.reset(seed=3)
        env.step([0, 0, 0])
        transition = env.step([0, 0, 0])
        assert transition.metrics.steps == 2
        assert [task.label for task in env.env.task_queue.tasks] == ["A1", "B1"]
    finally:
        env.close()


def test_phase3_dynamic_ingress_terminates_once_the_completion_target_is_met():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        priority_schedule=None,
        batch_interval=8,
        batch_size_range=(1, 1),
        initial_priority_label="A",
        request_queue_size=1,
        task_completion_target=2,
        include_priority_features=True,
    )
    try:
        env.reset(seed=3)
        for task in env.env.task_queue.active_tasks:
            env.env.task_queue.complete(task.task_id, completed_step=0)
        env.env._refresh_request_queue()
        transition = env.step([0, 0, 0])
        assert transition.terminated
        assert not transition.truncated
        assert transition.metrics.completed_tasks == 2
        assert transition.metrics.task_completion_target == 2
        assert transition.metrics.completion_rate == 1.0
        assert transition.metrics.success
        assert transition.info["task_target_reached"]
    finally:
        env.close()


def test_phase3_config_disables_astar_kl_and_enables_rule_labels():
    config = Phase3TrainingConfig()
    assert config.ppo.reservation_kl_coefficient == 0.0
    assert config.ppo.engagement_coefficient > 0.0


def test_phase3_round_robins_isolated_training_seed_groups():
    config = Phase3TrainingConfig(seed=7, training_seed_groups=(100, 101, 102))
    assert _training_episode_seed(config, 0) == (1_000_000, 100, 0)
    assert _training_episode_seed(config, 1) == (1_010_000, 101, 0)
    assert _training_episode_seed(config, 3) == (1_000_001, 100, 1)


def test_phase3_rejects_duplicate_training_seed_groups():
    with pytest.raises(ValueError, match="duplicates"):
        _validate_training_seed_groups((100, 100))


def test_phase3b_config_enables_only_astar_kl_ablation():
    config = Phase3TrainingConfig.from_yaml("configs/phase3b_dual_head_astar_kl.yaml")
    assert config.phase == "3b"
    assert config.n_agents == 3
    assert config.priority_schedule == ("A", "B", "C")
    assert config.ppo.reservation_kl_coefficient == 0.05
    assert config.ppo.engagement_coefficient == 0.1


def test_phase3_r2_configs_differ_only_in_phase_and_path_kl():
    phase3a = Phase3TrainingConfig.from_yaml("configs/phase3a_r2_semantic.yaml")
    phase3b = Phase3TrainingConfig.from_yaml(
        "configs/phase3b_r2_semantic_astar_kl.yaml"
    )
    assert phase3a.phase == "3a"
    assert phase3b.phase == "3b"
    assert phase3a.output_dir != phase3b.output_dir
    assert phase3a.ppo.reservation_kl_coefficient == 0.0
    assert phase3b.ppo.reservation_kl_coefficient == 0.05
    assert phase3a.ppo.engagement_coefficient == phase3b.ppo.engagement_coefficient
    assert phase3a.priority_schedule == phase3b.priority_schedule


def test_phase3_dynamic_configs_keep_ingress_fixed_for_the_kl_ablation():
    phase3a = Phase3TrainingConfig.from_yaml(
        "configs/phase3a_r2_dynamic_ingress.yaml"
    )
    phase3b = Phase3TrainingConfig.from_yaml(
        "configs/phase3b_r2_dynamic_ingress_astar_kl.yaml"
    )
    assert phase3a.priority_schedule is None
    assert phase3b.priority_schedule is None
    assert phase3a.batch_interval == phase3b.batch_interval == 40
    assert phase3a.batch_size_range == phase3b.batch_size_range == (1, 3)
    assert phase3a.initial_priority_label == phase3b.initial_priority_label == "A"
    assert phase3a.request_queue_size == phase3b.request_queue_size == 4
    assert phase3a.task_completion_target == phase3b.task_completion_target == 9
    assert phase3a.ppo.reservation_kl_coefficient == 0.0
    assert phase3b.ppo.reservation_kl_coefficient == 0.05


def test_phase3_csv_writer_preserves_late_priority_columns(tmp_path):
    path = tmp_path / "episodes.csv"
    _write_csv(
        path,
        [
            {"episode": 1, "task_completion_rate": 0.0},
            {
                "episode": 2,
                "task_completion_rate": 1.0,
                "priority_A_mean_completion_steps": 10.0,
            },
        ],
    )
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["priority_A_mean_completion_steps"] == "10.0"
    assert rows[0]["priority_A_mean_completion_steps"] == ""


def test_phase3_evaluation_rejects_nonpositive_engagement_sample_rate():
    with pytest.raises(ValueError, match="engagement_sample_rate"):
        evaluate_phase3(None, {}, (), engagement_sample_rate=0)
