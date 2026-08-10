import numpy as np

from llm_mappo.mappo import DualHeadMAPPOPolicy, PPOHyperparameters
from llm_mappo.phase2 import ACTION_COUNT, Phase2Warehouse
from llm_mappo.phase3_training import Phase3TrainingConfig


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


def test_phase3_config_disables_astar_kl_and_enables_rule_labels():
    config = Phase3TrainingConfig()
    assert config.ppo.reservation_kl_coefficient == 0.0
    assert config.ppo.engagement_coefficient > 0.0
