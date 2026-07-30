import numpy as np

from llm_mappo.mappo import MAPPOPolicy, MAPPOUpdater, PPOHyperparameters, RolloutBuffer
from llm_mappo.phase2 import ACTION_COUNT, Phase2Warehouse
from llm_mappo.phase2_training import Phase2TrainingConfig, train_phase2


def test_phase2_adapter_provides_oracle_features_and_single_b_priority():
    env = Phase2Warehouse(n_agents=3, max_steps=8)
    try:
        observations = env.reset(seed=3)
        assert observations.shape == (3, env.actor_observation_dim)
        assert observations.dtype == np.float32
        assert {task.label[0] for task in env.env.task_queue.active_tasks} == {"B"}

        transition = env.step([0, 0, 0])
        assert transition.observations.shape == observations.shape
        assert transition.metrics.created_tasks == 3
        assert transition.team_reward == np.mean([transition.team_reward])
    finally:
        env.close()


def test_mappo_update_handles_a_shared_three_agent_rollout():
    observation_dim = 12
    policy = MAPPOPolicy(observation_dim, ACTION_COUNT)
    updater = MAPPOUpdater(
        policy,
        PPOHyperparameters(update_epochs=1, minibatch_steps=2),
    )
    buffer = RolloutBuffer(n_agents=3)
    for step in range(3):
        observations = np.full((3, observation_dim), step, dtype=np.float32)
        actions, log_probs, value = policy.act(observations)
        buffer.add(
            observations,
            actions,
            log_probs,
            reward=1.0,
            done=step == 2,
            value=value,
        )

    losses = updater.update(buffer, last_value=0.0)
    assert not len(buffer)
    assert set(losses) == {"policy_loss", "value_loss", "entropy"}
    assert all(np.isfinite(loss) for loss in losses.values())


def test_short_training_run_writes_a_checkpoint_and_metrics(tmp_path):
    config = Phase2TrainingConfig(
        seed=11,
        n_agents=1,
        max_steps=3,
        episodes=2,
        rollout_steps=2,
        checkpoint_interval=1,
        output_dir=str(tmp_path),
        ppo=PPOHyperparameters(update_epochs=1, minibatch_steps=2),
    )
    summary = train_phase2(config)

    assert summary["episodes"] == 2
    assert (tmp_path / "seed_011" / "checkpoint_final.pt").is_file()
    assert (tmp_path / "seed_011" / "episodes.csv").is_file()
    assert (tmp_path / "seed_011" / "updates.csv").is_file()
