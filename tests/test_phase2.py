import numpy as np

from llm_mappo.mappo import MAPPOPolicy, MAPPOUpdater, PPOHyperparameters, RolloutBuffer
from llm_mappo.phase2 import ACTION_COUNT, Phase2Warehouse
from llm_mappo.phase2_training import Phase2TrainingConfig, train_phase2
from llm_mappo.phase2_expert import AStarExpert, collect_expert_episodes
from llm_mappo.visualization import render_warehouse_frame
from rware.warehouse import Action


def test_astar_expert_reuses_unchanged_state_and_target():
    env = Phase2Warehouse(n_agents=3, max_steps=8)
    expert = AStarExpert()
    try:
        env.reset(seed=4)
        masks = env.action_masks()
        expert.act(env, masks)
        env.env._cur_steps = 1
        expert.act(env, masks)
    finally:
        env.close()
    assert expert.cache_misses == 1
    assert expert.cache_hits == 1


def test_phase2_adapter_provides_oracle_features_and_single_b_priority():
    env = Phase2Warehouse(n_agents=3, max_steps=8)
    try:
        observations = env.reset(seed=3)
        assert observations.shape == (3, env.actor_observation_dim)
        assert observations.dtype == np.float32
        assert env.actor_observation_dim > len(env._raw_observations[0])
        assert {task.label[0] for task in env.env.task_queue.active_tasks} == {"B"}

        transition = env.step([0, 0, 0])
        assert transition.observations.shape == observations.shape
        assert transition.metrics.created_tasks == 3
        assert transition.team_reward == np.mean([transition.team_reward])
    finally:
        env.close()


def test_phase2_adapter_scales_waypoint_rewards(monkeypatch):
    env = Phase2Warehouse(n_agents=1, max_steps=8, waypoint_reward=0.05)
    try:
        env.reset(seed=9)
        monkeypatch.setattr(env, "_waypoint_distances", lambda: [2])
        assert np.allclose(env._movement_rewards([3]), [0.05])
        assert np.allclose(env._movement_rewards([2]), [0.0])
    finally:
        env.close()


def test_phase2_action_mask_forces_valid_assigned_shelf_pickup():
    env = Phase2Warehouse(n_agents=1, max_steps=8)
    try:
        env.reset(seed=9)
        task = env.env.task_queue.task_for_agent(1)
        shelf = env.env.shelfs[task.shelf_id - 1]
        env.env.agents[0].x, env.env.agents[0].y = shelf.x, shelf.y
        env.env._recalc_grid()

        mask = env.action_masks()[0]
        assert mask.tolist() == [False, False, False, False, True]
        transition = env.step([Action.TOGGLE_LOAD])
        assert env.env.agents[0].carrying_shelf is shelf
        assert transition.metrics.picked_tasks == 1
    finally:
        env.close()


def test_astar_expert_completes_a_small_single_agv_task_without_collisions():
    env = Phase2Warehouse(
        env_id="llm-mappo-small-1ag-v1", n_agents=1, max_steps=200
    )
    try:
        dataset, summary = collect_expert_episodes(env, episodes=5, seed=3)
        assert len(dataset) > 0
        assert summary["task_completion_rate"] == 1.0
        assert summary["pickup_delivery_match"]
        assert summary["mean_collisions"] == 0.0
    finally:
        env.close()


def test_astar_expert_completes_a_medium_single_agv_task_without_collisions():
    env = Phase2Warehouse(
        env_id="llm-mappo-medium-3ag-v1", n_agents=1, max_steps=400
    )
    try:
        dataset, summary = collect_expert_episodes(env, episodes=5, seed=3)
        assert len(dataset) > 0
        assert summary["task_completion_rate"] == 1.0
        assert summary["pickup_delivery_match"]
        assert summary["mean_collisions"] == 0.0
    finally:
        env.close()


def test_phase2_software_rendering_returns_a_nonblank_rgb_frame():
    env = Phase2Warehouse(env_id="llm-mappo-small-1ag-v1", n_agents=1, max_steps=8)
    try:
        env.reset(seed=3)
        frame = render_warehouse_frame(env.env)
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        assert frame.shape[-1] == 3
        assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 5
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
