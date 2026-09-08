import numpy as np
import pytest
import torch

from llm_mappo.mappo_baseline.fp_state import (
    FPStateSchema,
    build_active_masks,
    build_fp_state,
)
from llm_mappo.mappo_baseline.networks import MAPPOActor, MAPPOFPCritic
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage
from llm_mappo.mappo_baseline.structured_observation import (
    StructuredObservationSchema,
    build_structured_observations,
)
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer
from llm_mappo.mappo_baseline.value_norm import ValueNorm
from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.phase2 import Phase2Warehouse
from rware.warehouse import Direction


def _environment():
    environment = Phase2Warehouse(
        n_agents=3,
        max_steps=1000,
        env_id="llm-mappo-medium-3ag-v1",
        batch_interval=40,
        batch_size_range=(1, 3),
        initial_priority_label="A",
        request_queue_size=4,
        task_completion_target=9,
        battery_cost_scale=1.10,
        charge_threshold=0.30,
        charge_release_threshold=0.80,
        deadlock_steps=180,
        waypoint_reward=0.10,
        reward_version="legacy-v1",
        observation_schema=ObservationSchema.DIRECT_GOAL_V1,
    )
    environment.reset(seed=7)
    return environment


def test_fp_state_is_deterministic_agent_specific_and_teacher_free():
    environment = _environment()
    try:
        schema = FPStateSchema(n_agents=3, n_shelves=144)
        first = build_fp_state(environment, schema)
        second = build_fp_state(environment, schema)
        assert first.shape == (3, schema.dimension)
        np.testing.assert_array_equal(first, second)
        assert not np.array_equal(first[0], first[1])
        np.testing.assert_array_equal(
            first[:, :schema.common_dimension],
            np.repeat(first[:1, :schema.common_dimension], 3, axis=0),
        )
        forbidden = ("astar", "llm", "semantic", "calibration", "reward", "future")
        assert not any(
            token in name.lower()
            for name in schema.feature_names
            for token in forbidden
        )
    finally:
        environment.close()


def test_fp_state_records_current_shelf_and_station_occupancy() -> None:
    environment = _environment()
    try:
        schema = FPStateSchema(n_agents=3, n_shelves=144)
        assert "shelf_0_x" in schema.common_feature_names
        assert "shelf_0_carried" in schema.common_feature_names
        assert "charger_0_occupied" in schema.common_feature_names
        assert "picking_0_occupied" in schema.common_feature_names
    finally:
        environment.close()


def test_structured_actor_observation_is_teacher_free_and_pads_missing_peers():
    environment = _environment()
    try:
        schema = StructuredObservationSchema()
        observations = build_structured_observations(environment, schema)
        assert observations.shape == (3, 208)
        forbidden = (
            "astar", "waypoint", "planner", "llm", "semantic", "yield",
            "queue", "future", "calibration",
        )
        assert not any(
            token in name.lower()
            for name in schema.feature_names
            for token in forbidden
        )
        assert environment.planner_query_counter.count == 0
        peer_width = len(schema.peer_feature_names) // schema.n_neighbours
        final_peer_start = (
            len(schema.self_feature_names)
            + len(schema.goal_feature_names)
            + len(schema.map_feature_names)
            + 2 * peer_width
        )
        np.testing.assert_array_equal(
            observations[:, final_peer_start:final_peer_start + peer_width],
            np.zeros((3, peer_width), dtype=np.float32),
        )
    finally:
        environment.close()


def test_structured_actor_peer_order_uses_distance_then_relative_coordinates():
    environment = _environment()
    try:
        first, second, third = environment.env.agents
        first.x, first.y, first.dir = 5, 5, Direction.UP
        second.x, second.y, second.dir = 6, 5, Direction.RIGHT
        third.x, third.y, third.dir = 5, 4, Direction.DOWN
        environment.env._recalc_grid()
        schema = StructuredObservationSchema()
        observation = build_structured_observations(environment, schema)[0]
        peer_start = (
            len(schema.self_feature_names)
            + len(schema.goal_feature_names)
            + len(schema.map_feature_names)
        )
        scale = np.hypot(19, 23)
        # Both peers are one cell away; (dx=0, dy=-1) precedes (dx=1, dy=0).
        np.testing.assert_allclose(observation[peer_start:peer_start + 2], [1 / scale, 0])
    finally:
        environment.close()


def test_fp_active_mask_marks_dead_agent_without_erasing_team_state():
    environment = _environment()
    try:
        environment.env.agents[1].dead = True
        np.testing.assert_array_equal(
            build_active_masks(environment), np.asarray([1.0, 0.0, 1.0])
        )
        states = build_fp_state(
            environment, FPStateSchema(n_agents=3, n_shelves=144)
        )
        assert states[0].any()
        assert states[1].any()
        assert not states[1, : FPStateSchema(3).common_dimension].any()
    finally:
        environment.close()


def test_actor_critic_and_value_norm_keep_agent_axis():
    torch.manual_seed(3)
    actor = MAPPOActor(observation_dim=613, action_dim=5)
    critic = MAPPOFPCritic(state_dim=68)
    observations = torch.randn(4, 3, 613)
    states = torch.randn(4, 3, 68)
    logits, actor_hidden = actor(observations, torch.zeros(4, 3, 64))
    values, critic_hidden = critic(states, torch.zeros(4, 3, 64))
    assert logits.shape == (4, 3, 5)
    assert actor_hidden.shape == (4, 3, 64)
    assert values.shape == (4, 3)
    assert critic_hidden.shape == (4, 3, 64)
    assert not torch.equal(values[:, 0], values[:, 1])
    actor_projection = actor.action_head.weight.detach()
    torch.testing.assert_close(
        actor_projection @ actor_projection.T,
        torch.eye(5) * 0.01**2,
        atol=1e-7,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        critic.value_head.weight.detach().norm(), torch.tensor(1.0),
        atol=1e-6, rtol=1e-6,
    )

    normalizer = ValueNorm(shape=(1,))
    returns = torch.tensor([[-3.0], [1.0], [4.0], [9.0]])
    normalizer.update(returns)
    restored = normalizer.denormalize(normalizer.normalize(returns))
    torch.testing.assert_close(restored, returns, atol=1e-5, rtol=1e-5)


def test_rollout_gae_is_per_worker_per_agent_and_stops_at_episode_boundary():
    storage = MAPPORolloutStorage(
        rollout_length=3,
        num_workers=2,
        num_agents=2,
        observation_dim=2,
        state_dim=3,
        action_dim=2,
    )
    for step in range(3):
        rewards = np.asarray(
            [[1.0, 10.0], [100.0, 1000.0]], dtype=np.float32
        )
        dones = np.zeros((2, 2), dtype=bool)
        if step == 1:
            dones[0] = True
        storage.insert(
            observations=np.zeros((2, 2, 2), dtype=np.float32),
            fp_states=np.zeros((2, 2, 3), dtype=np.float32),
            actions=np.zeros((2, 2), dtype=np.int64),
            action_masks=np.ones((2, 2, 2), dtype=bool),
            log_probs=np.zeros((2, 2), dtype=np.float32),
            values=np.zeros((2, 2), dtype=np.float32),
            rewards=rewards,
            dones=dones,
            active_masks=np.ones((2, 2), dtype=np.float32),
            actor_hidden_states=np.zeros((2, 2, 64), dtype=np.float32),
            critic_hidden_states=np.zeros((2, 2, 64), dtype=np.float32),
        )
    batch = storage.compute_returns(
        next_values=np.zeros((2, 2), dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_array_equal(batch.advantages[:, 0, 0], [2.0, 1.0, 1.0])
    np.testing.assert_array_equal(batch.advantages[:, 0, 1], [20.0, 10.0, 10.0])
    np.testing.assert_array_equal(batch.advantages[:, 1, 0], [300.0, 200.0, 100.0])
    np.testing.assert_array_equal(batch.advantages[:, 1, 1], [3000.0, 2000.0, 1000.0])


def test_trainer_uses_separate_optimizers_and_reports_canonical_losses():
    config = MAPPOConfig(
        observation_dim=613,
        state_dim=68,
        action_dim=5,
        ppo_epochs=2,
        num_minibatches=1,
    )
    trainer = MAPPOTrainer(config, device="cpu")
    assert trainer.actor_optimizer is not trainer.critic_optimizer
    assert not set(map(id, trainer.actor.parameters())) & set(
        map(id, trainer.critic.parameters())
    )

    storage = MAPPORolloutStorage(2, 1, 3, 613, 68, 5)
    rng = np.random.default_rng(4)
    actor_hidden = np.zeros((1, 3, 64), dtype=np.float32)
    critic_hidden = np.zeros((1, 3, 64), dtype=np.float32)
    for _ in range(2):
        observations = rng.normal(size=(1, 3, 613)).astype(np.float32)
        states = rng.normal(size=(1, 3, 68)).astype(np.float32)
        masks = np.ones((1, 3, 5), dtype=bool)
        actions, log_probs, values, value_predictions, next_actor, next_critic = trainer.act(
            observations, states, masks, actor_hidden, critic_hidden, deterministic=False
        )
        storage.insert(
            observations=observations,
            fp_states=states,
            actions=actions,
            action_masks=masks,
            log_probs=log_probs,
            values=values,
            value_predictions=value_predictions,
            rewards=np.ones((1, 3), dtype=np.float32),
            dones=np.zeros((1, 3), dtype=bool),
            active_masks=np.ones((1, 3), dtype=np.float32),
            actor_hidden_states=actor_hidden,
            critic_hidden_states=critic_hidden,
        )
        actor_hidden = next_actor
        critic_hidden = next_critic
    batch = storage.compute_returns(
        np.zeros((1, 3), dtype=np.float32), gamma=0.99, gae_lambda=0.95
    )
    metrics = trainer.update(batch)
    assert metrics["optimizer_steps"] == 2
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "actor_grad_norm",
        "critic_grad_norm",
    ):
        assert np.isfinite(metrics[key])


def test_storage_rejects_wrong_agent_axis():
    storage = MAPPORolloutStorage(1, 1, 3, 2, 3, 2)
    with pytest.raises(ValueError, match="observations"):
        storage.insert(
            observations=np.zeros((1, 2, 2), dtype=np.float32),
            fp_states=np.zeros((1, 3, 3), dtype=np.float32),
            actions=np.zeros((1, 3), dtype=np.int64),
            action_masks=np.ones((1, 3, 2), dtype=bool),
            log_probs=np.zeros((1, 3), dtype=np.float32),
            values=np.zeros((1, 3), dtype=np.float32),
            rewards=np.zeros((1, 3), dtype=np.float32),
            dones=np.zeros((1, 3), dtype=bool),
            active_masks=np.ones((1, 3), dtype=np.float32),
        )
