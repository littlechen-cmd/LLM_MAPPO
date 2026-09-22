"""R1-B contract tests for the versioned Reward-v2 implementation."""

import numpy as np
import pytest

from llm_mappo.warehouse_environment import Phase2Warehouse
from llm_mappo.reward_v2 import (
    RewardGoalSnapshot,
    reward_v2_progress_deltas,
    reward_v2_team_reward,
    reward_v2_team_reward_breakdown,
    static_grid_distance,
)
from rware.warehouse import Action, Direction


def test_reward_v2_matches_the_frozen_team_and_local_formula():
    raw_rewards = np.asarray([10.0, -0.05, -2.0, 0.0, 0.0])

    reward = reward_v2_team_reward(
        raw_rewards=raw_rewards,
        events=[
            {"type": "task_completed", "task_id": 11, "agent_id": 1},
            {"type": "blocked_forward", "agent_id": 2},
            {"type": "collision", "agent_id": 3},
        ],
        task_weights={11: 2.0},
        pickup_weights={5: 0.5},
        progress_deltas=np.asarray([0.0, 1.0, -1.0, 0.0, 0.0]),
        low_battery_penalties=np.asarray([0.0, 0.0, 0.0, -0.5, 0.0]),
        legacy_blocked_forward_penalty=0.05,
    )

    # Reward-v2 replaces old raw blockers/collisions with -0.1 each and
    # retains the shared completion, pickup, energy and real-step terms.
    assert reward == pytest.approx(9.955)


def test_reward_v2_progress_is_signed_and_zero_when_the_goal_switches():
    before = [
        RewardGoalSnapshot(("task", 1), (3, 0), 3),
        RewardGoalSnapshot(("task", 2), (3, 0), 2),
        RewardGoalSnapshot(("task", 3), (3, 0), 1),
    ]
    after = [
        RewardGoalSnapshot(("task", 1), (3, 0), 2),
        RewardGoalSnapshot(("task", 2), (3, 0), 3),
        RewardGoalSnapshot(("delivery", 3), (3, 0), 0),
    ]

    assert reward_v2_progress_deltas(before, after).tolist() == [1.0, -1.0, 0.0]


def test_reward_v2_successful_pickup_is_priority_independent_and_zero_switch_progress():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        reward_version="reward-v2",
    )
    try:
        environment.reset(seed=9)
        task = environment.env.task_queue.task_for_agent(1)
        shelf = environment.env.shelfs[task.shelf_id - 1]
        agent = environment.env.agents[0]
        agent.x, agent.y = shelf.x, shelf.y
        environment.env._recalc_grid()

        transition = environment.step([Action.TOGGLE_LOAD])

        assert transition.metrics.picked_tasks == 1
        assert transition.team_reward == pytest.approx(0.495)
    finally:
        environment.close()


def test_reward_v2_blocked_forward_is_minus_point_one_before_step_cost():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        reward_version="reward-v2",
    )
    try:
        environment.reset(seed=3)
        agent = environment.env.agents[0]
        agent.x, agent.y, agent.dir = 0, 0, Direction.LEFT
        environment.env._recalc_grid()

        transition = environment.step([Action.FORWARD])

        assert transition.info["blocked_forwards"] == 1
        assert transition.team_reward == pytest.approx(-0.105)
    finally:
        environment.close()


def test_reward_v2_completion_is_not_diluted_by_the_agent_mean():
    environment = Phase2Warehouse(
        n_agents=2,
        max_steps=8,
        reward_version="reward-v2",
    )
    try:
        environment.reset(seed=8)
        warehouse = environment.env
        task = warehouse.task_queue.task_for_agent(1)
        shelf = warehouse.shelfs[task.shelf_id - 1]
        goal_x, goal_y = warehouse.goals[0]
        first, second = warehouse.agents
        first.x, first.y, first.dir = goal_x, goal_y - 1, Direction.DOWN
        second.x, second.y = 9, 0
        shelf.x, shelf.y = goal_x, goal_y - 1
        first.carrying_shelf = shelf
        warehouse._recalc_grid()

        transition = environment.step([Action.FORWARD, Action.NOOP])

        assert transition.metrics.completed_tasks == 1
        # +5 team completion, +0.1 local progress averaged over two AGVs,
        # and -0.005 team step cost.
        assert transition.team_reward == pytest.approx(5.045)
    finally:
        environment.close()


def test_reward_v2_completed_delivery_is_better_than_equally_long_stagnation():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        reward_version="reward-v2",
    )
    try:
        environment.reset(seed=8)
        warehouse = environment.env
        task = warehouse.task_queue.task_for_agent(1)
        shelf = warehouse.shelfs[task.shelf_id - 1]
        goal_x, goal_y = warehouse.goals[0]
        agent = warehouse.agents[0]
        agent.x, agent.y, agent.dir = goal_x, goal_y - 1, Direction.DOWN
        shelf.x, shelf.y = goal_x, goal_y - 1
        agent.carrying_shelf = shelf
        warehouse._recalc_grid()

        productive = environment.step([Action.FORWARD]).team_reward
        environment.reset(seed=8)
        stagnant = environment.step([Action.NOOP]).team_reward

        assert productive > stagnant
    finally:
        environment.close()


def test_reward_v2_low_battery_penalty_is_charged_only_on_threshold_entry():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        charge_threshold=0.3,
        reward_version="reward-v2",
    )
    try:
        environment.reset(seed=3)
        agent = environment.env.agents[0]
        agent.battery = 0.3001

        entry = environment.step([Action.NOOP])
        sustained = environment.step([Action.NOOP])

        assert entry.team_reward == pytest.approx(-0.505)
        assert sustained.team_reward == pytest.approx(-0.005)
    finally:
        environment.close()


def test_reward_v2_low_battery_delivery_outranks_equal_length_stagnation():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        charge_threshold=0.3,
        reward_version="reward-v2",
    )
    try:
        environment.reset(seed=8)
        warehouse = environment.env
        task = warehouse.task_queue.task_for_agent(1)
        shelf = warehouse.shelfs[task.shelf_id - 1]
        goal_x, goal_y = warehouse.goals[0]
        agent = warehouse.agents[0]
        agent.x, agent.y, agent.dir = goal_x, goal_y - 1, Direction.DOWN
        agent.battery = 0.301
        shelf.x, shelf.y = goal_x, goal_y - 1
        agent.carrying_shelf = shelf
        warehouse._recalc_grid()

        productive = environment.step([Action.FORWARD]).team_reward
        environment.reset(seed=8)
        stagnant = environment.step([Action.NOOP]).team_reward

        assert productive > stagnant
    finally:
        environment.close()


def test_reward_v2_rewards_only_a_real_charge_event():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        charge_threshold=0.3,
        reward_version="reward-v2",
        charge_event_reward=0.05,
    )
    try:
        environment.reset(seed=3)
        agent = environment.env.agents[0]
        charger = environment.env.charging_stations[0]
        agent.x, agent.y = charger
        agent.battery = 0.29
        environment.env._recalc_grid()

        transition = environment.step([Action.NOOP])

        assert transition.metrics.reward_charging == pytest.approx(0.05)
        assert transition.team_reward == pytest.approx(0.045)
    finally:
        environment.close()


def test_reward_v2_accepts_configured_block_and_collision_penalties():
    breakdown = reward_v2_team_reward_breakdown(
        raw_rewards=np.asarray([-2.05]),
        events=[
            {"type": "blocked_forward", "agent_id": 1},
            {"type": "collision", "agent_id": 1},
        ],
        task_weights={},
        pickup_weights={},
        progress_deltas=np.asarray([0.0]),
        low_battery_penalties=np.asarray([0.0]),
        charge_rewards=np.asarray([0.0]),
        legacy_blocked_forward_penalty=0.05,
        blocked_forward_penalty=0.15,
        collision_penalty=0.5,
    )

    assert breakdown.blocked_forward == pytest.approx(-0.15)
    assert breakdown.collision == pytest.approx(-0.5)


def test_reward_v2_applies_terminal_deadlock_penalty_once():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        deadlock_steps=1,
        reward_version="reward-v2",
        reward_deadlock_penalty=5.0,
    )
    try:
        environment.reset(seed=3)

        first = environment.step([Action.NOOP])
        assert first.metrics.deadlocked is True
        assert first.metrics.reward_deadlock == pytest.approx(-5.0)
        assert first.team_reward == pytest.approx(-5.005)

        second = environment.step([Action.NOOP])
        assert second.metrics.reward_deadlock == pytest.approx(-5.0)
        assert second.team_reward == pytest.approx(-0.005)
    finally:
        environment.close()


def test_rware_delivery_replenishment_keeps_training_request_queue_full():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=16,
        deadlock_steps=15,
        reward_version="reward-v2",
        batch_interval=40,
        batch_size_range=(1, 1),
        request_queue_size=1,
        task_completion_target=6,
        replenishment_mode="rware_delivery",
    )
    try:
        environment.reset(seed=3)
        task = environment.env.task_queue.active_tasks[0]
        delivered_shelf = environment.env.shelfs[task.shelf_id - 1]
        agent = environment.env.agents[0]
        goal_x, goal_y = environment.env.goals[0]
        agent.x, agent.y, agent.dir = goal_x, goal_y - 1, Direction.DOWN
        agent.carrying_shelf = delivered_shelf
        delivered_shelf.x, delivered_shelf.y = agent.x, agent.y
        environment.env._recalc_grid()

        transition = environment.step([Action.FORWARD])

        active = environment.env.task_queue.active_tasks
        assert transition.metrics.completed_tasks == 1
        assert len(active) == 1
        assert active[0].shelf_id != delivered_shelf.id
        assert len(environment.env.request_queue) == 1
        assert environment.env.request_queue[0].id == active[0].shelf_id
    finally:
        environment.close()


@pytest.mark.parametrize("n_agents", (1, 5))
def test_reward_v2_fixed_seed_rollout_reconciles_the_full_reward_ledger(n_agents):
    """A deterministic rollout cannot hide a reward term outside the ledger."""
    snapshots = []
    for _ in range(2):
        environment = Phase2Warehouse(
            n_agents=n_agents,
            max_steps=12,
            deadlock_steps=11,
            reward_version="reward-v2",
            replenishment_mode="rware_delivery",
        )
        try:
            environment.reset(seed=41)
            rewards = []
            events = []
            for _ in range(8):
                transition = environment.step([Action.NOOP] * n_agents)
                rewards.append(transition.team_reward)
                events.append(tuple(transition.info["events"]))
            metrics = transition.metrics
            ledger_total = sum(
                getattr(metrics, name)
                for name in (
                    "reward_completion",
                    "reward_pickup",
                    "reward_progress",
                    "reward_blocked_forward",
                    "reward_collision",
                    "reward_charging",
                    "reward_deadlock",
                    "reward_energy",
                    "reward_step_cost",
                )
            )
            assert metrics.reward == pytest.approx(sum(rewards))
            assert ledger_total == pytest.approx(metrics.reward)
            snapshots.append((rewards, events, metrics.as_dict()))
        finally:
            environment.close()

    assert snapshots[0] == snapshots[1]


def test_static_grid_distance_rewards_a_necessary_manhattan_detour():
    """A valid first detour step can increase Manhattan distance yet progress."""
    start = (0, 0)
    goal = (2, 0)
    blocked = {(1, 0), (1, 1)}

    before = static_grid_distance(start, goal, width=3, height=3, blocked=blocked)
    after = static_grid_distance((0, 1), goal, width=3, height=3, blocked=blocked)

    assert before == 6
    assert after == 5
    assert abs(goal[0] - 0) + abs(goal[1] - 1) > 2


def test_reward_versions_are_explicit_and_legacy_remains_available():
    with pytest.raises(ValueError, match="reward_version"):
        Phase2Warehouse(n_agents=1, max_steps=8, reward_version="unknown")

    legacy = Phase2Warehouse(n_agents=1, max_steps=8, reward_version="legacy-v1")
    try:
        legacy.reset(seed=3)
        agent = legacy.env.agents[0]
        agent.x, agent.y, agent.dir = 0, 0, Direction.LEFT
        legacy.env._recalc_grid()
        transition = legacy.step([Action.FORWARD])
        assert transition.team_reward == pytest.approx(-0.06)
    finally:
        legacy.close()


def test_reward_v2_noop_has_only_the_team_step_cost():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        reward_version="reward-v2",
    )
    try:
        environment.reset(seed=3)
        transition = environment.step([Action.NOOP])
        assert transition.team_reward == pytest.approx(-0.005)
    finally:
        environment.close()


def test_reward_v2_breakdown_records_every_candidate_component():
    breakdown = reward_v2_team_reward_breakdown(
    raw_rewards=np.asarray([10.0, -2.05]),
        events=[
            {"type": "task_completed", "task_id": 1, "agent_id": 1},
            {"type": "blocked_forward", "agent_id": 2},
            {"type": "collision", "agent_id": 2},
        ],
        task_weights={1: 2.0},
        pickup_weights={1: 2.0},
        progress_deltas=np.asarray([1.0, -1.0]),
        low_battery_penalties=np.asarray([0.0, -0.5]),
        legacy_blocked_forward_penalty=0.05,
    )

    assert breakdown.completion == pytest.approx(10.0)
    assert breakdown.pickup == pytest.approx(0.25)
    assert breakdown.progress == pytest.approx(0.0)
    assert breakdown.blocked_forward == pytest.approx(-0.05)
    assert breakdown.collision == pytest.approx(-0.05)
    assert breakdown.energy == pytest.approx(-0.25)
    assert breakdown.step_cost == pytest.approx(-0.005)
    assert breakdown.team_reward == pytest.approx(9.895)
