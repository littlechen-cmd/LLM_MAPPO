"""R1-B contract tests for the versioned Reward-v2 implementation."""

import numpy as np
import pytest
import sys
from types import SimpleNamespace

from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.reward_v2 import (
    RewardGoalSnapshot,
    reward_v2_progress_deltas,
    reward_v2_team_reward,
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

    # Local mean: mean([0, -0.05, -2.1, -0.5, +1]) = -0.33.
    # Team terms: +20 completion and -0.01 real-step cost.
    assert reward == pytest.approx(19.66)


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


def test_reward_v2_successful_pickup_uses_priority_weight_and_zero_switch_progress():
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
        assert transition.team_reward == pytest.approx(1.99)
    finally:
        environment.close()


def test_reward_v2_blocked_forward_is_minus_point_fifteen_before_step_cost():
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
        assert transition.team_reward == pytest.approx(-0.16)
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
        # +10 team completion, +0.1 local progress averaged over two AGVs,
        # and -0.01 team step cost.
        assert transition.team_reward == pytest.approx(10.04)
    finally:
        environment.close()


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


def test_e1_vector_worker_receives_the_selected_reward_version():
    from llm_mappo.e1_vector_env import _new_environment

    values = {
        "environment_id": "llm-mappo-medium-3ag-v1",
        "n_agents": 1,
        "max_steps": 8,
        "charge_threshold": 0.3,
        "charge_release_threshold": 0.8,
        "battery_cost_scale": 1.1,
        "deadlock_steps": 7,
        "dynamic_ingress_interval": 9,
        "batch_size_range": [1, 1],
        "queue_size": 1,
        "task_target": 1,
        "reward_version": "reward-v2",
    }
    run = SimpleNamespace(observation_schema="direct-goal-observation-v1")
    environment = _new_environment(values, run)
    try:
        assert environment.reward_version == "reward-v2"
    finally:
        environment.close()


def test_reward_version_changes_the_shadow_configuration_identity():
    from llm_mappo.shadow_state import ShadowStateAdapter

    legacy = Phase2Warehouse(n_agents=1, max_steps=8, reward_version="legacy-v1")
    revised = Phase2Warehouse(n_agents=1, max_steps=8, reward_version="reward-v2")
    try:
        legacy_hash = ShadowStateAdapter(legacy, code_commit="test").config_hash()
        revised_hash = ShadowStateAdapter(revised, code_commit="test").config_hash()
        assert legacy_hash != revised_hash
    finally:
        legacy.close()
        revised.close()


def test_e1_owner_runner_defaults_to_reward_v2_and_allows_legacy(monkeypatch):
    from scripts.run_e1_training import _arguments

    base = ["run_e1_training.py", "--records", "records.jsonl",
            "--run", "MAPPO-DG:9107", "--output-root", "artifacts"]
    monkeypatch.setattr(sys, "argv", base)
    assert _arguments().reward_version == "reward-v2"

    monkeypatch.setattr(sys, "argv", [*base, "--reward-version", "legacy-v1"])
    assert _arguments().reward_version == "legacy-v1"


def test_reward_v2_noop_has_only_the_team_step_cost():
    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=8,
        reward_version="reward-v2",
    )
    try:
        environment.reset(seed=3)
        transition = environment.step([Action.NOOP])
        assert transition.team_reward == pytest.approx(-0.01)
    finally:
        environment.close()
