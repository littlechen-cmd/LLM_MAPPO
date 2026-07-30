import gymnasium as gym
import pytest

import rware
from llm_mappo.environment import DynamicWarehouse
from llm_mappo.planner import AStarPlanner
from llm_mappo.rules import TaskQueue
from llm_mappo.types import PlannerEvent, PriorityAdjustment, TaskStatus
from rware.warehouse import Action, Direction, RewardType


def make_env(n_agents=2, **kwargs):
    return DynamicWarehouse(
        3,
        2,
        1,
        n_agents,
        0,
        1,
        n_agents,
        None,
        100,
        RewardType.INDIVIDUAL,
        batch_interval=kwargs.pop("batch_interval", 20),
        batch_size_range=kwargs.pop("batch_size_range", (1, 1)),
        charging_stations=kwargs.pop(
            "charging_stations", ((0, 0), (9, 0))[:n_agents]
        ),
        **kwargs,
    )


def test_registered_dynamic_environment_exposes_phase1_info():
    env = gym.make("llm-mappo-medium-3ag-v1")
    observations, info = env.reset(seed=5)

    assert len(observations) == 3
    assert info["queue"] == ["A1", "A2", "A3"]
    assert len(info["charging_stations"]) == 3
    assert all(agent["battery"] == 1.0 for agent in info["agents"])

    env.close()


def test_task_queue_filters_low_battery_and_validates_adjustments_atomically():
    queue = TaskQueue()
    queue.create_batch([10, 11], batch_id=1, letter="A", arrival_step=0)
    queue.create_batch([12], batch_id=2, letter="B", arrival_step=5)

    assert queue.assign_next(agent_id=1, battery=0.09) is None
    assert queue.assign_next(agent_id=1, battery=0.1).label == "A1"

    changed = queue.apply_adjustments(
        [
            PriorityAdjustment("A1", "B1", "swap"),
            PriorityAdjustment("B1", "A1", "swap"),
        ]
    )
    assert {task.label for task in changed} == {"A1", "B1"}

    before = [task.label for task in queue.active_tasks]
    with pytest.raises(ValueError, match="numeric suffix"):
        queue.apply_adjustments([PriorityAdjustment("A2", "B9", "invalid")])
    assert [task.label for task in queue.active_tasks] == before


def test_dynamic_batches_and_charging_are_reported():
    env = make_env(batch_interval=2)
    env.reset(seed=4)
    env.agents[0].x, env.agents[0].y = 0, 0
    env.agents[0].battery = 0.5
    env._recalc_grid()

    _, _, _, _, info = env.step([Action.NOOP, Action.NOOP])
    assert info["agents"][0]["battery"] == pytest.approx(0.52)

    _, _, _, _, info = env.step([Action.NOOP, Action.NOOP])
    assert "B1" in info["queue"]
    assert any(event["type"] == "batch_arrived" for event in info["events"])


def test_collision_penalty_is_applied_to_each_forward_initiator():
    env = make_env(batch_interval=100)
    env.reset(seed=3)
    env.agents[0].x, env.agents[0].y, env.agents[0].dir = 1, 0, Direction.RIGHT
    env.agents[1].x, env.agents[1].y, env.agents[1].dir = 2, 0, Direction.LEFT
    env._recalc_grid()

    _, rewards, _, _, info = env.step([Action.FORWARD, Action.FORWARD])

    assert info["collisions"] == 2
    assert rewards[0] == pytest.approx(-2.0)
    assert rewards[1] == pytest.approx(-2.0)


def test_delivery_completes_task_and_enforces_picking_lock():
    env = make_env(batch_interval=100)
    env.reset(seed=8)
    task = env.task_queue.task_for_agent(1)
    shelf = env.shelfs[task.shelf_id - 1]
    goal_x, goal_y = env.goals[0]
    env.agents[0].x, env.agents[0].y, env.agents[0].dir = (
        goal_x,
        goal_y - 1,
        Direction.DOWN,
    )
    env.agents[1].x, env.agents[1].y = 9, 0
    shelf.x, shelf.y = goal_x, goal_y - 1
    env.agents[0].carrying_shelf = shelf
    env._recalc_grid()

    _, rewards, _, _, info = env.step([Action.FORWARD, Action.NOOP])

    completed = next(item for item in info["tasks"] if item["task_id"] == task.task_id)
    assert completed["status"] == TaskStatus.COMPLETED.value
    assert info["agents"][0]["picking_lock_steps"] == 3
    assert rewards[0] == pytest.approx(5.0)

    position = env.agents[0].x, env.agents[0].y
    env.step([Action.FORWARD, Action.NOOP])
    assert (env.agents[0].x, env.agents[0].y) == position
    assert env.agents[0].picking_lock_steps == 2

    env.step([Action.NOOP, Action.NOOP])
    _, _, _, _, info = env.step([Action.NOOP, Action.NOOP])
    assert env.agents[0].picking_lock_steps == 0
    assert env.agents[0].carrying_shelf is None
    assert any(event["type"] == "picking_complete" for event in info["events"])


def test_astar_returns_orientation_aware_preferences():
    env = make_env(batch_interval=100)
    env.reset(seed=1)
    env.agents[0].x, env.agents[0].y, env.agents[0].dir = 0, 0, Direction.RIGHT
    env.agents[1].x, env.agents[1].y = 9, 3
    env._recalc_grid()

    planner = AStarPlanner()
    plan = planner.plan(env, agent_id=1, goal=(2, 0))

    assert plan.waypoints[0] == (0, 0)
    assert plan.waypoints[-1] == (2, 0)
    assert plan.action_preferences[Action.FORWARD.value] == pytest.approx(0.82)
    assert planner.status_for_progress(30, False) == PlannerEvent.STALLED
    assert planner.status_for_progress(0, True) == PlannerEvent.BLOCKED
