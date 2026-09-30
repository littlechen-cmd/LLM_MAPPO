"""Regression checks for the provisional three-scene MAPPO contract."""

from dataclasses import replace
from collections import deque
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import yaml

from llm_mappo.formal_training import (
    EVALUATION_SEEDS, SCENES, SCENE_TRAINING_BUDGETS, make_diagnostic_experiment,
)
from llm_mappo.formal_layout import FORMAL_MAP_SHA256, FORMAL_MAP_PATH
from llm_mappo.mappo_baseline.fp_state import FPStateSchema, build_fp_state
from llm_mappo.mappo_baseline.runner import (
    BaselineRun, _continuous_episode_window, load_baseline_experiment,
)
from llm_mappo.mappo_baseline.vector_env import new_baseline_environment
from llm_mappo.planner import AStarPlanner
from rware.warehouse import Action, Direction


CONFIG_ROOT = Path("configs/formal_environment_training")


def test_formal_map_and_all_twelve_provisional_configs() -> None:
    assert sha256(FORMAL_MAP_PATH.read_bytes()).hexdigest() == FORMAL_MAP_SHA256
    source = Path("docs/formal_environment/environment.json")
    assert source.read_bytes() == FORMAL_MAP_PATH.read_bytes()
    paths = sorted(CONFIG_ROOT.glob("*.yaml"))
    assert len(paths) == 12
    contracts = [load_baseline_experiment(path) for path in paths]
    assert {item.environment["scene_id"] for item in contracts} == set(SCENES)
    assert all(item.approval_status == "provisional" for item in contracts)
    for item in contracts:
        budget = SCENE_TRAINING_BUDGETS[item.environment["scene_id"]]
        assert item.training["environment_steps"] == budget
        assert item.training["optimization_schedule_steps"] == budget
        assert item.training["milestone_checkpoint_steps"] == [budget // 2, budget]
        assert item.training["post_schedule_learning_rate"] == 0.0
    assert all(item.training["num_env_workers"] == 16 for item in contracts)
    assert all(item.training["rollout_length"] == 64 for item in contracts)
    assert all(tuple(item.evaluation["seeds"]) == EVALUATION_SEEDS
               for item in contracts)
    assert len(set(EVALUATION_SEEDS)) == 200


@pytest.mark.parametrize("field,value", [
    ("environment_steps", 600_000),
    ("optimization_schedule_steps", 300_000),
    ("milestone_checkpoint_steps", [300_000, 450_000]),
    ("post_schedule_learning_rate", 0.00001),
    ("checkpoint_interval_steps", 50_000),
])
def test_scene_budget_and_schedule_cannot_drift(tmp_path, field, value) -> None:
    source = CONFIG_ROOT / "tiny-3ag-baseline.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["training"][field] = value
    path = tmp_path / "altered.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_baseline_experiment(path)


@pytest.mark.parametrize("scene_id", tuple(SCENES))
def test_scene_reset_and_fp_dimensions(scene_id: str) -> None:
    scene = SCENES[scene_id]
    experiment = load_baseline_experiment(CONFIG_ROOT / f"{scene_id}-baseline.yaml")
    environment = new_baseline_environment(experiment.environment)
    try:
        first = environment.reset(seed=9300)
        initial = (
            [(agent.x, agent.y, agent.dir) for agent in environment.env.agents],
            [(task.shelf_id, task.label) for task in
             environment.env.task_queue.active_tasks],
        )
        second = environment.reset(seed=9300)
        assert np.array_equal(first, second)
        assert (
            [(agent.x, agent.y, agent.dir) for agent in environment.env.agents],
            [(task.shelf_id, task.label) for task in
             environment.env.task_queue.active_tasks],
        ) == initial
        assert len(environment.env.shelfs) == scene["n_shelves"]
        assert len(environment.env.task_queue.active_tasks) == scene["n_agents"]
        assert tuple(environment.env.charging_stations) == scene["charging_stations"]
        assert tuple(environment.env.picking_stations) == scene["picking_stations"]
        schema = FPStateSchema(
            scene["n_agents"], scene["n_shelves"],
            len(scene["charging_stations"]), len(scene["picking_stations"]),
        )
        assert build_fp_state(environment, schema).shape == (
            scene["n_agents"], schema.dimension
        )
        highways = environment.env.highways
        traversable = {(x, y) for y, row in enumerate(highways)
                       for x, value in enumerate(row) if value}
        if scene_id == "formal-8ag":
            assert len(traversable) == 896
        assert set(scene["charging_stations"]).issubset(traversable)
        assert set(scene["picking_stations"]).issubset(traversable)
        start = next(iter(traversable))
        reached = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x + 1, y), (x - 1, y),
                             (x, y + 1), (x, y - 1)):
                if neighbor in traversable and neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        assert reached == traversable
        for _ in range(3):
            environment.step(np.zeros(scene["n_agents"], dtype=int))
        assert not environment.env._dynamic_ingress_enabled()
    finally:
        environment.close()


def test_provisional_config_refuses_long_training_before_artifacts(tmp_path) -> None:
    experiment = load_baseline_experiment(CONFIG_ROOT / "tiny-3ag-baseline.yaml")
    with pytest.raises(ValueError, match="provisional configuration"):
        BaselineRun(experiment, device="cpu", output_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_diagnostic_contract_is_bounded() -> None:
    experiment = load_baseline_experiment(CONFIG_ROOT / "tiny-3ag-baseline.yaml")
    diagnostic = make_diagnostic_experiment(experiment, environment_steps=50_000)
    assert diagnostic.training["environment_steps"] == 50_000
    assert diagnostic.evaluation["seeds"] == [9300, 9301, 9302]
    assert diagnostic.config_sha256 != experiment.config_sha256
    assert diagnostic.config_sha256 == make_diagnostic_experiment(
        experiment, environment_steps=50_000
    ).config_sha256
    assert diagnostic.config_sha256 != make_diagnostic_experiment(
        experiment, environment_steps=49_000
    ).config_sha256
    with pytest.raises(ValueError, match="up to 50k"):
        make_diagnostic_experiment(experiment, environment_steps=50_001)
    with pytest.raises(ValueError, match="up to 50k"):
        make_diagnostic_experiment(
            replace(experiment, training={**experiment.training, "seed": 17}),
            environment_steps=1000,
        )


def test_continuous_evaluation_uses_actual_steps_as_denominator() -> None:
    window = _continuous_episode_window([
        {"steps": 800, "completed_tasks": 4, "energy_deaths": 1},
        {"steps": 200, "completed_tasks": 1, "energy_deaths": 0},
    ])
    assert window["completed_tasks_per_1000_actual_steps"] == 5.0
    assert window["total_energy_deaths"] == 1.0
    assert "mean_task_completion_rate" not in window


@pytest.mark.parametrize("field,value", [
    ("charge_threshold", 0.29),
    ("reward_energy_death_aggregation", "team-broadcast"),
    ("observation_schema", "direct-goal-observation-v1"),
])
def test_shared_scene_fields_cannot_drift(tmp_path, field, value) -> None:
    source = CONFIG_ROOT / "formal-8ag-baseline.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["environment"][field] = value
    path = tmp_path / "altered.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        load_baseline_experiment(path)


@pytest.mark.parametrize("scene_id", tuple(SCENES))
def test_delivery_replenishment_and_charging_cycle(scene_id: str) -> None:
    experiment = load_baseline_experiment(CONFIG_ROOT / f"{scene_id}-baseline.yaml")
    environment = new_baseline_environment(experiment.environment)
    try:
        environment.reset(seed=9301)
        warehouse = environment.env
        task = next(
            task for task in warehouse.task_queue.active_tasks
            if task.assigned_agent_id is not None
        )
        agent_index = task.assigned_agent_id - 1
        agent = warehouse.agents[agent_index]
        shelf = warehouse.shelfs[task.shelf_id - 1]
        home = warehouse._shelf_home[shelf.id]
        goal_x, goal_y = warehouse.goals[0]
        for other in warehouse.agents:
            if other is agent:
                continue
            station = warehouse.charging_stations[other.id - 1]
            other.x, other.y = station
        agent.x, agent.y, agent.dir = goal_x, goal_y - 1, Direction.DOWN
        agent.carrying_shelf = shelf
        shelf.x, shelf.y = agent.x, agent.y
        warehouse._recalc_grid()
        actions = [Action.NOOP] * environment.n_agents
        actions[agent_index] = Action.FORWARD
        completed = environment.step(actions)
        assert completed.metrics.completed_tasks == 1
        assert len(warehouse.task_queue.active_tasks) == environment.n_agents
        for _ in range(3):
            environment.step([Action.NOOP] * environment.n_agents)
        agent.x, agent.y = home
        shelf.x, shelf.y = home
        warehouse._recalc_grid()
        actions[agent_index] = Action.TOGGLE_LOAD
        environment.step(actions)
        assert agent.carrying_shelf is None
        assert warehouse.task_queue.task_for_agent(agent.id) is not None
        agent.x, agent.y = warehouse.charging_stations[agent_index]
        agent.battery = 0.25
        warehouse._recalc_grid()
        before = agent.battery
        environment.step([Action.NOOP] * environment.n_agents)
        assert agent.battery > before
        assert any(event["type"] == "charged" for event in warehouse.last_events)
    finally:
        environment.close()


def _follow_plan(environment, agent_id, target, *, limit=250):
    warehouse = environment.env
    index = agent_id - 1
    planner = AStarPlanner()
    actions = [Action.NOOP] * environment.n_agents
    for _ in range(limit):
        agent = warehouse.agents[index]
        if (agent.x, agent.y) == target:
            return
        plan = planner.plan(warehouse, agent_id, target)
        assert plan.reached_goal, f"No path to {target} from {(agent.x, agent.y)}"
        actions[index] = Action(int(np.argmax(plan.action_preferences)))
        environment.step(actions)
    pytest.fail(f"AGV did not reach {target} within {limit} actions")


@pytest.mark.parametrize("scene_id", tuple(SCENES))
@pytest.mark.parametrize("seed", (9300, 9301, 9302))
def test_real_pickup_loaded_motion_delivery_and_replenishment(
    scene_id: str, seed: int
) -> None:
    experiment = load_baseline_experiment(CONFIG_ROOT / f"{scene_id}-baseline.yaml")
    environment = new_baseline_environment(experiment.environment)
    try:
        environment.reset(seed=seed)
        warehouse = environment.env
        planner = AStarPlanner()
        candidate = None
        for task in warehouse.task_queue.active_tasks:
            if task.assigned_agent_id is None:
                continue
            agent = warehouse.agents[task.assigned_agent_id - 1]
            shelf = warehouse.shelfs[task.shelf_id - 1]
            if not planner.plan(warehouse, agent.id, (shelf.x, shelf.y)).reached_goal:
                continue
            original = (agent.x, agent.y, agent.carrying_shelf)
            agent.x, agent.y = shelf.x, shelf.y
            agent.carrying_shelf = shelf
            warehouse._recalc_grid()
            goals = tuple(
                goal for goal in warehouse.picking_stations
                if planner.plan(warehouse, agent.id, goal).reached_goal
            )
            agent.x, agent.y, agent.carrying_shelf = original
            warehouse._recalc_grid()
            if goals:
                candidate = task, min(
                    goals,
                    key=lambda goal: abs(goal[0] - shelf.x) + abs(goal[1] - shelf.y),
                )
                break
        assert candidate is not None, f"No viable assigned task: {scene_id} {seed}"
        task, goal = candidate
        agent_id = task.assigned_agent_id
        agent = warehouse.agents[agent_id - 1]
        shelf = warehouse.shelfs[task.shelf_id - 1]
        _follow_plan(environment, agent_id, (shelf.x, shelf.y), limit=350)
        actions = [Action.NOOP] * environment.n_agents
        actions[agent_id - 1] = Action.TOGGLE_LOAD
        environment.step(actions)
        assert agent.carrying_shelf is shelf
        start = (agent.x, agent.y)
        _follow_plan(environment, agent_id, goal, limit=350)
        assert (agent.x, agent.y) != start
        assert warehouse._completed_task_count() == 1
        assert len(warehouse.task_queue.active_tasks) == environment.n_agents
    finally:
        environment.close()


@pytest.mark.parametrize("scene_id", ("medium-5ag", "formal-8ag"))
def test_larger_maps_have_pickup_and_loaded_delivery_routes(scene_id: str) -> None:
    experiment = load_baseline_experiment(CONFIG_ROOT / f"{scene_id}-baseline.yaml")
    environment = new_baseline_environment(experiment.environment)
    try:
        environment.reset(seed=9302)
        warehouse = environment.env
        task = next(task for task in warehouse.task_queue.active_tasks
                    if task.assigned_agent_id is not None)
        agent = warehouse.agents[task.assigned_agent_id - 1]
        shelf = warehouse.shelfs[task.shelf_id - 1]
        planner = AStarPlanner()
        assert planner.plan(warehouse, agent.id, (shelf.x, shelf.y)).reached_goal
        for other in warehouse.agents:
            if other is not agent:
                other.x, other.y = warehouse.charging_stations[other.id - 1]
        agent.x, agent.y = shelf.x, shelf.y
        agent.carrying_shelf = shelf
        warehouse._recalc_grid()
        assert any(planner.plan(warehouse, agent.id, goal).reached_goal
                   for goal in warehouse.picking_stations)
    finally:
        environment.close()
