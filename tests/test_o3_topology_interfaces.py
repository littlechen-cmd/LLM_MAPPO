"""O3-D deterministic, planner-free, semantic, Teacher, and safety checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from llm_mappo.o3_topologies import (
    O3_ENVIRONMENT_IDS,
    get_o3_topology,
    make_o3_evaluation_environment,
)
from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.optimization_training import OptimizationTrainer
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.pure_motion_teacher import PureMotionTeacher
from llm_mappo.shadow_state import ShadowStateAdapter
from rware.warehouse import Action, Direction


TEST_SEEDS = (9301, 9302)


def _new_o3(environment_id, schema=ObservationSchema.DIRECT_GOAL_V1):
    return make_o3_evaluation_environment(
        environment_id,
        observation_schema=schema,
    )


def _task_state(environment):
    warehouse = environment.env
    return {
        "agents": tuple(
            (
                item.id,
                item.x,
                item.y,
                item.dir.name,
                item.task_id,
                float(item.battery),
            )
            for item in warehouse.agents
        ),
        "tasks": warehouse.task_queue.as_dict(),
        "queue": tuple(item.id for item in warehouse.request_queue),
        "events": tuple(
            tuple(sorted(event.items())) for event in warehouse.last_events
        ),
    }


@pytest.mark.parametrize("environment_id", O3_ENVIRONMENT_IDS)
@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_o3_reset_and_dynamic_ingress_short_trajectory_are_byte_deterministic(
    environment_id,
    seed,
):
    """Catch topology-specific reset or ingress randomness escaping seed control."""
    first = _new_o3(environment_id)
    second = _new_o3(environment_id)
    try:
        first_observation = first.reset(seed=seed)
        second_observation = second.reset(seed=seed)
        assert first_observation.tobytes() == second_observation.tobytes()
        assert _task_state(first) == _task_state(second)
        assert first.action_masks().tobytes() == second.action_masks().tobytes()

        actions = np.zeros(5, dtype=np.int64)
        for _ in range(41):
            first_step = first.step(actions)
            second_step = second.step(actions)
            assert first_step.observations.tobytes() == (
                second_step.observations.tobytes()
            )
            assert first.action_masks().tobytes() == second.action_masks().tobytes()
            assert first_step.team_reward == second_step.team_reward
            assert first_step.terminated == second_step.terminated
            assert first_step.truncated == second_step.truncated
            assert first_step.info == second_step.info
            assert _task_state(first) == _task_state(second)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("environment_id", O3_ENVIRONMENT_IDS)
def test_o3_direct_goal_and_no_goal_hint_keep_the_frozen_613d_boundary(
    environment_id,
):
    """Catch padding, truncation, or non-geometry slots changing on O3 maps."""
    direct = _new_o3(environment_id, ObservationSchema.DIRECT_GOAL_V1)
    no_goal = _new_o3(
        environment_id,
        ObservationSchema.NO_GEOMETRIC_GOAL_HINT_V1,
    )
    try:
        direct_values = direct.reset(seed=9301)
        no_goal_values = no_goal.reset(seed=9301)
        assert direct_values.shape == no_goal_values.shape == (5, 613)
        assert direct_values.dtype == no_goal_values.dtype == np.float32
        assert np.array_equal(direct_values[:, :586], no_goal_values[:, :586])
        assert np.array_equal(direct_values[:, 595:], no_goal_values[:, 595:])
        assert np.array_equal(no_goal_values[:, 586:595], np.zeros((5, 9)))
        assert np.array_equal(direct_values[:, 588:595], np.zeros((5, 7)))
    finally:
        direct.close()
        no_goal.close()


@pytest.mark.parametrize("environment_id", O3_ENVIRONMENT_IDS)
def test_o3_interfaces_match_core_schema_without_planner_queries(environment_id):
    """Catch O3-only action, mask, info, or planner behavior."""
    class ThrowingPlanner:
        def plan(self, *args, **kwargs):
            raise AssertionError("O3 DirectGoal must not query a planner")

    o3_environment = _new_o3(environment_id)
    core = Phase2Warehouse(
        env_id="llm-mappo-medium-3ag-v1",
        n_agents=5,
        max_steps=1000,
        charge_threshold=0.30,
        charge_release_threshold=0.80,
        battery_cost_scale=1.10,
        deadlock_steps=180,
        batch_interval=40,
        batch_size_range=(4, 8),
        request_queue_size=8,
        task_completion_target=50,
        observation_schema=ObservationSchema.DIRECT_GOAL_V1,
    )
    try:
        o3_environment._planner = ThrowingPlanner()
        assert o3_environment.reset(seed=9301).shape == (5, 613)
        assert core.reset(seed=9301).shape == (5, 613)
        o3_step = o3_environment.step([Action.NOOP] * 5)
        core_step = core.step([Action.NOOP] * 5)
        assert [space.n for space in o3_environment.env.action_space] == [5] * 5
        assert o3_environment.action_masks().shape == core.action_masks().shape == (
            5,
            5,
        )
        assert o3_environment.action_masks().dtype == bool
        assert set(o3_step.info) == set(core_step.info)
        assert isinstance(o3_step.team_reward, float)
        assert isinstance(o3_step.terminated, bool)
        assert isinstance(o3_step.truncated, bool)
        assert o3_environment.planner_query_counter.count == 0
    finally:
        o3_environment.close()
        core.close()


@pytest.mark.parametrize("environment_id", O3_ENVIRONMENT_IDS)
def test_o3_semantic_view_and_pure_teacher_use_the_effective_layout_hash(
    environment_id,
):
    """Catch semantic or Teacher queries using stale provenance or wrong shapes."""
    environment = _new_o3(environment_id)
    teacher = PureMotionTeacher()

    class RecordingTeacher:
        def __init__(self):
            self.queries = []

        def query(self, query):
            self.queries.append(query)
            return teacher.query(query)

    recording = RecordingTeacher()
    trainer_view = object.__new__(OptimizationTrainer)
    trainer_teacher = SimpleNamespace(teacher=recording)
    try:
        environment.reset(seed=9301)
        views = OptimizationTrainer._semantic_views(trainer_view, environment)
        preferences, valid = OptimizationTrainer._teacher_batch(
            trainer_teacher,
            environment,
        )
        expected_hash = get_o3_topology(environment_id).effective_layout_hash
        assert np.stack([view.vector for view in views]).shape == (5, 61)
        assert all(view.json_view["layout_hash"] == expected_hash for view in views)
        assert preferences.shape == (5, 3)
        assert valid.shape == (5,)
        assert len(recording.queries) == 5
        assert all(query.layout_hash == expected_hash for query in recording.queries)
    finally:
        environment.close()


@pytest.mark.parametrize("environment_id", O3_ENVIRONMENT_IDS)
def test_o3_shadow_snapshot_restores_state_and_rng_under_the_same_layout(
    environment_id,
):
    """Catch O3 snapshots omitting the effective layout hash or mutable RNG state."""
    source = _new_o3(environment_id)
    target = _new_o3(environment_id)
    try:
        source.reset(seed=9302)
        target.reset(seed=9302)
        source.step([Action.NOOP] * 5)
        source_adapter = ShadowStateAdapter(source, code_commit="o3-interface-test")
        target_adapter = ShadowStateAdapter(target, code_commit="o3-interface-test")
        snapshot = source_adapter.capture(
            run_seed=9302,
            episode_index=0,
            episode_seed=9302,
            environment_index=0,
            real_global_step=1,
            episode_step=1,
        )
        assert snapshot.payload["layout_hash"] == (
            get_o3_topology(environment_id).effective_layout_hash
        )
        target_adapter.restore_bytes(snapshot.to_bytes())
        assert target_adapter.state_hash() == snapshot.state_hash
        first = source.step([Action.NOOP] * 5)
        second = target.step([Action.NOOP] * 5)
        assert first.observations.tobytes() == second.observations.tobytes()
        assert first.info == second.info
    finally:
        source.close()
        target.close()


def test_o3_safety_smoke_covers_boundary_conflict_and_charging():
    """Catch custom-layout construction bypassing shared physical safety rules."""
    environment = _new_o3(O3_ENVIRONMENT_IDS[0])
    try:
        environment.reset(seed=9301)
        warehouse = environment.env
        safe = [(0, 3), (1, 3), (3, 3), (4, 3), (5, 3)]
        for agent, position in zip(warehouse.agents, safe):
            agent.x, agent.y = position
        warehouse.agents[0].dir = Direction.RIGHT
        warehouse.agents[1].dir = Direction.LEFT
        warehouse._recalc_grid()
        environment.step([Action.FORWARD, Action.FORWARD] + [Action.NOOP] * 3)
        assert (warehouse.agents[0].x, warehouse.agents[0].y) == (0, 3)
        assert (warehouse.agents[1].x, warehouse.agents[1].y) == (1, 3)
        assert warehouse.total_collisions >= 1

        station = warehouse.charging_stations[0]
        warehouse.agents[0].x, warehouse.agents[0].y = station
        warehouse.agents[0].battery = 0.5
        warehouse._recalc_grid()
        transition = environment.step([Action.NOOP] * 5)
        assert warehouse.agents[0].battery > 0.5
        assert any(event["type"] == "charged" for event in transition.info["events"])

        warehouse.agents[0].x, warehouse.agents[0].y = (0, 0)
        warehouse.agents[0].dir = Direction.UP
        warehouse._recalc_grid()
        environment.step([Action.FORWARD] + [Action.NOOP] * 4)
        assert (warehouse.agents[0].x, warehouse.agents[0].y) == (0, 0)
    finally:
        environment.close()


def test_o3_toggle_load_mask_and_transition_use_the_shared_task_contract():
    """Catch a custom map disconnecting mandatory pickup from TOGGLE_LOAD."""
    environment = _new_o3(O3_ENVIRONMENT_IDS[1])
    try:
        environment.reset(seed=9302)
        warehouse = environment.env
        agent = warehouse.agents[0]
        task = warehouse.task_queue.task_for_agent(agent.id)
        shelf = next(item for item in warehouse.shelfs if item.id == task.shelf_id)
        agent.x, agent.y = shelf.x, shelf.y
        other_highways = [(0, 0), (19, 0), (0, 23), (19, 23)]
        for other, position in zip(warehouse.agents[1:], other_highways):
            other.x, other.y = position
        warehouse._recalc_grid()

        mask = environment.action_masks()[0]
        assert np.array_equal(mask, [False, False, False, False, True])
        transition = environment.step(
            [Action.TOGGLE_LOAD] + [Action.NOOP] * 4
        )
        assert warehouse.agents[0].carrying_shelf is shelf
        assert transition.observations.shape == (5, 613)
    finally:
        environment.close()
