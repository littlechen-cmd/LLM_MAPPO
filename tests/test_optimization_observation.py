import numpy as np

from llm_mappo.phase2 import Phase2Warehouse


def test_direct_goal_schema_replaces_only_the_legacy_waypoint_block():
    env = Phase2Warehouse(
        env_id="llm-mappo-medium-3ag-v1",
        n_agents=5,
        max_steps=8,
        observation_schema="direct-goal-observation-v1",
    )
    try:
        observations = env.reset(seed=31)
        warehouse = env.env
        height, width = warehouse.grid_size
        agent = warehouse.agents[0]
        target, _ = env._target_for_agent(agent.id)
        expected_goal = np.asarray(
            [
                (target[0] - agent.x) / max(width - 1, 1),
                (target[1] - agent.y) / max(height - 1, 1),
            ],
            dtype=np.float32,
        )

        assert observations.shape == (5, 613)
        assert np.allclose(observations[0, 586:588], expected_goal)
        assert np.array_equal(observations[0, 588:595], np.zeros(7))
    finally:
        env.close()


def test_no_geometric_goal_hint_zeros_all_nine_geometry_slots():
    direct = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        observation_schema="direct-goal-observation-v1",
    )
    no_goal_hint = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        observation_schema="no-geometric-goal-hint-v1",
    )
    try:
        direct_observations = direct.reset(seed=41)
        no_goal_hint_observations = no_goal_hint.reset(seed=41)

        assert direct_observations.shape == no_goal_hint_observations.shape
        assert np.array_equal(no_goal_hint_observations[:, 586:595], np.zeros((3, 9)))
        assert np.array_equal(
            direct_observations[:, :586], no_goal_hint_observations[:, :586]
        )
        assert np.array_equal(
            direct_observations[:, 595:], no_goal_hint_observations[:, 595:]
        )
    finally:
        direct.close()
        no_goal_hint.close()


def test_direct_goal_observations_do_not_query_a_planner():
    class ThrowingPlanner:
        def plan(self, *args, **kwargs):
            raise AssertionError("DirectGoal must not query a planner")

    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        observation_schema="direct-goal-observation-v1",
    )
    try:
        env._planner = ThrowingPlanner()
        env.reset(seed=9)
        transition = env.step([0, 0, 0])

        assert transition.observations.shape == (3, 613)
        assert env.planner_query_counter.count == 0
    finally:
        env.close()


def test_legacy_schema_remains_the_default_and_unknown_schema_is_rejected():
    legacy = Phase2Warehouse(n_agents=3, max_steps=8)
    explicit_legacy = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        observation_schema="legacy-waypoint-v1",
    )
    try:
        assert np.array_equal(legacy.reset(seed=7), explicit_legacy.reset(seed=7))
    finally:
        legacy.close()
        explicit_legacy.close()

    try:
        Phase2Warehouse(n_agents=1, max_steps=8, observation_schema="unknown-v1")
    except ValueError as error:
        assert "observation_schema" in str(error)
    else:
        raise AssertionError("unknown observation schema must be rejected")
