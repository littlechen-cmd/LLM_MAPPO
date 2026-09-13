"""Verify AStarExpert coordinator and action-pipeline behavior."""
import numpy as np

from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase2_expert import AStarExpert
from rware.warehouse import Action, Direction


def test_coordinator_reports_head_on_swap_as_edge_swap_yield():
    """Catch edge swaps being mislabeled as generic occupied-target yields."""
    env = Phase2Warehouse(n_agents=2, max_steps=8)
    try:
        env.reset(seed=4)
        first, second = env.env.agents
        first.x, first.y, first.dir = 0, 1, Direction.RIGHT
        second.x, second.y, second.dir = 1, 1, Direction.LEFT
        env.env._recalc_grid()

        coordinated, reasons = AStarExpert._coordinate_actions(
            env,
            np.asarray(
                [Action.FORWARD.value, Action.FORWARD.value], dtype=np.int64
            ),
            return_reasons=True,
        )
    finally:
        env.close()

    assert coordinated.tolist() == [Action.NOOP.value, Action.NOOP.value]
    assert reasons == {
        0: "coordinator_edge_swap_yield",
        1: "coordinator_edge_swap_yield",
    }


def test_right_switch_changes_only_the_controlled_yield_action():
    """Catch the diagnostic switch changing conflict selection or reason labels."""
    env = Phase2Warehouse(n_agents=2, max_steps=8)
    try:
        env.reset(seed=4)
        first, second = env.env.agents
        first.x, first.y, first.dir = 0, 1, Direction.RIGHT
        second.x, second.y, second.dir = 1, 1, Direction.LEFT
        env.env._recalc_grid()
        requested = np.asarray(
            [Action.FORWARD.value, Action.FORWARD.value], dtype=np.int64
        )

        noop_actions, noop_reasons = AStarExpert._coordinate_actions(
            env, requested, yield_action="noop", return_reasons=True
        )
        right_actions, right_reasons = AStarExpert._coordinate_actions(
            env, requested, yield_action="right", return_reasons=True
        )
    finally:
        env.close()

    assert noop_actions.tolist() == [Action.NOOP.value, Action.NOOP.value]
    assert right_actions.tolist() == [Action.RIGHT.value, Action.RIGHT.value]
    assert right_reasons == noop_reasons


def test_coordinator_distinguishes_vertex_and_occupied_yields():
    """Catch distinct immediate conflict causes collapsing into one reason code."""
    env = Phase2Warehouse(n_agents=2, max_steps=8)
    try:
        env.reset(seed=4)
        first, second = env.env.agents
        first.x, first.y, first.dir = 0, 1, Direction.RIGHT
        second.x, second.y, second.dir = 2, 1, Direction.LEFT
        env.env._recalc_grid()
        vertex_actions, vertex_reasons = AStarExpert._coordinate_actions(
            env,
            np.asarray(
                [Action.FORWARD.value, Action.FORWARD.value], dtype=np.int64
            ),
            return_reasons=True,
        )

        second.x, second.y, second.dir = 1, 1, Direction.LEFT
        env.env._recalc_grid()
        occupied_actions, occupied_reasons = AStarExpert._coordinate_actions(
            env,
            np.asarray([Action.FORWARD.value, Action.NOOP.value], dtype=np.int64),
            return_reasons=True,
        )
    finally:
        env.close()

    assert vertex_actions.tolist() == [Action.FORWARD.value, Action.NOOP.value]
    assert vertex_reasons == {1: "coordinator_vertex_yield"}
    assert occupied_actions.tolist() == [Action.NOOP.value, Action.NOOP.value]
    assert occupied_reasons == {0: "coordinator_occupied_yield"}


def test_action_pipeline_records_a_real_mask_override(monkeypatch):
    """Catch planner actions being recorded after, rather than before, masking."""
    env = Phase2Warehouse(n_agents=1, max_steps=8)
    expert = AStarExpert()
    try:
        env.reset(seed=3)
        preferences = np.zeros((1, len(Action)), dtype=np.float32)
        preferences[0, Action.FORWARD.value] = 1.0
        preferences[0, Action.NOOP.value] = 0.1
        monkeypatch.setattr(expert, "action_preferences", lambda ignored: preferences)
        masks = np.zeros_like(preferences, dtype=bool)
        masks[0, Action.NOOP.value] = True

        actions, _ = expert.act(env, masks)
    finally:
        env.close()

    assert actions.tolist() == [Action.NOOP.value]
    assert expert.last_action_pipeline == [
        {
            "planner_action": Action.FORWARD.value,
            "masked_action": Action.NOOP.value,
            "coordinated_action": Action.NOOP.value,
            "planner_masked": True,
            "coordinator_reason": None,
        }
    ]
