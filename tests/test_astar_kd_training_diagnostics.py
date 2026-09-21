"""Regression coverage for Phase 2 episode behavior diagnostics."""

import numpy as np

from llm_mappo.mappo_baseline.runner import BaselineRun, _episode_window
from llm_mappo.warehouse_environment import (
    EPISODE_DIAGNOSTIC_FIELDS,
    EpisodeMetrics,
    Phase2Warehouse,
)
from rware.warehouse import Action, Direction


def test_episode_diagnostics_classify_every_requested_action_by_pre_step_phase(
    monkeypatch,
):
    """Catch action diagnostics using the post-transition target phase."""
    env = Phase2Warehouse(
        env_id="llm-mappo-small-1ag-v1", n_agents=1, max_steps=24
    )
    phases = ("idle", "task", "delivery", "charging")
    actions = tuple(Action)

    def target_for_current_step(agent_id, charging_targets=None):
        del charging_targets
        agent = env.env.agents[agent_id - 1]
        phase_index = min(env.env._cur_steps // len(actions), len(phases) - 1)
        phase = phases[phase_index]
        return (agent.x, agent.y), phase

    monkeypatch.setattr(env, "_target_for_agent", target_for_current_step)
    try:
        env.reset(seed=17)
        for phase in phases:
            for action in actions:
                transition = env.step([action])
                assert not transition.terminated
                assert not transition.truncated

        metrics = transition.metrics.as_dict()
    finally:
        env.close()

    for phase in phases:
        for action in actions:
            key = action.name.lower()
            assert metrics[f"{phase}_{key}_count"] == 1
            assert metrics[f"{phase}_{key}_rate"] == 0.2


def test_direction_reversal_uses_only_successful_displacement_edges():
    """Catch turns or blocked forwards overwriting the last successful edge."""
    env = Phase2Warehouse(
        env_id="llm-mappo-small-1ag-v1", n_agents=1, max_steps=12
    )
    try:
        env.reset(seed=3)
        agent = env.env.agents[0]
        agent.x, agent.y, agent.dir = 0, 1, Direction.RIGHT
        env.env._recalc_grid()

        env.step([Action.FORWARD])
        env.step([Action.LEFT])
        env.step([Action.LEFT])
        agent.x, agent.y, agent.dir = 0, 1, Direction.LEFT
        env.env._recalc_grid()
        env.step([Action.FORWARD])
        agent.x, agent.y, agent.dir = 1, 1, Direction.LEFT
        env.env._recalc_grid()
        transition = env.step([Action.FORWARD])
    finally:
        env.close()

    assert transition.metrics.as_dict()["direction_reversals"] == 1


def test_motion_steps_per_task_excludes_idle_and_is_undefined_without_completion():
    """Catch idle movement entering the active rate or a zero denominator."""
    metrics = EpisodeMetrics(
        completed_tasks=2,
        motion_steps=8,
        active_motion_steps=6,
    )

    assert metrics.motion_steps_per_task == 4.0
    assert metrics.active_motion_steps_per_task == 3.0

    metrics.completed_tasks = 0
    assert metrics.motion_steps_per_task is None
    assert metrics.active_motion_steps_per_task is None


def test_snapshot_restore_keeps_episode_diagnostic_accumulators(monkeypatch):
    """Catch snapshots that lose counts or the preceding successful edge."""
    source = Phase2Warehouse(
        env_id="llm-mappo-small-1ag-v1", n_agents=1, max_steps=8
    )
    restored = Phase2Warehouse(
        env_id="llm-mappo-small-1ag-v1", n_agents=1, max_steps=8
    )

    def fixed_task_target(agent_id, charging_targets=None):
        del charging_targets
        agent = source.env.agents[agent_id - 1]
        return (agent.x, agent.y), "task"

    monkeypatch.setattr(source, "_target_for_agent", fixed_task_target)
    try:
        source.reset(seed=5)
        agent = source.env.agents[0]
        agent.x, agent.y, agent.dir = 0, 1, Direction.RIGHT
        source.env._recalc_grid()
        source.step([Action.FORWARD])
        state = source.export_shadow_state()
        restored.reset(seed=5)
        restored.import_shadow_state(state)
    finally:
        source.close()
        restored.close()

    assert restored.metrics.as_dict()["task_forward_count"] == 1
    assert np.array_equal(
        restored._last_successful_moves, source._last_successful_moves
    )


def test_runner_persists_and_aggregates_episode_diagnostic_fields():
    """Catch diagnostic fields being dropped from CSV or evaluation windows."""
    assert {
        "task_forward_count",
        "charging_toggle_load_rate",
        "direction_reversals",
        "motion_steps_per_task",
        "active_motion_steps_per_task",
    } <= set(BaselineRun.EPISODE_FIELDS)

    window = _episode_window(
        [
            {
                "steps": "8",
                "completed_tasks": "2",
                "reward": "1.0",
                "task_completion_rate": "0.5",
                "motion_steps": "8",
                "active_motion_steps": "6",
            },
            {
                "steps": "8",
                "completed_tasks": "0",
                "reward": "2.0",
                "task_completion_rate": "0.0",
                "motion_steps": "4",
                "active_motion_steps": "3",
            },
        ]
    )

    assert window["motion_steps_per_task"] == 6.0
    assert window["active_motion_steps_per_task"] == 4.5


def test_episode_window_ignores_blank_diagnostics_from_legacy_csv_migration():
    """Keep a resumed pre-diagnostics episode log summarizable."""
    row = {
        "steps": "8",
        "completed_tasks": "2",
        "reward": "1.0",
        "task_completion_rate": "0.5",
        **{field: "" for field in EPISODE_DIAGNOSTIC_FIELDS},
    }

    window = _episode_window([row])

    assert window["steps_per_task"] == 4.0
    assert "motion_steps_per_task" not in window
    assert "direction_reversals" not in window
