from collections import Counter
import json
import sys

import numpy as np

import eval.evaluate_dynamic_ingress_astar as astar_eval
from eval.evaluate_dynamic_ingress_astar import (
    _aggregate_stall_diagnostics,
    _record_step_diagnostics,
    evaluate_dynamic_astar,
)
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase2_expert import AStarExpert
from llm_mappo.phase3_training import Phase3TrainingConfig
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


def test_stall_aggregation_preserves_pipeline_counts_and_conservation():
    """Catch action-stage override counts being dropped from the JSON summary."""
    records = [
        {
            "stall_diagnostics": {
                "termination_reason": "time_limit",
                "stationary_reason_counts": {
                    "planner_noop": 1,
                    "unknown_stationary": 1,
                },
                "pipeline_counts": {
                    "planner_to_mask_overrides": 1,
                    "planner_to_mask_unchanged": 3,
                    "mask_to_coordinator_overrides": 2,
                    "mask_to_coordinator_unchanged": 2,
                    "coordinator_to_executed_overrides": 1,
                    "coordinator_to_executed_unchanged": 3,
                },
                "agent_steps": 4,
                "stationary_agent_steps": 2,
            }
        }
    ]

    summary = _aggregate_stall_diagnostics(records)

    assert summary["pipeline_counts"] == records[0]["stall_diagnostics"][
        "pipeline_counts"
    ]
    assert summary["stationary_agent_steps"] == 2
    assert summary["termination_reason_counts"]["time_limit"] == 1
    assert summary["conservation_errors"] == []


def test_stall_aggregation_marks_legacy_records_unavailable():
    """Catch missing legacy diagnostics being silently represented as zero data."""
    summary = _aggregate_stall_diagnostics([{"episodes": 1}])

    assert summary == {
        "schema_version": 1,
        "available": False,
        "unavailable_reason": "stall_diagnostics_not_collected",
        "records_with_diagnostics": 0,
    }


def test_stall_aggregation_detects_missing_termination_reason():
    """Catch a diagnosed episode being omitted from the termination partition."""
    summary = _aggregate_stall_diagnostics(
        [
            {
                "stall_diagnostics": {
                    "stationary_reason_counts": {},
                    "pipeline_counts": {},
                    "agent_steps": 0,
                    "stationary_agent_steps": 0,
                }
            }
        ]
    )

    assert summary["conservation_errors"] == ["termination_reason_total"]


def test_dynamic_astar_diagnostics_conserve_every_action_stage():
    """Catch missing unchanged counts or omitted unknown-stationary schema keys."""
    config = Phase3TrainingConfig(
        n_agents=1,
        max_steps=2,
        env_id="llm-mappo-small-1ag-v1",
        priority_schedule=None,
        batch_interval=2,
        batch_size_range=(1, 1),
        initial_priority_label="A",
        request_queue_size=2,
        task_completion_target=2,
    )

    result = evaluate_dynamic_astar(
        config,
        seeds=(300,),
        episodes_per_seed=1,
        collect_stall_diagnostics=True,
    )
    diagnostics = result["seeds"][0]["stall_diagnostics"]

    assert diagnostics["available"] is True
    assert diagnostics["agent_steps"] == 2
    assert diagnostics["conservation_errors"] == []
    assert "unknown_stationary" in diagnostics["stationary_reason_counts"]
    assert sum(diagnostics["stationary_reason_counts"].values()) == diagnostics[
        "stationary_agent_steps"
    ]
    for stage in (
        "planner_to_mask",
        "mask_to_coordinator",
        "coordinator_to_executed",
    ):
        assert (
            diagnostics["pipeline_counts"][f"{stage}_overrides"]
            + diagnostics["pipeline_counts"][f"{stage}_unchanged"]
            == diagnostics["agent_steps"]
        )


def test_environment_blocked_forward_is_an_executed_action_override():
    """Catch a failed forward attempt being counted as an executed forward action."""
    env = Phase2Warehouse(n_agents=1, max_steps=2)
    try:
        env.reset(seed=9)
        agent = env.env.agents[0]
        agent.x, agent.y, agent.dir = 0, 1, Direction.LEFT
        env.env._recalc_grid()
        before = [(agent.x, agent.y)]
        actions = np.asarray([Action.FORWARD.value], dtype=np.int64)
        transition = env.step(actions)
        stall_counts = Counter()
        pipeline_counts = Counter()
        _record_step_diagnostics(
            env,
            [
                {
                    "planner_action": Action.FORWARD.value,
                    "masked_action": Action.FORWARD.value,
                    "coordinated_action": Action.FORWARD.value,
                    "planner_masked": False,
                    "coordinator_reason": None,
                }
            ],
            before,
            actions,
            transition,
            stall_counts,
            pipeline_counts,
        )
    finally:
        env.close()

    assert stall_counts == {"environment_blocked_forward": 1}
    assert pipeline_counts["coordinator_to_executed_overrides"] == 1


def test_diagnostics_and_explicit_noop_preserve_episode_outcomes():
    """Catch diagnostic collection or the default switch changing controller output."""
    config = Phase3TrainingConfig(
        n_agents=1,
        max_steps=2,
        env_id="llm-mappo-small-1ag-v1",
        priority_schedule=None,
        batch_interval=2,
        batch_size_range=(1, 1),
        initial_priority_label="A",
        request_queue_size=2,
        task_completion_target=2,
    )
    default = evaluate_dynamic_astar(config, seeds=(300,), episodes_per_seed=1)
    explicit_noop = evaluate_dynamic_astar(
        config,
        seeds=(300,),
        episodes_per_seed=1,
        coordinator_yield_action="noop",
    )
    diagnosed = evaluate_dynamic_astar(
        config,
        seeds=(300,),
        episodes_per_seed=1,
        collect_stall_diagnostics=True,
    )

    assert _outcome_projection(default) == _outcome_projection(explicit_noop)
    assert _outcome_projection(default) == _outcome_projection(diagnosed)


def _outcome_projection(result):
    top_level = {
        key: result[key]
        for key in (
            "episodes",
            "task_completion_rate",
            "mean_collisions_per_episode",
            "deadlock_rate",
            "mean_energy_deaths_per_episode",
        )
    }
    seed = result["seeds"][0]
    top_level["seed"] = {
        key: seed[key]
        for key in (
            "seed",
            "episodes",
            "task_completion_rate",
            "mean_collisions",
            "deadlock_rate",
            "success_rate",
            "mean_energy_deaths",
            "path_livelocks",
            "state_deadlocks",
        )
    }
    top_level["expert"] = {
        key: value
        for key, value in result["reservation_teacher"].items()
        if key not in {"planning_time_ms_total", "planning_time_ms_p95"}
    }
    return top_level


def test_cli_wires_diagnostic_switches_and_writes_json(monkeypatch, tmp_path):
    """Catch CLI flags being parsed but not forwarded to the evaluator."""
    calls = []

    def fake_evaluate(config, seeds, episodes_per_seed, **options):
        calls.append((config, seeds, episodes_per_seed, options))
        return {"episodes": 1}

    output = tmp_path / "nested" / "diagnostics.json"
    monkeypatch.setattr(astar_eval, "evaluate_dynamic_astar", fake_evaluate)
    monkeypatch.setattr(
        astar_eval.Phase3TrainingConfig,
        "from_yaml",
        staticmethod(lambda path: "CONFIG"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_dynamic_ingress_astar.py",
            "--config",
            "config.yaml",
            "--seeds",
            "300",
            "301",
            "--episodes-per-seed",
            "1",
            "--reservation-mode",
            "legacy",
            "--coordinator-yield-action",
            "right",
            "--stall-diagnostics",
            "--output",
            str(output),
        ],
    )

    astar_eval.main()

    assert calls == [
        (
            "CONFIG",
            [300, 301],
            1,
            {
                "legacy_terminal_reservation": True,
                "coordinator_yield_action": "right",
                "collect_stall_diagnostics": True,
            },
        )
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "episodes": 1,
        "config": "config.yaml",
    }
