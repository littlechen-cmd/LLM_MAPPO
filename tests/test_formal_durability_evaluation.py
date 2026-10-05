"""Durability windows preserve real early stops and cumulative boundaries."""

from pathlib import Path

import torch

from llm_mappo.formal_training import make_durability_experiment
from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment


def test_window_boundaries_subtract_initial_load_and_never_double_count():
    from llm_mappo.formal_training import build_durability_windows

    snapshots = [
        {"steps": 0, "created_tasks": 8, "completed_tasks": 0},
        {"steps": 1000, "created_tasks": 11, "completed_tasks": 3},
        {"steps": 2000, "created_tasks": 15, "completed_tasks": 7},
        {"steps": 2250, "created_tasks": 16, "completed_tasks": 8},
    ]
    windows = build_durability_windows(snapshots)
    assert [row["actual_steps"] for row in windows] == [1000, 1000, 250, 0, 0]
    assert [row["status"] for row in windows] == [
        "complete", "complete", "partial", "not_run", "not_run",
    ]
    assert [row["completed_tasks"] for row in windows] == [3, 4, 1, None, None]
    assert [row["created_tasks"] for row in windows] == [3, 4, 1, None, None]
    assert windows[2]["completed_tasks_per_1000_actual_steps"] == 4.0
    assert windows[3]["completed_tasks_per_1000_actual_steps"] is None


def test_exact_boundary_early_stop_leaves_next_window_unrun():
    from llm_mappo.formal_training import build_durability_windows

    windows = build_durability_windows([
        {"steps": 0, "created_tasks": 8},
        {"steps": 1000, "created_tasks": 8, "energy_deaths": 2},
    ])
    assert windows[0]["actual_steps"] == 1000
    assert windows[0]["energy_deaths"] == 2
    assert windows[1]["status"] == "not_run"
    assert windows[1]["energy_deaths"] is None


def test_real_durability_noop_policy_reports_180_step_deadlock_without_restart(tmp_path):
    experiment = make_durability_experiment(load_baseline_experiment(
        Path("configs/formal_environment_training/formal-8ag-baseline.yaml")
    ))
    run = BaselineRun(experiment, device="cpu", output_root=tmp_path,
                      evaluation_only=True)
    try:
        with torch.no_grad():
            for parameter in run.trainer.actor.parameters():
                parameter.zero_()
            run.trainer.actor.action_head.bias[0] = 100.0
        episode = run._evaluation_episode(400)
        assert episode["steps"] == 180
        assert episode["ran_full_horizon"] is False
        assert episode["termination"]["reason"] == "deadlock"
        assert episode["teacher_query_count"] == 0
        assert episode["durability_windows"][0]["actual_steps"] == 180
        assert episode["durability_windows"][0]["created_tasks"] == 0
        assert [row["status"] for row in episode["durability_windows"]] == [
            "partial", "not_run", "not_run", "not_run", "not_run",
        ]
    finally:
        run.close()


def test_formal_main_evaluation_does_not_emit_durability_protocol_fields(tmp_path):
    experiment = load_baseline_experiment(
        Path("configs/formal_environment_training/formal-8ag-baseline.yaml")
    )
    run = BaselineRun(experiment, device="cpu", output_root=tmp_path,
                      evaluation_only=True)
    try:
        with torch.no_grad():
            for parameter in run.trainer.actor.parameters():
                parameter.zero_()
            run.trainer.actor.action_head.bias[0] = 100.0
        episode = run._evaluation_episode(200000)
        assert episode["steps"] == 180
        assert "durability_windows" not in episode
        assert "ran_full_horizon" not in episode
        assert "termination" not in episode
    finally:
        run.close()
