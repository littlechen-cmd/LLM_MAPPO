"""Deterministic evidence classification for Group 2 reward diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from llm_mappo.reward_diagnostic_analysis import analyze_completed_run
from llm_mappo.reward_diagnostics import classify_candidate_effect


def _seed(completed, steps_per_task, collisions=0.0, blocked=0.0, deaths=0):
    return {
        "completed_tasks": completed,
        "steps_per_task": steps_per_task,
        "collisions": collisions,
        "blocked_forwards": blocked,
        "deadlocked": 0,
        "energy_deaths": deaths,
    }


def test_classifier_labels_consistent_safe_paired_task_wins_positive():
    """Catch a report downgrading a candidate that wins two paired seeds safely."""
    reference = {
        7: _seed(10.0, 80.0, collisions=2.0, blocked=3.0),
        17: _seed(11.0, 75.0, collisions=2.0, blocked=3.0),
        27: _seed(10.0, 78.0, collisions=2.0, blocked=3.0),
    }
    candidate = {
        7: _seed(12.0, 70.0, collisions=1.0, blocked=2.0),
        17: _seed(12.0, 70.0, collisions=2.0, blocked=2.0),
        27: _seed(10.0, 78.0, collisions=2.0, blocked=3.0),
    }

    result = classify_candidate_effect(
        reference,
        candidate,
        training_trend_consistent=True,
        numerical_health=True,
    )

    assert result["label"] == "positive"
    assert result["wins"] == 2
    assert result["losses"] == 0


def test_classifier_marks_conflicting_safety_evidence_no_clear_effect():
    """Catch a task gain being reported positive despite repeated blocked regressions."""
    reference = {seed: _seed(10.0, 80.0, blocked=1.0) for seed in (7, 17, 27)}
    candidate = {
        7: _seed(11.0, 75.0, blocked=2.0),
        17: _seed(11.0, 75.0, blocked=2.0),
        27: _seed(10.0, 80.0, blocked=1.0),
    }

    result = classify_candidate_effect(
        reference,
        candidate,
        training_trend_consistent=True,
        numerical_health=True,
    )

    assert result["label"] == "no-clear-effect"
    assert result["blocked_regression_seeds"] == [7, 17]


def test_completed_run_analysis_reports_rates_ledger_and_reduced_windows(tmp_path):
    """A short completed run must retain null trend evidence rather than invent it."""
    run_directory = Path(tmp_path) / "seed_0007"
    evaluation_directory = run_directory / "evaluation"
    evaluation_directory.mkdir(parents=True)
    identity = {
        "code_commit": "abc123",
        "config_sha256": "c" * 64,
        "reward_definition_sha256": "r" * 64,
        "seed": 7,
    }
    (run_directory / "run_manifest.json").write_text(
        json.dumps({"identity": identity, "evaluation": {"seeds": [9400, 9401]}}),
        encoding="utf-8",
    )
    (run_directory / "state.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (run_directory / "checkpoint_final.pt").write_bytes(b"checkpoint")
    _write_csv(
        run_directory / "episodes.csv",
        [
            {
                "global_environment_steps": 10,
                "worker_index": 0,
                "episode_index": 0,
                "steps": 10,
                "completed_tasks": 2,
                "picked_tasks": 2,
                "progress_positive_events": 3,
                "progress_negative_events": 1,
                "blocked_forwards": 1,
                "collisions": 0,
                "deadlocked": 0,
                "low_battery_triggers": 0,
                "charged_events": 0,
                "energy_deaths": 0,
                "reward": 0.5,
                "reward_completion": 1.0,
                "reward_step_cost": -0.5,
                "steps_per_task": 5.0,
            },
            {
                "global_environment_steps": 20,
                "worker_index": 0,
                "episode_index": 1,
                "steps": 10,
                "completed_tasks": 0,
                "picked_tasks": 0,
                "progress_positive_events": 0,
                "progress_negative_events": 0,
                "blocked_forwards": 2,
                "collisions": 1,
                "deadlocked": 1,
                "low_battery_triggers": 1,
                "charged_events": 1,
                "energy_deaths": 1,
                "reward": -1.5,
                "reward_completion": 0.0,
                "reward_step_cost": -1.5,
                "steps_per_task": "",
            },
            {
                "global_environment_steps": 30,
                "worker_index": 0,
                "episode_index": 2,
                "steps": 10,
                "completed_tasks": 1,
                "picked_tasks": 1,
                "progress_positive_events": 2,
                "progress_negative_events": 0,
                "blocked_forwards": 0,
                "collisions": 0,
                "deadlocked": 0,
                "low_battery_triggers": 0,
                "charged_events": 0,
                "energy_deaths": 0,
                "reward": 0.2,
                "reward_completion": 0.5,
                "reward_step_cost": -0.3,
                "steps_per_task": 10.0,
            },
        ],
    )
    _write_csv(
        run_directory / "updates.csv",
        [{"raw_advantage_max_abs": 3.0, "raw_return_max_abs": 4.0}],
    )
    (evaluation_directory / "evaluation.json").write_text(
        json.dumps({"episodes": [{"seed": 9400}, {"seed": 9401}]}),
        encoding="utf-8",
    )

    result = analyze_completed_run(run_directory)

    assert result["identity"] == identity
    assert result["trend"]["reduced_window"] is True
    assert result["trend"]["window_size"] == 1
    assert result["events"]["blocked_forwards"]["total"] == 3.0
    assert result["events"]["completed_tasks"]["per_completed_task"] == 1.0
    assert result["ledger"]["reward_step_cost"]["total"] == -2.3
    assert result["ledger"]["reward_step_cost"]["per_completed_task"] == -2.3 / 3.0
    assert result["checkpoint_sha256"] is not None
    assert result["evaluation_seeds"] == [9400, 9401]

    output_directory = Path(tmp_path) / "report"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_reward_diagnostics.py",
            "--run-directory",
            str(run_directory),
            "--output-directory",
            str(output_directory),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_directory / "reward_diagnostic_report.json").read_text())
    assert report["training_seeds"] == [7]
    assert "Evidence completeness" in (
        output_directory / "reward_diagnostic_report.md"
    ).read_text(encoding="utf-8")


def test_completed_run_analysis_rejects_incomplete_artifacts(tmp_path):
    run_directory = Path(tmp_path) / "incomplete"
    run_directory.mkdir()
    (run_directory / "state.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="not complete"):
        analyze_completed_run(run_directory)


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
