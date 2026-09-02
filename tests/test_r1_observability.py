"""R1-A regressions for complete episodes and atomic evidence."""

import csv
import json

import pytest
import torch


def _episode(*, worker: int, episode: int, step: int, completion: float) -> dict:
    return {
        "real_env_steps": step,
        "worker_index": worker,
        "episode_index": episode,
        "episode_seed": 9107 + worker * 1_000_003 + episode,
        "completed_tasks": int(50 * completion),
        "created_tasks": 50,
        "task_completion_target": 50,
        "task_completion_rate": completion,
        "reward": 12.5 + episode,
        "collisions": episode,
        "deadlocked": False,
        "agent_deaths": 0,
        "picked_tasks": 4,
        "blocked_forwards": 2,
        "low_battery_triggers": 1,
        "charging_target_steps": 3,
        "charging_exposure_rate": 0.02,
        "charger_arrivals": 1,
        "charged_events": 1,
        "charging_wait_steps": 0,
        "task_recoveries": 0,
        "energy_deaths": 0,
        "minimum_battery": 0.41,
        "steps": 1000,
        "success": completion >= 1.0,
    }


def _update(step: int) -> dict:
    return {
        "real_env_steps": step,
        "policy_loss": -0.1,
        "value_loss": 0.2,
        "entropy": 0.8,
        "astar_loss": 0.0,
        "semantic_loss": 0.0,
        "semantic_valid_denominator": 0,
        "lambda_a": 0.01,
        "lambda_l": 0.02,
        "num_env_workers": 16,
        "rollout_length": 32,
        "global_environment_steps": step,
        "environment_steps_per_second": 100.0,
        "rollout_wall_time": 1.0,
        "policy_inference_time": 0.1,
        "ppo_update_time": 0.2,
        "total_elapsed_time": 1.3,
        "peak_cuda_memory_allocated": 10,
        "peak_cuda_memory_reserved": 20,
    }


def test_completed_episode_record_preserves_terminal_worker_identity():
    from llm_mappo.e1_vector_env import completed_episode_record

    metrics = {
        "completed_tasks": 3,
        "created_tasks": 12,
        "task_completion_target": 9,
        "task_completion_rate": 1 / 3,
        "reward": 7.5,
        "collisions": 2,
        "deadlocked": True,
        "agent_deaths": 1,
        "picked_tasks": 4,
        "blocked_forwards": 5,
        "low_battery_triggers": 1,
        "charging_target_steps": 6,
        "charging_exposure_rate": 0.03,
        "charger_arrivals": 1,
        "charged_events": 2,
        "charging_wait_steps": 3,
        "task_recoveries": 1,
        "energy_deaths": 0,
        "minimum_battery": 0.2,
        "steps": 180,
        "success": False,
    }

    record = completed_episode_record(
        metrics,
        worker_index=4,
        episode_index=2,
        episode_seed=4_009_121,
        terminal_global_step=513,
    )

    assert record == {
        "real_env_steps": 513,
        "worker_index": 4,
        "episode_index": 2,
        "episode_seed": 4_009_121,
        **metrics,
    }


def test_checkpoint_evidence_reconciliation_is_idempotent(tmp_path):
    from llm_mappo.e1_evidence import E1EvidenceWriter

    writer = E1EvidenceWriter.create(
        tmp_path / "run",
        {"identity": {"group": "MAPPO-DG"}, "requires_completed_episodes": True},
    )
    evidence = {
        "update_row": _update(512),
        "episode_rows": [_episode(worker=0, episode=0, step=500, completion=0.5)],
    }

    first = writer.reconcile_checkpoint_evidence(evidence)
    second = writer.reconcile_checkpoint_evidence(evidence)

    assert first == {"update_appended": True, "episodes_appended": 1}
    assert second == {"update_appended": False, "episodes_appended": 0}
    with (writer.directory / "updates.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    with (writer.directory / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1

    conflicting = {**evidence, "update_row": {**evidence["update_row"], "policy_loss": -9.0}}
    with pytest.raises(ValueError, match="conflicts"):
        writer.reconcile_checkpoint_evidence(conflicting)


def test_complete_fails_closed_without_episode_rows_and_aggregates_real_rows(tmp_path):
    from llm_mappo.e1_evidence import E1EvidenceWriter

    empty = E1EvidenceWriter.create(
        tmp_path / "empty",
        {"identity": {"group": "MAPPO-DG"}, "requires_completed_episodes": True},
    )
    with pytest.raises(ValueError, match="completed episode"):
        empty.complete({"group": "MAPPO-DG"})
    assert json.loads((empty.directory / "state.json").read_text(encoding="utf-8")) == {
        "status": "running"
    }

    writer = E1EvidenceWriter.create(
        tmp_path / "run",
        {"identity": {"group": "MAPPO-DG"}, "requires_completed_episodes": True},
    )
    rows = [
        _episode(worker=0, episode=0, step=500, completion=0.2),
        _episode(worker=1, episode=0, step=512, completion=0.6),
    ]
    writer.reconcile_checkpoint_evidence({"update_row": _update(512), "episode_rows": rows})

    summary = writer.complete({"group": "MAPPO-DG", "latest_episode_metrics": {"bad": True}})

    assert summary["completed_episodes"] == 2
    assert summary["latest_episode_metrics"] == rows[-1]
    assert summary["episode_window"] == {
        "window_size": 2,
        "mean_task_completion_rate": pytest.approx(0.4),
        "mean_completed_tasks": pytest.approx(20.0),
        "mean_reward": pytest.approx(12.5),
        "mean_collisions": pytest.approx(0.0),
        "deadlock_rate": pytest.approx(0.0),
        "success_rate": pytest.approx(0.0),
    }


def test_tensorboard_records_update_and_complete_episode_axes(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from llm_mappo.e1_evidence import E1TensorBoardWriter

    logdir = tmp_path / "tensorboard"
    board = E1TensorBoardWriter(logdir)
    board.add_update(_update(512))
    board.add_episode(_episode(worker=3, episode=2, step=500, completion=0.6))
    board.close()

    accumulator = EventAccumulator(str(logdir)).Reload()
    tags = set(accumulator.Tags()["scalars"])
    assert {
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "performance/environment_steps_per_second",
        "episode/task_completion_rate",
        "episode/completed_tasks",
        "episode/team_reward",
        "episode/collisions",
    } <= tags
    completion = accumulator.Scalars("episode/task_completion_rate")
    assert [(item.step, item.value) for item in completion] == [(500, pytest.approx(0.6))]


def test_checkpoint_round_trips_the_evidence_commit_batch(tmp_path):
    from llm_mappo.e1_evidence import load_e1_checkpoint, save_e1_checkpoint
    from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor

    actor, critic = O0StudentActor(), O0CentralizedCritic()
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
    identity = {"group": "MAPPO-DG", "seed": 9107, "raw_records_sha256": "abc"}
    evidence = {
        "update_row": _update(512),
        "episode_rows": [_episode(worker=0, episode=0, step=500, completion=0.5)],
    }
    path = tmp_path / "checkpoint.pt"

    save_e1_checkpoint(
        path,
        identity=identity,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        schedule_state={"schedule_version": "linear-env-step-v1"},
        calibration_state=None,
        trainer_state={"schema": "e1-runtime-v2"},
        evidence_state=evidence,
    )
    restored = load_e1_checkpoint(
        path,
        expected_identity=identity,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
    )

    assert restored["evidence_state"] == evidence
