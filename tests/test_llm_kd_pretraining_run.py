"""Runnable Phase 3 semantic-pretraining smoke contracts."""

from __future__ import annotations

import json
from pathlib import Path
from hashlib import sha256

import numpy as np
import pytest
import torch

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.pretraining_run import (
    PRETRAINING_RUNNER_V1,
    PretrainingRunConfig,
    run_semantic_pretraining,
)
import llm_mappo.mappo_llm_kd.pretraining_run as pretraining_run


def _selection_config(tmp_path, *, min_delta=0.0):
    dataset, labels = _labeled_dataset(tmp_path)
    return PretrainingRunConfig(
        schema=PRETRAINING_RUNNER_V1,
        dataset_directory=str(dataset), label_directory=str(labels),
        artifact_root=str(tmp_path / "artifacts"), config_sha256="a" * 64,
        approval_status="approved", seed=7, batch_size=1, max_epochs=3,
        learning_rate=0.001, dimension_weights=(1.0, 1.0, 1.0, 1.0),
        early_stopping_patience=2, early_stopping_min_delta=min_delta,
    )


def _controlled_validation(monkeypatch, losses):
    """Force a nonmonotonic selection boundary while training real parameters."""
    observed_states = []
    sequence = iter(losses)

    def evaluate(model, samples, *_):
        assert {sample["partition"] for sample in samples} == {"validation"}
        observed_states.append({
            name: value.detach().clone()
            for name, value in model.actor.state_dict().items()
            if name.startswith(("encoder.", "gru."))
        })
        return next(sequence)

    monkeypatch.setattr(pretraining_run, "_evaluate", evaluate)
    return observed_states


@pytest.mark.parametrize("losses,min_delta,best_epoch", [
    ([1.0, 0.9, 1.1], 0.2, 2), ([1.0, 1.0, 1.1], 0.0, 1),
])
def test_export_uses_actual_validation_minimum_and_retains_earliest_tie(
    tmp_path, monkeypatch, losses, min_delta, best_epoch,
):
    config = _selection_config(tmp_path, min_delta=min_delta)
    states = _controlled_validation(monkeypatch, losses)
    result = run_semantic_pretraining(config, device="cpu", run_id="selection")
    directory = Path(result["run_directory"])
    exported = torch.load(directory / "representation.pt", weights_only=True)
    latest = torch.load(directory / "checkpoint_latest.pt", weights_only=True)
    for name, expected in states[best_epoch - 1].items():
        assert torch.equal(exported["actor_representation"][name], expected)
        assert torch.equal(latest["actor_representation"][name], states[-1][name])
    assert any(
        not torch.equal(states[best_epoch - 1][name], states[-1][name])
        for name in states[-1]
    )
    manifest = result["manifest"]
    assert manifest["best_epoch"] == best_epoch
    assert manifest["best_validation_loss"] == min(losses)
    assert manifest["completed_epochs"] == 3
    assert manifest["best_checkpoint"]["file_sha256"] == sha256(
        (directory / "checkpoint_best.pt").read_bytes()
    ).hexdigest()
    assert manifest["representation"]["file_sha256"] == sha256(
        (directory / "representation.pt").read_bytes()
    ).hexdigest()
    assert latest["early_stopping"]["bad_epochs"] == 2
    assert latest["early_stopping"]["best_validation_loss"] == 1.0
    assert latest["progress"] == {
        "global_step": 3, "epoch": 4, "batch_in_epoch": 0,
    }


def test_resume_keeps_prior_best_when_later_validation_is_worse(
    tmp_path, monkeypatch,
):
    config = _selection_config(tmp_path)
    states = _controlled_validation(monkeypatch, [1.0])
    original_save = pretraining_run.save_pretraining_checkpoint

    def interrupt_in_second_epoch(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if (Path(path).name == "checkpoint_latest.pt"
                and kwargs["progress"] == {
                    "global_step": 2, "epoch": 2, "batch_in_epoch": 1,
                }):
            raise RuntimeError("fixture interruption after prior best")

    monkeypatch.setattr(pretraining_run, "save_pretraining_checkpoint",
                        interrupt_in_second_epoch)
    with pytest.raises(RuntimeError, match="prior best"):
        run_semantic_pretraining(config, device="cpu", run_id="resumed")
    monkeypatch.setattr(pretraining_run, "save_pretraining_checkpoint", original_save)
    _controlled_validation(monkeypatch, [1.1, 1.2])
    directory = Path(config.artifact_root) / "resumed"
    result = run_semantic_pretraining(config, device="cpu", resume=directory)
    exported = torch.load(directory / "representation.pt", weights_only=True)
    for name, expected in states[0].items():
        assert torch.equal(exported["actor_representation"][name], expected)
    assert result["manifest"]["best_epoch"] == 1
    assert result["manifest"]["best_validation_loss"] == 1.0
    assert result["manifest"]["completed_epochs"] == 3
    assert result["manifest"]["global_step"] == 3
    assert [row["epoch"] for row in result["manifest"]["history"]] == [1, 2, 3]
    assert result["manifest"]["history_missing_epochs"] == []


@pytest.mark.parametrize("boundary", [
    "best", "terminal_latest", "missing_best", "ahead_best", "missing_history",
    "unordered_history", "nonfinite_history",
])
def test_resume_handles_checkpoint_write_boundaries_without_extra_updates(
    tmp_path, monkeypatch, boundary,
):
    config = _selection_config(tmp_path, min_delta=0.2)
    states = _controlled_validation(monkeypatch, [1.0, 0.9, 1.1])
    original_save = pretraining_run.save_pretraining_checkpoint
    epoch_one_latest = None

    def interrupt_at_boundary(path, *args, **kwargs):
        nonlocal epoch_one_latest
        original_save(path, *args, **kwargs)
        progress = kwargs["progress"]
        if Path(path).name == "checkpoint_latest.pt" and progress == {
            "global_step": 1, "epoch": 2, "batch_in_epoch": 0,
        }:
            epoch_one_latest = Path(path).read_bytes()
        selected = (
            boundary in {"best", "ahead_best"}
            and Path(path).name == "checkpoint_best.pt"
            and progress["epoch"] == 2
        ) or (
            boundary not in {"best", "ahead_best"}
            and Path(path).name == "checkpoint_latest.pt"
            and progress["epoch"] == 4
        )
        if selected:
            raise RuntimeError("fixture checkpoint boundary")

    monkeypatch.setattr(pretraining_run, "save_pretraining_checkpoint",
                        interrupt_at_boundary)
    with pytest.raises(RuntimeError, match="checkpoint boundary"):
        run_semantic_pretraining(config, device="cpu", run_id="boundary")
    monkeypatch.setattr(pretraining_run, "save_pretraining_checkpoint", original_save)
    directory = Path(config.artifact_root) / "boundary"
    if boundary == "ahead_best":
        (directory / "checkpoint_latest.pt").write_bytes(epoch_one_latest)
        with pytest.raises(ValueError, match="ahead of latest"):
            run_semantic_pretraining(config, device="cpu", resume=directory)
        return
    if boundary in {"unordered_history", "nonfinite_history"}:
        history_path = directory / "history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if boundary == "unordered_history":
            history["history"].reverse()
        else:
            history["history"][0]["validation_loss"] = float("nan")
        history_path.write_text(json.dumps(history), encoding="utf-8")
        with pytest.raises(ValueError, match="history"):
            run_semantic_pretraining(config, device="cpu", resume=directory)
        return
    if boundary == "missing_history":
        (directory / "history.json").unlink()
    if boundary == "missing_best":
        (directory / "checkpoint_best.pt").unlink()
        with pytest.raises(ValueError, match="prior best checkpoint is missing"):
            run_semantic_pretraining(config, device="cpu", resume=directory)
        return
    latest_before = (directory / "checkpoint_latest.pt").read_bytes()
    # Re-evaluate the same epoch after a best-before-latest interruption.
    _controlled_validation(monkeypatch, [0.9, 1.1] if boundary == "best" else [])
    result = run_semantic_pretraining(config, device="cpu", resume=directory)
    exported = torch.load(directory / "representation.pt", weights_only=True)
    for name, expected in states[1].items():
        assert torch.equal(exported["actor_representation"][name], expected)
    assert result["manifest"]["global_step"] == 3
    assert result["manifest"]["best_epoch"] == 2
    assert result["manifest"]["completed_epochs"] == 3
    if boundary == "missing_history":
        assert result["manifest"]["history_missing_epochs"] == [1, 2, 3]
        assert result["manifest"]["history"] == []
    else:
        assert [row["epoch"] for row in result["manifest"]["history"]] == [1, 2, 3]
    if boundary == "terminal_latest":
        assert (directory / "checkpoint_latest.pt").read_bytes() == latest_before


def _window(partition, episode_id, row):
    return {
        "partition": partition,
        "episode_id": episode_id,
        "agent_id": 1,
        "start_timestep": 0,
        "end_timestep": 0,
        "observations": [row.tolist()],
        "reset_masks": [True],
        "stratum": "fixture",
    }


def _view_id(window):
    return (
        f"{window['partition']}:{window['episode_id']}:agent-{window['agent_id']}"
        f":start-{window['start_timestep']}:end-{window['end_timestep']}"
    )


def _labeled_dataset(tmp_path, train_count=1):
    dataset = tmp_path / "dataset"
    labels = dataset / "labels" / "approved" / "fixture-provider"
    labels.mkdir(parents=True)
    schema = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    )
    index = {name: position for position, name in enumerate(schema.feature_names)}
    row = np.zeros(209, dtype=np.float32)
    row[index["self_orientation_up"]] = 1.0
    row[index["self_phase_task"]] = 1.0
    row[index["self_previous_action_0"]] = 1.0
    row[index["goal_type_task"]] = 1.0
    row[index["self_priority_weight"]] = 0.5
    row[index["goal_normalized_distance"]] = 0.5
    windows = [
        _window("train", f"episode-{index}", row)
        for index in range(1, train_count + 1)
    ] + [_window("validation", "episode-validation", row)]
    windows_path = dataset / "windows.jsonl"
    windows_path.write_text(
        "".join(
            json.dumps(window, sort_keys=True) + "\n" for window in windows
        ),
        encoding="utf-8",
    )
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "llm-dataset-build-run-v1",
                "distribution": "native-209d-collection-main-v1",
                "main_pretraining_eligible": True,
                "windows_sha256": sha256(windows_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (dataset / "state.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    scores = {
        "mission_urgency": 0.1,
        "concession_appropriateness": 0.2,
        "interaction_risk": 0.3,
        "deadlock_severity": 0.4,
    }
    labels_path = labels / "labels.jsonl"
    labels_path.write_text(
        "".join(
            json.dumps({"status": "ok", "view_id": _view_id(window), "scores": scores})
            + "\n"
            for window in windows
        ),
        encoding="utf-8",
    )
    (labels / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "llm-dataset-label-run-v1",
                "dataset_manifest_sha256": sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "labels_sha256": sha256(labels_path.read_bytes()).hexdigest(),
                "successful_records": len(windows),
            }
        ),
        encoding="utf-8",
    )
    (labels / "state.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    return dataset, labels


def test_pretraining_runner_writes_finite_checkpoint_and_representation(tmp_path):
    dataset, labels = _labeled_dataset(tmp_path)
    config = PretrainingRunConfig(
        schema=PRETRAINING_RUNNER_V1,
        dataset_directory=str(dataset),
        label_directory=str(labels),
        artifact_root=str(tmp_path / "artifacts"),
        config_sha256="a" * 64,
        approval_status="approved",
        seed=17,
        batch_size=1,
        max_epochs=2,
        learning_rate=0.001,
        dimension_weights=(1.0, 1.0, 1.0, 1.0),
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
    )

    result = run_semantic_pretraining(config, device="cpu", run_id="tiny")

    directory = tmp_path / "artifacts" / "tiny"
    assert result["run_directory"] == str(directory.resolve())
    assert (directory / "checkpoint_latest.pt").is_file()
    assert (directory / "representation.pt").is_file()
    assert result["manifest"]["global_step"] == 2
    assert np.isfinite(result["manifest"]["best_validation_loss"])


def test_pretraining_runner_refuses_provisional_configuration(tmp_path):
    config = PretrainingRunConfig(
        schema=PRETRAINING_RUNNER_V1,
        dataset_directory=str(tmp_path / "dataset"),
        label_directory=str(tmp_path / "labels"),
        artifact_root=str(tmp_path / "artifacts"),
        config_sha256="a" * 64,
        approval_status="provisional",
        seed=7,
        batch_size=256,
        max_epochs=100,
        learning_rate=0.001,
        dimension_weights=(1.0, 1.0, 1.0, 1.0),
        early_stopping_patience=10,
        early_stopping_min_delta=0.0001,
    )

    with pytest.raises(ValueError, match="not frozen"):
        run_semantic_pretraining(config, device="cpu", run_id="not-run")


def test_pretraining_resume_skips_completed_update_and_matches_continuous_run(
    tmp_path, monkeypatch
):
    dataset, labels = _labeled_dataset(tmp_path, train_count=2)
    common = dict(
        schema=PRETRAINING_RUNNER_V1,
        dataset_directory=str(dataset),
        label_directory=str(labels),
        config_sha256="a" * 64,
        approval_status="approved",
        seed=17,
        batch_size=1,
        max_epochs=2,
        learning_rate=0.001,
        dimension_weights=(1.0, 1.0, 1.0, 1.0),
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
    )
    continuous_config = PretrainingRunConfig(
        **common, artifact_root=str(tmp_path / "continuous")
    )
    resumed_config = PretrainingRunConfig(
        **common, artifact_root=str(tmp_path / "resumed")
    )
    continuous = run_semantic_pretraining(
        continuous_config, device="cpu", run_id="run"
    )

    original_save = pretraining_run.save_pretraining_checkpoint
    saves = 0

    def interrupt_after_first_update(*args, **kwargs):
        nonlocal saves
        original_save(*args, **kwargs)
        saves += 1
        if saves == 1:
            raise RuntimeError("fixture interruption")

    monkeypatch.setattr(
        pretraining_run, "save_pretraining_checkpoint", interrupt_after_first_update
    )
    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_semantic_pretraining(resumed_config, device="cpu", run_id="run")
    monkeypatch.setattr(
        pretraining_run, "save_pretraining_checkpoint", original_save
    )

    resumed_directory = tmp_path / "resumed" / "run"
    resumed = run_semantic_pretraining(
        resumed_config, device="cpu", resume=resumed_directory
    )

    assert continuous["manifest"]["global_step"] == 4
    assert resumed["manifest"]["global_step"] == 4
    continuous_payload = torch.load(
        tmp_path / "continuous" / "run" / "representation.pt",
        map_location="cpu",
        weights_only=True,
    )
    resumed_payload = torch.load(
        resumed_directory / "representation.pt",
        map_location="cpu",
        weights_only=True,
    )
    for name, tensor in continuous_payload["actor_representation"].items():
        assert torch.equal(tensor, resumed_payload["actor_representation"][name])
