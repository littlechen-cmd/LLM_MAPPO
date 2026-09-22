"""Runnable Phase 3 semantic-pretraining smoke contracts."""

from __future__ import annotations

import json
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
