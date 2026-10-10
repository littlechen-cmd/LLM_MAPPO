"""Resumable real-dataset LLM labeling contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.dataset_labeling import run_dataset_labeling
from llm_mappo.mappo_llm_kd.provider import (
    OpenAICompatibleProvider,
    ProviderConfig,
)


def _provider_config():
    return ProviderConfig(
        provider_id="fixture-provider",
        requested_model="fixture-model",
        base_url="https://provider.invalid/v1",
        endpoint_path="/chat/completions",
        api_key_environment_variable="PHASE3_DATASET_LABEL_TEST_KEY",
        temperature=0.0,
        seed=None,
        timeout_seconds=5.0,
        retry_limit=0,
        max_completion_tokens=500,
        max_prompt_characters=60000,
        response_format="json_object",
        extra_body={},
    )


def _response():
    dimensions = (
        "mission_urgency",
        "concession_appropriateness",
        "interaction_risk",
        "deadlock_severity",
    )
    return {
        "id": "chatcmpl-fixture",
        "model": "fixture-served-model",
        "choices": [{"message": {"content": json.dumps({
            "scores": {name: 0.5 for name in dimensions},
            "evidence": {name: ["summary.blocked_count"] for name in dimensions},
        })}}],
    }


def _dataset_directory(tmp_path):
    directory = tmp_path / "dataset"
    directory.mkdir()
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
    (directory / "manifest.json").write_text(
        json.dumps({
            "schema": "llm-dataset-build-run-v1",
            "distribution": "native-209d-collection-main-v1",
            "main_pretraining_eligible": True,
        }),
        encoding="utf-8",
    )
    (directory / "state.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (directory / "windows.jsonl").write_text(
        json.dumps({
            "partition": "train",
            "episode_id": "episode-1",
            "agent_id": 1,
            "start_timestep": 0,
            "end_timestep": 0,
            "observations": [row.tolist()],
            "reset_masks": [True],
            "stratum": "fixture",
        }) + "\n",
        encoding="utf-8",
    )
    return directory


def test_dataset_labeling_uses_only_persisted_windows_and_resumes(monkeypatch, tmp_path):
    monkeypatch.setenv("PHASE3_DATASET_LABEL_TEST_KEY", "unit-secret")
    calls = []

    def factory(config):
        def transport(endpoint, body, headers, timeout):
            calls.append((endpoint, body, headers, timeout))
            return _response()

        return OpenAICompatibleProvider(config, transport=transport)

    directory = _dataset_directory(tmp_path)
    result = run_dataset_labeling(
        directory,
        provider_config=_provider_config(),
        label_config_sha256="a" * 64,
        provider_factory=factory,
    )
    resumed = run_dataset_labeling(
        directory,
        provider_config=_provider_config(),
        label_config_sha256="a" * 64,
        provider_factory=factory,
    )

    output = directory / "labels" / ("a" * 12) / "fixture-provider"
    records = [
        json.loads(line)
        for line in (output / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 1
    assert result["successful_records"] == 1
    assert resumed["successful_records"] == 1
    assert records[0]["status"] == "ok"
    assert records[0]["view_id"] == "train:episode-1:agent-1:start-0:end-0"
    assert result["returned_models"] == ["fixture-served-model"]


def test_dataset_labeling_cli_requires_an_approved_provider_choice():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_llm_kd_dataset_labeling",
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--provider-id" in completed.stdout
