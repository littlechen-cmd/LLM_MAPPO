"""Strict, resumable provider labeling over persisted main-dataset windows."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)

from .labeling import AtomicLabelStore, LabelingFailure, label_view
from .provider import OpenAICompatibleProvider, ProviderConfig
from .temporal_view import serialize_temporal_semantic_view


DATASET_LABEL_RUN_V1 = "llm-dataset-label-run-v1"
_SCHEMA = StructuredObservationSchema(observation_schema=DIRECT_GOAL_OBSERVATION_V2)


def run_dataset_labeling(
    dataset_directory: str | Path,
    *,
    provider_config: ProviderConfig,
    label_config_sha256: str,
    provider_factory=OpenAICompatibleProvider,
) -> dict[str, object]:
    """Label persisted native-209D windows without reading an environment or A*."""

    directory = Path(dataset_directory)
    dataset_manifest = _load_completed_dataset_manifest(directory)
    _validate_digest(label_config_sha256, "Label config SHA-256")
    if not isinstance(provider_config, ProviderConfig):
        raise ValueError("Dataset labeling requires a ProviderConfig.")
    output = (
        directory / "labels" / label_config_sha256[:12] / provider_config.provider_id
    )
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "state.json", {"status": "running"})
    store = AtomicLabelStore(output / "labels.jsonl")
    provider = provider_factory(provider_config)
    if not isinstance(provider, OpenAICompatibleProvider):
        raise ValueError(
            "Dataset labeling provider factory returned an invalid provider."
        )
    windows = _load_windows(directory / "windows.jsonl")
    failures = 0
    try:
        for window in windows:
            try:
                label_view(
                    _window_id(window),
                    _semantic_view(window),
                    provider,
                    store,
                    cache_namespace="main-dataset",
                )
            except LabelingFailure:
                failures += 1
        records = store.records()
        successful = [record for record in records if record.get("status") == "ok"]
        if failures or len(successful) != len(windows):
            raise RuntimeError(
                "Dataset labeling did not produce complete success coverage."
            )
        receipt = {
            "schema": DATASET_LABEL_RUN_V1,
            "dataset_manifest_sha256": sha256(
                (directory / "manifest.json").read_bytes()
            ).hexdigest(),
            "dataset_distribution": dataset_manifest["distribution"],
            "label_config_sha256": label_config_sha256,
            "provider": provider_config.public_identity(),
            "successful_records": len(successful),
            "failed_records": len(records) - len(successful),
            "labels_sha256": sha256(
                (output / "labels.jsonl").read_bytes()
            ).hexdigest(),
            "returned_models": sorted(
                {record["receipt"]["returned_model"] for record in successful}
            ),
        }
        _atomic_json(output / "manifest.json", receipt)
        _atomic_json(output / "state.json", {"status": "completed"})
    except Exception as error:
        _atomic_json(
            output / "state.json",
            {"status": "failed", "error_type": type(error).__name__},
        )
        raise
    return receipt


def _load_completed_dataset_manifest(directory: Path) -> dict[str, object]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Dataset artifact cannot be read safely.") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "llm-dataset-build-run-v1"
        or manifest.get("distribution") != "native-209d-collection-main-v1"
        or manifest.get("main_pretraining_eligible") is not True
    ):
        raise ValueError("Dataset artifact is not an eligible main 209D dataset.")
    if not isinstance(state, dict) or state.get("status") != "completed":
        raise ValueError("Dataset artifact is not complete.")
    return manifest


def _load_windows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ValueError("Dataset artifact has no windows.")
    windows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            window = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Dataset window {line_number} is invalid JSON.") from error
        _validate_window(window)
        windows.append(window)
    if not windows:
        raise ValueError("Dataset artifact has no windows.")
    return windows


def _validate_window(window: object) -> None:
    expected = {
        "partition", "episode_id", "agent_id", "start_timestep", "end_timestep",
        "observations", "reset_masks", "stratum",
    }
    if not isinstance(window, dict) or set(window) != expected:
        raise ValueError("Dataset window fields are invalid.")
    if window["partition"] not in {"train", "validation", "test"}:
        raise ValueError("Dataset window partition is invalid.")
    if not isinstance(window["episode_id"], str) or not window["episode_id"]:
        raise ValueError("Dataset window episode identity is invalid.")
    if type(window["agent_id"]) is not int or window["agent_id"] < 1:
        raise ValueError("Dataset window agent identity is invalid.")
    if any(type(window[name]) is not int for name in ("start_timestep", "end_timestep")):
        raise ValueError("Dataset window timestep identity is invalid.")
    observations = np.asarray(window["observations"], dtype=np.float32)
    masks = np.asarray(window["reset_masks"], dtype=np.bool_)
    if (
        observations.ndim != 2
        or observations.shape[1] != 209
        or observations.shape[0] < 1
        or masks.shape != (observations.shape[0],)
        or not np.isfinite(observations).all()
    ):
        raise ValueError("Dataset window tensor payload is invalid.")


def _semantic_view(window: Mapping[str, object]) -> dict[str, object]:
    return serialize_temporal_semantic_view(
        np.asarray(window["observations"], dtype=np.float32),
        np.asarray(window["reset_masks"], dtype=np.bool_),
        _SCHEMA.feature_names,
        requested_history_length=len(window["observations"]),
    )


def _window_id(window: Mapping[str, object]) -> str:
    return (
        f"{window['partition']}:{window['episode_id']}:agent-{window['agent_id']}"
        f":start-{window['start_timestep']}:end-{window['end_timestep']}"
    )


def _validate_digest(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is invalid.")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
