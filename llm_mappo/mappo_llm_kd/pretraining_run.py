"""Small, strict runnable loop for Phase 3 semantic representation training."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
import torch
import yaml

from llm_mappo.mappo_baseline.networks import MAPPOActor

from .dataset import MAIN_BASELINE_DISTRIBUTION
from .pretraining import (
    ACTOR_ARCHITECTURE_V1,
    HIDDEN_DIM,
    OBSERVATION_DIM,
    SEMANTIC_DIMENSIONS,
    SEMANTIC_ONTOLOGY_V1,
    SEMANTIC_ONTOLOGY_V2,
    SEMANTIC_CONTRACT_V2,
    SEMANTIC_CONTRACT_V3,
    SEMANTIC_ONTOLOGY_V3,
    SEMANTIC_DIMENSIONS_V3,
    SEMANTIC_SOURCES_V3,
    LOSS_AGGREGATION_V3,
    DATASET_BUILD_V3,
    LABEL_RUN_V3,
    MAIN_BASELINE_DISTRIBUTION_V3,
    UNKNOWN_REASONS_V3,
    PretrainingBatch,
    SemanticPretrainingModel,
    masked_semantic_loss,
    validate_pretraining_identity,
)
from .pretraining_checkpoint import (
    load_pretraining_checkpoint,
    save_pretraining_checkpoint,
)
from .representation_export import export_actor_representation
from .medium_extension_dataset import LINEAGE_SCHEMA
from .native_derived_adapter_v3 import verify_native_derived_dataset


PRETRAINING_RUNNER_V1 = "semantic-pretraining-runner-v1"
_WINDOW_FIELDS = (
    "partition", "episode_id", "agent_id", "start_timestep", "end_timestep",
    "observations", "reset_masks", "stratum",
)
_SCORE_FIELDS = SEMANTIC_DIMENSIONS


@dataclass(frozen=True)
class PretrainingRunConfig:
    """Explicit, hash-bound configuration for a semantic-pretraining run."""

    schema: str
    dataset_directory: str
    label_directory: str
    artifact_root: str
    config_sha256: str
    approval_status: str
    seed: int
    batch_size: int
    max_epochs: int
    learning_rate: float
    dimension_weights: tuple[float, float, float, float]
    early_stopping_patience: int
    early_stopping_min_delta: float
    source_artifact_root: str | None = None


def load_pretraining_run_config(path: str | Path) -> PretrainingRunConfig:
    """Read a complete pretraining config and bind it to its file digest."""

    source = Path(path)
    raw = source.read_bytes()
    values = yaml.safe_load(raw)
    expected = {
        "schema", "dataset_directory", "label_directory", "artifact_root",
        "approval_status", "seed", "batch_size", "max_epochs", "learning_rate",
        "dimension_weights", "early_stopping_patience",
        "early_stopping_min_delta",
    }
    if (not isinstance(values, dict) or not expected <= set(values)
            or set(values) - expected - {"source_artifact_root"}):
        raise ValueError("Pretraining config has unexpected top-level keys.")
    config = PretrainingRunConfig(
        schema=values["schema"],
        dataset_directory=values["dataset_directory"],
        label_directory=values["label_directory"],
        artifact_root=values["artifact_root"],
        config_sha256=sha256(raw).hexdigest(),
        approval_status=values["approval_status"],
        seed=values["seed"],
        batch_size=values["batch_size"],
        max_epochs=values["max_epochs"],
        learning_rate=values["learning_rate"],
        dimension_weights=tuple(values["dimension_weights"]),
        early_stopping_patience=values["early_stopping_patience"],
        early_stopping_min_delta=values["early_stopping_min_delta"],
        source_artifact_root=values.get("source_artifact_root"),
    )
    _validate_config(config)
    return config


def run_semantic_pretraining(  # noqa: C901
    config: PretrainingRunConfig,
    *,
    device: str | torch.device,
    run_id: str | None = None,
    resume: str | Path | None = None,
) -> dict[str, object]:
    """Train Encoder+GRU only from complete main-dataset LLM labels.

    Labels supervise just a window's final frame.  Earlier frames remain recurrent
    context, preventing one window label from being incorrectly copied backward.
    """

    _validate_config(config)
    if config.approval_status != "approved":
        raise ValueError("Pretraining configuration is not frozen and approved.")
    target_device = torch.device(device)
    dataset = Path(config.dataset_directory)
    labels = Path(config.label_directory)
    windows = _load_eligible_windows(
        dataset, source_artifact_root=config.source_artifact_root,
    )
    _validate_label_artifact(dataset, labels)
    label_manifest = json.loads((labels / "manifest.json").read_text("utf-8"))
    contract = label_manifest.get("semantic_contract_version")
    v3 = contract == SEMANTIC_CONTRACT_V3
    if v3 and config.dimension_weights != (1.0, 1.0, 1.0, 0.25):
        raise ValueError("Pretraining v3 dimension_weights differ from the contract.")
    scores = _load_scores(
        labels / "labels.jsonl", contract=contract,
        view_ids={_window_id(window) for window in windows} if v3 else None,
    )
    samples = _attach_scores(windows, scores)
    train = [sample for sample in samples if sample["partition"] == "train"]
    validation = [
        sample for sample in samples if sample["partition"] == "validation"
    ]
    if not train or not validation:
        raise ValueError("Pretraining requires non-empty train and validation windows.")
    identity = _identity(config, dataset, labels)
    if resume is not None and run_id is not None:
        raise ValueError("Pretraining resume cannot also select a new run_id.")
    directory = (
        _resume_directory(config, resume)
        if resume is not None
        else _prepare_directory(config, run_id)
    )
    _seed(config.seed)
    actor = MAPPOActor(OBSERVATION_DIM, 5, HIDDEN_DIM).to(target_device)
    model = SemanticPretrainingModel(actor).to(target_device)
    optimizer = torch.optim.Adam(
        list(model.actor.encoder.parameters())
        + list(model.actor.gru.parameters())
        + list(model.semantic_head.parameters()),
        lr=float(config.learning_rate),
    )
    weights = torch.tensor(
        config.dimension_weights, dtype=torch.float32, device=target_device
    )
    global_step = 0
    start_epoch = 1
    start_batch = 0
    best_validation_loss = float(np.finfo(np.float64).max)
    selected_validation_loss = best_validation_loss
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    if resume is not None:
        best_restored = None
        if (directory / "checkpoint_best.pt").is_file():
            best_restored = load_pretraining_checkpoint(
                directory / "checkpoint_best.pt", model, optimizer,
                expected_identity=identity, map_location=target_device,
            )
            selected_validation_loss = float(
                best_restored["early_stopping"]["best_validation_loss"]
            )
            best_epoch = int(best_restored["progress"]["epoch"])
        restored = load_pretraining_checkpoint(
            directory / "checkpoint_latest.pt",
            model,
            optimizer,
            expected_identity=identity,
            map_location=target_device,
        )
        progress = restored["progress"]
        early_stopping = restored["early_stopping"]
        global_step = int(progress["global_step"])
        start_epoch = max(int(progress["epoch"]), 1)
        start_batch = int(progress["batch_in_epoch"])
        best_validation_loss = float(
            early_stopping["best_validation_loss"]
        )
        bad_epochs = int(early_stopping["bad_epochs"])
        if best_restored is not None:
            if best_restored["progress"]["global_step"] > global_step:
                raise ValueError("Pretraining best checkpoint is ahead of latest.")
        elif start_epoch > 1:
            raise ValueError(
                "Pretraining prior best checkpoint is missing; discarded "
                "validation weights cannot be recovered."
            )
    completed_epochs = start_epoch - 1
    if resume is not None:
        history = _load_run_history(directory, completed_epochs, v3=v3)
    _atomic_json(directory / "state.json", {"status": "running"})
    try:
        train_batches = list(_batches(train, config.batch_size))
        if start_batch > len(train_batches):
            raise ValueError("Pretraining resume batch position is out of range.")
        for epoch in range(start_epoch, config.max_epochs + 1):
            if completed_epochs > 0 and bad_epochs >= config.early_stopping_patience:
                break
            model.train()
            train_losses = []
            batch_offset = start_batch if epoch == start_epoch else 0
            skipped_unknown_batches = (
                sum(not any(any(sample["known_masks"]) for sample in previous_batch)
                    for previous_batch in train_batches[:batch_offset]) if v3 else 0
            )
            for batch_index, batch_samples in enumerate(
                train_batches[batch_offset:], start=batch_offset
            ):
                batch = _pretraining_batch(batch_samples, target_device)
                if v3 and not batch.target_masks.any().item():
                    skipped_unknown_batches += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch)
                loss = masked_semantic_loss(
                    predictions, batch.targets, batch.target_masks, weights
                )
                _require_finite(float(loss.detach().cpu()), "training loss")
                loss.backward()
                _require_finite_gradient(model)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
                global_step += 1
                save_pretraining_checkpoint(
                    directory / "checkpoint_latest.pt",
                    model,
                    optimizer,
                    identity=identity,
                    progress={
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_index + 1,
                    },
                    early_stopping={
                        "best_validation_loss": best_validation_loss,
                        "bad_epochs": bad_epochs,
                        "patience": config.early_stopping_patience,
                        "minimum_improvement": config.early_stopping_min_delta,
                    },
                )
            validation_loss = _evaluate(
                model, validation, config.batch_size, target_device, weights
            )
            _require_finite(validation_loss, "validation loss")
            if validation_loss < selected_validation_loss:
                selected_validation_loss = validation_loss
                best_epoch = epoch
                save_pretraining_checkpoint(
                    directory / "checkpoint_best.pt", model, optimizer,
                    identity=identity,
                    progress={
                        "global_step": global_step, "epoch": epoch,
                        "batch_in_epoch": len(train_batches),
                    },
                    early_stopping={
                        "best_validation_loss": selected_validation_loss,
                        "bad_epochs": bad_epochs,
                        "patience": config.early_stopping_patience,
                        "minimum_improvement": config.early_stopping_min_delta,
                    },
                )
            improved = validation_loss < (
                best_validation_loss - config.early_stopping_min_delta
            )
            if improved:
                best_validation_loss = validation_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": (
                        float(np.mean(train_losses)) if train_losses else 0.0
                    ),
                    "validation_loss": validation_loss,
                }
            )
            if v3:
                history[-1]["skipped_unknown_batches"] = skipped_unknown_batches
            _atomic_json(directory / "history.json", {"history": history})
            save_pretraining_checkpoint(
                directory / "checkpoint_latest.pt",
                model,
                optimizer,
                identity=identity,
                progress={
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "batch_in_epoch": 0,
                },
                early_stopping={
                    "best_validation_loss": best_validation_loss,
                    "bad_epochs": bad_epochs,
                    "patience": config.early_stopping_patience,
                    "minimum_improvement": config.early_stopping_min_delta,
                },
            )
            completed_epochs = epoch
            if bad_epochs >= config.early_stopping_patience:
                break
            start_batch = 0
        load_pretraining_checkpoint(
            directory / "checkpoint_best.pt", model, optimizer,
            expected_identity=identity, map_location=target_device,
        )
        representation = export_actor_representation(
            directory / "representation.pt", model.actor, identity=identity
        )
        manifest = {
            "schema": PRETRAINING_RUNNER_V1,
            "identity": identity,
            "global_step": global_step,
            "completed_epochs": completed_epochs,
            "best_validation_loss": selected_validation_loss,
            "best_epoch": best_epoch,
            "best_checkpoint": {
                "path": "checkpoint_best.pt",
                "file_sha256": sha256(
                    (directory / "checkpoint_best.pt").read_bytes()
                ).hexdigest(),
            },
            "early_stopping_reference_loss": best_validation_loss,
            "history": history,
            "history_missing_epochs": sorted(
                set(range(1, completed_epochs + 1))
                - {int(row["epoch"]) for row in history}
            ),
            "representation": representation,
            "formal_performance_evidence": False,
        }
        if v3:
            manifest["skipped_unknown_batches"] = sum(
                row["skipped_unknown_batches"] for row in history
            )
        _atomic_json(directory / "manifest.json", manifest)
        _atomic_json(directory / "state.json", {"status": "completed"})
    except Exception as error:
        _atomic_json(
            directory / "state.json",
            {"status": "failed", "error_type": type(error).__name__},
        )
        raise
    return {"run_directory": str(directory.resolve()), "manifest": manifest}


def _load_run_history(  # noqa: C901
    directory: Path, completed_epochs: int, *, v3: bool = False,
) -> list[dict]:
    """Read known rows and discard validation not yet committed in latest."""
    path = directory / "history.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Pretraining history cannot be read.") from error
    if (
        not isinstance(payload, dict) or set(payload) != {"history"}
        or not isinstance(payload["history"], list)
    ):
        raise ValueError("Pretraining history fields are invalid.")
    previous_epoch = 0
    rows = payload["history"]
    fields = {"epoch", "train_loss", "validation_loss"}
    if v3:
        fields.add("skipped_unknown_batches")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != fields
            or type(row["epoch"]) is not int
            or row["epoch"] <= previous_epoch
        ):
            raise ValueError("Pretraining history epoch ordering is invalid.")
        for key in ("train_loss", "validation_loss"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Pretraining history loss is invalid.")
        if v3 and (type(row["skipped_unknown_batches"]) is not int
                   or row["skipped_unknown_batches"] < 0):
            raise ValueError("Pretraining history skip count is invalid.")
        previous_epoch = row["epoch"]
    return [row for row in rows if row["epoch"] <= completed_epochs]


def _validate_config(config: object) -> None:  # noqa: C901
    if not isinstance(config, PretrainingRunConfig):
        raise ValueError("Pretraining config is invalid.")
    if config.schema != PRETRAINING_RUNNER_V1:
        raise ValueError("Pretraining runner schema differs.")
    for name in ("dataset_directory", "label_directory", "artifact_root"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name):
            raise ValueError(f"Pretraining {name} is invalid.")
    if (config.source_artifact_root is not None
            and (not isinstance(config.source_artifact_root, str)
                 or not config.source_artifact_root)):
        raise ValueError("Pretraining source_artifact_root is invalid.")
    _validate_digest(config.config_sha256, "Pretraining config SHA-256")
    if config.approval_status not in {"provisional", "approved"}:
        raise ValueError("Pretraining approval_status is invalid.")
    for name in ("seed", "batch_size", "max_epochs", "early_stopping_patience"):
        value = getattr(config, name)
        if type(value) is not int or value < 0 or (
            name in {"batch_size", "max_epochs"} and value == 0
        ):
            raise ValueError(f"Pretraining {name} is invalid.")
    for name in ("learning_rate", "early_stopping_min_delta"):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Pretraining {name} is invalid.")
    if config.learning_rate == 0:
        raise ValueError("Pretraining learning_rate must be positive.")
    if (
        not isinstance(config.dimension_weights, tuple)
        or len(config.dimension_weights) != len(SEMANTIC_DIMENSIONS)
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in config.dimension_weights
        )
        or sum(config.dimension_weights) <= 0
    ):
        raise ValueError("Pretraining dimension_weights are invalid.")


def _load_eligible_windows(  # noqa: C901
    directory: Path,
    *, source_artifact_root: str | None = None,
) -> list[dict[str, object]]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        state = json.loads((directory / "state.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Pretraining dataset artifact cannot be read.") from error
    v3 = isinstance(manifest, dict) and manifest.get("schema") == DATASET_BUILD_V3
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != (
            DATASET_BUILD_V3 if v3 else "llm-dataset-build-run-v1"
        )
        or manifest.get("distribution") != (
            MAIN_BASELINE_DISTRIBUTION_V3 if v3 else MAIN_BASELINE_DISTRIBUTION
        )
        or manifest.get("main_pretraining_eligible") is not True
        or not isinstance(state, dict)
        or state.get("status") != "completed"
    ):
        raise ValueError("Pretraining requires a completed eligible main dataset.")
    if v3:
        _validate_v3_manifest_semantics(manifest)
    lineage = manifest.get("lineage_schema")
    if lineage == LINEAGE_SCHEMA:
        if not v3:
            raise ValueError("Native derived lineage requires the v3 semantic contract.")
        verify_native_derived_dataset(
            directory, manifest, artifact_root=source_artifact_root,
        )
    elif lineage is not None or source_artifact_root is not None:
        raise ValueError("Unknown dataset lineage or source relocation is not approved.")
    windows_path = directory / "windows.jsonl"
    try:
        lines = windows_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("Pretraining dataset has no windows.") from error
    result = []
    episodes = {}
    seen_ids = set()
    for number, line in enumerate(lines, 1):
        try:
            window = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pretraining window {number} is invalid JSON.") from error
        if v3:
            if (not isinstance(window, dict)
                    or window.get("partition") not in {"train", "validation", "test"}
                    or not isinstance(window.get("episode_id"), str)):
                raise ValueError("Pretraining window partition identity is invalid.")
            previous_partition = episodes.setdefault(
                window["episode_id"], window["partition"]
            )
            if previous_partition != window["partition"]:
                raise ValueError("Pretraining episode crosses dataset partitions.")
            if window["partition"] == "test":
                continue
        _validate_window(window)
        if v3 and _window_id(window) in seen_ids:
            raise ValueError("Pretraining windows have duplicate identities.")
        seen_ids.add(_window_id(window))
        result.append(window)
    if not result:
        raise ValueError("Pretraining dataset has no windows.")
    expected_digest = manifest.get("windows_sha256")
    if (
        not isinstance(expected_digest, str)
        or sha256(windows_path.read_bytes()).hexdigest() != expected_digest
    ):
        raise ValueError("Pretraining dataset windows digest differs.")
    return result


def _validate_window(window: object) -> None:
    if not isinstance(window, dict) or set(window) != set(_WINDOW_FIELDS):
        raise ValueError("Pretraining window fields are invalid.")
    if window["partition"] not in {"train", "validation", "test"}:
        raise ValueError("Pretraining window partition is invalid.")
    if not isinstance(window["episode_id"], str) or not window["episode_id"]:
        raise ValueError("Pretraining window episode identity is invalid.")
    if type(window["agent_id"]) is not int or window["agent_id"] < 1:
        raise ValueError("Pretraining window agent identity is invalid.")
    observations = np.asarray(window["observations"], dtype=np.float32)
    resets = np.asarray(window["reset_masks"], dtype=np.bool_)
    if (
        observations.ndim != 2
        or observations.shape[0] < 1
        or observations.shape[1] != OBSERVATION_DIM
        or resets.shape != (observations.shape[0],)
        or not np.isfinite(observations).all()
    ):
        raise ValueError("Pretraining window tensor payload is invalid.")


def _validate_label_artifact(dataset: Path, labels: Path) -> None:  # noqa: C901
    labels_path = labels / "labels.jsonl"
    try:
        manifest = json.loads((labels / "manifest.json").read_text("utf-8"))
        state = json.loads((labels / "state.json").read_text("utf-8"))
        labels_bytes = labels_path.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Pretraining label artifact cannot be read.") from error
    rule_artifact = (
        isinstance(manifest, dict)
        and manifest.get("schema") == "deterministic-semantic-rule-labels-v1"
        and manifest.get("teacher_type") == "deterministic-observable-rule-v1"
        and manifest.get("semantic_contract_version") == SEMANTIC_CONTRACT_V2
    )
    if (
        not isinstance(manifest, dict)
        or (manifest.get("schema") not in {"llm-dataset-label-run-v1", LABEL_RUN_V3}
            and not rule_artifact)
        or manifest.get("dataset_manifest_sha256")
        != sha256((dataset / "manifest.json").read_bytes()).hexdigest()
        or manifest.get("labels_sha256") != sha256(labels_bytes).hexdigest()
        or not isinstance(manifest.get("successful_records"), int)
        or manifest["successful_records"] < 1
        or not isinstance(state, dict)
        or state.get("status") != "completed"
    ):
        raise ValueError("Pretraining labels are not a completed bound artifact.")
    dataset_manifest = json.loads((dataset / "manifest.json").read_text("utf-8"))
    contract = dataset_manifest.get("semantic_contract_version")
    label_contract = manifest.get("semantic_contract_version")
    if (manifest.get("schema") == LABEL_RUN_V3
            and label_contract != SEMANTIC_CONTRACT_V3):
        raise ValueError("Pretraining v3 label artifact requires its v3 contract.")
    if contract == SEMANTIC_CONTRACT_V3 or label_contract == SEMANTIC_CONTRACT_V3:
        if (contract != SEMANTIC_CONTRACT_V3 or label_contract != contract
                or dataset_manifest.get("schema") != DATASET_BUILD_V3
                or manifest.get("schema") != LABEL_RUN_V3
                or manifest.get("pretraining_eligible") is not True):
            raise ValueError("Pretraining v3 semantic contract is incompatible.")
        _validate_v3_manifest_semantics(manifest)
        _validate_v3_quality_evidence(labels, manifest)
        if dataset_manifest.get("lineage_schema") == LINEAGE_SCHEMA:
            plan = json.loads((labels / "plan.json").read_text("utf-8"))
            required_code = {
                "medium_extension_dataset.py", "native_derived_adapter_v3.py"
            }
            if (manifest.get("lineage_schema") != LINEAGE_SCHEMA
                    or manifest.get("lineage_sha256")
                    != dataset_manifest.get("collection_identity_sha256")
                    or plan.get("lineage_schema") != LINEAGE_SCHEMA
                    or plan.get("lineage_sha256") != manifest["lineage_sha256"]
                    or plan.get("dataset_windows_sha256")
                    != dataset_manifest.get("windows_sha256")
                    or not isinstance(plan.get("code_sha256"), dict)
                    or not required_code <= set(plan["code_sha256"])
                    or any(plan["code_sha256"][name] != sha256(
                        (Path(__file__).with_name(name)).read_bytes()
                    ).hexdigest() for name in required_code)):
                raise ValueError("Pretraining native derived label lineage differs.")
        elif (dataset_manifest.get("lineage_schema") is not None
              or manifest.get("lineage_schema") is not None):
            raise ValueError("Pretraining label lineage is incompatible.")
        records = [
            json.loads(line) for line in labels_bytes.decode("utf-8").splitlines()
        ]
        if (any(not isinstance(record, dict) or record.get("status") != "ok"
                for record in records)
                or len(records) != manifest["successful_records"]):
            raise ValueError("Pretraining v3 label success coverage is invalid.")
        return
    if contract is not None or label_contract is not None:
        if (contract != SEMANTIC_CONTRACT_V2 or label_contract != contract
                or manifest.get("pretraining_eligible") is not True):
            raise ValueError("Pretraining semantic contract is incompatible.")
        records = [
            json.loads(line) for line in labels_bytes.decode("utf-8").splitlines()
        ]
        for record in records:
            if record.get("status") != "ok":
                continue
            if record.get("receipt", {}).get(
                "semantic_contract_version"
            ) != contract:
                raise ValueError(
                    "Pretraining label receipt contract is missing or mixed."
                )


def _load_scores(  # noqa: C901
    path: Path, *, contract: str | None = None, view_ids: set[str] | None = None,
) -> dict:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("Pretraining labels cannot be read.") from error
    result = {}
    for number, line in enumerate(lines, 1):
        try:
            label = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pretraining label {number} is invalid JSON.") from error
        if not isinstance(label, dict) or label.get("status") != "ok":
            continue
        view_id, values = label.get("view_id"), label.get("scores")
        if view_ids is not None and view_id not in view_ids:
            continue
        if not isinstance(view_id, str) or view_id in result:
            raise ValueError("Pretraining labels have duplicate or invalid view IDs.")
        if contract == SEMANTIC_CONTRACT_V3:
            result[view_id] = _v3_scores(label)
            continue
        if not isinstance(values, dict) or set(values) != set(_SCORE_FIELDS):
            raise ValueError("Pretraining label scores are invalid.")
        ordered = tuple(values[name] for name in _SCORE_FIELDS)
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or value > 1
            for value in ordered
        ):
            raise ValueError("Pretraining label scores must lie in [0, 1].")
        result[view_id] = tuple(float(value) for value in ordered)
    if not result:
        raise ValueError("Pretraining labels have no successful records.")
    return result


def _validate_v3_quality_evidence(labels: Path, manifest: Mapping[str, object]) -> None:
    """Retained relative evidence paths keep an authenticated bundle portable."""
    if (manifest.get("quality_report_path") != "quality-main.json"
            or manifest.get("plan_path") != "plan.json"):
        raise ValueError("Pretraining v3 requires retained quality and plan evidence.")
    quality_path, plan_path = labels / "quality-main.json", labels / "plan.json"
    try:
        quality_bytes, plan_bytes = quality_path.read_bytes(), plan_path.read_bytes()
        quality, plan = json.loads(quality_bytes), json.loads(plan_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Pretraining quality or plan evidence is missing.") from error
    if (sha256(quality_bytes).hexdigest() != manifest.get("quality_report_sha256")
            or sha256(plan_bytes).hexdigest() != manifest.get("plan_sha256")):
        raise ValueError("Pretraining v3 quality or plan evidence hash differs.")
    if (not isinstance(quality, dict) or quality.get("schema")
            != "semantic-quality-report-v3" or quality.get("stage") != "main"
            or quality.get("status") != "go"
            or quality.get("main_pretraining_eligible") is not True
            or not isinstance(quality.get("checks"), list) or not quality["checks"]
            or any(not isinstance(check, dict) or check.get("status") != "go"
                   for check in quality["checks"])):
        raise ValueError("Pretraining v3 quality evidence is not a complete main Go.")
    bindings = quality.get("bindings")
    if (not isinstance(plan, dict) or plan.get("schema") != "frozen-labeling-plan-v3"
            or plan.get("semantic_contract_version") != SEMANTIC_CONTRACT_V3
            or plan.get("dataset_manifest_sha256")
            != manifest["dataset_manifest_sha256"] or not isinstance(bindings, dict)
            or any(bindings.get(name) != manifest[name] for name in (
                "plan_sha256", "dataset_manifest_sha256", "labels_sha256"
            ))):
        raise ValueError("Pretraining v3 quality/plan bindings differ from the labels.")


def _validate_v3_manifest_semantics(manifest: Mapping[str, object]) -> None:
    if (manifest.get("semantic_contract_version") != SEMANTIC_CONTRACT_V3
            or manifest.get("semantic_dimensions") != list(SEMANTIC_DIMENSIONS_V3)
            or manifest.get("semantic_sources") != list(SEMANTIC_SOURCES_V3)):
        raise ValueError("Pretraining v3 manifest semantics are incompatible.")


def _v3_scores(label: Mapping[str, object]) -> dict:  # noqa: C901
    fields = ("scores", "known_masks", "sources", "evidence", "unknown_reasons")
    for field in fields:
        if (not isinstance(label.get(field), dict)
                or set(label[field]) != set(SEMANTIC_DIMENSIONS_V3)):
            raise ValueError(f"Pretraining v3 label {field} fields are invalid.")
    receipt = label.get("receipt")
    if (not isinstance(receipt, dict)
            or receipt.get("semantic_contract_version") != SEMANTIC_CONTRACT_V3):
        raise ValueError("Pretraining v3 label receipt contract is incompatible.")
    scores, masks = [], []
    for name, source in zip(SEMANTIC_DIMENSIONS_V3, SEMANTIC_SOURCES_V3):
        score = label["scores"][name]
        known = label["known_masks"][name]
        evidence = label["evidence"][name]
        reason = label["unknown_reasons"][name]
        if type(known) is not bool or label["sources"][name] != source:
            raise ValueError("Pretraining v3 known masks or label sources differ.")
        if (not isinstance(evidence, list)
                or any(not isinstance(path, str) or not path for path in evidence)):
            raise ValueError("Pretraining v3 evidence is invalid.")
        if (len(set(evidence)) != len(evidence)
                or (name != "motion-stagnation" and not 1 <= len(evidence) <= 4)):
            raise ValueError("Pretraining v3 evidence count or uniqueness is invalid.")
        if known:
            if (type(score) not in (int, float) or not math.isfinite(score)
                    or not 0 <= score <= 1 or reason is not None or not evidence):
                raise ValueError("Pretraining v3 known score is invalid.")
        elif (score is not None or not isinstance(reason, str)
              or reason not in UNKNOWN_REASONS_V3[name]):
            raise ValueError("Pretraining v3 unknown score must be null with a reason.")
        scores.append(float(score) if known else 0.0)
        masks.append(known)
    return {"scores": tuple(scores), "known_masks": tuple(masks)}


def _attach_scores(windows, scores):
    result = []
    for window in windows:
        view_id = _window_id(window)
        if view_id not in scores:
            raise ValueError("Pretraining windows do not have complete label coverage.")
        payload = scores[view_id]
        result.append({**window, **payload} if isinstance(payload, dict)
                      else {**window, "scores": payload})
    return result


def _window_id(window: Mapping[str, object]) -> str:
    return (
        f"{window['partition']}:{window['episode_id']}:agent-{window['agent_id']}"
        f":start-{window['start_timestep']}:end-{window['end_timestep']}"
    )


def _identity(config: PretrainingRunConfig, dataset: Path, labels: Path) -> dict:
    digest = sha256()
    for path in (
        dataset / "manifest.json",
        dataset / "windows.jsonl",
        labels / "labels.jsonl",
    ):
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    manifest = json.loads((labels / "manifest.json").read_text("utf-8"))
    contract = manifest.get("semantic_contract_version")
    v3 = contract == SEMANTIC_CONTRACT_V3
    if v3:
        content = (labels / "manifest.json").read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    if contract is not None and contract not in {
        SEMANTIC_CONTRACT_V2, SEMANTIC_CONTRACT_V3,
    }:
        raise ValueError("Pretraining label semantic contract is incompatible.")
    identity = {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": OBSERVATION_DIM,
        "ontology_schema": (
            SEMANTIC_ONTOLOGY_V3 if v3 else
            SEMANTIC_ONTOLOGY_V2 if contract else SEMANTIC_ONTOLOGY_V1
        ),
        "semantic_dimensions": SEMANTIC_DIMENSIONS_V3 if v3 else SEMANTIC_DIMENSIONS,
        "actor_architecture": ACTOR_ARCHITECTURE_V1,
        "hidden_dim": HIDDEN_DIM,
        "dataset_sha256": digest.hexdigest(),
        "config_sha256": config.config_sha256,
    }
    if contract:
        identity["semantic_contract_version"] = contract
    if v3:
        identity["semantic_sources"] = SEMANTIC_SOURCES_V3
        identity["loss_aggregation"] = LOSS_AGGREGATION_V3
    validate_pretraining_identity(identity)
    return identity


def _prepare_directory(config: PretrainingRunConfig, run_id: str | None) -> Path:
    identifier = run_id or "pretraining"
    if not isinstance(identifier, str) or not identifier or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in identifier
    ):
        raise ValueError("Pretraining run_id contains unsafe characters.")
    directory = Path(config.artifact_root) / identifier
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _resume_directory(config: PretrainingRunConfig, resume: str | Path) -> Path:
    directory = Path(resume).resolve()
    artifact_root = Path(config.artifact_root).resolve()
    if directory.parent != artifact_root or not directory.is_dir():
        raise ValueError("Pretraining resume directory is outside artifact_root.")
    try:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Pretraining resume state cannot be read.") from error
    if not isinstance(state, dict) or state.get("status") not in {"running", "failed"}:
        raise ValueError("Only an interrupted pretraining run can resume.")
    if not (directory / "checkpoint_latest.pt").is_file():
        raise ValueError("Pretraining resume checkpoint is missing.")
    return directory


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batches(items: Sequence[dict[str, object]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _pretraining_batch(samples, device: torch.device) -> PretrainingBatch:
    length = max(len(sample["observations"]) for sample in samples)
    batch_size = len(samples)
    observations = torch.zeros(batch_size, length, OBSERVATION_DIM, device=device)
    valid_masks = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    reset_masks = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    targets = torch.zeros(batch_size, length, len(SEMANTIC_DIMENSIONS), device=device)
    per_dimension = "known_masks" in samples[0]
    if any(("known_masks" in sample) != per_dimension for sample in samples):
        raise ValueError("Pretraining batch mixes semantic target mask contracts.")
    mask_shape = (batch_size, length, 4) if per_dimension else (batch_size, length)
    target_masks = torch.zeros(mask_shape, dtype=torch.bool, device=device)
    for index, sample in enumerate(samples):
        values = torch.as_tensor(
            sample["observations"], dtype=torch.float32, device=device
        )
        resets = torch.as_tensor(
            sample["reset_masks"], dtype=torch.bool, device=device
        )
        score = torch.as_tensor(sample["scores"], dtype=torch.float32, device=device)
        start = length - values.shape[0]
        observations[index, start:] = values
        valid_masks[index, start:] = True
        reset_masks[index, start:] = resets
        targets[index, -1] = score
        target_masks[index, -1] = (
            torch.as_tensor(sample["known_masks"], dtype=torch.bool, device=device)
            if per_dimension else True
        )
    return PretrainingBatch(
        observations=observations,
        valid_masks=valid_masks,
        reset_masks=reset_masks,
        targets=targets,
        initial_hidden=torch.zeros(batch_size, HIDDEN_DIM, device=device),
        target_masks=target_masks,
    )


def _evaluate(model, samples, batch_size, device, weights) -> float:
    model.eval()
    total_loss = 0.0
    target_count = 0
    dimension_sse = torch.zeros(4, dtype=torch.float64, device=device)
    dimension_counts = torch.zeros(4, dtype=torch.int64, device=device)
    per_dimension = bool(samples) and "known_masks" in samples[0]
    with torch.no_grad():
        for batch_samples in _batches(samples, batch_size):
            batch = _pretraining_batch(batch_samples, device)
            predictions = model(batch)
            if per_dimension:
                errors = (predictions.double() - batch.targets.double()).square()
                dimension_sse += errors.masked_fill(
                    ~batch.target_masks, 0.0
                ).sum(dim=(0, 1))
                dimension_counts += batch.target_masks.sum(dim=(0, 1))
                continue
            loss = masked_semantic_loss(
                predictions, batch.targets, batch.target_masks, weights
            )
            count = int(batch.target_masks.sum().item())
            total_loss += float(loss.detach().cpu()) * count
            target_count += count
    result = total_loss / target_count if target_count else float("nan")
    if per_dimension:
        active = (dimension_counts > 0) & (weights > 0)
        if active.any().item():
            means = dimension_sse / dimension_counts.clamp_min(1)
            result = float((
                (means[active] * weights[active]).sum() / weights[active].sum()
            ).cpu())
    _require_finite(result, "validation loss")
    return result


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"Pretraining {name} is non-finite.")


def _require_finite_gradient(model: SemanticPretrainingModel) -> None:
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise ValueError("Pretraining gradient is non-finite.")


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
