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
    if not isinstance(values, dict) or set(values) != expected:
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
    )
    _validate_config(config)
    return config


def run_semantic_pretraining(
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
    windows = _load_eligible_windows(dataset)
    _validate_label_artifact(dataset, labels)
    scores = _load_scores(labels / "labels.jsonl")
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
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    if resume is not None:
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
    _atomic_json(directory / "state.json", {"status": "running"})
    try:
        train_batches = list(_batches(train, config.batch_size))
        if start_batch > len(train_batches):
            raise ValueError("Pretraining resume batch position is out of range.")
        for epoch in range(start_epoch, config.max_epochs + 1):
            model.train()
            train_losses = []
            batch_offset = start_batch if epoch == start_epoch else 0
            for batch_index, batch_samples in enumerate(
                train_batches[batch_offset:], start=batch_offset
            ):
                batch = _pretraining_batch(batch_samples, target_device)
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
            if bad_epochs >= config.early_stopping_patience:
                break
            start_batch = 0
        representation = export_actor_representation(
            directory / "representation.pt", model.actor, identity=identity
        )
        manifest = {
            "schema": PRETRAINING_RUNNER_V1,
            "identity": identity,
            "global_step": global_step,
            "completed_epochs": len(history),
            "best_validation_loss": best_validation_loss,
            "history": history,
            "representation": representation,
            "formal_performance_evidence": False,
        }
        _atomic_json(directory / "manifest.json", manifest)
        _atomic_json(directory / "state.json", {"status": "completed"})
    except Exception as error:
        _atomic_json(
            directory / "state.json",
            {"status": "failed", "error_type": type(error).__name__},
        )
        raise
    return {"run_directory": str(directory.resolve()), "manifest": manifest}


def _validate_config(config: object) -> None:  # noqa: C901
    if not isinstance(config, PretrainingRunConfig):
        raise ValueError("Pretraining config is invalid.")
    if config.schema != PRETRAINING_RUNNER_V1:
        raise ValueError("Pretraining runner schema differs.")
    for name in ("dataset_directory", "label_directory", "artifact_root"):
        if not isinstance(getattr(config, name), str) or not getattr(config, name):
            raise ValueError(f"Pretraining {name} is invalid.")
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
) -> list[dict[str, object]]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        state = json.loads((directory / "state.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Pretraining dataset artifact cannot be read.") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "llm-dataset-build-run-v1"
        or manifest.get("distribution") != MAIN_BASELINE_DISTRIBUTION
        or manifest.get("main_pretraining_eligible") is not True
        or not isinstance(state, dict)
        or state.get("status") != "completed"
    ):
        raise ValueError("Pretraining requires a completed eligible main dataset.")
    windows_path = directory / "windows.jsonl"
    try:
        lines = windows_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError("Pretraining dataset has no windows.") from error
    result = []
    for number, line in enumerate(lines, 1):
        try:
            window = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Pretraining window {number} is invalid JSON.") from error
        _validate_window(window)
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


def _validate_label_artifact(dataset: Path, labels: Path) -> None:
    labels_path = labels / "labels.jsonl"
    try:
        manifest = json.loads((labels / "manifest.json").read_text("utf-8"))
        state = json.loads((labels / "state.json").read_text("utf-8"))
        labels_bytes = labels_path.read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Pretraining label artifact cannot be read.") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "llm-dataset-label-run-v1"
        or manifest.get("dataset_manifest_sha256")
        != sha256((dataset / "manifest.json").read_bytes()).hexdigest()
        or manifest.get("labels_sha256") != sha256(labels_bytes).hexdigest()
        or not isinstance(manifest.get("successful_records"), int)
        or manifest["successful_records"] < 1
        or not isinstance(state, dict)
        or state.get("status") != "completed"
    ):
        raise ValueError("Pretraining labels are not a completed bound artifact.")


def _load_scores(  # noqa: C901
    path: Path,
) -> dict[str, tuple[float, float, float, float]]:
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
        if not isinstance(view_id, str) or view_id in result:
            raise ValueError("Pretraining labels have duplicate or invalid view IDs.")
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


def _attach_scores(windows, scores):
    result = []
    for window in windows:
        view_id = _window_id(window)
        if view_id not in scores:
            raise ValueError("Pretraining windows do not have complete label coverage.")
        result.append({**window, "scores": scores[view_id]})
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
    identity = {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": OBSERVATION_DIM,
        "ontology_schema": SEMANTIC_ONTOLOGY_V1,
        "semantic_dimensions": SEMANTIC_DIMENSIONS,
        "actor_architecture": ACTOR_ARCHITECTURE_V1,
        "hidden_dim": HIDDEN_DIM,
        "dataset_sha256": digest.hexdigest(),
        "config_sha256": config.config_sha256,
    }
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
    target_masks = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
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
        target_masks[index, -1] = True
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
    losses = []
    with torch.no_grad():
        for batch_samples in _batches(samples, batch_size):
            batch = _pretraining_batch(batch_samples, device)
            loss = masked_semantic_loss(
                model(batch), batch.targets, batch.target_masks, weights
            )
            losses.append(float(loss.detach().cpu()))
    result = float(np.mean(losses))
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
