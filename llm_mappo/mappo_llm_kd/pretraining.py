"""Strict tensor and identity contracts for Phase 3 semantic pretraining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
)


PRETRAINING_RUN_V1 = "semantic-pretraining-run-v1"
PRETRAINING_CHECKPOINT_V1 = "semantic-pretraining-checkpoint-v1"
PRETRAINING_EXPORT_V1 = "actor-representation-export-v1"
SEMANTIC_ONTOLOGY_V1 = "agent-centric-four-score-ontology-v1"
SEMANTIC_ONTOLOGY_V2 = "agent-centric-observable-coordination-four-score-v2"
SEMANTIC_CONTRACT_V2 = "observable-coordination-four-score-v2"
ACTOR_ARCHITECTURE_V1 = "mappo-actor-encoder-gru-v1"

OBSERVATION_DIM = 209
HIDDEN_DIM = 64
SEMANTIC_DIM = 4
SEMANTIC_DIMENSIONS = (
    "mission_urgency",
    "concession_appropriateness",
    "interaction_risk",
    "deadlock_severity",
)
REQUIRED_PRETRAINING_IDENTITY_FIELDS = (
    "observation_schema",
    "observation_dim",
    "ontology_schema",
    "semantic_dimensions",
    "actor_architecture",
    "hidden_dim",
    "dataset_sha256",
    "config_sha256",
)


@dataclass(frozen=True)
class PretrainingBatch:
    """One batch-first, padded recurrent history batch and semantic targets."""

    observations: Tensor
    valid_masks: Tensor
    reset_masks: Tensor
    targets: Tensor
    initial_hidden: Tensor
    target_masks: Tensor | None = None

    def __post_init__(self) -> None:
        for name in (
            "observations", "valid_masks", "reset_masks", "targets",
            "initial_hidden",
        ):
            _require_tensor(name, getattr(self, name))
        if self.observations.ndim != 3:
            raise ValueError("observations must have rank 3 [batch,time,209].")
        if self.observations.shape[-1] != OBSERVATION_DIM:
            raise ValueError("observations must use the 209D v2 schema.")
        batch_time = tuple(self.observations.shape[:2])
        if any(size < 1 for size in batch_time):
            raise ValueError("Pretraining batch and time axes must be non-empty.")
        _validate_mask("valid_masks", self.valid_masks, batch_time)
        _validate_mask("reset_masks", self.reset_masks, batch_time)
        target_masks = _validated_target_masks(
            self.target_masks, self.valid_masks, batch_time
        )
        _validate_targets(self.targets, batch_time)
        _validate_initial_hidden(self.initial_hidden, batch_time[0])
        for name, value in (
            ("observations", self.observations),
            ("targets", self.targets),
            ("initial_hidden", self.initial_hidden),
        ):
            _validate_finite_floating(name, value)
        if torch.any((self.targets < 0.0) | (self.targets > 1.0)).item():
            raise ValueError("targets must be within [0, 1].")
        if not self.valid_masks[:, -1].all().item():
            raise ValueError("Every sequence must have a valid final frame.")
        _validate_contiguous_valid_suffix(self.valid_masks)
        if torch.any(self.reset_masks & ~self.valid_masks).item():
            raise ValueError("reset_masks cannot mark padded frames.")
        object.__setattr__(self, "target_masks", target_masks)

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[0])

    @property
    def time_steps(self) -> int:
        return int(self.observations.shape[1])


class SemanticPretrainingModel(nn.Module):
    """Reuse one MAPPO Actor representation with a temporary semantic head."""

    def __init__(self, actor: MAPPOActor) -> None:
        super().__init__()
        if not isinstance(actor, MAPPOActor):
            raise ValueError("Semantic pretraining requires one MAPPOActor instance.")
        if actor.observation_dim != OBSERVATION_DIM or actor.hidden_dim != HIDDEN_DIM:
            raise ValueError("Semantic pretraining requires the 209D/64D Actor.")
        self.actor = actor
        self.semantic_head = nn.Linear(actor.hidden_dim, SEMANTIC_DIM)

    def forward(self, batch: PretrainingBatch) -> Tensor:
        """Predict batch-first scores without evaluating left-padding frames."""

        if not isinstance(batch, PretrainingBatch):
            raise ValueError("Semantic pretraining requires a PretrainingBatch.")
        predictions = []
        for sample in range(batch.batch_size):
            first_valid = int(
                torch.nonzero(batch.valid_masks[sample], as_tuple=False)[0].item()
            )
            observations = batch.observations[sample, first_valid:].unsqueeze(1)
            resets = batch.reset_masks[sample, first_valid:].unsqueeze(1)
            continuation = (~resets).to(dtype=observations.dtype)
            encoded, _ = self.actor._sequence(
                observations,
                batch.initial_hidden[sample].unsqueeze(0),
                continuation,
            )
            valid_predictions = torch.sigmoid(
                self.semantic_head(encoded.squeeze(1))
            )
            predictions.append(F.pad(valid_predictions, (0, 0, first_valid, 0)))
        output = torch.stack(predictions)
        validate_prediction_tensor(output, batch)
        return output


def masked_semantic_loss(
    predictions: Tensor,
    targets: Tensor,
    valid_masks: Tensor,
    dimension_weights: Tensor,
) -> Tensor:
    """Return weighted MSE over valid frames and four explicit dimensions."""

    _require_tensor("predictions", predictions)
    _require_tensor("targets", targets)
    _require_tensor("valid_masks", valid_masks)
    _require_tensor("dimension_weights", dimension_weights)
    _validate_loss_tensors(predictions, targets, valid_masks)
    _validate_dimension_weights(dimension_weights)
    weights = dimension_weights.to(
        device=predictions.device, dtype=predictions.dtype
    )
    valid_errors = (
        predictions[valid_masks] - targets[valid_masks]
    ).square()
    return (valid_errors * weights).sum() / (
        valid_errors.shape[0] * weights.sum()
    )


def _validate_loss_tensors(
    predictions: Tensor, targets: Tensor, valid_masks: Tensor
) -> None:
    if predictions.ndim != 3 or predictions.shape[-1] != SEMANTIC_DIM:
        raise ValueError("predictions must have shape [batch,time,4].")
    if tuple(targets.shape) != tuple(predictions.shape):
        raise ValueError("targets must match prediction shape [batch,time,4].")
    if valid_masks.dtype is not torch.bool:
        raise ValueError("valid_masks must be a boolean tensor.")
    if tuple(valid_masks.shape) != tuple(predictions.shape[:2]):
        raise ValueError("valid_masks must match prediction batch/time axes.")
    if not valid_masks.any().item():
        raise ValueError("At least one valid frame is required.")
    if (
        predictions.device != targets.device
        or predictions.device != valid_masks.device
    ):
        raise ValueError("predictions, targets, and valid_masks must share a device.")


def _validate_dimension_weights(dimension_weights: Tensor) -> None:
    if dimension_weights.ndim != 1 or dimension_weights.numel() != SEMANTIC_DIM:
        raise ValueError("dimension_weights must contain exactly four values.")
    if not dimension_weights.is_floating_point():
        raise ValueError("dimension_weights must be floating-point values.")
    if not torch.isfinite(dimension_weights).all().item():
        raise ValueError("dimension_weights must contain only finite values.")
    if torch.any(dimension_weights < 0.0).item():
        raise ValueError("dimension_weights must be non-negative.")
    if dimension_weights.sum().item() <= 0.0:
        raise ValueError("dimension_weights must have a positive total.")


def validate_prediction_tensor(predictions: Tensor, batch: PretrainingBatch) -> None:
    """Reject a malformed SemanticHead output before loss calculation."""

    _require_tensor("predictions", predictions)
    expected = (batch.batch_size, batch.time_steps, SEMANTIC_DIM)
    if predictions.ndim != 3 or tuple(predictions.shape) != expected:
        raise ValueError("predictions must have shape [batch,time,4].")
    if not predictions.is_floating_point():
        raise ValueError("predictions must be a floating-point tensor.")
    if not torch.isfinite(predictions).all().item():
        raise ValueError("predictions must contain only finite values.")


def validate_pretraining_identity(identity: Mapping[str, object]) -> None:
    """Validate exact compatibility identity without filling omitted fields."""

    if not isinstance(identity, Mapping):
        raise ValueError("Pretraining identity must be a mapping.")
    versioned = "semantic_contract_version" in identity
    fields = REQUIRED_PRETRAINING_IDENTITY_FIELDS + (
        ("semantic_contract_version",) if versioned else ()
    )
    if tuple(identity.keys()) != fields:
        raise ValueError("Pretraining identity fields and order must be exact.")
    expected = {
        "observation_schema": DIRECT_GOAL_OBSERVATION_V2,
        "observation_dim": OBSERVATION_DIM,
        "ontology_schema": SEMANTIC_ONTOLOGY_V2 if versioned else SEMANTIC_ONTOLOGY_V1,
        "semantic_dimensions": SEMANTIC_DIMENSIONS,
        "actor_architecture": ACTOR_ARCHITECTURE_V1,
        "hidden_dim": HIDDEN_DIM,
    }
    if versioned:
        expected["semantic_contract_version"] = SEMANTIC_CONTRACT_V2
    for name, value in expected.items():
        supplied = identity[name]
        if name == "semantic_dimensions":
            supplied = (
                tuple(supplied)
                if isinstance(supplied, (list, tuple))
                else supplied
            )
        if supplied != value:
            raise ValueError(f"Incompatible pretraining identity field: {name}.")
    for name in ("dataset_sha256", "config_sha256"):
        value = identity[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} must be a lowercase SHA-256 identity.")


def _require_tensor(name: str, value: object) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor.")


def _validate_mask(name: str, value: Tensor, batch_time: tuple[int, int]) -> None:
    if value.dtype is not torch.bool:
        raise ValueError(f"{name} must be a boolean tensor.")
    if value.ndim != 2 or tuple(value.shape) != batch_time:
        raise ValueError(f"{name} must match observation batch/time axes.")


def _validated_target_masks(
    supplied: Tensor | None, valid_masks: Tensor, batch_time: tuple[int, int]
) -> Tensor:
    target_masks = valid_masks if supplied is None else supplied
    _require_tensor("target_masks", target_masks)
    _validate_mask("target_masks", target_masks, batch_time)
    if torch.any(target_masks & ~valid_masks).item():
        raise ValueError("target_masks cannot mark padded frames.")
    if not target_masks[:, -1].all().item():
        raise ValueError("Every sequence must supervise its final frame.")
    return target_masks


def _validate_targets(value: Tensor, batch_time: tuple[int, int]) -> None:
    if value.ndim != 3:
        raise ValueError("targets must have rank 3 [batch,time,4].")
    if value.shape[-1] != SEMANTIC_DIM:
        raise ValueError("targets must contain exactly four semantic scores.")
    if tuple(value.shape[:2]) != batch_time:
        raise ValueError("targets must match observation batch/time axes.")


def _validate_initial_hidden(value: Tensor, batch_size: int) -> None:
    if value.ndim != 2:
        raise ValueError("initial_hidden must have rank 2 [batch,64].")
    if value.shape[-1] != HIDDEN_DIM:
        raise ValueError("initial_hidden must use the 64D recurrent state.")
    if value.shape[0] != batch_size:
        raise ValueError("initial_hidden must match the observation batch axis.")


def _validate_finite_floating(name: str, value: Tensor) -> None:
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor.")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values.")


def _validate_contiguous_valid_suffix(valid_masks: Tensor) -> None:
    for row in valid_masks:
        first_valid = int(torch.nonzero(row, as_tuple=False)[0].item())
        if not row[first_valid:].all().item():
            raise ValueError("valid_masks must describe one contiguous valid suffix.")
