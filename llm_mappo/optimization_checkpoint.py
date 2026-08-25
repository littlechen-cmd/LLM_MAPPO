"""Strict, isolated checkpoint contract for the O0 Student."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


CHECKPOINT_SCHEMA = "o0-student-checkpoint-v1"


@dataclass(frozen=True)
class O0CheckpointV1:
    """Schema helpers for a resumable O0 optimization checkpoint."""

    schema: str = CHECKPOINT_SCHEMA

    @staticmethod
    def default_metadata(
        *,
        global_env_steps: int,
        update_count: int,
        completed_episodes: int,
        ema_state: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> dict:
        return {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "architecture_version": "o0-student-v1",
            "observation_schema": "direct-goal-observation-v1",
            "semantic_schema": "semantic-view-v3",
            "schedule_version": "linear-env-step-v1",
            "sampler_version": "calibration-sampler-v1",
            "pure_motion_teacher_version": "pure-motion-astar-v1",
            "k_motion": 12,
            "h_reward": 12,
            "expansion_budget": 512,
            "calibration_modulus": 16,
            "ema_decay": 0.99,
            "ema_minimum_scale": 1e-3,
            "ema_initialization_samples": 64,
            "physical_observation_dim": 613,
            "semantic_observation_dim": 61,
            "semantic_score_names": [
                "task_persistence",
                "yielding_preference",
                "coordination_risk",
            ],
            "action_names": ["NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE_LOAD"],
            "global_env_steps": int(global_env_steps),
            "update_count": int(update_count),
            "completed_episodes": int(completed_episodes),
            "ema_state": dict(ema_state),
            "provenance": dict(provenance),
            "rng_state": {
                "python": None,
                "numpy": None,
                "torch_cpu": None,
                "torch_cuda": None,
            },
        }


_REQUIRED_METADATA = {
    "checkpoint_schema",
    "architecture_version",
    "observation_schema",
    "semantic_schema",
    "schedule_version",
    "sampler_version",
    "pure_motion_teacher_version",
    "k_motion",
    "h_reward",
    "expansion_budget",
    "calibration_modulus",
    "ema_decay",
    "ema_minimum_scale",
    "ema_initialization_samples",
    "physical_observation_dim",
    "semantic_observation_dim",
    "semantic_score_names",
    "action_names",
    "global_env_steps",
    "update_count",
    "completed_episodes",
    "ema_state",
    "provenance",
    "rng_state",
}


def save_o0_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Mapping[str, Any],
) -> None:
    """Save a complete O0 checkpoint only after strict metadata validation."""

    validated = _validate_metadata(metadata)
    torch.save(
        {
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "metadata": validated,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        Path(path),
    )


def load_o0_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Reject incompatible checkpoints before strict model/optimizer loading."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA
    ):
        raise ValueError("checkpoint_schema must be o0-student-checkpoint-v1.")
    metadata = _validate_metadata(payload.get("metadata", {}))
    if "model_state" not in payload:
        raise ValueError("Checkpoint is missing model_state.")
    if optimizer is not None and "optimizer_state" not in payload:
        raise ValueError("Checkpoint is missing optimizer_state.")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return {"metadata": metadata, "non_resumable": optimizer is None}


def _validate_metadata(metadata: Mapping[str, Any]) -> dict:  # noqa: C901
    if not isinstance(metadata, Mapping):
        raise ValueError("Checkpoint metadata must be a mapping.")
    missing = sorted(_REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError(f"Checkpoint metadata is missing {missing[0]}.")
    validated = dict(metadata)
    if validated["checkpoint_schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint_schema is incompatible.")
    expected = {
        "architecture_version": "o0-student-v1",
        "observation_schema": "direct-goal-observation-v1",
        "semantic_schema": "semantic-view-v3",
        "schedule_version": "linear-env-step-v1",
        "sampler_version": "calibration-sampler-v1",
        "pure_motion_teacher_version": "pure-motion-astar-v1",
        "k_motion": 12,
        "h_reward": 12,
        "expansion_budget": 512,
        "calibration_modulus": 16,
        "ema_decay": 0.99,
        "ema_minimum_scale": 1e-3,
        "ema_initialization_samples": 64,
        "physical_observation_dim": 613,
        "semantic_observation_dim": 61,
    }
    for key, value in expected.items():
        if validated[key] != value:
            raise ValueError(f"Checkpoint {key} is incompatible.")
    if validated["semantic_score_names"] != [
        "task_persistence",
        "yielding_preference",
        "coordination_risk",
    ]:
        raise ValueError("Checkpoint semantic_score_names are incompatible.")
    if validated["action_names"] != [
        "NOOP",
        "FORWARD",
        "LEFT",
        "RIGHT",
        "TOGGLE_LOAD",
    ]:
        raise ValueError("Checkpoint action_names are incompatible.")
    if not isinstance(validated["ema_state"], Mapping):
        raise ValueError("Checkpoint ema_state is incompatible.")
    if not isinstance(validated["provenance"], Mapping):
        raise ValueError("Checkpoint provenance is incompatible.")
    if not isinstance(validated["rng_state"], Mapping):
        raise ValueError("Checkpoint rng_state is incompatible.")
    return validated
