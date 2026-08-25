"""Optimization-route rollout tensors, auxiliary losses and real-step schedule."""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as functional

from llm_mappo.optimization_student import (
    ACTION_DIM,
    MOTION_ACTION_DIM,
    PHYSICAL_OBSERVATION_DIM,
    SEMANTIC_OBSERVATION_DIM,
    SEMANTIC_SCORE_DIM,
)


@dataclass(frozen=True)
class OptimizationBatch:
    physical_observations: Tensor
    semantic_observations: Tensor
    actions: Tensor
    log_probs: Tensor
    action_masks: Tensor
    astar_preferences: Tensor
    astar_valid: Tensor
    calibration_selected: Tensor
    reward_confidence: Tensor
    semantic_targets: Tensor
    semantic_validity: Tensor
    ood_reliability: Tensor

    def astar_kl_loss(self, motion_logits: Tensor, lambda_a: float) -> Tensor:
        if motion_logits.shape != self.astar_preferences.shape:
            raise ValueError("Motion logits must match A* preference shape.")
        active = self.astar_valid & self.calibration_selected.unsqueeze(-1)
        denominator = active.sum()
        if not bool(denominator):
            return motion_logits.sum() * 0.0
        divergence = functional.kl_div(
            torch.log_softmax(motion_logits, dim=-1),
            self.astar_preferences,
            reduction="none",
        ).sum(dim=-1)
        weights = active.to(motion_logits.dtype) * self.reward_confidence.unsqueeze(-1)
        return float(lambda_a) * (weights * divergence).sum() / denominator

    def semantic_mse_loss(self, semantic_scores: Tensor, lambda_l: float) -> Tensor:
        if semantic_scores.shape != self.semantic_targets.shape:
            raise ValueError("Semantic scores must match semantic target shape.")
        denominator = self.semantic_validity.sum()
        if not bool(denominator):
            return semantic_scores.sum() * 0.0
        per_record = (semantic_scores - self.semantic_targets).square().mean(dim=-1)
        weights = self.semantic_validity * self.ood_reliability.unsqueeze(-1)
        return float(lambda_l) * (weights * per_record).sum() / denominator


class OptimizationRolloutBuffer:
    """Store only O0 tensors and preserve independent per-agent validity masks."""

    def __init__(self, n_agents: int) -> None:
        if n_agents < 1:
            raise ValueError("Optimization rollout requires at least one agent.")
        self.n_agents = n_agents
        self._records: List[dict] = []

    def add(
        self,
        *,
        physical_observations: np.ndarray,
        semantic_observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        action_masks: np.ndarray,
        astar_preferences: np.ndarray,
        astar_valid: np.ndarray,
        calibration_selected: bool,
        reward_confidence: float,
        semantic_targets: np.ndarray,
        semantic_validity: np.ndarray,
        ood_reliability: float,
    ) -> None:
        record = {
            "physical_observations": self._array(
                physical_observations, (self.n_agents, PHYSICAL_OBSERVATION_DIM)
            ),
            "semantic_observations": self._array(
                semantic_observations, (self.n_agents, SEMANTIC_OBSERVATION_DIM)
            ),
            "actions": self._array(actions, (self.n_agents,), np.int64),
            "log_probs": self._array(log_probs, (self.n_agents,)),
            "action_masks": self._array(
                action_masks, (self.n_agents, ACTION_DIM), bool
            ),
            "astar_preferences": self._array(
                astar_preferences, (self.n_agents, MOTION_ACTION_DIM)
            ),
            "astar_valid": self._array(astar_valid, (self.n_agents,), bool),
            "semantic_targets": self._array(
                semantic_targets, (self.n_agents, SEMANTIC_SCORE_DIM)
            ),
            "semantic_validity": self._array(
                semantic_validity, (self.n_agents,)
            ),
        }
        if not np.all(record["action_masks"].any(axis=-1)):
            raise ValueError("Every agent must retain one legal action.")
        if not np.isfinite(reward_confidence) or not 0.0 <= reward_confidence <= 1.0:
            raise ValueError("Reward confidence must be finite and in [0, 1].")
        if not np.isfinite(ood_reliability) or not 0.0 <= ood_reliability <= 1.0:
            raise ValueError("OOD reliability must be finite and in [0, 1].")
        if np.any(
            (record["semantic_targets"] < 0.0)
            | (record["semantic_targets"] > 1.0)
        ):
            raise ValueError("Semantic targets must be in [0, 1].")
        if np.any(
            (record["semantic_validity"] < 0.0)
            | (record["semantic_validity"] > 1.0)
        ):
            raise ValueError("Semantic validity must be in [0, 1].")
        valid_preferences = record["astar_preferences"][record["astar_valid"]]
        if valid_preferences.size and not np.allclose(
            valid_preferences.sum(axis=-1), 1.0
        ):
            raise ValueError("Each valid A* preference must sum to one.")
        record["calibration_selected"] = bool(calibration_selected)
        record["reward_confidence"] = float(reward_confidence)
        record["ood_reliability"] = float(ood_reliability)
        self._records.append(record)

    def tensors(self, device: str | torch.device) -> OptimizationBatch:
        if not self._records:
            raise ValueError("Cannot tensorize an empty optimization rollout.")

        def stack(name: str, dtype):
            return torch.as_tensor(
                np.stack([record[name] for record in self._records]),
                dtype=dtype,
                device=device,
            )

        return OptimizationBatch(
            physical_observations=stack("physical_observations", torch.float32),
            semantic_observations=stack("semantic_observations", torch.float32),
            actions=stack("actions", torch.long),
            log_probs=stack("log_probs", torch.float32),
            action_masks=stack("action_masks", torch.bool),
            astar_preferences=stack("astar_preferences", torch.float32),
            astar_valid=stack("astar_valid", torch.bool),
            calibration_selected=torch.as_tensor(
                [record["calibration_selected"] for record in self._records],
                dtype=torch.bool,
                device=device,
            ),
            reward_confidence=torch.as_tensor(
                [record["reward_confidence"] for record in self._records],
                dtype=torch.float32,
                device=device,
            ),
            semantic_targets=stack("semantic_targets", torch.float32),
            semantic_validity=stack("semantic_validity", torch.float32),
            ood_reliability=torch.as_tensor(
                [record["ood_reliability"] for record in self._records],
                dtype=torch.float32,
                device=device,
            ),
        )

    def _array(self, values, shape, dtype=np.float32) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if array.shape != shape:
            raise ValueError(f"Expected shape {shape}, got {array.shape}.")
        return array.copy()


@dataclass
class LinearEnvStepSchedule:
    """`linear-env-step-v1`, advanced only by real environment transitions."""

    total_env_steps: int
    global_env_steps: int = 0

    def __post_init__(self) -> None:
        if self.total_env_steps < 1:
            raise ValueError("total_env_steps must be positive.")
        if self.global_env_steps < 0:
            raise ValueError("global_env_steps must not be negative.")

    def advance_real_env_steps(self, count: int) -> None:
        if count < 0:
            raise ValueError("Real environment step count must not be negative.")
        self.global_env_steps += count

    def weights(self) -> tuple[float, float]:
        progress = min(max(self.global_env_steps / self.total_env_steps, 0.0), 1.0)
        return 0.05 * (1.0 - progress), 0.10 * (1.0 - progress)

    def state_dict(self) -> dict:
        lambda_a, lambda_l = self.weights()
        return {
            "schedule_version": "linear-env-step-v1",
            "total_env_steps": self.total_env_steps,
            "global_env_steps": self.global_env_steps,
            "lambda_a": lambda_a,
            "lambda_l": lambda_l,
        }
