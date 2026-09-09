"""A*KD loss primitives for the recurrent MAPPO actor update."""

from __future__ import annotations

import torch
from torch.nn import functional

from llm_mappo.mappo_astar_kd.networks import AStarKDActor
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer


def astar_kd_lambda(global_environment_steps: int, total_steps: int) -> float:
    """Linearly decay 0.05 to 0.01 in 70% of cumulative env transitions."""

    if total_steps < 1:
        raise ValueError("A*KD schedule requires a positive total step budget.")
    ratio = max(0.0, min(float(global_environment_steps) / total_steps, 1.0))
    if ratio >= 0.70:
        return 0.01
    return 0.05 + (0.01 - 0.05) * ratio / 0.70


def astar_kd_loss(
    motion_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    validity: torch.Tensor,
    attenuation: torch.Tensor,
    *,
    lambda_value: float,
) -> torch.Tensor:
    """KL(q||p), normalized by valid count rather than attenuation sum."""

    if motion_logits.shape != soft_targets.shape or motion_logits.shape[-1] != 3:
        raise ValueError("A*KD logits and targets must share a final width of three.")
    if validity.shape != motion_logits.shape[:-1]:
        raise ValueError("A*KD validity must match the non-action axes.")
    if attenuation.shape != validity.shape:
        raise ValueError("A*KD attenuation must match validity.")
    valid = validity.to(dtype=motion_logits.dtype)
    if not bool(valid.any()):
        return motion_logits.sum() * 0.0
    log_probabilities = functional.log_softmax(motion_logits, dim=-1)
    safe_targets = soft_targets.clamp_min(1e-12)
    kl = (soft_targets * (safe_targets.log() - log_probabilities)).sum(dim=-1)
    weighted = kl * valid * attenuation
    return float(lambda_value) * weighted.sum() / valid.sum().clamp_min(1.0)


class AStarKDTrainer(MAPPOTrainer):
    """MAPPO Actor update with a training-only MotionPriorHead auxiliary loss."""

    def __init__(
        self,
        config: MAPPOConfig,
        device: str | torch.device,
        *,
        total_environment_steps: int,
    ) -> None:
        super().__init__(config, device)
        if total_environment_steps < 1:
            raise ValueError("A*KD trainer needs a positive environment-step budget.")
        self.total_environment_steps = int(total_environment_steps)
        self.global_environment_steps = 0
        self.actor = AStarKDActor(
            config.observation_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, eps=1e-5
        )

    def set_global_environment_steps(self, steps: int) -> None:
        self.global_environment_steps = int(steps)

    def _actor_sequence(self, observations, hidden, continuation):
        logits, motion_logits, _ = self.actor.forward_sequence_with_motion(
            observations, hidden, continuation
        )
        return logits, motion_logits

    def _actor_auxiliary_loss(self, auxiliary, selected):
        if auxiliary is None:
            raise RuntimeError("A*KD Actor did not return MotionPriorHead logits.")
        required = ("astar_soft_targets", "astar_validity", "astar_attenuation")
        if any(name not in selected for name in required):
            raise ValueError("A*KD trainer received a rollout without teacher fields.")
        return astar_kd_loss(
            auxiliary,
            selected["astar_soft_targets"],
            selected["astar_validity"],
            selected["astar_attenuation"],
            lambda_value=astar_kd_lambda(
                self.global_environment_steps, self.total_environment_steps
            ),
        )
