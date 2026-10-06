"""A*KD loss primitives for the recurrent MAPPO actor update."""

from __future__ import annotations

import torch
from torch.nn import functional

from llm_mappo.mappo_astar_kd.networks import AStarKDActor
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer


def astar_kd_diagnostics(
    policy_logits: torch.Tensor,
    motion_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    root_costs: torch.Tensor,
    validity: torch.Tensor,
    supervision_scope: torch.Tensor,
    active_masks: torch.Tensor,
) -> dict[str, float | int]:
    """Measure teacher transfer without changing policy or teacher semantics."""

    if policy_logits.shape[:-1] != motion_logits.shape[:-1]:
        raise ValueError("Policy and motion diagnostics must share sample axes.")
    if policy_logits.shape[-1] != 5 or motion_logits.shape[-1] != 3:
        raise ValueError(
            "A*KD diagnostics require five policy and three motion logits."
        )
    if (
        soft_targets.shape != motion_logits.shape
        or root_costs.shape != motion_logits.shape
    ):
        raise ValueError("A*KD diagnostic targets and root costs must be three-wide.")
    if (
        validity.shape != motion_logits.shape[:-1]
        or supervision_scope.shape != validity.shape
        or active_masks.shape != validity.shape
    ):
        raise ValueError("A*KD diagnostic masks must match the sample axes.")
    teacher_valid = validity.bool() & supervision_scope.bool()
    policy_eligible = teacher_valid & (active_masks > 0.0)
    teacher_top = soft_targets.argmax(dim=-1)
    motion_top = motion_logits.argmax(dim=-1)
    if bool(teacher_valid.any()):
        motion_agreement = (
            (motion_top == teacher_top) & teacher_valid
        ).sum() / teacher_valid.sum()
        motion_agreement_value = float(motion_agreement.detach().cpu())
    else:
        motion_agreement_value = 0.0
    if not bool(policy_eligible.any()):
        return {
            "motion_prior_top1_agreement": motion_agreement_value,
            "policy_teacher_motion_agreement": 0.0,
            "static_route_regret": 0.0,
            "static_route_regret_count": 0,
            "static_invalid_top1_rate": 0.0,
        }

    policy_top = policy_logits[..., 1:4].argmax(dim=-1)
    policy_agreement = (
        (policy_top == teacher_top) & policy_eligible
    ).sum() / policy_eligible.sum()

    finite = torch.isfinite(root_costs)
    finite_costs = root_costs.masked_fill(~finite, float("inf"))
    minimum = finite_costs.min(dim=-1).values
    selected_cost = finite_costs.gather(-1, policy_top.unsqueeze(-1)).squeeze(-1)
    selected_finite = policy_eligible & torch.isfinite(selected_cost)
    regret_count = int(selected_finite.sum().detach().cpu())
    if regret_count:
        regret = (selected_cost[selected_finite] - minimum[selected_finite]).mean()
        regret_value = float(regret.detach().cpu())
    else:
        regret_value = 0.0
    invalid_rate = (
        policy_eligible & ~torch.isfinite(selected_cost)
    ).sum() / policy_eligible.sum()
    return {
        "motion_prior_top1_agreement": motion_agreement_value,
        "policy_teacher_motion_agreement": float(policy_agreement.detach().cpu()),
        "static_route_regret": regret_value,
        "static_route_regret_count": regret_count,
        "static_invalid_top1_rate": float(invalid_rate.detach().cpu()),
    }


def _gradient_norm(loss: torch.Tensor, parameters) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=True,
        allow_unused=True,
    )
    terms = [gradient.square().sum() for gradient in gradients if gradient is not None]
    if not terms:
        return loss.new_zeros(())
    return torch.stack(terms).sum().sqrt()


def astar_kd_lambda(
    global_environment_steps: int,
    total_steps: int,
    *,
    post_schedule_lambda: float = 0.01,
    stop_steps: int | None = None,
) -> float:
    """Linearly decay 0.05 to 0.01 in 70% of cumulative env transitions.

    Steps at or beyond the schedule horizon use ``post_schedule_lambda`` so a
    decoupled tail can hold or disable A*KD without stretching the curriculum.
    Optional ``stop_steps`` independently turns guidance off at a hard cutoff.
    """

    if total_steps < 1:
        raise ValueError("A*KD schedule requires a positive total step budget.")
    if stop_steps is not None:
        if (not isinstance(stop_steps, int) or isinstance(stop_steps, bool)
                or stop_steps < 1):
            raise ValueError("A*KD stop steps must be a positive integer.")
        if global_environment_steps >= stop_steps:
            return 0.0
    if global_environment_steps >= total_steps:
        return float(post_schedule_lambda)
    ratio = max(0.0, min(float(global_environment_steps) / total_steps, 1.0))
    if ratio >= 0.70:
        return 0.01
    return 0.05 + (0.01 - 0.05) * ratio / 0.70


def astar_kd_loss(
    motion_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    validity: torch.Tensor,
    attenuation: torch.Tensor,
    supervision_scope: torch.Tensor,
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
    if supervision_scope.shape != validity.shape:
        raise ValueError("A*KD supervision scope must match validity.")
    valid = (validity.bool() & supervision_scope.bool()).to(
        dtype=motion_logits.dtype
    )
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
        post_schedule_astar_lambda: float = 0.01,
        astar_stop_steps: int | None = None,
    ) -> None:
        super().__init__(config, device)
        if total_environment_steps < 1:
            raise ValueError("A*KD trainer needs a positive environment-step budget.")
        if post_schedule_astar_lambda < 0.0:
            raise ValueError("A*KD post-schedule lambda must be non-negative.")
        if astar_stop_steps is not None and (
            not isinstance(astar_stop_steps, int)
            or isinstance(astar_stop_steps, bool)
            or astar_stop_steps < 1
        ):
            raise ValueError("A*KD stop steps must be a positive integer.")
        self.astar_stop_steps = astar_stop_steps
        self.total_environment_steps = int(total_environment_steps)
        self.post_schedule_astar_lambda = float(post_schedule_astar_lambda)
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
        required = (
            "astar_soft_targets",
            "astar_validity",
            "astar_supervision_scope",
            "astar_attenuation",
        )
        if any(name not in selected for name in required):
            raise ValueError("A*KD trainer received a rollout without teacher fields.")
        return astar_kd_loss(
            auxiliary,
            selected["astar_soft_targets"],
            selected["astar_validity"],
            selected["astar_attenuation"],
            selected["astar_supervision_scope"],
            lambda_value=astar_kd_lambda(
                self.global_environment_steps,
                self.total_environment_steps,
                post_schedule_lambda=self.post_schedule_astar_lambda,
                stop_steps=self.astar_stop_steps,
            ),
        )

    def _actor_diagnostics(
        self,
        logits,
        auxiliary,
        selected,
        ppo_actor_loss,
        auxiliary_loss,
    ):
        metrics = astar_kd_diagnostics(
            logits,
            auxiliary,
            selected["astar_soft_targets"],
            selected["astar_root_costs"],
            selected["astar_validity"],
            selected["astar_supervision_scope"],
            selected["active"],
        )
        shared_parameters = tuple(self.actor.encoder.parameters()) + tuple(
            self.actor.gru.parameters()
        )
        ppo_norm = _gradient_norm(ppo_actor_loss, shared_parameters)
        astar_norm = _gradient_norm(auxiliary_loss, shared_parameters)
        metrics.update({
            "ppo_shared_grad_norm": float(ppo_norm.detach().cpu()),
            "astar_shared_grad_norm": float(astar_norm.detach().cpu()),
            "astar_to_ppo_shared_grad_ratio": float(
                (astar_norm / ppo_norm.clamp_min(1e-12)).detach().cpu()
            ),
        })
        return metrics
