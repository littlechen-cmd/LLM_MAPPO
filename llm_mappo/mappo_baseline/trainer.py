"""Canonical recurrent MAPPO update for the teacher-free baseline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional

from llm_mappo.mappo_baseline.networks import MAPPOActor, MAPPOFPCritic
from llm_mappo.mappo_baseline.storage import MAPPOBatch
from llm_mappo.mappo_baseline.value_norm import ValueNorm


@dataclass(frozen=True)
class MAPPOConfig:
    observation_dim: int
    state_dim: int
    action_dim: int = 5
    hidden_dim: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    clip_param: float = 0.2
    ppo_epochs: int = 5
    num_minibatches: int = 1
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    huber_delta: float = 10.0
    use_clipped_value_loss: bool = True


class MAPPOTrainer:
    """Shared Actor/Critic trained from intact worker/AGV time sequences."""

    def __init__(self, config: MAPPOConfig, device: str | torch.device) -> None:
        self.config = config
        self.device = torch.device(device)
        self.actor = MAPPOActor(
            config.observation_dim, config.action_dim, config.hidden_dim
        ).to(self.device)
        self.critic = MAPPOFPCritic(config.state_dim, config.hidden_dim).to(self.device)
        self.value_normalizer = ValueNorm((1,)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr, eps=1e-5
        )

    @torch.no_grad()
    def act(
        self,
        observations,
        fp_states,
        action_masks,
        actor_hidden_states,
        critic_hidden_states,
        *,
        deterministic: bool,
    ):
        observations = self._tensor(observations)
        fp_states = self._tensor(fp_states)
        masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        actor_hidden = self._tensor(actor_hidden_states)
        critic_hidden = self._tensor(critic_hidden_states)
        logits, next_actor_hidden = self.actor(observations, actor_hidden)
        logits = logits.masked_fill(~masks, -1e10)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)
        normalized_values, next_critic_hidden = self.critic(fp_states, critic_hidden)
        values = self.value_normalizer.denormalize(
            normalized_values.unsqueeze(-1)
        ).squeeze(-1)
        return tuple(
            value.detach().cpu().numpy()
            for value in (
                actions,
                log_probs,
                values,
                normalized_values,
                next_actor_hidden,
                next_critic_hidden,
            )
        )

    @torch.no_grad()
    def act_policy(
        self,
        observations,
        action_masks,
        actor_hidden_states,
        *,
        deterministic: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select actions without evaluating the centralized critic."""

        observations = self._tensor(observations)
        masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        actor_hidden = self._tensor(actor_hidden_states)
        logits, next_actor_hidden = self.actor(observations, actor_hidden)
        logits = logits.masked_fill(~masks, -1e10)
        actions = (
            logits.argmax(dim=-1)
            if deterministic
            else Categorical(logits=logits).sample()
        )
        return actions.cpu().numpy(), next_actor_hidden.cpu().numpy()

    @torch.no_grad()
    def values(self, fp_states, critic_hidden_states) -> tuple[np.ndarray, np.ndarray]:
        normalized, next_hidden = self.critic(
            self._tensor(fp_states), self._tensor(critic_hidden_states)
        )
        values = self.value_normalizer.denormalize(normalized.unsqueeze(-1)).squeeze(-1)
        return values.cpu().numpy(), next_hidden.cpu().numpy()

    def update(self, batch: MAPPOBatch) -> dict[str, float | int]:  # noqa: C901
        observations = self._tensor(batch.observations)
        fp_states = self._tensor(batch.fp_states)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        action_masks = torch.as_tensor(
            batch.action_masks, dtype=torch.bool, device=self.device
        )
        old_log_probs = self._tensor(batch.old_log_probs)
        old_values = self._tensor(batch.old_values)
        returns = self._tensor(batch.returns)
        advantages = self._tensor(batch.advantages)
        active_masks = self._tensor(batch.active_masks)
        dones = torch.as_tensor(batch.dones, dtype=torch.bool, device=self.device)
        actor_hidden = self._tensor(batch.actor_hidden_states)
        critic_hidden = self._tensor(batch.critic_hidden_states)
        if observations.ndim != 4:
            raise ValueError("Recurrent MAPPO batch needs time, worker and AGV axes.")
        active = active_masks > 0.0
        if not bool(active.any()):
            raise RuntimeError("MAPPO update has no active robot samples.")
        valid_advantages = advantages[active]
        valid_returns = returns[active]
        raw_diagnostics = {
            **_distribution_metrics("raw_advantage", valid_advantages),
            **_distribution_metrics("raw_return", valid_returns),
        }
        advantages = (advantages - valid_advantages.mean()) / (
            valid_advantages.std(unbiased=False) + 1e-5
        )
        self.value_normalizer.update(returns[active].unsqueeze(-1))
        normalized_returns = self.value_normalizer.normalize(
            returns.unsqueeze(-1)
        ).squeeze(-1)

        time_steps, workers, agents = observations.shape[:3]
        sequence_count = workers * agents
        continuation = torch.ones_like(active_masks)
        continuation[0] = active_masks[0]
        if time_steps > 1:
            continuation[1:] = active_masks[1:] * (~dones[:-1]).float()
        tensors = {
            "observations": observations.reshape(time_steps, sequence_count, -1),
            "fp_states": fp_states.reshape(time_steps, sequence_count, -1),
            "actions": actions.reshape(time_steps, sequence_count),
            "action_masks": action_masks.reshape(
                time_steps, sequence_count, self.config.action_dim
            ),
            "old_log_probs": old_log_probs.reshape(time_steps, sequence_count),
            "old_values": old_values.reshape(time_steps, sequence_count),
            "advantages": advantages.reshape(time_steps, sequence_count),
            "returns": normalized_returns.reshape(time_steps, sequence_count),
            "active": active.reshape(time_steps, sequence_count).float(),
            "continuation": continuation.reshape(time_steps, sequence_count),
            "actor_hidden": actor_hidden[0].reshape(sequence_count, -1),
            "critic_hidden": critic_hidden[0].reshape(sequence_count, -1),
        }
        if batch.astar_soft_targets is not None:
            if (
                batch.astar_root_costs is None
                or batch.astar_validity is None
                or batch.astar_supervision_scope is None
                or batch.astar_attenuation is None
            ):
                raise ValueError("A*KD rollout fields must be present together.")
            tensors.update({
                "astar_soft_targets": self._tensor(batch.astar_soft_targets).reshape(
                    time_steps, sequence_count, 3
                ),
                "astar_root_costs": self._tensor(batch.astar_root_costs).reshape(
                    time_steps, sequence_count, 3
                ),
                "astar_validity": torch.as_tensor(
                    batch.astar_validity, dtype=torch.bool, device=self.device
                ).reshape(time_steps, sequence_count),
                "astar_supervision_scope": torch.as_tensor(
                    batch.astar_supervision_scope,
                    dtype=torch.bool,
                    device=self.device,
                ).reshape(time_steps, sequence_count),
                "astar_attenuation": self._tensor(batch.astar_attenuation).reshape(
                    time_steps, sequence_count
                ),
            })
        chunks = self._sequence_chunks(sequence_count)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "total_loss": 0.0,
            "astar_loss": 0.0,
        }
        actor_diagnostics = {}
        optimizer_steps = 0
        for _ in range(self.config.ppo_epochs):
            for indices in chunks:
                selected = {
                    name: values[:, indices]
                    for name, values in tensors.items()
                    if name not in {"actor_hidden", "critic_hidden"}
                }
                selected["actor_hidden"] = tensors["actor_hidden"][indices]
                selected["critic_hidden"] = tensors["critic_hidden"][indices]
                active_weight = selected["active"]
                if not bool(active_weight.any()):
                    continue
                logits, auxiliary = self._actor_sequence(
                    selected["observations"],
                    selected["actor_hidden"],
                    selected["continuation"],
                )
                distribution = Categorical(
                    logits=logits.masked_fill(~selected["action_masks"], -1e10)
                )
                new_log_probs = distribution.log_prob(selected["actions"])
                log_ratio = new_log_probs - selected["old_log_probs"]
                ratio = log_ratio.exp()
                surrogate = ratio * selected["advantages"]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_param,
                    1.0 + self.config.clip_param,
                ) * selected["advantages"]
                policy_loss = -self._masked_mean(
                    torch.minimum(surrogate, clipped), active_weight
                )
                entropy = self._masked_mean(distribution.entropy(), active_weight)
                auxiliary_loss = self._actor_auxiliary_loss(auxiliary, selected)
                actor_loss = (
                    policy_loss - self.config.entropy_coef * entropy + auxiliary_loss
                )
                if not actor_diagnostics:
                    actor_diagnostics = self._actor_diagnostics(
                        logits,
                        auxiliary,
                        selected,
                        policy_loss - self.config.entropy_coef * entropy,
                        auxiliary_loss,
                    )
                predicted, _ = self.critic.forward_sequence(
                    selected["fp_states"],
                    selected["critic_hidden"],
                    selected["continuation"],
                )
                value_loss = self._value_loss(
                    predicted,
                    selected["old_values"],
                    selected["returns"],
                    active_weight,
                )
                if not torch.isfinite(actor_loss + value_loss):
                    raise RuntimeError("MAPPO update produced a non-finite loss.")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()
                _require_finite_parameters(self.actor, "actor")
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_loss_coef * value_loss).backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.critic_optimizer.step()
                _require_finite_parameters(self.critic, "critic")
                with torch.no_grad():
                    approx_kl = self._masked_mean(
                        (ratio - 1.0) - log_ratio, active_weight
                    )
                    clip_fraction = self._masked_mean(
                        ((ratio - 1.0).abs() > self.config.clip_param).float(),
                        active_weight,
                    )
                measured = (
                    policy_loss,
                    value_loss,
                    entropy,
                    approx_kl,
                    clip_fraction,
                    actor_grad_norm,
                    critic_grad_norm,
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy,
                    auxiliary_loss,
                )
                for name, value in zip(totals, measured):
                    totals[name] += float(value.detach().cpu())
                optimizer_steps += 1
        if not optimizer_steps:
            raise RuntimeError("MAPPO update did not execute an optimizer step.")
        with torch.no_grad():
            predictions, _ = self.critic.forward_sequence(
                tensors["fp_states"],
                tensors["critic_hidden"],
                tensors["continuation"],
            )
            denormalized = self.value_normalizer.denormalize(
                predictions.unsqueeze(-1)
            ).squeeze(-1).reshape_as(returns)
            variance = torch.var(returns[active], unbiased=False)
            explained_variance = 0.0 if float(variance) < 1e-8 else float(
                1.0
                - torch.var(returns[active] - denormalized[active], unbiased=False)
                / variance
            )
        return {
            **{name: value / optimizer_steps for name, value in totals.items()},
            **actor_diagnostics,
            "explained_variance": explained_variance,
            "optimizer_steps": optimizer_steps,
            "actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
            "critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"],
            **raw_diagnostics,
        }

    def _actor_sequence(self, observations, hidden, continuation):
        logits, _ = self.actor.forward_sequence(observations, hidden, continuation)
        return logits, None

    def _actor_auxiliary_loss(self, auxiliary, selected) -> Tensor:
        del auxiliary, selected
        return torch.zeros((), device=self.device)

    def _actor_diagnostics(
        self,
        logits,
        auxiliary,
        selected,
        ppo_actor_loss,
        auxiliary_loss,
    ) -> dict[str, float | int]:
        del logits, auxiliary, selected, ppo_actor_loss, auxiliary_loss
        return {}

    def _sequence_chunks(self, sequence_count: int) -> tuple[Tensor, ...]:
        chunk_count = min(max(int(self.config.num_minibatches), 1), sequence_count)
        order = torch.randperm(sequence_count, device=self.device)
        return tuple(
            chunk for chunk in torch.tensor_split(order, chunk_count) if chunk.numel()
        )

    def _value_loss(
        self,
        predicted: Tensor,
        old_values: Tensor,
        targets: Tensor,
        active_weight: Tensor,
    ) -> Tensor:
        original = functional.huber_loss(
            predicted, targets, delta=self.config.huber_delta, reduction="none"
        )
        if not self.config.use_clipped_value_loss:
            return self._masked_mean(original, active_weight)
        clipped_prediction = old_values + (predicted - old_values).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        clipped = functional.huber_loss(
            clipped_prediction,
            targets,
            delta=self.config.huber_delta,
            reduction="none",
        )
        return self._masked_mean(torch.maximum(original, clipped), active_weight)

    @staticmethod
    def _masked_mean(values: Tensor, weights: Tensor) -> Tensor:
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    def linear_lr_decay(
        self,
        completed_steps: int,
        total_steps: int,
        post_schedule_lr: float = 0.0,
    ) -> None:
        if completed_steps <= total_steps:
            fraction = max(0.0, 1.0 - completed_steps / max(total_steps, 1))
            actor_lr = self.config.actor_lr * fraction
            critic_lr = self.config.critic_lr * fraction
        else:
            actor_lr = post_schedule_lr
            critic_lr = post_schedule_lr
        self.actor_optimizer.param_groups[0]["lr"] = actor_lr
        self.critic_optimizer.param_groups[0]["lr"] = critic_lr

    def state_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Commit checkpoint state only after an isolated full validation load."""

        candidate = deepcopy(self)
        candidate._load_state_dict_unchecked(state)
        self._load_state_dict_unchecked(state)

    def _load_state_dict_unchecked(self, state: dict) -> None:
        if state.get("config") != asdict(self.config):
            raise ValueError("MAPPO checkpoint trainer configuration differs.")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.value_normalizer.load_state_dict(state["value_normalizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def _tensor(self, values) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)


def _distribution_metrics(prefix: str, values: Tensor) -> dict[str, float | int]:
    """Return a finite, pre-normalization distribution summary for one update."""

    detached = values.detach().cpu().numpy().astype(np.float64, copy=False)
    if detached.size == 0 or not np.all(np.isfinite(detached)):
        raise RuntimeError(f"MAPPO {prefix} contains non-finite values.")
    return {
        f"{prefix}_count": int(detached.size),
        f"{prefix}_mean": float(np.mean(detached)),
        f"{prefix}_std": float(np.std(detached)),
        f"{prefix}_min": float(np.min(detached)),
        f"{prefix}_max": float(np.max(detached)),
        f"{prefix}_p05": float(np.quantile(detached, 0.05)),
        f"{prefix}_p50": float(np.quantile(detached, 0.50)),
        f"{prefix}_p95": float(np.quantile(detached, 0.95)),
        f"{prefix}_max_abs": float(np.max(np.abs(detached))),
    }


def _require_finite_parameters(module: torch.nn.Module, name: str) -> None:
    """Fail the update before a non-finite parameter reaches the next rollout."""

    if not all(torch.isfinite(parameter).all() for parameter in module.parameters()):
        raise RuntimeError(f"MAPPO {name} parameters became non-finite.")
