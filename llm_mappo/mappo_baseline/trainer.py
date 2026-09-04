"""Canonical feed-forward MAPPO update for the teacher-free baseline."""

from __future__ import annotations

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
    hidden_dim: int = 128
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
    """Shared actor/critic with separate optimizers and per-agent losses."""

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
    def act(self, observations, fp_states, action_masks, *, deterministic: bool):
        observations = self._tensor(observations)
        fp_states = self._tensor(fp_states)
        masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        logits = self.actor(observations).masked_fill(~masks, -1e10)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)
        normalized_values = self.critic(fp_states)
        values = self.value_normalizer.denormalize(
            normalized_values.unsqueeze(-1)
        ).squeeze(-1)
        return tuple(
            value.detach().cpu().numpy()
            for value in (actions, log_probs, values, normalized_values)
        )

    @torch.no_grad()
    def values(self, fp_states) -> np.ndarray:
        normalized = self.critic(self._tensor(fp_states))
        return (
            self.value_normalizer.denormalize(normalized.unsqueeze(-1))
            .squeeze(-1)
            .cpu()
            .numpy()
        )

    def update(self, batch: MAPPOBatch) -> dict[str, float | int]:
        observations = self._tensor(batch.observations).reshape(
            -1, self.config.observation_dim
        )
        fp_states = self._tensor(batch.fp_states).reshape(-1, self.config.state_dim)
        actions = torch.as_tensor(
            batch.actions.reshape(-1), dtype=torch.long, device=self.device
        )
        action_masks = torch.as_tensor(
            batch.action_masks.reshape(-1, self.config.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        old_log_probs = self._tensor(batch.old_log_probs).reshape(-1)
        old_values = self._tensor(batch.old_values).reshape(-1)
        returns = self._tensor(batch.returns).reshape(-1)
        advantages = self._tensor(batch.advantages).reshape(-1)
        active_masks = self._tensor(batch.active_masks).reshape(-1)
        active = active_masks > 0.0
        if not bool(active.any()):
            raise RuntimeError("MAPPO update has no active robot samples.")
        valid_advantages = advantages[active]
        advantages = (advantages - valid_advantages.mean()) / (
            valid_advantages.std(unbiased=False) + 1e-5
        )
        self.value_normalizer.update(returns[active].unsqueeze(-1))
        normalized_returns = self.value_normalizer.normalize(
            returns.unsqueeze(-1)
        ).squeeze(-1)
        normalized_old_values = old_values
        sample_count = observations.shape[0]
        minibatch_size = max(sample_count // self.config.num_minibatches, 1)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "total_loss": 0.0,
        }
        optimizer_steps = 0
        for _ in range(self.config.ppo_epochs):
            order = torch.randperm(sample_count, device=self.device)
            for start in range(0, sample_count, minibatch_size):
                index = order[start:start + minibatch_size]
                index = index[active[index]]
                if index.numel() == 0:
                    continue
                logits = self.actor(observations[index]).masked_fill(
                    ~action_masks[index], -1e10
                )
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[index])
                log_ratio = new_log_probs - old_log_probs[index]
                ratio = log_ratio.exp()
                surrogate = ratio * advantages[index]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_param,
                    1.0 + self.config.clip_param,
                ) * advantages[index]
                policy_loss = -torch.minimum(surrogate, clipped).mean()
                entropy = distribution.entropy().mean()
                actor_loss = policy_loss - self.config.entropy_coef * entropy
                predicted = self.critic(fp_states[index])
                value_loss = self._value_loss(
                    predicted,
                    normalized_old_values[index],
                    normalized_returns[index],
                )
                if not torch.isfinite(actor_loss + value_loss):
                    raise RuntimeError("MAPPO update produced a non-finite loss.")
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.config.value_loss_coef * value_loss).backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                self.critic_optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip_param
                    ).float().mean()
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
                )
                for name, value in zip(totals, measured):
                    totals[name] += float(value.detach().cpu())
                optimizer_steps += 1
        if not optimizer_steps:
            raise RuntimeError("MAPPO update did not execute an optimizer step.")
        with torch.no_grad():
            prediction = self.value_normalizer.denormalize(
                self.critic(fp_states).unsqueeze(-1)
            ).squeeze(-1)
            variance = torch.var(returns[active], unbiased=False)
            explained_variance = 0.0 if float(variance) < 1e-8 else float(
                1.0
                - torch.var(returns[active] - prediction[active], unbiased=False)
                / variance
            )
        return {
            **{name: value / optimizer_steps for name, value in totals.items()},
            "explained_variance": explained_variance,
            "optimizer_steps": optimizer_steps,
            "actor_learning_rate": self.actor_optimizer.param_groups[0]["lr"],
            "critic_learning_rate": self.critic_optimizer.param_groups[0]["lr"],
        }

    def _value_loss(
        self, predicted: Tensor, old_values: Tensor, targets: Tensor
    ) -> Tensor:
        error = targets - predicted
        original = functional.huber_loss(
            predicted, targets, delta=self.config.huber_delta, reduction="none"
        )
        if not self.config.use_clipped_value_loss:
            return original.mean()
        clipped_prediction = old_values + (predicted - old_values).clamp(
            -self.config.clip_param, self.config.clip_param
        )
        clipped = functional.huber_loss(
            clipped_prediction,
            targets,
            delta=self.config.huber_delta,
            reduction="none",
        )
        del error
        return torch.maximum(original, clipped).mean()

    def linear_lr_decay(self, completed_steps: int, total_steps: int) -> None:
        fraction = max(0.0, 1.0 - completed_steps / max(total_steps, 1))
        self.actor_optimizer.param_groups[0]["lr"] = self.config.actor_lr * fraction
        self.critic_optimizer.param_groups[0]["lr"] = self.config.critic_lr * fraction

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
        if state.get("config") != asdict(self.config):
            raise ValueError("MAPPO checkpoint trainer configuration differs.")
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.value_normalizer.load_state_dict(state["value_normalizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])

    def _tensor(self, values) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)
