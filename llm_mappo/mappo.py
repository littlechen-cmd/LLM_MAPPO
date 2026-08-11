"""Custom PyTorch CTDE MAPPO components used by Phase 2."""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


def _mlp(
    sizes: Sequence[int], final_activation: nn.Module | None = None
) -> nn.Sequential:
    """Build a ReLU MLP with an optional final activation."""
    layers: List[nn.Module] = []
    for input_size, output_size in zip(sizes[:-1], sizes[1:]):
        layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
    layers.pop()
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class SharedActor(nn.Module):
    """One decentralized policy shared across all homogeneous AGVs."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = _mlp((observation_dim, hidden_dim, 64))
        self.logits = nn.Linear(64, action_dim)

    def forward(self, observations: Tensor) -> Tensor:
        return self.logits(self.encoder(observations))


class DualHeadActor(nn.Module):
    """Separate semantic and motion branches for the Phase 3 r2 ablation."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.motion_encoder = _mlp((observation_dim, hidden_dim, 64))
        self.engagement_encoder = _mlp((observation_dim, hidden_dim, 64))
        self.engagement_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.motion_head = nn.Linear(65, action_dim)

    def engagement(self, observations: Tensor) -> Tensor:
        encoded = self.engagement_encoder(observations)
        return self.engagement_head(encoded).squeeze(-1)

    def forward(self, observations: Tensor) -> Tensor:
        motion_features = self.motion_encoder(observations)
        engagement = self.engagement(observations).unsqueeze(-1)
        motion_input = torch.cat((motion_features, engagement.detach()), dim=-1)
        return self.motion_head(motion_input)


class CentralizedCritic(nn.Module):
    """Attention-pool every agent encoding before predicting the team value."""

    def __init__(self, observation_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.agent_encoder = _mlp((observation_dim, hidden_dim, hidden_dim))
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.value_head = _mlp((hidden_dim, 256, 128, 1))

    def forward(self, global_observations: Tensor) -> Tensor:
        encoded = self.agent_encoder(global_observations)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        pooled = attended.mean(dim=1)
        return self.value_head(pooled).squeeze(-1)


class MAPPOPolicy(nn.Module):
    """Shared Actor plus centralized Critic for homogeneous multi-agent PPO."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.actor = SharedActor(observation_dim, action_dim)
        self.critic = CentralizedCritic(observation_dim)
        self.device = torch.device(device)
        self.to(self.device)

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray | None = None,
        deterministic: bool = False,
    ):
        actor_obs = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        state = actor_obs.unsqueeze(0)
        logits = self._masked_logits(self.actor(actor_obs), action_masks)
        distribution = Categorical(logits=logits)
        actions = (
            torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        )
        return (
            actions.cpu().numpy(),
            distribution.log_prob(actions).cpu().numpy(),
            self.critic(state).item(),
        )

    def evaluate_actions(
        self, observations: Tensor, actions: Tensor, action_masks: Tensor
    ):
        distribution = Categorical(
            logits=self._masked_logits(self.actor(observations), action_masks)
        )
        return distribution.log_prob(actions), distribution.entropy()

    def _masked_logits(self, logits: Tensor, action_masks) -> Tensor:
        if action_masks is None:
            return logits
        masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        if masks.shape != logits.shape:
            raise ValueError("Action-mask shape must match the policy-logit shape.")
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("Every AGV must have at least one valid action.")
        return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)

    def values(self, states: Tensor) -> Tensor:
        return self.critic(states)


class DualHeadMAPPOPolicy(nn.Module):
    """CTDE MAPPO policy used by Phase 3a and its later ablations."""

    def __init__(self, observation_dim: int, action_dim: int, device: str = "cpu"):
        super().__init__()
        self.actor = DualHeadActor(observation_dim, action_dim)
        self.critic = CentralizedCritic(observation_dim)
        self.device = torch.device(device)
        self.to(self.device)

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray | None = None,
        deterministic: bool = False,
    ):
        actor_obs = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        logits = self._masked_logits(self.actor(actor_obs), action_masks)
        distribution = Categorical(logits=logits)
        actions = (
            torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        )
        return (
            actions.cpu().numpy(),
            distribution.log_prob(actions).cpu().numpy(),
            self.critic(actor_obs.unsqueeze(0)).item(),
            self.actor.engagement(actor_obs).cpu().numpy(),
        )

    def evaluate_actions(
        self, observations: Tensor, actions: Tensor, action_masks: Tensor
    ):
        distribution = Categorical(
            logits=self._masked_logits(self.actor(observations), action_masks)
        )
        return distribution.log_prob(actions), distribution.entropy()

    def engagement(self, observations: Tensor) -> Tensor:
        return self.actor.engagement(observations)

    def _masked_logits(self, logits: Tensor, action_masks) -> Tensor:
        if action_masks is None:
            return logits
        masks = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        if masks.shape != logits.shape:
            raise ValueError("Action-mask shape must match the policy-logit shape.")
        if not torch.all(masks.any(dim=-1)):
            raise ValueError("Every AGV must have at least one valid action.")
        return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)

    def values(self, states: Tensor) -> Tensor:
        return self.critic(states)


@dataclass
class PPOHyperparameters:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_steps: int = 64
    reservation_kl_coefficient: float = 0.0
    reservation_kl_decay_interval: int = 100
    reservation_kl_decay_factor: float = 0.90
    reservation_kl_minimum: float = 0.03
    engagement_coefficient: float = 0.0


class RolloutBuffer:
    """Collect whole-team transitions and calculate centralized GAE returns."""

    def __init__(self, n_agents: int, action_dim: int = 5):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.action_masks: List[np.ndarray] = []
        self.reservation_preferences: List[np.ndarray] = []
        self.engagement_targets: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        action_masks: np.ndarray | None = None,
        reservation_preferences: np.ndarray | None = None,
        engagement_targets: np.ndarray | None = None,
    ) -> None:
        self.observations.append(np.asarray(observations, dtype=np.float32).copy())
        self.actions.append(np.asarray(actions, dtype=np.int64).copy())
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32).copy())
        if action_masks is None:
            action_masks = np.ones((self.n_agents, self.action_dim), dtype=bool)
        self.action_masks.append(np.asarray(action_masks, dtype=bool).copy())
        if reservation_preferences is None:
            preferences = np.ones(
                (self.n_agents, self.action_dim), dtype=np.float32
            )
            preferences /= float(self.action_dim)
        else:
            preferences = np.asarray(reservation_preferences, dtype=np.float32)
            if preferences.shape != (self.n_agents, self.action_dim):
                raise ValueError(
                    "Reservation preferences must match the team action shape."
                )
            preferences = preferences.copy()
        self.reservation_preferences.append(preferences)
        if engagement_targets is None:
            targets = np.full(self.n_agents, -1.0, dtype=np.float32)
        else:
            targets = np.asarray(engagement_targets, dtype=np.float32)
            if targets.shape != (self.n_agents,):
                raise ValueError("Engagement targets must contain one value per AGV.")
            if np.any((targets < 0.0) | (targets > 1.0)):
                raise ValueError("Engagement targets must be within [0, 1].")
            targets = targets.copy()
        self.engagement_targets.append(targets)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.rewards)

    def tensors(self, last_value: float, hyperparameters: PPOHyperparameters, device):
        if not self.rewards:
            raise ValueError("Cannot update MAPPO with an empty rollout.")
        advantages = np.zeros(len(self), dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)
        for index in reversed(range(len(self))):
            mask = 1.0 - float(self.dones[index])
            delta = self.rewards[index] + hyperparameters.gamma * next_value * mask
            delta -= self.values[index]
            gae = delta + hyperparameters.gamma * hyperparameters.gae_lambda * mask * gae
            advantages[index] = gae
            next_value = self.values[index]
        returns = advantages + np.asarray(self.values, dtype=np.float32)
        return {
            "states": torch.as_tensor(
                np.stack(self.observations), dtype=torch.float32, device=device
            ),
            "actions": torch.as_tensor(
                np.stack(self.actions), dtype=torch.long, device=device
            ),
            "log_probs": torch.as_tensor(
                np.stack(self.log_probs), dtype=torch.float32, device=device
            ),
            "action_masks": torch.as_tensor(
                np.stack(self.action_masks), dtype=torch.bool, device=device
            ),
            "reservation_preferences": torch.as_tensor(
                np.stack(self.reservation_preferences),
                dtype=torch.float32,
                device=device,
            ),
            "engagement_targets": torch.as_tensor(
                np.stack(self.engagement_targets),
                dtype=torch.float32,
                device=device,
            ),
            "advantages": torch.as_tensor(
                advantages, dtype=torch.float32, device=device
            ),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        }

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.action_masks.clear()
        self.reservation_preferences.clear()
        self.engagement_targets.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()


class MAPPOUpdater:
    """Run clipped PPO updates over shared Actor and centralized Critic."""

    def __init__(self, policy: MAPPOPolicy, hyperparameters: PPOHyperparameters):
        self.policy = policy
        self.hyperparameters = hyperparameters
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=hyperparameters.learning_rate
        )

    def update(self, buffer: RolloutBuffer, last_value: float) -> Dict[str, float]:
        data = buffer.tensors(last_value, self.hyperparameters, self.policy.device)
        advantages = data["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        data["advantages"] = advantages
        total_steps = data["states"].shape[0]
        metric_sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        use_engagement = self.hyperparameters.engagement_coefficient > 0
        if use_engagement:
            if not hasattr(self.policy, "engagement"):
                raise TypeError("Engagement loss requires a dual-head MAPPO policy.")
            metric_sums["engagement_loss"] = 0.0
        use_reservation_kl = self.hyperparameters.reservation_kl_coefficient > 0
        if use_reservation_kl:
            metric_sums["reservation_kl"] = 0.0
        updates = 0
        for _ in range(self.hyperparameters.update_epochs):
            order = torch.randperm(total_steps, device=self.policy.device)
            for start in range(0, total_steps, self.hyperparameters.minibatch_steps):
                indices = order[start:start + self.hyperparameters.minibatch_steps]
                states = data["states"][indices]
                batch_agents = states.shape[1]
                actor_observations = states.reshape(-1, states.shape[-1])
                actions = data["actions"][indices].reshape(-1)
                old_log_probs = data["log_probs"][indices].reshape(-1)
                action_masks = data["action_masks"][indices].reshape(
                    -1, data["action_masks"].shape[-1]
                )
                reservation_preferences = data["reservation_preferences"][indices]
                reservation_preferences = reservation_preferences.reshape(
                    -1, reservation_preferences.shape[-1]
                )
                engagement_targets = data["engagement_targets"][indices].reshape(-1)
                batch_advantages = data["advantages"][indices].repeat_interleave(
                    batch_agents
                )
                log_probs, entropy = self.policy.evaluate_actions(
                    actor_observations, actions, action_masks
                )
                ratio = torch.exp(log_probs - old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.hyperparameters.clip_ratio,
                    1.0 + self.hyperparameters.clip_ratio,
                )
                policy_loss = -torch.minimum(
                    ratio * batch_advantages, clipped_ratio * batch_advantages
                ).mean()
                values = self.policy.values(states)
                value_loss = nn.functional.mse_loss(values, data["returns"][indices])
                entropy_bonus = entropy.mean()
                reservation_kl = torch.zeros((), device=self.policy.device)
                if use_reservation_kl:
                    masked_logits = self.policy._masked_logits(
                        self.policy.actor(actor_observations), action_masks
                    )
                    actor_log_probs = torch.log_softmax(masked_logits, dim=-1)
                    valid_teacher = reservation_preferences * action_masks.float()
                    normalizer = valid_teacher.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    teacher_probs = valid_teacher / normalizer
                    teacher_log_probs = torch.where(
                        teacher_probs > 0,
                        torch.log(teacher_probs.clamp_min(1e-8)),
                        torch.zeros_like(teacher_probs),
                    )
                    reservation_kl = (
                        teacher_probs * (teacher_log_probs - actor_log_probs)
                    ).sum(dim=-1).mean()
                engagement_loss = torch.zeros((), device=self.policy.device)
                if use_engagement:
                    valid_targets = engagement_targets >= 0.0
                    if torch.any(valid_targets):
                        engagement_values = self.policy.engagement(actor_observations)
                        engagement_loss = nn.functional.mse_loss(
                            engagement_values[valid_targets],
                            engagement_targets[valid_targets],
                        )
                loss = (
                    policy_loss
                    + self.hyperparameters.value_coefficient * value_loss
                    - self.hyperparameters.entropy_coefficient * entropy_bonus
                    + self.hyperparameters.reservation_kl_coefficient
                    * reservation_kl
                    + self.hyperparameters.engagement_coefficient * engagement_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.hyperparameters.max_grad_norm
                )
                self.optimizer.step()
                metric_sums["policy_loss"] += policy_loss.item()
                metric_sums["value_loss"] += value_loss.item()
                metric_sums["entropy"] += entropy_bonus.item()
                if use_reservation_kl:
                    metric_sums["reservation_kl"] += reservation_kl.item()
                if use_engagement:
                    metric_sums["engagement_loss"] += engagement_loss.item()
                updates += 1
        buffer.clear()
        metrics = {key: value / updates for key, value in metric_sums.items()}
        if use_reservation_kl:
            metrics["reservation_coefficient"] = float(
                self.hyperparameters.reservation_kl_coefficient
            )
        return metrics
