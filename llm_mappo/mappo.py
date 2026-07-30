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
    def act(self, observations: np.ndarray, deterministic: bool = False):
        actor_obs = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        state = actor_obs.unsqueeze(0)
        logits = self.actor(actor_obs)
        distribution = Categorical(logits=logits)
        actions = (
            torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
        )
        return (
            actions.cpu().numpy(),
            distribution.log_prob(actions).cpu().numpy(),
            self.critic(state).item(),
        )

    def evaluate_actions(self, observations: Tensor, actions: Tensor):
        distribution = Categorical(logits=self.actor(observations))
        return distribution.log_prob(actions), distribution.entropy()

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


class RolloutBuffer:
    """Collect whole-team transitions and calculate centralized GAE returns."""

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.observations: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
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
    ) -> None:
        self.observations.append(np.asarray(observations, dtype=np.float32).copy())
        self.actions.append(np.asarray(actions, dtype=np.int64).copy())
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32).copy())
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
            "advantages": torch.as_tensor(
                advantages, dtype=torch.float32, device=device
            ),
            "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
        }

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
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
                batch_advantages = data["advantages"][indices].repeat_interleave(
                    batch_agents
                )
                log_probs, entropy = self.policy.evaluate_actions(
                    actor_observations, actions
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
                loss = (
                    policy_loss
                    + self.hyperparameters.value_coefficient * value_loss
                    - self.hyperparameters.entropy_coefficient * entropy_bonus
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
                updates += 1
        buffer.clear()
        return {key: value / updates for key, value in metric_sums.items()}
