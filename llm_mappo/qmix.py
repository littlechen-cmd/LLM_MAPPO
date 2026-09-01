"""A compact QMIX-WP baseline sharing the Phase 2 warehouse interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class AgentQNetwork(nn.Module):
    """Shared decentralised action-value network for homogeneous AGVs."""

    def __init__(self, observation_dim: int, action_count: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class QMIXMixer(nn.Module):
    """Monotonic state-conditioned mixer used by the frozen QMIX baseline."""

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size = agent_values.shape[0]
        weights_one = torch.abs(self.hyper_w1(state)).view(
            batch_size, self.n_agents, self.embed_dim
        )
        hidden = torch.bmm(agent_values.unsqueeze(1), weights_one).squeeze(1)
        hidden = torch.nn.functional.elu(hidden + self.hyper_b1(state))
        weights_two = torch.abs(self.hyper_w2(state)).view(
            batch_size, self.embed_dim, 1
        )
        mixed = torch.bmm(hidden.unsqueeze(1), weights_two).squeeze(1)
        return mixed + self.hyper_b2(state)


@dataclass
class QMIXHyperparameters:
    gamma: float = 0.99
    learning_rate: float = 3e-4
    target_update_interval: int = 1000


class QMIXLearner:
    """One-step QMIX learner; replay ownership stays in the training entry point."""

    def __init__(
        self, observation_dim: int, action_count: int, n_agents: int,
        hyperparameters: QMIXHyperparameters, device: torch.device,
    ) -> None:
        self.n_agents = n_agents
        self.device = device
        self.hyperparameters = hyperparameters
        state_dim = observation_dim * n_agents
        self.agent = AgentQNetwork(observation_dim, action_count).to(device)
        self.mixer = QMIXMixer(n_agents, state_dim).to(device)
        self.target_agent = AgentQNetwork(observation_dim, action_count).to(device)
        self.target_mixer = QMIXMixer(n_agents, state_dim).to(device)
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer = torch.optim.Adam(
            list(self.agent.parameters()) + list(self.mixer.parameters()),
            lr=hyperparameters.learning_rate,
        )

    def select_actions(
        self, observations: np.ndarray, action_masks: np.ndarray, epsilon: float,
        generator: np.random.Generator,
    ) -> np.ndarray:
        with torch.no_grad():
            values = self.agent(torch.as_tensor(observations, device=self.device))
        actions = values.argmax(dim=1).cpu().numpy()
        for index, mask in enumerate(action_masks):
            valid = np.flatnonzero(mask)
            if generator.random() < epsilon:
                actions[index] = generator.choice(valid)
            elif not mask[actions[index]]:
                actions[index] = valid[np.argmax(values[index, valid].cpu().numpy())]
        return actions.astype(np.int64, copy=False)

    def update(self, batch: dict[str, np.ndarray]) -> float:
        observations = torch.as_tensor(batch["observations"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(batch["rewards"], device=self.device).unsqueeze(1)
        next_observations = torch.as_tensor(
            batch["next_observations"], device=self.device
        )
        dones = torch.as_tensor(batch["dones"], device=self.device).unsqueeze(1)
        masks = torch.as_tensor(batch["masks"], device=self.device, dtype=torch.bool)
        next_masks = torch.as_tensor(
            batch["next_masks"], device=self.device, dtype=torch.bool
        )
        state = observations.flatten(start_dim=1)
        next_state = next_observations.flatten(start_dim=1)
        values = self.agent(observations.flatten(end_dim=1)).view(*actions.shape, -1)
        chosen = values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        mixed = self.mixer(chosen, state)
        with torch.no_grad():
            target_values = self.target_agent(next_observations.flatten(end_dim=1))
            target_values = target_values.view(*actions.shape, -1)
            target_values = target_values.masked_fill(~next_masks, -torch.inf)
            next_chosen = target_values.max(dim=2).values
            target_values = self.target_mixer(next_chosen, next_state)
            target = rewards + self.hyperparameters.gamma * (1.0 - dones) * target_values
        del masks
        loss = torch.nn.functional.mse_loss(mixed, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.parameters()) + list(self.mixer.parameters()), 10.0
        )
        self.optimizer.step()
        return float(loss.detach().cpu())

    def update_targets(self) -> None:
        self.target_agent.load_state_dict(self.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def checkpoint(self) -> dict:
        return {
            "algorithm": "qmix-wp-v1",
            "agent": self.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "hyperparameters": self.hyperparameters.__dict__,
        }

    def state_dict(self) -> dict:
        """Complete resumable state for E1's DirectGoal QMIX baseline."""
        return {"agent": self.agent.state_dict(), "mixer": self.mixer.state_dict(),
                "target_agent": self.target_agent.state_dict(), "target_mixer": self.target_mixer.state_dict(),
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.agent.load_state_dict(state["agent"]); self.mixer.load_state_dict(state["mixer"])
        self.target_agent.load_state_dict(state["target_agent"]); self.target_mixer.load_state_dict(state["target_mixer"])
        self.optimizer.load_state_dict(state["optimizer"])
