"""Shared feed-forward MAPPO actor and FP centralized critic."""

from __future__ import annotations

import numpy as np
from torch import Tensor, nn


def _orthogonal(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    gain = float(np.sqrt(2.0))
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        _orthogonal(nn.Linear(input_dim, hidden_dim), gain),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
        _orthogonal(nn.Linear(hidden_dim, hidden_dim), gain),
        nn.ReLU(),
    )


class MAPPOActor(nn.Module):
    """Parameter-shared decentralized policy over local DirectGoal inputs."""

    def __init__(
        self, observation_dim: int, action_dim: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.encoder = _mlp(self.observation_dim, hidden_dim)
        self.action_head = _orthogonal(
            nn.Linear(hidden_dim, self.action_dim), gain=0.01
        )

    def forward(self, observations: Tensor) -> Tensor:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("Actor observation dimension does not match its contract.")
        return self.action_head(self.encoder(observations))


class MAPPOFPCritic(nn.Module):
    """Shared centralized value function yielding one value per robot."""

    def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.encoder = _mlp(self.state_dim, hidden_dim)
        self.value_head = _orthogonal(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, fp_states: Tensor) -> Tensor:
        if fp_states.shape[-1] != self.state_dim:
            raise ValueError("Critic FP-state dimension does not match its contract.")
        return self.value_head(self.encoder(fp_states)).squeeze(-1)
