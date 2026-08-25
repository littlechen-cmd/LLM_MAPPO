"""Frozen O0 Student modules isolated from legacy MAPPO actors."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


PHYSICAL_OBSERVATION_DIM = 613
SEMANTIC_OBSERVATION_DIM = 61
SEMANTIC_SCORE_DIM = 3
ACTION_DIM = 5
MOTION_ACTION_DIM = 3


def _relu_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


@dataclass(frozen=True)
class O0StudentOutput:
    """Separate outputs preserve the frozen PPO, A* and LLM gradient boundaries."""

    action_logits: Tensor
    motion_logits: Tensor
    semantic_scores: Tensor


class O0StudentActor(nn.Module):
    """613D motion and 61D semantic branches with detached late fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.motion_encoder = _relu_mlp(PHYSICAL_OBSERVATION_DIM, 128, 64)
        self.motion_prior_head = nn.Linear(64, MOTION_ACTION_DIM)
        self.semantic_encoder = _relu_mlp(SEMANTIC_OBSERVATION_DIM, 128, 64)
        self.semantic_head = nn.Sequential(
            nn.Linear(64, SEMANTIC_SCORE_DIM), nn.Sigmoid()
        )
        self.semantic_adapter = nn.Sequential(
            nn.Linear(SEMANTIC_SCORE_DIM, 16), nn.ReLU()
        )
        self.action_head = nn.Linear(80, ACTION_DIM)

    def forward(
        self, physical_observations: Tensor, semantic_observations: Tensor
    ) -> O0StudentOutput:
        _validate_last_dimension(
            physical_observations, PHYSICAL_OBSERVATION_DIM, "physical_observations"
        )
        _validate_last_dimension(
            semantic_observations, SEMANTIC_OBSERVATION_DIM, "semantic_observations"
        )
        if physical_observations.shape[:-1] != semantic_observations.shape[:-1]:
            raise ValueError(
                "Physical and semantic observation batch shapes must match."
            )
        motion_features = self.motion_encoder(physical_observations)
        motion_logits = self.motion_prior_head(motion_features)
        semantic_scores = self.semantic_head(
            self.semantic_encoder(semantic_observations)
        )
        semantic_features = self.semantic_adapter(semantic_scores.detach())
        action_logits = self.action_head(
            torch.cat((motion_features, semantic_features), dim=-1)
        )
        return O0StudentOutput(action_logits, motion_logits, semantic_scores)


class O0CentralizedCritic(nn.Module):
    """Physical-observation-only 4-head centralized value function."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_encoder = _relu_mlp(PHYSICAL_OBSERVATION_DIM, 128, 128)
        self.attention = nn.MultiheadAttention(
            128, num_heads=4, batch_first=True
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, physical_observations: Tensor) -> Tensor:
        if physical_observations.ndim != 3:
            raise ValueError("Centralized critic expects [batch, agents, 613].")
        _validate_last_dimension(physical_observations, PHYSICAL_OBSERVATION_DIM,
                                 "physical_observations")
        encoded = self.agent_encoder(physical_observations)
        attended, _ = self.attention(encoded, encoded, encoded, need_weights=False)
        return self.value_head(attended.mean(dim=1)).squeeze(-1)


def _validate_last_dimension(values: Tensor, expected: int, name: str) -> None:
    if values.shape[-1] != expected:
        raise ValueError(f"{name} must end in dimension {expected}.")
