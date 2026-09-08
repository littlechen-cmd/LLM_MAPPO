"""Recurrent, parameter-shared MAPPO Actor and FP centralized Critic."""

from __future__ import annotations

import numpy as np
import torch
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


class _RecurrentEncoder(nn.Module):
    """MLP encoder plus one GRU layer with explicit worker/AGV state axes."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = _mlp(self.input_dim, self.hidden_dim)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.constant_(parameter, 0.0)

    def _hidden(self, inputs: Tensor, hidden: Tensor | None) -> Tensor:
        expected = (*inputs.shape[:-1], self.hidden_dim)
        if hidden is None:
            return torch.zeros(expected, dtype=inputs.dtype, device=inputs.device)
        if tuple(hidden.shape) != expected:
            raise ValueError(
                "Recurrent hidden state must preserve every worker and AGV axis."
            )
        return hidden

    def _step(self, inputs: Tensor, hidden: Tensor | None) -> tuple[Tensor, Tensor]:
        if inputs.shape[-1] != self.input_dim:
            raise ValueError("Network input dimension does not match its contract.")
        previous = self._hidden(inputs, hidden)
        leading_shape = inputs.shape[:-1]
        encoded = self.encoder(inputs).reshape(-1, 1, self.hidden_dim)
        output, next_hidden = self.gru(
            encoded, previous.reshape(1, -1, self.hidden_dim).contiguous()
        )
        return (
            output.reshape(*leading_shape, self.hidden_dim),
            next_hidden.reshape(*leading_shape, self.hidden_dim),
        )

    def _sequence(
        self,
        inputs: Tensor,
        initial_hidden: Tensor,
        continuation_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if inputs.ndim < 3:
            raise ValueError("Recurrent sequence requires time and batch axes.")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError("Network input dimension does not match its contract.")
        expected_hidden = (*inputs.shape[1:-1], self.hidden_dim)
        if tuple(initial_hidden.shape) != expected_hidden:
            raise ValueError("Initial recurrent state does not match sequence axes.")
        if tuple(continuation_masks.shape) != tuple(inputs.shape[:-1]):
            raise ValueError("Continuation masks must have time, worker and AGV axes.")

        hidden = initial_hidden
        outputs = []
        for step in range(inputs.shape[0]):
            hidden = hidden * continuation_masks[step].unsqueeze(-1)
            output, hidden = self._step(inputs[step], hidden)
            outputs.append(output)
        return torch.stack(outputs), hidden


class MAPPOActor(_RecurrentEncoder):
    """Decentralized recurrent policy with one hidden state per worker/AGV."""

    def __init__(
        self, observation_dim: int, action_dim: int, hidden_dim: int = 64
    ) -> None:
        super().__init__(observation_dim, hidden_dim)
        self.observation_dim = self.input_dim
        self.action_dim = int(action_dim)
        self.action_head = _orthogonal(
            nn.Linear(self.hidden_dim, self.action_dim), gain=0.01
        )

    def forward(
        self, observations: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        encoded, next_hidden = self._step(observations, hidden)
        return self.action_head(encoded), next_hidden

    def forward_sequence(
        self,
        observations: Tensor,
        initial_hidden: Tensor,
        continuation_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        encoded, next_hidden = self._sequence(
            observations, initial_hidden, continuation_masks
        )
        return self.action_head(encoded), next_hidden


class MAPPOFPCritic(_RecurrentEncoder):
    """Training-only recurrent FP value function with independent parameters."""

    def __init__(self, state_dim: int, hidden_dim: int = 64) -> None:
        super().__init__(state_dim, hidden_dim)
        self.state_dim = self.input_dim
        self.value_head = _orthogonal(nn.Linear(self.hidden_dim, 1), gain=1.0)

    def forward(
        self, fp_states: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        encoded, next_hidden = self._step(fp_states, hidden)
        return self.value_head(encoded).squeeze(-1), next_hidden

    def forward_sequence(
        self,
        fp_states: Tensor,
        initial_hidden: Tensor,
        continuation_masks: Tensor,
    ) -> tuple[Tensor, Tensor]:
        encoded, next_hidden = self._sequence(
            fp_states, initial_hidden, continuation_masks
        )
        return self.value_head(encoded).squeeze(-1), next_hidden
