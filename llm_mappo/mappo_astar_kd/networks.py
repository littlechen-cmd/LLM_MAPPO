"""Phase 2 Actor with a training-only geometric motion head."""

from __future__ import annotations

from torch import Tensor, nn

from llm_mappo.mappo_baseline.networks import MAPPOActor, _orthogonal


class AStarKDActor(MAPPOActor):
    """Keep the Phase 1 policy path intact and expose a parallel 3-way head."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__(observation_dim, action_dim, hidden_dim)
        self.motion_prior_head = _orthogonal(nn.Linear(hidden_dim, 3), gain=0.01)

    def forward_with_motion(
        self, observations: Tensor, hidden: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded, next_hidden = self._step(observations, hidden)
        return self.action_head(encoded), self.motion_prior_head(encoded), next_hidden

    def forward_sequence_with_motion(
        self,
        observations: Tensor,
        initial_hidden: Tensor,
        continuation_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        encoded, next_hidden = self._sequence(
            observations, initial_hidden, continuation_masks
        )
        return self.action_head(encoded), self.motion_prior_head(encoded), next_hidden
