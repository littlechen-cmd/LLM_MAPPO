"""Debiased running normalization for MAPPO value targets."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ValueNorm(nn.Module):
    """Track return moments without adding trainable parameters."""

    def __init__(
        self,
        shape: tuple[int, ...] = (1,),
        beta: float = 0.99999,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_mean_sq", torch.zeros(shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, values: Tensor) -> None:
        values = values.detach().reshape(-1, *self.running_mean.shape)
        batch_mean = values.mean(dim=0)
        batch_mean_sq = values.square().mean(dim=0)
        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(
            batch_mean_sq * (1.0 - self.beta)
        )
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def _moments(self) -> tuple[Tensor, Tensor]:
        debias = self.debiasing_term.clamp_min(self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        variance = (mean_sq - mean.square()).clamp_min(1e-2)
        return mean, variance

    def normalize(self, values: Tensor) -> Tensor:
        mean, variance = self._moments()
        return (values - mean) / torch.sqrt(variance)

    def denormalize(self, values: Tensor) -> Tensor:
        mean, variance = self._moments()
        return values * torch.sqrt(variance) + mean
