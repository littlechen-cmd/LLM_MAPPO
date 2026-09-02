"""Reward-v2 primitives for the R1 convergence-recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


REWARD_V2 = "reward-v2"
LEGACY_REWARD_V1 = "legacy-v1"
SUPPORTED_REWARD_VERSIONS = frozenset({LEGACY_REWARD_V1, REWARD_V2})

COMPLETION_REWARD = 10.0
PICKUP_REWARD = 2.0
PROGRESS_REWARD = 0.1
BLOCKED_FORWARD_PENALTY = 0.15
STEP_COST = 0.01
LEGACY_COMPLETION_REWARD = 5.0


@dataclass(frozen=True)
class RewardGoalSnapshot:
    """Goal identity and distance at one side of an environment transition."""

    identity: tuple[Hashable, ...]
    coordinate: tuple[int, int]
    distance: int


def reward_v2_progress_deltas(
    before: Sequence[RewardGoalSnapshot],
    after: Sequence[RewardGoalSnapshot],
) -> np.ndarray:
    """Return signed distance progress, suppressing every goal switch."""
    if len(before) != len(after):
        raise ValueError("Reward-v2 goal snapshots must preserve agent cardinality.")
    return np.asarray(
        [
            float(prior.distance - current.distance)
            if prior.identity == current.identity
            else 0.0
            for prior, current in zip(before, after)
        ],
        dtype=np.float64,
    )


def reward_v2_team_reward(
    *,
    raw_rewards: Sequence[float],
    events: Sequence[Mapping],
    task_weights: Mapping[Hashable, float],
    pickup_weights: Mapping[int, float],
    progress_deltas: Sequence[float],
    low_battery_penalties: Sequence[float],
    legacy_blocked_forward_penalty: float,
) -> float:
    """Replace legacy shaping while preserving collision and energy rewards."""
    local = np.asarray(raw_rewards, dtype=np.float64).copy()
    progress = np.asarray(progress_deltas, dtype=np.float64)
    energy = np.asarray(low_battery_penalties, dtype=np.float64)
    if local.ndim != 1 or progress.shape != local.shape or energy.shape != local.shape:
        raise ValueError("Reward-v2 per-agent vectors must have identical shapes.")
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(progress)):
        raise ValueError("Reward-v2 inputs must be finite.")

    team_completion = 0.0
    for event in events:
        event_type = event.get("type")
        if event_type == "task_completed":
            task_id = event["task_id"]
            if task_id not in task_weights:
                raise ValueError(
                    "Reward-v2 completion lacks its pre-step priority weight."
                )
            weight = float(task_weights[task_id])
            agent_index = int(event["agent_id"]) - 1
            local[agent_index] -= LEGACY_COMPLETION_REWARD * weight
            team_completion += COMPLETION_REWARD * weight
        elif event_type == "blocked_forward":
            agent_index = int(event["agent_id"]) - 1
            local[agent_index] -= (
                BLOCKED_FORWARD_PENALTY - legacy_blocked_forward_penalty
            )

    for agent_id, weight in pickup_weights.items():
        local[int(agent_id) - 1] += PICKUP_REWARD * float(weight)
    local += PROGRESS_REWARD * progress
    local += energy

    reward = team_completion + float(np.mean(local)) - STEP_COST
    if not np.isfinite(reward):
        raise ValueError("Reward-v2 produced a non-finite team reward.")
    return reward
