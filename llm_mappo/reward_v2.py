"""Reward-v2 primitives for the R1 convergence-recovery experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np


REWARD_V2 = "reward-v2"
LEGACY_REWARD_V1 = "legacy-v1"
SUPPORTED_REWARD_VERSIONS = frozenset({LEGACY_REWARD_V1, REWARD_V2})

COMPLETION_REWARD = 5.0
PICKUP_REWARD = 0.5
PROGRESS_REWARD = 0.1
BLOCKED_FORWARD_PENALTY = 0.1
COLLISION_PENALTY = 0.1
STEP_COST = 0.005
LEGACY_COMPLETION_REWARD = 5.0
LEGACY_COLLISION_PENALTY = 2.0


@dataclass(frozen=True)
class RewardV2Breakdown:
    """Auditable scalar contributions to one shared Reward-v2 transition."""

    completion: float
    pickup: float
    progress: float
    blocked_forward: float
    collision: float
    energy: float
    step_cost: float

    @property
    def team_reward(self) -> float:
        return (
            self.completion
            + self.pickup
            + self.progress
            + self.blocked_forward
            + self.collision
            + self.energy
            + self.step_cost
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "reward_completion": self.completion,
            "reward_pickup": self.pickup,
            "reward_progress": self.progress,
            "reward_blocked_forward": self.blocked_forward,
            "reward_collision": self.collision,
            "reward_energy": self.energy,
            "reward_step_cost": self.step_cost,
        }


@dataclass(frozen=True)
class RewardGoalSnapshot:
    """Goal identity and distance at one side of an environment transition."""

    identity: tuple[Hashable, ...]
    coordinate: tuple[int, int]
    distance: int


def static_grid_distance(
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
) -> int:
    """Return static four-neighbour route distance, or ``-1`` when unreachable.

    This intentionally excludes peer AGV positions, reservations, priorities and
    plans.  It captures only geometry which a carrying AGV must respect, so a
    route around shelves is rewarded even when its next cell increases Manhattan
    distance to the target.
    """
    if start == goal:
        return 0
    blocked = set(blocked)
    blocked.discard(start)
    blocked.discard(goal)
    frontier = deque([(start, 0)])
    visited = {start}
    while frontier:
        (x, y), distance = frontier.popleft()
        for next_point in ((x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)):
            nx, ny = next_point
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if next_point in blocked or next_point in visited:
                continue
            if next_point == goal:
                return distance + 1
            visited.add(next_point)
            frontier.append((next_point, distance + 1))
    return -1


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
            if (
                prior.identity == current.identity
                and prior.distance >= 0
                and current.distance >= 0
            )
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
    return reward_v2_team_reward_breakdown(
        raw_rewards=raw_rewards,
        events=events,
        task_weights=task_weights,
        pickup_weights=pickup_weights,
        progress_deltas=progress_deltas,
        low_battery_penalties=low_battery_penalties,
        legacy_blocked_forward_penalty=legacy_blocked_forward_penalty,
    ).team_reward


def reward_v2_team_reward_breakdown(
    *,
    raw_rewards: Sequence[float],
    events: Sequence[Mapping],
    task_weights: Mapping[Hashable, float],
    pickup_weights: Mapping[int, float],
    progress_deltas: Sequence[float],
    low_battery_penalties: Sequence[float],
    legacy_blocked_forward_penalty: float,
) -> RewardV2Breakdown:
    """Compute Reward-v2 and retain each term in shared-reward units.

    Completion is moved out of the legacy per-agent vector before aggregation.
    All local consequences are averaged over the AGV team, matching the Phase 1
    shared-reward contract. ``pickup_weights`` supplies only successful-pickup
    identities; pickup reward itself is deliberately priority-independent.
    """
    local = np.asarray(raw_rewards, dtype=np.float64).copy()
    progress = np.asarray(progress_deltas, dtype=np.float64)
    energy = np.asarray(low_battery_penalties, dtype=np.float64)
    if local.ndim != 1 or progress.shape != local.shape or energy.shape != local.shape:
        raise ValueError("Reward-v2 per-agent vectors must have identical shapes.")
    if not np.all(np.isfinite(local)) or not np.all(np.isfinite(progress)):
        raise ValueError("Reward-v2 inputs must be finite.")

    agent_count = local.size
    completion = 0.0
    raw_completion = 0.0
    blocked_events = 0
    collision_events = 0
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
            raw_completion += LEGACY_COMPLETION_REWARD * weight
            completion += COMPLETION_REWARD * weight
        elif event_type == "blocked_forward":
            agent_index = int(event["agent_id"]) - 1
            local[agent_index] -= (
                BLOCKED_FORWARD_PENALTY - legacy_blocked_forward_penalty
            )
            blocked_events += 1
        elif event_type == "collision":
            collision_events += 1

    for agent_id in pickup_weights:
        local[int(agent_id) - 1] += PICKUP_REWARD
    local += PROGRESS_REWARD * progress
    local += energy

    pickup = PICKUP_REWARD * len(pickup_weights) / agent_count
    progress_component = PROGRESS_REWARD * float(np.mean(progress))
    blocked = -BLOCKED_FORWARD_PENALTY * blocked_events / agent_count
    collision = -COLLISION_PENALTY * collision_events / agent_count
    raw_energy = (
        float(np.sum(np.asarray(raw_rewards, dtype=np.float64)))
        - raw_completion
        + legacy_blocked_forward_penalty * blocked_events
        + LEGACY_COLLISION_PENALTY * collision_events
    ) / agent_count
    energy_component = raw_energy + float(np.mean(energy))
    breakdown = RewardV2Breakdown(
        completion=completion,
        pickup=pickup,
        progress=progress_component,
        blocked_forward=blocked,
        collision=collision,
        energy=energy_component,
        step_cost=-STEP_COST,
    )
    if not np.isfinite(breakdown.team_reward):
        raise ValueError("Reward-v2 produced a non-finite team reward.")
    return breakdown
