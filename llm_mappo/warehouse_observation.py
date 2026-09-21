"""Versioned planner-free physical observation builders for the optimization route."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class ObservationSchema(str, Enum):
    """Physical-observation schemas with fixed meanings and widths."""

    LEGACY_WAYPOINT_V1 = "legacy-waypoint-v1"
    DIRECT_GOAL_V1 = "direct-goal-observation-v1"
    NO_GEOMETRIC_GOAL_HINT_V1 = "no-geometric-goal-hint-v1"


@dataclass
class PlannerQueryCounter:
    """Explicit instrumentation for forbidden optimization execution queries."""

    count: int = 0

    def record_query(self) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


def build_direct_goal_observation(
    raw_observation: np.ndarray,
    own_features: np.ndarray,
    direction_features: np.ndarray,
    goal_dx: float,
    goal_dy: float,
    nearby_features: np.ndarray,
    global_features: np.ndarray,
) -> np.ndarray:
    """Build the 613D DirectGoal observation without planner-derived fields."""

    geometry = np.asarray(
        [goal_dx, goal_dy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    return _concatenate_observation(
        raw_observation,
        own_features,
        direction_features,
        geometry,
        nearby_features,
        global_features,
    )


def build_no_goal_hint_observation(
    raw_observation: np.ndarray,
    own_features: np.ndarray,
    direction_features: np.ndarray,
    nearby_features: np.ndarray,
    global_features: np.ndarray,
) -> np.ndarray:
    """Build NoGoalHint with all nine geometric slots set to immutable zeros."""

    return _concatenate_observation(
        raw_observation,
        own_features,
        direction_features,
        np.zeros(9, dtype=np.float32),
        nearby_features,
        global_features,
    )


def _concatenate_observation(*parts: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32, copy=False)
