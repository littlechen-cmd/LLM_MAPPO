"""Stable data contracts shared by the dynamic warehouse components."""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Optional, Tuple


LABEL_PATTERN = re.compile(r"^[A-Z][0-9]+$")


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"


class PlannerEvent(str, Enum):
    STALLED = "stalled"
    BLOCKED = "blocked"
    REPLAN_REQUIRED = "replan_required"


@dataclass(frozen=True)
class Task:
    """A warehouse delivery request associated with one RWARE shelf."""

    task_id: str
    shelf_id: int
    batch_id: int
    label: str
    arrival_step: int
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[int] = None
    completed_step: Optional[int] = None


@dataclass(frozen=True)
class PriorityAdjustment:
    """A validated request to change only the letter in a task label."""

    task: str
    new_label: str
    reason: str


@dataclass(frozen=True)
class EngagementLabel:
    """An offline LLM engagement preference used by the Phase 4 trainer."""

    scenario_id: str
    observation_version: str
    value: float
    model: str
    created_at: str

    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("EngagementLabel.value must be within [0, 1].")


@dataclass(frozen=True)
class SemanticPreferenceLabel:
    """Two independent offline semantic preferences for Phase 4."""

    scenario_id: str
    observation_version: str
    task_commitment: float
    local_assertiveness: float
    model: str
    created_at: str

    def __post_init__(self):
        for name in ("task_commitment", "local_assertiveness"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"SemanticPreferenceLabel.{name} must be in [0, 1].")


@dataclass(frozen=True)
class PathPlan:
    """An A* path and the action preference for its immediate waypoint."""

    waypoints: Tuple[Tuple[int, int], ...]
    action_preferences: Tuple[float, float, float, float, float]
    event: Optional[PlannerEvent] = None
