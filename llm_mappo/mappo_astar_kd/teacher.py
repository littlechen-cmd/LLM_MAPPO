"""Cached orientation-aware geometric labels for Phase 2 A*KD.

The cache deliberately contains only static map geometry.  It neither observes
other AGVs nor models their future motion; dynamic occupancy belongs solely to
the training-side attenuation diagnostic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

import numpy as np


TEACHER_VERSION = "astar-geometric-cost-v2"
MOTION_ACTIONS = ("FORWARD", "LEFT", "RIGHT")
ORIENTATIONS = ("UP", "RIGHT", "DOWN", "LEFT")
_DIRECTION_INDEX = {name: index for index, name in enumerate(ORIENTATIONS)}
_DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
_Footprint = Literal["unloaded", "loaded"]


@dataclass(frozen=True)
class StaticMap:
    """Immutable facilities needed to identify a static geometric table."""

    width: int
    height: int
    highway: frozenset[tuple[int, int]]
    workstations: frozenset[tuple[int, int]]
    chargers: frozenset[tuple[int, int]]
    static_obstacles: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("Static map dimensions must be positive.")
        for point in (
            self.highway | self.workstations | self.chargers | self.static_obstacles
        ):
            if not self.in_bounds(point):
                raise ValueError("Static map contains an out-of-bounds coordinate.")

    def in_bounds(self, point: tuple[int, int]) -> bool:
        return 0 <= point[0] < self.width and 0 <= point[1] < self.height

    @property
    def map_hash(self) -> str:
        payload = {
            "width": self.width,
            "height": self.height,
            "highway": sorted(self.highway),
            "workstations": sorted(self.workstations),
            "chargers": sorted(self.chargers),
            "static_obstacles": sorted(self.static_obstacles),
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AStarKDQuery:
    """One pre-transition static geometric teacher request."""

    pose: tuple[int, int]
    orientation: str
    goal: tuple[int, int]
    footprint: _Footprint
    dead: bool = False
    picking_lock: bool = False
    mandatory_toggle_load: bool = False


@dataclass(frozen=True)
class AStarKDResult:
    """A three-action label plus fail-closed diagnostic provenance."""

    soft_target: np.ndarray
    valid: bool
    failure_reason: str
    root_costs: tuple[float | None, float | None, float | None]
    diagnostics: dict[str, object]


class AStarKDTeacher:
    """Build and reuse exact reverse-BFS cost-to-go tables per static key."""

    def __init__(self, static_map: StaticMap, *, tau: float = 1.0) -> None:
        if tau <= 0.0:
            raise ValueError("A*KD temperature must be positive.")
        self.static_map = static_map
        self.tau = float(tau)
        self._tables: dict[tuple[str, str, tuple[int, int], str], dict] = {}
        self.cache_build_count = 0
        self.cache_hit_count = 0

    def query(
        self,
        query: AStarKDQuery,
        *,
        use_semantic_validity: bool = True,
    ) -> AStarKDResult:
        """Return a min-max, tau=1 soft label or an exact zero invalid label."""

        semantic_failure = self._semantic_failure(query)
        if use_semantic_validity and semantic_failure is not None:
            return self._invalid(semantic_failure, cache_hit=False)
        technical_failure = self._technical_failure(query)
        if technical_failure is not None:
            return self._invalid(technical_failure, cache_hit=False)
        key = (TEACHER_VERSION, self.static_map.map_hash, query.goal, query.footprint)
        table = self._tables.get(key)
        cache_hit = table is not None
        if table is None:
            try:
                table = self._build_table(query.goal, query.footprint)
            except (TypeError, ValueError):
                return self._invalid("cache_build_error", cache_hit=False)
            self._tables[key] = table
            self.cache_build_count += 1
        else:
            self.cache_hit_count += 1
        root_costs = self._root_costs(query, table)
        finite = [cost for cost in root_costs if cost is not None]
        if not finite:
            return self._invalid(
                "no_finite_root_cost", cache_hit=cache_hit, root_costs=root_costs
            )
        target = self._soft_target(root_costs)
        if not np.isfinite(target).all() or not np.isclose(target.sum(), 1.0):
            return self._invalid(
                "non_finite_target", cache_hit=cache_hit, root_costs=root_costs
            )
        preferred = min(
            (index for index, cost in enumerate(root_costs) if cost is not None),
            key=lambda index: (root_costs[index], index),
        )
        return AStarKDResult(
            soft_target=target,
            valid=True,
            failure_reason="ok",
            root_costs=root_costs,
            diagnostics={
                "teacher_version": TEACHER_VERSION,
                "map_hash": self.static_map.map_hash,
                "cache_hit": cache_hit,
                "preferred_root_action": MOTION_ACTIONS[preferred],
                "finite_root_count": len(finite),
            },
        )

    def _semantic_failure(self, query: AStarKDQuery) -> str | None:
        if query.dead:
            return "dead"
        if query.picking_lock:
            return "picking_lock"
        if query.mandatory_toggle_load:
            return "mandatory_toggle_load"
        if query.pose == query.goal:
            return "already_at_goal"
        return None

    def _technical_failure(self, query: AStarKDQuery) -> str | None:
        if query.orientation not in _DIRECTION_INDEX:
            return "invalid_orientation"
        if query.footprint not in {"unloaded", "loaded"}:
            return "invalid_footprint"
        if not self.static_map.in_bounds(query.pose):
            return "invalid_pose"
        if not self.static_map.in_bounds(query.goal):
            return "invalid_goal"
        if query.pose in self.static_map.static_obstacles:
            return "invalid_pose"
        if query.goal in self.static_map.static_obstacles:
            return "invalid_goal"
        return None

    def _build_table(self, goal: tuple[int, int], footprint: _Footprint) -> dict:
        states = [
            (x, y, direction)
            for y in range(self.static_map.height)
            for x in range(self.static_map.width)
            if (x, y) not in self.static_map.static_obstacles
            for direction in ORIENTATIONS
        ]
        reverse: dict[tuple[int, int, str], list[tuple[int, int, str]]] = {
            state: [] for state in states
        }
        for state in states:
            for action in MOTION_ACTIONS:
                successor = self._transition(state, action, footprint)
                if successor is not None:
                    reverse[successor].append(state)
        distances: dict[tuple[int, int, str], float] = {}
        frontier = deque()
        for orientation in ORIENTATIONS:
            state = (*goal, orientation)
            if state in reverse:
                distances[state] = 0.0
                frontier.append(state)
        while frontier:
            state = frontier.popleft()
            for predecessor in reverse[state]:
                if predecessor not in distances:
                    distances[predecessor] = distances[state] + 1.0
                    frontier.append(predecessor)
        return distances

    def _root_costs(self, query: AStarKDQuery, table: dict):
        state = (*query.pose, query.orientation)
        costs: list[float | None] = []
        for action in MOTION_ACTIONS:
            successor = self._transition(state, action, query.footprint)
            if successor is None or successor not in table:
                costs.append(None)
            else:
                costs.append(1.0 + float(table[successor]))
        return tuple(costs)

    def _transition(self, state, action: str, footprint: _Footprint):
        x, y, orientation = state
        if action == "LEFT":
            return x, y, ORIENTATIONS[(_DIRECTION_INDEX[orientation] - 1) % 4]
        if action == "RIGHT":
            return x, y, ORIENTATIONS[(_DIRECTION_INDEX[orientation] + 1) % 4]
        dx, dy = _DELTAS[orientation]
        candidate = x + dx, y + dy
        if (
            not self.static_map.in_bounds(candidate)
            or candidate in self.static_map.static_obstacles
        ):
            return None
        if footprint == "loaded" and candidate not in self.static_map.highway:
            return None
        return candidate[0], candidate[1], orientation

    def _soft_target(self, root_costs) -> np.ndarray:
        finite = np.asarray([cost for cost in root_costs if cost is not None])
        lower, upper = float(finite.min()), float(finite.max())
        weights = np.zeros(3, dtype=np.float64)
        for index, cost in enumerate(root_costs):
            if cost is None:
                continue
            normalized = 0.0 if upper == lower else (cost - lower) / (upper - lower)
            weights[index] = np.exp(-normalized / self.tau)
        return (weights / weights.sum()).astype(np.float32)

    @staticmethod
    def _invalid(reason, *, cache_hit, root_costs=(None, None, None)) -> AStarKDResult:
        return AStarKDResult(
            soft_target=np.zeros(3, dtype=np.float32),
            valid=False,
            failure_reason=reason,
            root_costs=tuple(root_costs),
            diagnostics={
                "teacher_version": TEACHER_VERSION,
                "cache_hit": cache_hit,
                "preferred_root_action": None,
            },
        )
