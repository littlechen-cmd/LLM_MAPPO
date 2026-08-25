"""Deterministic, coordination-free local geometric motion teacher."""

from dataclasses import dataclass
import hashlib
import heapq
import json
from time import perf_counter
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


TEACHER_VERSION = "pure-motion-astar-v1"
K_MOTION = 12
EXPANSION_BUDGET = 512
TAU_MOTION = 1.0
_ACTION_INDEX = {"FORWARD": 1, "LEFT": 2, "RIGHT": 3}
_ACTION_ORDER = ("FORWARD", "LEFT", "RIGHT")
_DIRECTION_ORDER = ("UP", "RIGHT", "DOWN", "LEFT")
_DIRECTION_INDEX = {direction: index for index, direction in enumerate(_DIRECTION_ORDER)}
_DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}


@dataclass(frozen=True)
class PureMotionQuery:
    """Only geometry, anonymous occupancy and physical action legality enter a query."""

    layout_hash: str
    width: int
    height: int
    blocked_coordinates: Tuple[Tuple[int, int], ...]
    own_pose: Tuple[int, int]
    orientation: str
    goal: Tuple[int, int]
    occupied_coordinates: Tuple[Tuple[int, int], ...]
    pure_motion_mask: Tuple[bool, bool, bool, bool, bool]
    dead: bool = False
    picking_lock: bool = False
    mandatory_toggle_load: bool = False
    footprint_class: str | None = None


@dataclass(frozen=True)
class PureMotionResult:
    """Fail-closed label and diagnostics for one independent robot query."""

    motion_preferences: np.ndarray
    valid: bool
    failure_reason: str
    diagnostics: Mapping[str, object]


class PureMotionTeacher:
    """One shared-budget orientation-aware search preserving root provenance."""

    def __init__(
        self,
        *,
        k_motion: int = K_MOTION,
        expansion_budget: int = EXPANSION_BUDGET,
    ) -> None:
        if k_motion < 1:
            raise ValueError("k_motion must be positive.")
        if expansion_budget < 1:
            raise ValueError("expansion_budget must be positive.")
        self.k_motion = k_motion
        self.expansion_budget = expansion_budget
        self._cache: Dict[str, PureMotionResult] = {}

    def query(self, query: PureMotionQuery) -> PureMotionResult:  # noqa: C901
        query_hash = self._query_hash(query)
        cached = self._cache.get(query_hash)
        if cached is not None:
            return self._with_cache_hit(cached)
        started = perf_counter()
        invalid = self._preflight_failure(query)
        if invalid is not None:
            result = self._invalid_result(query_hash, invalid, 0, False, started)
            self._cache[query_hash] = result
            return result

        static_distances = self._static_distances(query)
        start_distance = static_distances.get(query.own_pose)
        if start_distance is None:
            result = self._invalid_result(
                query_hash, "static_unreachable", 0, False, started
            )
            self._cache[query_hash] = result
            return result

        root_status = ["unsupported_action", "physical_illegal", "physical_illegal",
                       "physical_illegal", "unsupported_action"]
        root_costs: Dict[str, float] = {}
        root_details: Dict[str, tuple[int, int, int]] = {}
        frontier = []
        best_cost = {}
        for action in _ACTION_ORDER:
            action_index = _ACTION_INDEX[action]
            if not query.pure_motion_mask[action_index]:
                continue
            successor = self._transition(
                query, query.own_pose, query.orientation, action
            )
            if successor is None:
                continue
            position, direction = successor
            state = position[0], position[1], direction, 1, action
            key = position[0], position[1], direction, 1, action
            best_cost[key] = 1
            h_value = static_distances.get(position, float("inf"))
            heapq.heappush(
                frontier,
                self._priority_tuple(1, h_value, action, position, direction, state),
            )
            root_status[action_index] = "search_exhausted"

        if not frontier:
            result = self._invalid_result(
                query_hash, "no_physical_root_action", 0, False, started, root_status
            )
            self._cache[query_hash] = result
            return result

        expanded_nodes = 0
        saw_progress_candidate = False
        budget_exhausted = False
        while frontier and expanded_nodes < self.expansion_budget:
            _, _, _, _, _, _, _, state = heapq.heappop(frontier)
            x, y, direction, depth, root_action = state
            key = x, y, direction, depth, root_action
            if best_cost.get(key) != depth or root_action in root_costs:
                continue
            expanded_nodes += 1
            position = x, y
            end_distance = static_distances.get(position, float("inf"))
            if position == query.goal or depth == self.k_motion:
                if end_distance < start_distance:
                    root_costs[root_action] = float(depth + end_distance)
                    root_status[_ACTION_INDEX[root_action]] = "finite"
                    root_details[root_action] = (
                        depth,
                        int(end_distance),
                        int(start_distance - end_distance),
                    )
                else:
                    saw_progress_candidate = True
                continue
            for action in _ACTION_ORDER:
                successor = self._transition(query, position, direction, action)
                if successor is None:
                    continue
                next_position, next_direction = successor
                next_depth = depth + 1
                next_key = (
                    next_position[0],
                    next_position[1],
                    next_direction,
                    next_depth,
                    root_action,
                )
                if next_key in best_cost and best_cost[next_key] <= next_depth:
                    continue
                best_cost[next_key] = next_depth
                h_value = static_distances.get(next_position, float("inf"))
                next_state = (*next_key,)
                heapq.heappush(
                    frontier,
                    self._priority_tuple(
                        next_depth,
                        h_value,
                        root_action,
                        next_position,
                        next_direction,
                        next_state,
                    ),
                )
        if frontier and expanded_nodes >= self.expansion_budget:
            budget_exhausted = True
            for action in _ACTION_ORDER:
                if (
                    action not in root_costs
                    and root_status[_ACTION_INDEX[action]] == "search_exhausted"
                ):
                    root_status[_ACTION_INDEX[action]] = "budget_exceeded"

        if not root_costs:
            reason = "budget_exceeded" if budget_exhausted else (
                "no_progress_trajectory"
                if saw_progress_candidate
                else "search_exhausted"
            )
            result = self._invalid_result(
                query_hash,
                reason,
                expanded_nodes,
                budget_exhausted,
                started,
                root_status,
            )
            self._cache[query_hash] = result
            return result

        result = self._valid_result(
            query_hash,
            root_costs,
            root_details,
            root_status,
            expanded_nodes,
            budget_exhausted,
            started,
        )
        self._cache[query_hash] = result
        return result

    def _preflight_failure(self, query: PureMotionQuery) -> str | None:
        if query.dead:
            return "dead"
        if query.picking_lock:
            return "picking_lock"
        if query.mandatory_toggle_load:
            return "mandatory_toggle_load"
        if not self._in_bounds(query, query.own_pose) or not self._in_bounds(
            query, query.goal
        ):
            return "invalid_goal"
        if query.goal in set(query.blocked_coordinates):
            return "invalid_goal"
        if query.own_pose == query.goal:
            return "already_at_goal"
        if query.orientation not in _DIRECTION_INDEX:
            return "non_finite_output"
        if len(query.pure_motion_mask) != 5:
            return "non_finite_output"
        return None

    def _static_distances(self, query: PureMotionQuery) -> Dict[Tuple[int, int], int]:
        blocked = set(query.blocked_coordinates)
        distances = {query.goal: 0}
        frontier = [query.goal]
        while frontier:
            point = frontier.pop(0)
            for neighbour in self._static_neighbours(query, point, blocked):
                if neighbour not in distances:
                    distances[neighbour] = distances[point] + 1
                    frontier.append(neighbour)
        return distances

    def _static_neighbours(
        self,
        query: PureMotionQuery,
        point: Tuple[int, int],
        blocked: set[Tuple[int, int]],
    ) -> Iterable[Tuple[int, int]]:
        for direction in _DIRECTION_ORDER:
            dx, dy = _DELTAS[direction]
            candidate = point[0] + dx, point[1] + dy
            if self._in_bounds(query, candidate) and candidate not in blocked:
                yield candidate

    def _transition(
        self,
        query: PureMotionQuery,
        position: Tuple[int, int],
        direction: str,
        action: str,
    ) -> Tuple[Tuple[int, int], str] | None:
        if action == "LEFT":
            return position, _DIRECTION_ORDER[(_DIRECTION_INDEX[direction] - 1) % 4]
        if action == "RIGHT":
            return position, _DIRECTION_ORDER[(_DIRECTION_INDEX[direction] + 1) % 4]
        dx, dy = _DELTAS[direction]
        candidate = position[0] + dx, position[1] + dy
        if (
            not self._in_bounds(query, candidate)
            or candidate in set(query.blocked_coordinates)
            or candidate in set(query.occupied_coordinates)
        ):
            return None
        return candidate, direction

    def _priority_tuple(self, depth, h_value, root_action, position, direction, state):
        return (
            depth + h_value,
            h_value,
            -depth,
            _ACTION_ORDER.index(root_action),
            position[0],
            position[1],
            _DIRECTION_INDEX[direction],
            state,
        )

    def _valid_result(
        self,
        query_hash: str,
        root_costs: Mapping[str, float],
        root_details: Mapping[str, tuple[int, int, int]],
        root_status: list[str],
        expanded_nodes: int,
        budget_exhausted: bool,
        started: float,
    ) -> PureMotionResult:
        finite_costs = np.asarray(list(root_costs.values()), dtype=np.float64)
        lower, upper = float(finite_costs.min()), float(finite_costs.max())
        preferences = np.zeros(5, dtype=np.float64)
        normalized = [None] * 5
        for action, cost in root_costs.items():
            scaled = 0.0 if upper == lower else (cost - lower) / (upper - lower)
            normalized[_ACTION_INDEX[action]] = scaled
            preferences[_ACTION_INDEX[action]] = np.exp(-scaled / TAU_MOTION)
        preferences /= preferences.sum()
        if not np.all(np.isfinite(preferences)) or not np.isclose(
            preferences.sum(), 1.0
        ):
            return self._invalid_result(query_hash, "non_finite_output", expanded_nodes,
                                        budget_exhausted, started, root_status)
        raw_costs = [None] * 5
        path_lengths = [None] * 5
        h_ends = [None] * 5
        progress = [None] * 5
        for action, cost in root_costs.items():
            index = _ACTION_INDEX[action]
            raw_costs[index] = cost
            path_lengths[index], h_ends[index], progress[index] = root_details[action]
        preferred = min(
            root_costs,
            key=lambda action: (root_costs[action], _ACTION_ORDER.index(action)),
        )
        diagnostics = self._diagnostics(
            query_hash, expanded_nodes, budget_exhausted, root_status, raw_costs,
            normalized, path_lengths, h_ends, progress, preferred, started,
        )
        return PureMotionResult(preferences.astype(np.float32), True, "ok", diagnostics)

    def _invalid_result(self, query_hash, reason, expanded_nodes, budget_exhausted,
                        started, root_status=None) -> PureMotionResult:
        statuses = root_status or ["unsupported_action"] * 5
        diagnostics = self._diagnostics(
            query_hash, expanded_nodes, budget_exhausted, statuses, [None] * 5,
            [None] * 5, [None] * 5, [None] * 5, [None] * 5, None, started,
        )
        return PureMotionResult(
            np.zeros(5, dtype=np.float32), False, reason, diagnostics
        )

    def _diagnostics(self, query_hash, expanded_nodes, budget_exhausted, root_status,
                     raw_costs, normalized, path_lengths, h_ends, progress, preferred,
                     started) -> dict:
        return {
            "teacher_version": TEACHER_VERSION,
            "query_hash": query_hash,
            "cache_hit": False,
            "expanded_nodes": expanded_nodes,
            "planning_time_ms": (perf_counter() - started) * 1000.0,
            "root_costs_raw": raw_costs,
            "root_costs_normalized": normalized,
            "root_status": root_status,
            "root_path_length_actions": path_lengths,
            "root_h_static_end": h_ends,
            "root_progress_delta": progress,
            "feasible_root_actions": [value == "finite" for value in root_status],
            "preferred_root_action": preferred,
            "budget_exhausted": budget_exhausted,
            "non_finite_detected": False,
            "K_motion": self.k_motion,
            "expansion_budget": self.expansion_budget,
            "cost_normalization": "minmax-v1",
            "tau_motion": TAU_MOTION,
        }

    def _query_hash(self, query: PureMotionQuery) -> str:
        payload = {
            "layout_hash": query.layout_hash,
            "own_pose": query.own_pose,
            "orientation": query.orientation,
            "resolved_goal": query.goal,
            "occupied_coordinates": sorted(set(query.occupied_coordinates)),
            "pure_motion_mask": query.pure_motion_mask,
            "K_motion": self.k_motion,
            "expansion_budget": self.expansion_budget,
            "cost_normalization": "minmax-v1",
            "tau_motion": TAU_MOTION,
            "teacher_version": TEACHER_VERSION,
            "footprint_class": query.footprint_class,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _in_bounds(query: PureMotionQuery, point: Tuple[int, int]) -> bool:
        return 0 <= point[0] < query.width and 0 <= point[1] < query.height

    @staticmethod
    def _with_cache_hit(result: PureMotionResult) -> PureMotionResult:
        diagnostics = dict(result.diagnostics)
        diagnostics["cache_hit"] = True
        return PureMotionResult(result.motion_preferences.copy(), result.valid,
                                result.failure_reason, diagnostics)
