"""A* path teacher for the orientation-based RWARE action space."""

import heapq
from collections import defaultdict
from typing import Dict, Optional, Sequence, Set, Tuple

from llm_mappo.types import PathPlan, PlannerEvent
from rware.warehouse import Action, Direction


_DIRECTIONS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


class ReservationTable:
    """Time-indexed cell and directed-edge reservations for prioritized A*."""

    def __init__(self, horizon: int):
        if horizon < 1:
            raise ValueError("reservation horizon must be positive.")
        self.horizon = horizon
        self.cells: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        self.edges: Dict[int, Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = (
            defaultdict(set)
        )

    def reserve(self, path: Sequence[Tuple[int, int]]) -> None:
        if not path:
            raise ValueError("cannot reserve an empty path.")
        expanded = list(path)
        expanded.extend([expanded[-1]] * (self.horizon + 1 - len(expanded)))
        for time, position in enumerate(expanded[:self.horizon + 1]):
            self.cells[time].add(position)
            if time:
                self.edges[time].add((expanded[time - 1], position))

    def is_available(self, position: Tuple[int, int], time: int) -> bool:
        return position not in self.cells[time]

    def permits_edge(
        self, start: Tuple[int, int], end: Tuple[int, int], time: int
    ) -> bool:
        return (end, start) not in self.edges[time]


class AStarPlanner:
    """Grid A* that returns waypoints and a smooth five-action preference."""

    def plan(self, env, agent_id: int, goal: Tuple[int, int]) -> PathPlan:
        agent = env.agents[agent_id - 1]
        start = (agent.x, agent.y)
        blocked = self._blocked_cells(env, agent_id)
        blocked.discard(start)
        blocked.discard(goal)
        path = self._search(start, goal, env.grid_size, blocked)
        if path is None:
            return PathPlan((), self._noop_preferences(), PlannerEvent.BLOCKED)
        return PathPlan(tuple(path), self.action_preferences(agent.dir, path), None)

    def plan_with_reservations(
        self,
        env,
        agent_id: int,
        goal: Tuple[int, int],
        reservations: ReservationTable,
    ) -> PathPlan:
        """Plan a path that avoids all earlier AGVs' time-indexed reservations."""
        agent = env.agents[agent_id - 1]
        start = (agent.x, agent.y)
        blocked = self._static_blocked_cells(env, agent_id)
        blocked.discard(start)
        blocked.discard(goal)
        path = self._temporal_search(
            start, goal, env.grid_size, blocked, reservations
        )
        if path is None:
            return PathPlan((), self._noop_preferences(), PlannerEvent.BLOCKED)
        preferences = self._preferences_for_timed_path(agent.dir, path)
        return PathPlan(tuple(path), preferences, None)

    def action_preferences(
        self, direction: Direction, path: Sequence[Tuple[int, int]]
    ) -> Tuple[float, float, float, float, float]:
        if len(path) < 2:
            return (0.05, 0.05, 0.02, 0.82, 0.06)
        current, next_point = path[0], path[1]
        desired = self._direction_between(current, next_point)
        if desired == direction:
            preferred = Action.FORWARD
        elif self._turn(direction, right=False) == desired:
            preferred = Action.LEFT
        else:
            preferred = Action.RIGHT
        return self._smooth_preferences(preferred)

    def status_for_progress(
        self, unchanged_steps: int, all_actions_blocked: bool
    ) -> Optional[PlannerEvent]:
        if all_actions_blocked:
            return PlannerEvent.BLOCKED
        if unchanged_steps >= 30:
            return PlannerEvent.STALLED
        return None

    def _blocked_cells(self, env, agent_id: int) -> Set[Tuple[int, int]]:
        blocked = self._static_blocked_cells(env, agent_id)
        blocked.update(
            (agent.x, agent.y)
            for agent in env.agents
            if agent.id != agent_id
        )
        return blocked

    @staticmethod
    def _static_blocked_cells(env, agent_id: int) -> Set[Tuple[int, int]]:
        blocked = set()
        agent = env.agents[agent_id - 1]
        if agent.carrying_shelf:
            blocked.update((shelf.x, shelf.y) for shelf in env.shelfs)
        return blocked

    def _search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_size: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
    ) -> Optional[Sequence[Tuple[int, int]]]:
        frontier = [(0, start)]
        came_from = {start: None}
        cost = {start: 0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct(came_from, current)
            for neighbour in self._neighbours(current, grid_size):
                if neighbour in blocked:
                    continue
                new_cost = cost[current] + 1
                if neighbour not in cost or new_cost < cost[neighbour]:
                    cost[neighbour] = new_cost
                    priority = new_cost + self._manhattan(neighbour, goal)
                    heapq.heappush(frontier, (priority, neighbour))
                    came_from[neighbour] = current
        return None

    def _temporal_search(
        self,
        start,
        goal,
        grid_size,
        blocked,
        reservations: ReservationTable,
    ) -> Optional[Sequence[Tuple[int, int]]]:
        frontier = [(0, 0, start)]
        came_from = {(start, 0): None}
        cost = {(start, 0): 0}
        best_state = (start, 0)
        best_distance = self._manhattan(start, goal)
        while frontier:
            _, time, current = heapq.heappop(frontier)
            state = current, time
            if current == goal:
                return self._reconstruct_timed(came_from, state)
            distance = self._manhattan(current, goal)
            if distance < best_distance:
                best_state = state
                best_distance = distance
            if time >= reservations.horizon:
                continue
            for next_state in self._temporal_neighbours(
                current, time, grid_size, blocked, reservations
            ):
                candidate, next_time = next_state
                new_cost = cost[state] + 1
                if next_state not in cost or new_cost < cost[next_state]:
                    cost[next_state] = new_cost
                    priority = new_cost + self._manhattan(candidate, goal)
                    heapq.heappush(frontier, (priority, next_time, candidate))
                    came_from[next_state] = state
        if best_state != (start, 0):
            return self._reconstruct_timed(came_from, best_state)
        return None

    def _temporal_neighbours(
        self, current, time, grid_size, blocked, reservations: ReservationTable
    ):
        next_time = time + 1
        candidates = list(self._neighbours(current, grid_size)) + [current]
        for candidate in candidates:
            if candidate in blocked:
                continue
            if not reservations.is_available(candidate, next_time):
                continue
            if reservations.permits_edge(current, candidate, next_time):
                yield candidate, next_time

    @staticmethod
    def _reconstruct_timed(came_from, state):
        path = []
        while state is not None:
            path.append(state[0])
            state = came_from[state]
        path.reverse()
        return path

    def expand_for_orientation(
        self, direction: Direction, path: Sequence[Tuple[int, int]]
    ) -> Tuple[Tuple[int, int], ...]:
        """Turn spatial moves into time steps that respect the action semantics."""
        if not path:
            return ()
        expanded = [path[0]]
        current_direction = direction
        for current, next_point in zip(path, path[1:]):
            if current == next_point:
                expanded.append(current)
                continue
            desired = self._direction_between(current, next_point)
            while current_direction != desired:
                expanded.append(current)
                turn_right = self._turn(current_direction, right=False) != desired
                current_direction = self._turn(current_direction, right=turn_right)
            expanded.append(next_point)
        return tuple(expanded)

    def _preferences_for_timed_path(self, direction, path):
        if len(path) < 2 or path[0] == path[1]:
            return self._noop_preferences()
        return self.action_preferences(direction, path[:2])

    @staticmethod
    def _neighbours(point, grid_size):
        x, y = point
        height, width = grid_size
        for dx, dy in _DIRECTIONS.values():
            candidate = x + dx, y + dy
            if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                yield candidate

    @staticmethod
    def _reconstruct(came_from, current):
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    @staticmethod
    def _manhattan(first, second):
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    @staticmethod
    def _direction_between(first, second):
        delta = second[0] - first[0], second[1] - first[1]
        for direction, vector in _DIRECTIONS.items():
            if vector == delta:
                return direction
        raise ValueError("A* waypoints must be adjacent.")

    @staticmethod
    def _turn(direction: Direction, right: bool) -> Direction:
        order = (Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT)
        offset = 1 if right else -1
        return order[(order.index(direction) + offset) % len(order)]

    @staticmethod
    def _smooth_preferences(
        preferred: Action,
    ) -> Tuple[float, float, float, float, float]:
        values = [0.05, 0.05, 0.05, 0.05, 0.05]
        values[preferred.value] = 0.82
        values[Action.NOOP.value] = 0.03
        return tuple(values)

    @staticmethod
    def _noop_preferences() -> Tuple[float, float, float, float, float]:
        return (0.05, 0.05, 0.05, 0.05, 0.80)
