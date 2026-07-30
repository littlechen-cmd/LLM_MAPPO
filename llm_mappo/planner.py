"""A* path teacher for the orientation-based RWARE action space."""

import heapq
from typing import Optional, Sequence, Set, Tuple

from llm_mappo.types import PathPlan, PlannerEvent
from rware.warehouse import Action, Direction


_DIRECTIONS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


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
        blocked = {
            (agent.x, agent.y)
            for agent in env.agents
            if agent.id != agent_id
        }
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
