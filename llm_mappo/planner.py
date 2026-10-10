"""A* path teacher for the orientation-based RWARE action space."""

import heapq
from dataclasses import dataclass
from itertools import count
from time import perf_counter
from typing import Optional, Sequence, Set, Tuple

from llm_mappo.types import PathPlan, PlannerEvent
from rware.warehouse import Action, Direction


_DIRECTIONS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}
_DIRECTION_ORDER = (
    Direction.UP,
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
)
_DIRECTION_INDEX = {direction: index for index, direction in enumerate(_DIRECTION_ORDER)}


@dataclass(frozen=True)
class TemporalSearchResult:
    """Internal time-expanded plan used by prioritized A*."""

    timed_positions: Tuple[Tuple[int, int], ...]
    actions: Tuple[Action, ...]
    reached_goal: bool
    failure_reason: Optional[str]
    expanded_nodes: int

    @property
    def first_action(self) -> Action:
        return self.actions[0] if self.actions else Action.NOOP

    @property
    def waypoints(self) -> Tuple[Tuple[int, int], ...]:
        collapsed = []
        for position in self.timed_positions:
            if not collapsed or position != collapsed[-1]:
                collapsed.append(position)
        return tuple(collapsed)


class ReservationTable:
    """Time-indexed cell and directed-edge reservations for prioritized A*."""

    def __init__(self, horizon: int):
        if horizon < 1:
            raise ValueError("reservation horizon must be positive.")
        self.horizon = horizon
        self.cells: list[Set[Tuple[int, int]]] = [
            set() for _ in range(horizon + 1)
        ]
        self.edges: list[Set[Tuple[Tuple[int, int], Tuple[int, int]]]] = [
            set() for _ in range(horizon + 1)
        ]
        self.terminal_cells: list[Set[Tuple[int, int]]] = [
            set() for _ in range(horizon + 1)
        ]
        self.terminal_conflicts = 0

    def reserve(
        self,
        path: Sequence[Tuple[int, int]],
        terminal_hold_steps: int = 2,
        persistent: bool = False,
    ) -> None:
        if not path:
            raise ValueError("cannot reserve an empty path.")
        if terminal_hold_steps < 0:
            raise ValueError("terminal hold steps cannot be negative.")
        last_index = min(len(path) - 1, self.horizon)
        final_time = (
            self.horizon
            if persistent
            else min(self.horizon, last_index + terminal_hold_steps)
        )
        previous = path[0]
        for time in range(final_time + 1):
            position = path[min(time, last_index)]
            self.cells[time].add(position)
            if time > last_index:
                self.terminal_cells[time].add(position)
            if time:
                self.edges[time].add((previous, position))
            previous = position

    def is_available(self, position: Tuple[int, int], time: int) -> bool:
        return time <= self.horizon and position not in self.cells[time]

    def permits_edge(
        self, start: Tuple[int, int], end: Tuple[int, int], time: int
    ) -> bool:
        return time <= self.horizon and (end, start) not in self.edges[time]

    def permits_transition(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        start_time: int,
        step_cost: int,
    ) -> bool:
        """Check all turn/wait cells and the final directed edge in one pass."""
        arrival = start_time + step_cost
        if arrival > self.horizon:
            return False
        for time in range(start_time + 1, arrival):
            if start in self.cells[time]:
                return False
        if end in self.cells[arrival]:
            if end in self.terminal_cells[arrival]:
                self.terminal_conflicts += 1
            return False
        return (end, start) not in self.edges[arrival]


class AStarPlanner:
    """Grid A* that returns waypoints and a smooth five-action preference."""

    def __init__(self):
        self._oriented_neighbour_cache = {}

    def plan(self, env, agent_id: int, goal: Tuple[int, int]) -> PathPlan:
        agent = env.agents[agent_id - 1]
        start = (agent.x, agent.y)
        blocked = self._blocked_cells(env, agent_id)
        blocked.discard(start)
        blocked.discard(goal)
        path = self._search(start, agent.dir, goal, env.grid_size, blocked)
        if path is None:
            return PathPlan(
                (),
                self._noop_preferences(),
                PlannerEvent.BLOCKED,
                failure_reason="topology_blocked",
            )
        return PathPlan(
            tuple(path),
            self.action_preferences(agent.dir, path),
            reached_goal=True,
        )

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
        started = perf_counter()
        result = self._temporal_search(
            start, agent.dir, goal, env.grid_size, blocked, reservations
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        static_reachable = True
        if not result.reached_goal:
            static_reachable = self._search(
                start, agent.dir, goal, env.grid_size, blocked
            ) is not None
        failure_reason = result.failure_reason
        if not result.reached_goal and not static_reachable:
            failure_reason = "topology_blocked"
        event = None if result.reached_goal else PlannerEvent.REPLAN_REQUIRED
        if not result.timed_positions:
            event = PlannerEvent.BLOCKED
        return PathPlan(
            waypoints=result.waypoints,
            action_preferences=self._preferences_for_action(
                result.first_action
            ),
            event=event,
            timed_positions=result.timed_positions,
            first_action=result.first_action.value,
            reached_goal=result.reached_goal,
            failure_reason=failure_reason,
            expanded_nodes=result.expanded_nodes,
            planning_time_ms=elapsed_ms,
            reservation_false_no_path=(
                static_reachable and not result.reached_goal
            ),
        )

    def action_preferences(
        self, direction: Direction, path: Sequence[Tuple[int, int]]
    ) -> Tuple[float, float, float, float, float]:
        if len(path) < 2:
            return self._noop_preferences()
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
        start_dir: Direction,
        goal: Tuple[int, int],
        grid_size: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
    ) -> Optional[Sequence[Tuple[int, int]]]:
        """Orientation-aware A* that counts LEFT/RIGHT turns in g(n) and h(n).

        Search state is ``(x, y, direction)``. Moving forward costs 1 step; a
        90-degree turn costs 1 extra step. The heuristic adds the minimum turns
        needed to face the goal, which stays admissible (never overestimates)
        while pruning turn-heavy detours early.
        """
        start_state = (start[0], start[1], start_dir)
        sequence = count()
        frontier = [(0, next(sequence), start_state)]
        came_from = {start_state: None}
        cost = {start_state: 0}
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if (current[0], current[1]) == goal:
                return self._reconstruct_oriented(came_from, current)
            cx, cy, cdir = current
            for neighbour, ndir, step_cost in self._oriented_neighbours(
                (cx, cy), cdir, grid_size
            ):
                if neighbour in blocked:
                    continue
                next_state = (neighbour[0], neighbour[1], ndir)
                new_cost = cost[current] + step_cost
                if next_state not in cost or new_cost < cost[next_state]:
                    cost[next_state] = new_cost
                    priority = new_cost + self._heuristic(neighbour, ndir, goal)
                    heapq.heappush(frontier, (priority, next(sequence), next_state))
                    came_from[next_state] = current
        return None

    def _temporal_search(  # noqa: C901
        self,
        start: Tuple[int, int],
        start_dir: Direction,
        goal: Tuple[int, int],
        grid_size: Tuple[int, int],
        blocked: Set[Tuple[int, int]],
        reservations: ReservationTable,
    ) -> TemporalSearchResult:
        """Orientation-aware temporal A* over executable one-step actions.

        State is ``(position, direction, time)``. Forward moves cost 1 time
        step; LEFT, RIGHT, and a true NOOP wait each consume one step in place.
        This preserves the exact action timing used by the environment.
        """
        start_key = (start, start_dir, 0)
        sequence = count()
        frontier = [(0, next(sequence), 0, start, start_dir)]
        came_from = {start_key: None}
        cost = {start_key: 0}
        best_key = start_key
        best_distance = self._manhattan(start, goal)
        first_wait_key = None
        expanded_nodes = 0
        while frontier:
            _, _, time, current, cdir = heapq.heappop(frontier)
            state_key = (current, cdir, time)
            expanded_nodes += 1
            if current == goal:
                positions, actions = self._reconstruct_temporal_plan(
                    came_from, state_key
                )
                return TemporalSearchResult(
                    positions, actions, True, None, expanded_nodes
                )
            distance = self._manhattan(current, goal)
            if distance < best_distance:
                best_key = state_key
                best_distance = distance
            if time >= reservations.horizon:
                continue
            for neighbour, ndir, action in self._temporal_action_neighbours(
                current, cdir, grid_size
            ):
                if neighbour in blocked:
                    continue
                if not reservations.permits_transition(
                    current, neighbour, time, 1
                ):
                    continue
                next_time = time + 1
                next_key = (neighbour, ndir, next_time)
                new_cost = cost[state_key] + 1
                if next_key not in cost or new_cost < cost[next_key]:
                    cost[next_key] = new_cost
                    priority = new_cost + self._heuristic(
                        neighbour, ndir, goal
                    )
                    heapq.heappush(
                        frontier,
                        (priority, next(sequence), next_time, neighbour, ndir),
                    )
                    came_from[next_key] = state_key, action
                    if action == Action.NOOP and state_key == start_key:
                        first_wait_key = next_key
        if best_key != start_key:
            fallback_key = best_key
        else:
            fallback_key = first_wait_key
        if fallback_key is not None:
            positions, actions = self._reconstruct_temporal_plan(
                came_from, fallback_key
            )
            return TemporalSearchResult(
                positions,
                actions,
                False,
                "horizon_exhausted",
                expanded_nodes,
            )
        return TemporalSearchResult(
            (), (), False, "reservation_blocked", expanded_nodes
        )

    def _temporal_action_neighbours(
        self,
        point: Tuple[int, int],
        direction: Direction,
        grid_size: Tuple[int, int],
    ):
        """Yield exact one-step environment transitions in deterministic order."""
        dx, dy = _DIRECTIONS[direction]
        candidate = point[0] + dx, point[1] + dy
        height, width = grid_size
        if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
            yield candidate, direction, Action.FORWARD
        yield point, self._turn(direction, right=False), Action.LEFT
        yield point, self._turn(direction, right=True), Action.RIGHT
        yield point, direction, Action.NOOP

    def _temporal_neighbours(
        self, current, time, grid_size, blocked, reservations: ReservationTable
    ):
        """Deprecated: kept for backward compatibility with tests that call it
        directly. New code should use :meth:`_oriented_neighbours` instead.
        """
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
        """Deprecated: kept for backward compatibility. New temporal search uses
        :meth:`_reconstruct_timed_oriented`.
        """
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

    def _preferences_for_action(self, action: Action):
        if action == Action.NOOP:
            return self._noop_preferences()
        return self._smooth_preferences(action)

    @staticmethod
    def _neighbours(point, grid_size):
        x, y = point
        height, width = grid_size
        for dx, dy in _DIRECTIONS.values():
            candidate = x + dx, y + dy
            if 0 <= candidate[0] < width and 0 <= candidate[1] < height:
                yield candidate

    def _oriented_neighbours(
        self,
        point: Tuple[int, int],
        direction: Direction,
        grid_size: Tuple[int, int],
    ):
        """Yield ``(neighbour, new_direction, step_cost)`` for every reachable
        adjacent cell plus an in-place turn.

        ``step_cost`` is 1 for a forward move (no turn) or ``1 + turn_steps``
        for a move that requires turning first (turn steps + the forward step).
        A pure in-place turn is yielded as ``(point, turned_dir, turn_steps)``
        so the search can still explore rotational shortcuts when boxed in.
        """
        key = point, direction, grid_size
        cached = self._oriented_neighbour_cache.get(key)
        if cached is not None:
            return cached
        x, y = point
        height, width = grid_size
        neighbours = []
        for target_dir, (dx, dy) in _DIRECTIONS.items():
            candidate = x + dx, y + dy
            if not (0 <= candidate[0] < width and 0 <= candidate[1] < height):
                continue
            turn_steps = self._turn_steps(direction, target_dir)
            step_cost = 1 + turn_steps  # turns + 1 forward
            neighbours.append((candidate, target_dir, step_cost))
        # In-place turn option (useful when surrounded but need to reorient)
        left = self._turn(direction, right=False)
        right = self._turn(direction, right=True)
        neighbours.append((point, left, 1))
        if right != left:  # 4-direction system: right != left
            neighbours.append((point, right, 1))
        result = tuple(neighbours)
        self._oriented_neighbour_cache[key] = result
        return result

    @staticmethod
    def _turn_steps(src: Direction, dst: Direction) -> int:
        """Minimum LEFT/RIGHT actions to rotate from ``src`` to ``dst``.

        In the 4-direction cycle UP→RIGHT→DOWN→LEFT→UP the maximum is 2 (a
        180-degree U-turn requires two turns).
        """
        if src == dst:
            return 0
        diff = abs(_DIRECTION_INDEX[src] - _DIRECTION_INDEX[dst])
        return min(diff, len(_DIRECTION_ORDER) - diff)

    def _heuristic(
        self,
        position: Tuple[int, int],
        direction: Direction,
        goal: Tuple[int, int],
    ) -> float:
        """Admissible heuristic = Manhattan distance + minimum turns to face goal.

        The turn lower bound is 0 when already aligned with the dominant axis,
        1 for a 90-degree offset, or 2 when facing away (180 degrees). This
        never overestimates because any real path must cover the Manhattan
        distance *and* perform at least this many turns before moving forward.
        """
        distance = self._manhattan(position, goal)
        if distance == 0:
            return 0
        dx = goal[0] - position[0]
        dy = goal[1] - position[1]
        # Pick the axis with the larger displacement as the "dominant" direction
        # the AGV ultimately needs to travel along.
        if abs(dx) >= abs(dy):
            if dx > 0:
                target_dir = Direction.RIGHT
            elif dx < 0:
                target_dir = Direction.LEFT
            else:  # dx == 0, dy dominates but abs tie-break went here
                target_dir = (
                    Direction.DOWN if dy > 0 else Direction.UP
                )
        else:
            target_dir = (
                Direction.DOWN if dy > 0 else Direction.UP
            )
        return distance + self._turn_steps(direction, target_dir)

    @staticmethod
    def _reconstruct_oriented(came_from, state):
        """Rebuild a spatial path from orientation-aware search state.

        When consecutive states share the same position (in-place turns), only
        the final position is kept, yielding a clean spatial waypoint list.
        """
        path = []
        seen_pos = None
        while state is not None:
            pos = (state[0], state[1])
            if pos != seen_pos:
                path.append(pos)
                seen_pos = pos
            state = came_from[state]
        path.reverse()
        return path

    @staticmethod
    def _reconstruct_timed_oriented(came_from, state):
        """Rebuild a timed spatial path from orientation-aware temporal search.

        State is ``((x, y), direction, time)``; consecutive same-position
        entries (turns/waiting) are collapsed to keep the waypoint list clean
        while preserving the first occurrence so timing is visible if needed.
        """
        path = []
        seen_pos = None
        while state is not None:
            pos = state[0]
            if pos != seen_pos:
                path.append(pos)
                seen_pos = pos
            state = came_from[state]
        path.reverse()
        return path

    @staticmethod
    def _reconstruct_temporal_plan(came_from, state):
        positions = []
        actions = []
        while state is not None:
            positions.append(state[0])
            link = came_from[state]
            if link is None:
                state = None
            else:
                state, action = link
                actions.append(action)
        positions.reverse()
        actions.reverse()
        return tuple(positions), tuple(actions)

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
        offset = 1 if right else -1
        return _DIRECTION_ORDER[
            (_DIRECTION_INDEX[direction] + offset) % len(_DIRECTION_ORDER)
        ]

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
        return (0.80, 0.05, 0.05, 0.05, 0.05)
