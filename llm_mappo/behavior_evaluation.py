"""Natural-rollout behavior groups for Phase 3 and later policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

from llm_mappo.phase2 import Phase2Warehouse
from rware.warehouse import Action, Direction


Position = Tuple[int, int]


@dataclass(frozen=True)
class AgentState:
    """Minimal pre-action state used to classify a policy decision."""

    agent_id: int
    position: Position
    direction: Direction
    battery: float
    loaded: bool
    priority: str | None
    target: Position
    target_kind: str


def evaluate_behavior_groups(
    policy,
    config: Dict[str, object],
    seeds: Iterable[int],
    episodes_per_seed: int = 5,
) -> dict:
    """Evaluate naturally occurring safety and priority decisions.

    This evaluator does not modify reset states or inject a curriculum.  A
    zero sample count means a behavior did not occur in the supplied rollout;
    it is intentionally reported as uncovered instead of a passing score.
    """
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive.")
    env = _environment_from_config(config)
    groups = {
        "narrow_corridor_yielding": _new_group(),
        "priority_intersection_passage": _new_group(),
        "low_battery_charging_diversion": _new_group(),
    }
    episodes = 0
    try:
        for seed in seeds:
            for offset in range(episodes_per_seed):
                observations = env.reset(seed=seed * 10_000 + offset)
                while True:
                    before = _agent_states(env)
                    actions = policy.act(
                        observations, env.action_masks(), deterministic=True
                    )[0]
                    _record_corridor_yielding(env, before, actions, groups)
                    _record_priority_intersection(env, before, actions, groups)
                    transition = env.step(actions)
                    _record_charging_diversion(
                        env, before, transition.info, groups
                    )
                    observations = transition.observations
                    if (
                        transition.terminated
                        or transition.truncated
                        or transition.metrics.deadlocked
                    ):
                        episodes += 1
                        break
    finally:
        env.close()
    return {
        "episodes": episodes,
        "natural_rollout_only": True,
        "groups": {
            name: _finalize_group(group) for name, group in groups.items()
        },
    }


def _environment_from_config(config: Dict[str, object]) -> Phase2Warehouse:
    batch_size_range = config.get("batch_size_range")
    priority_schedule = config.get("priority_schedule")
    return Phase2Warehouse(
        n_agents=int(config.get("n_agents", 3)),
        max_steps=int(config["max_steps"]),
        env_id=str(config["env_id"]),
        charge_threshold=float(config.get("charge_threshold", 0.2)),
        waypoint_reward=float(config.get("waypoint_reward", 0.01)),
        oracle_interaction_mask=bool(config.get("oracle_interaction_mask", True)),
        deadlock_steps=int(config.get("deadlock_steps", 180)),
        priority_schedule=(
            tuple(priority_schedule) if priority_schedule else None
        ),
        batch_interval=config.get("batch_interval"),
        batch_size_range=(
            tuple(batch_size_range) if batch_size_range is not None else None
        ),
        initial_priority_label=str(config.get("initial_priority_label", "B")),
        request_queue_size=config.get("request_queue_size"),
        task_completion_target=config.get("task_completion_target"),
        include_priority_features=True,
    )


def _agent_states(env: Phase2Warehouse) -> tuple[AgentState, ...]:
    states = []
    for agent in env.env.agents:
        task = env.env.task_queue.task_for_agent(agent.id)
        target, target_kind = env._target_for_agent(agent.id)
        states.append(
            AgentState(
                agent_id=agent.id,
                position=(agent.x, agent.y),
                direction=agent.dir,
                battery=float(agent.battery),
                loaded=agent.carrying_shelf is not None,
                priority=task.label[0] if task is not None else None,
                target=target,
                target_kind=target_kind,
            )
        )
    return tuple(states)


def _record_corridor_yielding(
    env: Phase2Warehouse,
    states: Sequence[AgentState],
    actions: Sequence[int],
    groups: dict,
) -> None:
    for first, second in _pairs(states):
        if first.loaded == second.loaded:
            continue
        empty, loaded = (first, second) if not first.loaded else (second, first)
        if not (
            _is_corridor_cell(env, empty.position)
            and _is_corridor_cell(env, loaded.position)
            and _facing_each_other(empty, loaded)
            and _manhattan(empty.position, loaded.position) <= 2
        ):
            continue
        empty_action = Action(int(actions[empty.agent_id - 1]))
        loaded_action = Action(int(actions[loaded.agent_id - 1]))
        group = groups["narrow_corridor_yielding"]
        group["samples"] += 1
        group["successes"] += int(
            empty_action in (Action.NOOP, Action.LEFT, Action.RIGHT)
            and loaded_action == Action.FORWARD
        )


def _record_priority_intersection(
    env: Phase2Warehouse,
    states: Sequence[AgentState],
    actions: Sequence[int],
    groups: dict,
) -> None:
    for first, second in _pairs(states):
        if (
            not first.priority
            or not second.priority
            or first.priority == second.priority
        ):
            continue
        common = _shared_adjacent_intersection(env, first.position, second.position)
        if common is None:
            continue
        high, low = (
            (first, second) if first.priority < second.priority else (second, first)
        )
        high_action = Action(int(actions[high.agent_id - 1]))
        low_action = Action(int(actions[low.agent_id - 1]))
        group = groups["priority_intersection_passage"]
        group["samples"] += 1
        group["successes"] += int(
            high_action == Action.FORWARD and low_action != Action.FORWARD
        )


def _record_charging_diversion(
    env: Phase2Warehouse,
    before: Sequence[AgentState],
    info: dict,
    groups: dict,
) -> None:
    states_after = _agent_states(env)
    threshold = env.charge_threshold
    for prior, current in zip(before, states_after):
        if not (
            prior.loaded
            and prior.battery < threshold
            and prior.target_kind == "charging"
        ):
            continue
        prior_distance = _manhattan(prior.position, prior.target)
        next_distance = _manhattan(current.position, prior.target)
        group = groups["low_battery_charging_diversion"]
        group["samples"] += 1
        group["successes"] += int(next_distance < prior_distance)
        if any(
            event["type"] == "charged" and event["agent_id"] == prior.agent_id
            for event in info["events"]
        ):
            group["charged_events"] += 1


def _new_group() -> dict:
    return {"samples": 0, "successes": 0, "charged_events": 0}


def _finalize_group(group: dict) -> dict:
    samples = int(group["samples"])
    return {
        "samples": samples,
        "successes": int(group["successes"]),
        "rate": float(group["successes"] / samples) if samples else None,
        "charged_events": int(group["charged_events"]),
        "covered": bool(samples),
    }


def _pairs(items: Sequence[AgentState]):
    for index, first in enumerate(items):
        for second in items[index + 1 :]:
            yield first, second


def _is_corridor_cell(env: Phase2Warehouse, position: Position) -> bool:
    return len(_highway_neighbours(env, position)) == 2


def _shared_adjacent_intersection(
    env: Phase2Warehouse, first: Position, second: Position
) -> Position | None:
    common = set(_highway_neighbours(env, first)) & set(_highway_neighbours(env, second))
    intersections = [
        point for point in common if len(_highway_neighbours(env, point)) >= 3
    ]
    return intersections[0] if len(intersections) == 1 else None


def _highway_neighbours(env: Phase2Warehouse, position: Position) -> list[Position]:
    x, y = position
    warehouse = env.env
    return [
        (nx, ny)
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y))
        if 0 <= nx < warehouse.grid_size[1]
        and 0 <= ny < warehouse.grid_size[0]
        and warehouse.highways[ny, nx]
    ]


def _facing_each_other(first: AgentState, second: AgentState) -> bool:
    return (
        _next_position(first.position, first.direction) == second.position
        and _next_position(second.position, second.direction) == first.position
    )


def _next_position(position: Position, direction: Direction) -> Position:
    dx, dy = {
        Direction.UP: (0, -1),
        Direction.DOWN: (0, 1),
        Direction.LEFT: (-1, 0),
        Direction.RIGHT: (1, 0),
    }[direction]
    return position[0] + dx, position[1] + dy


def _manhattan(first: Position, second: Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])
