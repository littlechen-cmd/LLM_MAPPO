"""Convert the current DirectGoal rule state into A*KD rollout fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_mappo.mappo_astar_kd.teacher import (
    AStarKDQuery,
    AStarKDTeacher,
    StaticMap,
)


_DELTAS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}


@dataclass(frozen=True)
class TeacherLabels:
    """Axis-preserving labels emitted at the pre-transition environment state."""

    soft_targets: np.ndarray
    root_costs: np.ndarray
    validity: np.ndarray
    supervision_scope: np.ndarray
    attenuation: np.ndarray
    failure_reasons: tuple[str, ...]
    query_count: int
    cache_build_count: int
    cache_hit_count: int


class AStarKDLabeler:
    """One worker-local labeler; its cache never crosses processes."""

    def __init__(self, environment) -> None:
        self.environment = environment
        self.teacher = AStarKDTeacher(_static_map(environment))

    def labels(self, *, use_semantic_validity: bool) -> TeacherLabels:
        warehouse = self.environment.env
        charging_targets = self.environment._charging_targets()
        targets = np.zeros((len(warehouse.agents), 3), dtype=np.float32)
        root_costs = np.full(
            (len(warehouse.agents), 3), np.nan, dtype=np.float32
        )
        validity = np.zeros(len(warehouse.agents), dtype=bool)
        supervision_scope = np.zeros(len(warehouse.agents), dtype=bool)
        attenuation = np.ones(len(warehouse.agents), dtype=np.float32)
        reasons = []
        peer_positions = {(agent.x, agent.y) for agent in warehouse.agents}
        build_before = self.teacher.cache_build_count
        hit_before = self.teacher.cache_hit_count
        for index, agent in enumerate(warehouse.agents):
            goal, target_kind = self.environment._target_for_agent(
                agent.id, charging_targets=charging_targets
            )
            requires_pickup = self.environment._requires_pickup(agent.id)
            footprint = "loaded" if agent.carrying_shelf is not None else "unloaded"
            if requires_pickup:
                footprint = "loaded"
                if target_kind == "task":
                    goal = self.environment._nearest(
                        (int(agent.x), int(agent.y)), warehouse.picking_stations
                    )
            query = AStarKDQuery(
                pose=(int(agent.x), int(agent.y)),
                orientation=agent.dir.name,
                goal=(int(goal[0]), int(goal[1])),
                footprint=footprint,
                dead=bool(agent.dead),
                picking_lock=bool(agent.picking_lock_steps),
                mandatory_toggle_load=False,
            )
            result = self.teacher.query(
                query, use_semantic_validity=use_semantic_validity
            )
            targets[index] = result.soft_target
            root_costs[index] = np.asarray(
                [np.nan if cost is None else cost for cost in result.root_costs],
                dtype=np.float32,
            )
            validity[index] = result.valid
            # Scope describes route applicability. Temporary action locks belong
            # to semantic validity and must not sever the post-pickup route signal.
            supervision_scope[index] = bool(
                target_kind != "idle"
                and (int(agent.x), int(agent.y)) != query.goal
            )
            reasons.append(result.failure_reason)
            if (
                result.valid
                and result.diagnostics["preferred_root_action"] == "FORWARD"
                and _forward_cell(agent) in (peer_positions - {(agent.x, agent.y)})
            ):
                attenuation[index] = 0.25
        return TeacherLabels(
            soft_targets=targets,
            root_costs=root_costs,
            validity=validity,
            supervision_scope=supervision_scope,
            attenuation=attenuation,
            failure_reasons=tuple(reasons),
            query_count=len(warehouse.agents),
            cache_build_count=self.teacher.cache_build_count - build_before,
            cache_hit_count=self.teacher.cache_hit_count - hit_before,
        )


def _static_map(environment) -> StaticMap:
    """Freeze only warehouse layout/facilities, never mutable shelf or AGV poses."""

    warehouse = environment.env
    height, width = map(int, warehouse.grid_size)
    highway = frozenset(
        (x, y)
        for y in range(height)
        for x in range(width)
        if warehouse._is_highway(x, y)
    )
    return StaticMap(
        width=width,
        height=height,
        highway=highway,
        workstations=frozenset(tuple(point) for point in warehouse.picking_stations),
        chargers=frozenset(tuple(point) for point in warehouse.charging_stations),
        static_obstacles=frozenset(),
    )


def _forward_cell(agent) -> tuple[int, int]:
    dx, dy = _DELTAS[agent.dir.name]
    return int(agent.x) + dx, int(agent.y) + dy
