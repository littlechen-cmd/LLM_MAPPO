"""Feature-pruned, agent-specific centralized state for MAPPO.

The common prefix is built once from current environment truth. Each critic row
appends only the robot identity and its legal-action mask. This retains the
team configuration relevant to the shared cooperative value function without
repeating decentralized Actor inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_DIRECTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
_TARGET_KINDS = ("idle", "task", "delivery", "charging")


@dataclass(frozen=True)
class FPStateSchema:
    """Frozen feature ledger for one project-specific FP critic row."""

    n_agents: int
    n_shelves: int = 144
    n_chargers: int | None = None
    n_picking_stations: int | None = None

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            raise ValueError("FP state requires at least one agent.")
        if self.n_shelves < 1:
            raise ValueError("FP state requires at least one shelf slot.")
        if self.n_chargers is not None and self.n_chargers < 1:
            raise ValueError("FP state requires at least one charger slot.")
        if self.n_picking_stations is not None and self.n_picking_stations < 1:
            raise ValueError("FP state requires at least one picking-station slot.")

    @property
    def charger_count(self) -> int:
        return 8 if self.n_chargers is None else self.n_chargers

    @property
    def picking_station_count(self) -> int:
        return 2 if self.n_picking_stations is None else self.n_picking_stations

    @property
    def common_feature_names(self) -> tuple[str, ...]:
        header = (
            "episode_progress",
            "task_completion_progress",
            "active_task_density",
            "pending_task_fraction",
            "assigned_task_fraction",
            "occupied_charger_fraction",
        )
        agent_fields = (
            "present",
            "x",
            "y",
            "direction_up",
            "direction_down",
            "direction_left",
            "direction_right",
            "battery",
            "loaded",
            "dead",
            "picking_lock",
            "target_x",
            "target_y",
            "target_idle",
            "target_task",
            "target_delivery",
            "target_charging",
            "requires_pickup",
        )
        agents = tuple(
            f"team_agent_{agent_index}_{field}"
            for agent_index in range(self.n_agents)
            for field in agent_fields
        )
        shelves = tuple(
            f"shelf_{shelf_index}_{field}"
            for shelf_index in range(self.n_shelves)
            for field in ("x", "y", "carried", "active_priority")
        )
        chargers = tuple(
            f"charger_{charger_index}_occupied"
            for charger_index in range(self.charger_count)
        )
        picking = tuple(
            f"picking_{station_index}_occupied"
            for station_index in range(self.picking_station_count)
        )
        return header + agents + shelves + chargers + picking

    @property
    def agent_feature_names(self) -> tuple[str, ...]:
        return tuple(f"critic_agent_{index}" for index in range(self.n_agents)) + tuple(
            f"legal_action_{index}" for index in range(5)
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.common_feature_names + self.agent_feature_names

    @property
    def common_dimension(self) -> int:
        return len(self.common_feature_names)

    @property
    def dimension(self) -> int:
        return len(self.feature_names)


def build_active_masks(environment) -> np.ndarray:
    """Return one PPO active mask per robot; dead robots do not train losses."""

    return np.asarray(
        [not bool(agent.dead) for agent in environment.env.agents],
        dtype=np.float32,
    )


def build_fp_state(  # noqa: C901
    environment, schema: FPStateSchema | None = None
) -> np.ndarray:
    """Build `[agents, fp_dim]` from current, non-predictive environment state."""

    warehouse = environment.env
    schema = schema or FPStateSchema(
        n_agents=environment.n_agents,
        n_chargers=len(warehouse.charging_stations),
        n_picking_stations=len(warehouse.picking_stations),
    )
    if schema.n_agents != environment.n_agents:
        raise ValueError("FP schema and environment agent counts differ.")
    if schema.n_shelves != len(warehouse.shelfs):
        raise ValueError("FP schema and environment shelf counts differ.")
    if schema.charger_count != len(warehouse.charging_stations):
        raise ValueError("FP schema and environment charger counts differ.")
    if schema.picking_station_count != len(warehouse.picking_stations):
        raise ValueError("FP schema and environment picking-station counts differ.")
    height, width = map(int, warehouse.grid_size)
    active_tasks = tuple(warehouse.task_queue.active_tasks)
    pending = sum(task.status.value == "pending" for task in active_tasks)
    assigned = sum(task.status.value == "assigned" for task in active_tasks)
    completed = sum(
        task.status.value == "completed" for task in warehouse.task_queue.tasks
    )
    task_scale = max(int(warehouse.task_completion_target), 1)
    active_scale = max(task_scale, int(environment.request_queue_size or 1))
    occupied_chargers = sum(
        (agent.x, agent.y) in warehouse.charging_stations
        for agent in warehouse.agents
    )
    active_count = max(len(active_tasks), 1)
    common = [
        min(float(warehouse._cur_steps) / max(environment.max_steps, 1), 1.0),
        min(float(completed) / task_scale, 1.0),
        min(float(len(active_tasks)) / active_scale, 1.0),
        float(pending) / active_count,
        float(assigned) / active_count,
        float(occupied_chargers) / schema.n_agents,
    ]
    charging_targets = environment._charging_targets()
    for agent in sorted(warehouse.agents, key=lambda item: item.id):
        target, kind = environment._target_for_agent(
            agent.id, charging_targets=charging_targets
        )
        direction = [float(agent.dir.name == name) for name in _DIRECTION_NAMES]
        target_kind = [float(kind == name) for name in _TARGET_KINDS]
        common.extend(
            [
                1.0,
                float(agent.x) / max(width - 1, 1),
                float(agent.y) / max(height - 1, 1),
                *direction,
                float(agent.battery),
                float(agent.carrying_shelf is not None),
                float(agent.dead),
                min(float(agent.picking_lock_steps) / 3.0, 1.0),
                float(target[0]) / max(width - 1, 1),
                float(target[1]) / max(height - 1, 1),
                *target_kind,
                float(environment._requires_pickup(agent.id)),
            ]
        )
    active_shelf_priorities = np.zeros(schema.n_shelves, dtype=np.float32)
    for task in active_tasks:
        active_shelf_priorities[int(task.shelf_id) - 1] = (
            float(warehouse.task_queue.priority_weight(task.label)) / 2.0
        )
    carried_shelves = {
        agent.carrying_shelf.id
        for agent in warehouse.agents
        if agent.carrying_shelf is not None
    }
    for shelf in sorted(warehouse.shelfs, key=lambda item: item.id):
        common.extend(
            [
                float(shelf.x) / max(width - 1, 1),
                float(shelf.y) / max(height - 1, 1),
                float(shelf.id in carried_shelves),
                float(active_shelf_priorities[shelf.id - 1]),
            ]
        )
    agent_positions = {(agent.x, agent.y) for agent in warehouse.agents}
    common.extend(
        float(station in agent_positions)
        for station in sorted(warehouse.charging_stations)
    )
    common.extend(
        float(station in agent_positions)
        for station in sorted(warehouse.picking_stations)
    )
    common_array = np.asarray(common, dtype=np.float32)
    if common_array.shape != (schema.common_dimension,):
        raise RuntimeError("FP common-state feature ledger is inconsistent.")
    legal_masks = np.asarray(environment.action_masks(), dtype=np.float32)
    rows = []
    for index in range(schema.n_agents):
        identity = np.zeros(schema.n_agents, dtype=np.float32)
        identity[index] = 1.0
        if warehouse.agents[index].dead:
            # MAPPO death mask: zeros plus identity remove ambiguous stale state.
            row_common = np.zeros_like(common_array)
            row_actions = np.zeros(5, dtype=np.float32)
        else:
            row_common = common_array
            row_actions = legal_masks[index]
        rows.append(np.concatenate((row_common, identity, row_actions)))
    result = np.stack(rows).astype(np.float32, copy=False)
    if result.shape != (schema.n_agents, schema.dimension):
        raise RuntimeError("FP state shape does not match its frozen schema.")
    if not np.isfinite(result).all():
        raise RuntimeError("FP state contains a non-finite value.")
    return result
