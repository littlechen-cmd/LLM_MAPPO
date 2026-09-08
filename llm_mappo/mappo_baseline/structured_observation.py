"""Teacher-free, structured decentralized observations for recurrent MAPPO."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np


_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
_TARGET_KINDS = ("idle", "task", "delivery", "charging")
_ACTION_DIMENSION = 5


@dataclass(frozen=True)
class StructuredObservationSchema:
    """Exact feature ledger for a decentralized, current-only Actor input."""

    n_neighbours: int = 3
    local_radius: int = 2

    @property
    def self_feature_names(self) -> tuple[str, ...]:
        return (
            *(f"self_orientation_{name.lower()}" for name in _DIRECTIONS),
            "self_carrying",
            "self_battery",
            *(f"self_phase_{kind}" for kind in _TARGET_KINDS),
            *(f"self_previous_action_{index}" for index in range(_ACTION_DIMENSION)),
            "self_previous_blocked",
        )

    @property
    def goal_feature_names(self) -> tuple[str, ...]:
        return (
            "goal_relative_forward",
            "goal_relative_right",
            "goal_normalized_distance",
            *(f"goal_type_{kind}" for kind in _TARGET_KINDS),
        )

    @property
    def map_feature_names(self) -> tuple[str, ...]:
        names = []
        for forward in range(-self.local_radius, self.local_radius + 1):
            for right in range(-self.local_radius, self.local_radius + 1):
                for channel in (
                    "static_obstacle", "shelf", "workstation", "charger", "other_agv"
                ):
                    names.append(f"map_f{forward}_r{right}_{channel}")
        return tuple(names)

    @property
    def peer_feature_names(self) -> tuple[str, ...]:
        fields = (
            "relative_forward",
            "relative_right",
            *(f"orientation_{name.lower()}" for name in _DIRECTIONS),
            "carrying",
            "battery",
            *(f"phase_{kind}" for kind in _TARGET_KINDS),
            "priority_weight",
            *(f"previous_action_{index}" for index in range(_ACTION_DIMENSION)),
            "previous_blocked",
            "valid",
        )
        return tuple(
            f"peer_{peer_index}_{field}"
            for peer_index in range(self.n_neighbours)
            for field in fields
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (
            self.self_feature_names
            + self.goal_feature_names
            + self.map_feature_names
            + self.peer_feature_names
        )

    @property
    def dimension(self) -> int:
        return len(self.feature_names)


def build_structured_observations(
    environment, schema: StructuredObservationSchema | None = None
) -> np.ndarray:
    """Build one local Actor row per AGV without teachers or future information."""

    warehouse = environment.env
    schema = schema or StructuredObservationSchema()
    height, width = map(int, warehouse.grid_size)
    charging_targets = environment._charging_targets()
    rows = []
    for agent in warehouse.agents:
        target, phase = environment._target_for_agent(
            agent.id, charging_targets=charging_targets
        )
        rows.append(
            np.asarray(
                _self_features(environment, agent, phase)
                + _goal_features(agent, target, phase, width, height)
                + _local_map(environment, agent, schema, width, height)
                + _peer_features(environment, agent, schema, width, height),
                dtype=np.float32,
            )
        )
    result = np.stack(rows)
    if result.shape != (len(warehouse.agents), schema.dimension):
        raise RuntimeError("Structured Actor observation shape does not match schema.")
    if not np.isfinite(result).all():
        raise RuntimeError("Structured Actor observation contains a non-finite value.")
    return result


def _self_features(environment, agent, phase: str) -> list[float]:
    return (
        _one_hot(agent.dir.name, _DIRECTIONS)
        + [float(agent.carrying_shelf is not None), float(agent.battery)]
        + _one_hot(phase, _TARGET_KINDS)
        + _action_one_hot(environment.previous_action(agent.id))
        + [float(environment.previous_blocked(agent.id))]
    )


def _goal_features(agent, target, phase: str, width: int, height: int) -> list[float]:
    forward, right = _ego_relative(agent, target[0], target[1])
    scale = max(float(hypot(width - 1, height - 1)), 1.0)
    distance = hypot(target[0] - agent.x, target[1] - agent.y) / scale
    return [
        forward / scale,
        right / scale,
        distance,
        *_one_hot(phase, _TARGET_KINDS),
    ]


def _local_map(environment, agent, schema, width: int, height: int) -> list[float]:
    warehouse = environment.env
    shelves = {(shelf.x, shelf.y) for shelf in warehouse.shelfs}
    workstations = set(warehouse.picking_stations)
    chargers = set(warehouse.charging_stations)
    other_agents = {
        (peer.x, peer.y) for peer in warehouse.agents if peer.id != agent.id
    }
    features = []
    for forward in range(-schema.local_radius, schema.local_radius + 1):
        for right in range(-schema.local_radius, schema.local_radius + 1):
            x, y = _world_from_ego(agent, forward, right)
            outside = not (0 <= x < width and 0 <= y < height)
            features.extend(
                [
                    float(outside),
                    float(not outside and (x, y) in shelves),
                    float(not outside and (x, y) in workstations),
                    float(not outside and (x, y) in chargers),
                    float(not outside and (x, y) in other_agents),
                ]
            )
    return features


def _peer_features(environment, agent, schema, width: int, height: int) -> list[float]:
    warehouse = environment.env
    peers = [peer for peer in warehouse.agents if peer.id != agent.id]
    peers.sort(
        key=lambda peer: (
            hypot(peer.x - agent.x, peer.y - agent.y),
            peer.x - agent.x,
            peer.y - agent.y,
            _DIRECTIONS.index(peer.dir.name),
        )
    )
    scale = max(float(hypot(width - 1, height - 1)), 1.0)
    features = []
    for peer in peers[:schema.n_neighbours]:
        target, phase = environment._target_for_agent(peer.id)
        del target
        forward, right = _ego_relative(agent, peer.x, peer.y)
        features.extend(
            [
                forward / scale,
                right / scale,
                *_one_hot(peer.dir.name, _DIRECTIONS),
                float(peer.carrying_shelf is not None),
                float(peer.battery),
                *_one_hot(phase, _TARGET_KINDS),
                _priority_weight(environment, peer.id),
                *_action_one_hot(environment.previous_action(peer.id)),
                float(environment.previous_blocked(peer.id)),
                1.0,
            ]
        )
    peer_width = len(schema.peer_feature_names) // schema.n_neighbours
    features.extend([0.0] * (peer_width * schema.n_neighbours - len(features)))
    return features


def _priority_weight(environment, agent_id: int) -> float:
    task = environment.env.task_queue.task_for_agent(agent_id)
    if task is None:
        return 0.0
    return float(environment.env.task_queue.priority_weight(task.label) / 2.0)


def _one_hot(value: str, choices: tuple[str, ...]) -> list[float]:
    return [float(value == choice) for choice in choices]


def _action_one_hot(action: int) -> list[float]:
    if not 0 <= int(action) < _ACTION_DIMENSION:
        raise ValueError("Previous action is outside the warehouse action space.")
    return [float(index == int(action)) for index in range(_ACTION_DIMENSION)]


def _ego_relative(agent, x: int, y: int) -> tuple[int, int]:
    dx = int(x) - int(agent.x)
    dy = int(y) - int(agent.y)
    if agent.dir.name == "UP":
        return -dy, dx
    if agent.dir.name == "DOWN":
        return dy, -dx
    if agent.dir.name == "LEFT":
        return -dx, -dy
    if agent.dir.name == "RIGHT":
        return dx, dy
    raise ValueError("Unknown AGV orientation.")


def _world_from_ego(agent, forward: int, right: int) -> tuple[int, int]:
    if agent.dir.name == "UP":
        return agent.x + right, agent.y - forward
    if agent.dir.name == "DOWN":
        return agent.x - right, agent.y + forward
    if agent.dir.name == "LEFT":
        return agent.x - forward, agent.y - right
    if agent.dir.name == "RIGHT":
        return agent.x + forward, agent.y + right
    raise ValueError("Unknown AGV orientation.")
