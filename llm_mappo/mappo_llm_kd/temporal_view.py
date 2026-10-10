"""Deterministic temporal semantic view derived only from Actor observations."""

from __future__ import annotations

from math import hypot
from typing import Sequence

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)


TEMPORAL_SEMANTIC_VIEW_V1 = "temporal-semantic-view-v1"
_ORIENTATIONS = ("UP", "DOWN", "LEFT", "RIGHT")
_PHASES = ("idle", "task", "delivery", "charging")
_ACTIONS = ("NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE_LOAD")
_CARDINALS = (
    ("forward", 1, 0),
    ("right", 0, 1),
    ("back", -1, 0),
    ("left", 0, -1),
)
_MAP_CHANNELS = (
    "static_obstacle",
    "shelf",
    "workstation",
    "charger",
    "other_agv",
)
_EXPECTED_LEDGER = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
).feature_names


def serialize_temporal_semantic_view(
    observations: np.ndarray,
    reset_masks: np.ndarray,
    feature_ledger: Sequence[str],
    requested_history_length: int,
) -> dict[str, object]:
    """Serialize the latest same-episode window without privileged inputs.

    A true reset mask marks the first observation of an episode. If the caller
    supplies frames from multiple episodes, all frames before the latest reset
    are discarded before the requested history horizon is applied.
    """

    if tuple(feature_ledger) != _EXPECTED_LEDGER:
        raise ValueError("The exact versioned v2 feature ledger is required.")
    if not isinstance(requested_history_length, int) or requested_history_length < 1:
        raise ValueError("requested_history_length must be a positive integer.")
    rows = np.asarray(observations, dtype=np.float32)
    masks = np.asarray(reset_masks)
    if rows.ndim != 2 or rows.shape[1] != len(_EXPECTED_LEDGER):
        raise ValueError("observations must have shape [time, 209].")
    if rows.shape[0] == 0:
        raise ValueError("At least one observation is required.")
    if masks.dtype != np.bool_ or masks.shape != (rows.shape[0],):
        raise ValueError("reset_masks must be a boolean vector matching time.")
    if not np.isfinite(rows).all():
        raise ValueError("observations must contain only finite values.")

    reset_indices = np.flatnonzero(masks)
    episode_start = int(reset_indices[-1]) if reset_indices.size else 0
    window_start = max(episode_start, rows.shape[0] - requested_history_length)
    selected = rows[window_start:]
    start_is_known = reset_indices.size > 0 and window_start == episode_start
    index = {name: position for position, name in enumerate(feature_ledger)}
    frames = [_decode_frame(row, index) for row in selected]
    summary = _summarize(
        frames,
        requested_history_length=requested_history_length,
        history_left_censored=not start_is_known,
    )
    return {
        "schema_version": TEMPORAL_SEMANTIC_VIEW_V1,
        "observation_schema": DIRECT_GOAL_OBSERVATION_V2,
        "history": frames,
        "summary": summary,
    }


def _decode_frame(row: np.ndarray, index: dict[str, int]) -> dict[str, object]:
    orientation = _one_hot_value(row, index, "self_orientation_", _ORIENTATIONS)
    phase = _one_hot_value(row, index, "self_phase_", _PHASES)
    action = _indexed_one_hot_value(row, index, "self_previous_action_", _ACTIONS)
    peers = [_decode_peer(row, index, slot) for slot in range(3)]
    cells = _decode_cells(row, index)
    cell_by_offset = {(cell["forward"], cell["right"]): cell for cell in cells}
    static_blocked = {}
    peer_occupied = {}
    for name, forward, right in _CARDINALS:
        cell = cell_by_offset[(forward, right)]
        static_blocked[name] = bool(cell["static_obstacle"] or cell["shelf"])
        peer_occupied[name] = bool(cell["other_agv"])
    return {
        "self": {
            "orientation": orientation,
            "carrying": _binary(row, index, "self_carrying"),
            "battery": _number(row, index, "self_battery"),
            "task_phase": phase,
            "priority_weight": _number(row, index, "self_priority_weight"),
            "previous_requested_action": action,
            "previous_blocked": _binary(row, index, "self_previous_blocked"),
        },
        "goal": {
            "relative_forward": _number(row, index, "goal_relative_forward"),
            "relative_right": _number(row, index, "goal_relative_right"),
            "normalized_distance": _number(row, index, "goal_normalized_distance"),
            "goal_type": _one_hot_value(row, index, "goal_type_", _PHASES),
        },
        "peers": peers,
        "local": {
            "cells": cells,
            "static_or_shelf_blocked": static_blocked,
            "peer_occupied": peer_occupied,
            "non_peer_free_degree": sum(not value for value in static_blocked.values()),
            "visible_peer_cell_count": sum(cell["other_agv"] for cell in cells),
        },
    }


def _decode_peer(
    row: np.ndarray, index: dict[str, int], slot: int
) -> dict[str, object]:
    prefix = f"peer_{slot}_"
    if not _binary(row, index, prefix + "valid"):
        return {"valid": False}
    return {
        "valid": True,
        "relative_forward": _number(row, index, prefix + "relative_forward"),
        "relative_right": _number(row, index, prefix + "relative_right"),
        "orientation": _one_hot_value(
            row, index, prefix + "orientation_", _ORIENTATIONS
        ),
        "carrying": _binary(row, index, prefix + "carrying"),
        "battery": _number(row, index, prefix + "battery"),
        "task_phase": _one_hot_value(row, index, prefix + "phase_", _PHASES),
        "priority_weight": _number(row, index, prefix + "priority_weight"),
        "previous_requested_action": _indexed_one_hot_value(
            row, index, prefix + "previous_action_", _ACTIONS
        ),
        "previous_blocked": _binary(row, index, prefix + "previous_blocked"),
    }


def _decode_cells(row: np.ndarray, index: dict[str, int]) -> list[dict[str, object]]:
    cells = []
    for forward in range(-2, 3):
        for right in range(-2, 3):
            prefix = f"map_f{forward}_r{right}_"
            cells.append(
                {
                    "forward": forward,
                    "right": right,
                    **{
                        channel: _binary(row, index, prefix + channel)
                        for channel in _MAP_CHANNELS
                    },
                }
            )
    return cells


def _summarize(
    frames: list[dict[str, object]],
    *,
    requested_history_length: int,
    history_left_censored: bool,
) -> dict[str, object]:
    actions = [frame["self"]["previous_requested_action"] for frame in frames]
    blocked = [frame["self"]["previous_blocked"] for frame in frames]
    orientations = [frame["self"]["orientation"] for frame in frames]
    visible_counts = [frame["local"]["visible_peer_cell_count"] for frame in frames]
    phase = frames[-1]["self"]["task_phase"]
    phase_run = _consecutive_final(
        [frame["self"]["task_phase"] == phase for frame in frames]
    )
    goal_types = [frame["goal"]["goal_type"] for frame in frames]
    final_goal_type = goal_types[-1]
    goal_segment_length = _consecutive_final(
        [goal_type == final_goal_type for goal_type in goal_types]
    )
    goal_segment = frames[-goal_segment_length:]
    first_distance = goal_segment[0]["goal"]["normalized_distance"]
    final_distance = goal_segment[-1]["goal"]["normalized_distance"]
    nearest_distances = _nearest_peer_distances(frames)
    if nearest_distances:
        peer_first = nearest_distances[0]
        peer_final = nearest_distances[-1]
        peer_decrease = peer_first - peer_final
    else:
        peer_first = peer_final = peer_decrease = None
    return {
        "observed_history_length": len(frames),
        "requested_history_length": requested_history_length,
        "history_left_censored": history_left_censored,
        "action_counts": {action: actions.count(action) for action in _ACTIONS},
        "blocked_count": sum(blocked),
        "consecutive_noop": _consecutive_final(
            [action == "NOOP" for action in actions]
        ),
        "consecutive_blocked": _consecutive_final(blocked),
        "direction_change_count": sum(
            before != after for before, after in zip(orientations, orientations[1:])
        ),
        "goal_distance_trend": {
            "first": first_distance,
            "final": final_distance,
            "net_decrease": first_distance - final_distance,
            "segment_length": goal_segment_length,
            "target_identity_available": False,
        },
        "task_phase_age_lower_bound": {
            "frames": phase_run,
            "left_censored": phase_run == len(frames) and history_left_censored,
        },
        "visible_peer_cell_statistics": {
            "current": visible_counts[-1],
            "minimum": min(visible_counts),
            "maximum": max(visible_counts),
            "mean": float(sum(visible_counts) / len(visible_counts)),
        },
        "anonymous_nearest_peer_distance": {
            "first": peer_first,
            "final": peer_final,
            "net_decrease": peer_decrease,
            "peer_identity_available": False,
        },
    }


def _nearest_peer_distances(frames: list[dict[str, object]]) -> list[float]:
    distances = []
    for frame in frames:
        per_frame = [
            hypot(peer["relative_forward"], peer["relative_right"])
            for peer in frame["peers"]
            if peer["valid"]
        ]
        if per_frame:
            distances.append(float(min(per_frame)))
    return distances


def _consecutive_final(values: Sequence[bool]) -> int:
    count = 0
    for value in reversed(values):
        if not value:
            break
        count += 1
    return count


def _one_hot_value(
    row: np.ndarray,
    index: dict[str, int],
    prefix: str,
    labels: Sequence[str],
) -> str:
    names = [prefix + label.lower() for label in labels]
    values = [_number(row, index, name) for name in names]
    _require_one_hot(prefix, values)
    return labels[int(np.argmax(values))]


def _indexed_one_hot_value(
    row: np.ndarray,
    index: dict[str, int],
    prefix: str,
    labels: Sequence[str],
) -> str:
    values = [_number(row, index, prefix + str(number)) for number in range(len(labels))]
    _require_one_hot(prefix, values)
    return labels[int(np.argmax(values))]


def _require_one_hot(name: str, values: Sequence[float]) -> None:
    if any(value not in (0.0, 1.0) for value in values) or sum(values) != 1.0:
        raise ValueError(f"{name} must be exactly one-hot encoded.")


def _binary(row: np.ndarray, index: dict[str, int], name: str) -> bool:
    value = _number(row, index, name)
    if value not in (0.0, 1.0):
        raise ValueError(f"{name} must be binary.")
    return bool(value)


def _number(row: np.ndarray, index: dict[str, int], name: str) -> float:
    return float(row[index[name]])
