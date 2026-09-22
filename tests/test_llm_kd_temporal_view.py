"""Contracts for the privileged-information-free LLM KD temporal view."""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.temporal_view import (
    TEMPORAL_SEMANTIC_VIEW_V1,
    serialize_temporal_semantic_view,
)
from llm_mappo.mappo_llm_kd.trajectory import (
    ACTOR_OBSERVATION_TRAJECTORY_V1,
    ActorObservationTrajectory,
)


LEDGER = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
).feature_names
INDEX = {name: index for index, name in enumerate(LEDGER)}
ACTIONS = ("NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE_LOAD")


def _frame(
    *,
    orientation: str = "up",
    phase: str = "task",
    action: int = 0,
    blocked: bool = False,
    distance: float = 0.5,
    priority: float = 0.5,
) -> np.ndarray:
    row = np.zeros(len(LEDGER), dtype=np.float32)
    row[INDEX[f"self_orientation_{orientation}"]] = 1.0
    row[INDEX[f"self_phase_{phase}"]] = 1.0
    row[INDEX[f"self_previous_action_{action}"]] = 1.0
    row[INDEX["self_previous_blocked"]] = float(blocked)
    row[INDEX["self_battery"]] = 0.75
    row[INDEX["self_priority_weight"]] = priority
    row[INDEX["goal_relative_forward"]] = 0.25
    row[INDEX["goal_relative_right"]] = -0.125
    row[INDEX["goal_normalized_distance"]] = distance
    row[INDEX[f"goal_type_{phase}"]] = 1.0
    return row


def _set_peer(row: np.ndarray, slot: int, *, forward: float, right: float) -> None:
    prefix = f"peer_{slot}_"
    row[INDEX[prefix + "relative_forward"]] = forward
    row[INDEX[prefix + "relative_right"]] = right
    row[INDEX[prefix + "orientation_left"]] = 1.0
    row[INDEX[prefix + "battery"]] = 0.6
    row[INDEX[prefix + "phase_delivery"]] = 1.0
    row[INDEX[prefix + "priority_weight"]] = 1.0
    row[INDEX[prefix + "previous_action_1"]] = 1.0
    row[INDEX[prefix + "valid"]] = 1.0


def _serialize(rows, masks, requested=8):
    return serialize_temporal_semantic_view(
        observations=np.stack(rows),
        reset_masks=np.asarray(masks, dtype=np.bool_),
        feature_ledger=LEDGER,
        requested_history_length=requested,
    )


def test_raw_trajectory_has_only_actor_observable_versioned_fields():
    rows = np.stack([_frame(), _frame(action=1)])
    trajectory = ActorObservationTrajectory(
        episode_id="episode-7",
        agent_id=2,
        timesteps=np.asarray([0, 1]),
        observations=rows,
        reset_masks=np.asarray([True, False]),
        behavior_policy_id="phase2-baseline-sha256:abc",
        environment_sha256="a" * 64,
        config_sha256="b" * 64,
    )

    record = trajectory.to_record()
    assert list(record) == [
        "schema_version",
        "observation_schema",
        "episode_id",
        "agent_id",
        "timesteps",
        "behavior_policy_id",
        "environment_sha256",
        "config_sha256",
        "reset_masks",
        "observations",
    ]
    assert record["schema_version"] == ACTOR_OBSERVATION_TRAJECTORY_V1
    assert record["observation_schema"] == DIRECT_GOAL_OBSERVATION_V2
    assert "reward" not in json.dumps(record)
    assert "astar" not in json.dumps(record).lower()
    assert trajectory.observations.flags.writeable is False


@pytest.mark.parametrize(
    "change, message",
    [
        ({"observations": np.zeros((2, 208))}, "209"),
        ({"reset_masks": np.asarray([True])}, "reset"),
        ({"reset_masks": np.asarray([1, 0])}, "boolean"),
        ({"timesteps": np.asarray([1, 1])}, "timesteps"),
        ({"timesteps": np.asarray([0.5, 1.5])}, "integer"),
        ({"environment_sha256": "not-a-hash"}, "SHA-256"),
    ],
)
def test_raw_trajectory_rejects_malformed_identity_or_shape(change, message):
    kwargs = {
        "episode_id": "episode-7",
        "agent_id": 1,
        "timesteps": np.asarray([0, 1]),
        "observations": np.stack([_frame(), _frame()]),
        "reset_masks": np.asarray([True, False]),
        "behavior_policy_id": "baseline:test",
        "environment_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    kwargs.update(change)
    with pytest.raises(ValueError, match=message):
        ActorObservationTrajectory(**kwargs)


def test_temporal_view_exact_fields_order_and_invalid_peer_minimality():
    row = _frame(action=1, blocked=True)
    _set_peer(row, 0, forward=0.3, right=0.4)
    row[INDEX["map_f1_r0_shelf"]] = 1.0
    row[INDEX["map_f0_r1_other_agv"]] = 1.0

    view = _serialize([row], [True])

    assert list(view) == [
        "schema_version",
        "observation_schema",
        "history",
        "summary",
    ]
    assert view["schema_version"] == TEMPORAL_SEMANTIC_VIEW_V1
    frame = view["history"][0]
    assert list(frame) == ["self", "goal", "peers", "local"]
    assert frame["self"] == {
        "orientation": "UP",
        "carrying": False,
        "battery": 0.75,
        "task_phase": "task",
        "priority_weight": 0.5,
        "previous_requested_action": "FORWARD",
        "previous_blocked": True,
    }
    assert frame["peers"][0]["valid"] is True
    assert frame["peers"][1] == {"valid": False}
    assert frame["peers"][2] == {"valid": False}
    assert frame["local"]["static_or_shelf_blocked"] == {
        "forward": True,
        "right": False,
        "back": False,
        "left": False,
    }
    assert frame["local"]["peer_occupied"]["right"] is True
    assert frame["local"]["non_peer_free_degree"] == 3
    assert frame["local"]["visible_peer_cell_count"] == 1
    assert len(frame["local"]["cells"]) == 25
    assert list(frame["local"]["cells"][0]) == [
        "forward",
        "right",
        "static_obstacle",
        "shelf",
        "workstation",
        "charger",
        "other_agv",
    ]


def test_summary_math_censoring_and_anonymous_peer_null_rules():
    rows = [
        _frame(orientation="up", action=0, distance=0.8),
        _frame(orientation="right", action=1, distance=0.6, blocked=True),
        _frame(orientation="right", action=0, distance=0.5, blocked=True),
    ]
    summary = _serialize(rows, [True, False, False], requested=8)["summary"]

    assert list(summary) == [
        "observed_history_length",
        "requested_history_length",
        "history_left_censored",
        "action_counts",
        "blocked_count",
        "consecutive_noop",
        "consecutive_blocked",
        "direction_change_count",
        "goal_distance_trend",
        "task_phase_age_lower_bound",
        "visible_peer_cell_statistics",
        "anonymous_nearest_peer_distance",
    ]
    assert list(summary["action_counts"]) == list(ACTIONS)
    assert summary["action_counts"] == {
        "NOOP": 2,
        "FORWARD": 1,
        "LEFT": 0,
        "RIGHT": 0,
        "TOGGLE_LOAD": 0,
    }
    assert summary["observed_history_length"] == 3
    assert summary["requested_history_length"] == 8
    assert summary["history_left_censored"] is False
    assert summary["blocked_count"] == 2
    assert summary["consecutive_noop"] == 1
    assert summary["consecutive_blocked"] == 2
    assert summary["direction_change_count"] == 1
    assert summary["goal_distance_trend"] == {
        "first": pytest.approx(0.8),
        "final": pytest.approx(0.5),
        "net_decrease": pytest.approx(0.3),
        "segment_length": 3,
        "target_identity_available": False,
    }
    assert summary["task_phase_age_lower_bound"] == {
        "frames": 3,
        "left_censored": False,
    }
    assert summary["anonymous_nearest_peer_distance"] == {
        "first": None,
        "final": None,
        "net_decrease": None,
        "peer_identity_available": False,
    }


def test_history_never_crosses_latest_episode_reset_and_reports_left_censoring():
    rows = [_frame(distance=value) for value in (0.9, 0.8, 0.7, 0.6, 0.5)]

    after_reset = _serialize(rows, [True, False, False, True, False], requested=8)
    assert len(after_reset["history"]) == 2
    assert after_reset["history"][0]["goal"]["normalized_distance"] == pytest.approx(
        0.6
    )
    assert after_reset["summary"]["history_left_censored"] is False

    truncated = _serialize(rows, [True, False, False, False, False], requested=3)
    assert len(truncated["history"]) == 3
    assert truncated["summary"]["history_left_censored"] is True
    assert truncated["summary"]["task_phase_age_lower_bound"]["left_censored"] is True


def test_peer_slots_are_anonymous_across_frames_and_distance_is_per_frame_minimum():
    first = _frame()
    second = _frame()
    _set_peer(first, 0, forward=0.6, right=0.8)
    _set_peer(first, 1, forward=0.0, right=0.5)
    _set_peer(second, 0, forward=0.3, right=0.4)
    summary = _serialize([first, second], [True, False])["summary"]

    assert summary["anonymous_nearest_peer_distance"] == {
        "first": pytest.approx(0.5),
        "final": pytest.approx(0.5),
        "net_decrease": pytest.approx(0.0, abs=1e-7),
        "peer_identity_available": False,
    }


def test_serializer_is_deterministic_strict_and_has_no_privileged_input():
    rows = np.stack([_frame(), _frame()])
    masks = np.asarray([True, False])
    first = serialize_temporal_semantic_view(rows, masks, LEDGER, 8)
    second = serialize_temporal_semantic_view(rows, masks, LEDGER, 8)
    assert json.dumps(first, separators=(",", ":")) == json.dumps(
        second, separators=(",", ":")
    )

    with pytest.raises(ValueError, match="feature ledger"):
        serialize_temporal_semantic_view(rows, masks, LEDGER + ("reward",), 8)
    with pytest.raises(TypeError):
        serialize_temporal_semantic_view(
            observations=rows,
            reset_masks=masks,
            feature_ledger=LEDGER,
            requested_history_length=8,
            environment=object(),
        )
