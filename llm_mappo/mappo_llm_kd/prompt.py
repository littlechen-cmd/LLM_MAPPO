"""Versioned, observation-only prompt construction for Phase 3 labels."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping


LLM_LABEL_PROMPT_V3 = "llm-label-prompt-v3"
_DIMENSIONS = (
    "mission_urgency",
    "concession_appropriateness",
    "interaction_risk",
    "deadlock_severity",
)
_FRAME_KEYS = {"self", "goal", "peers", "local"}
_SELF_KEYS = {
    "orientation",
    "carrying",
    "battery",
    "task_phase",
    "priority_weight",
    "previous_requested_action",
    "previous_blocked",
}
_GOAL_KEYS = {
    "relative_forward",
    "relative_right",
    "normalized_distance",
    "goal_type",
}
_PEER_KEYS = {
    "valid",
    "relative_forward",
    "relative_right",
    "orientation",
    "carrying",
    "battery",
    "task_phase",
    "priority_weight",
    "previous_requested_action",
    "previous_blocked",
}
_LOCAL_KEYS = {
    "cells",
    "static_or_shelf_blocked",
    "peer_occupied",
    "non_peer_free_degree",
    "visible_peer_cell_count",
}
_CELL_KEYS = {
    "forward",
    "right",
    "static_obstacle",
    "shelf",
    "workstation",
    "charger",
    "other_agv",
}
_SUMMARY_KEYS = {
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
}


@dataclass(frozen=True)
class LabelPrompt:
    """Prompt texts and content identities without credentials."""

    prompt_version: str
    system_text: str
    user_text: str
    system_sha256: str
    user_sha256: str
    semantic_view_sha256: str
    score_dimensions: tuple[str, ...]
    available_evidence_paths: frozenset[str]


def build_label_prompt(semantic_view: Mapping[str, object]) -> LabelPrompt:
    """Build the strict four-score prompt from one exact temporal view."""

    _validate_temporal_view(semantic_view)
    canonical = _canonical_json(semantic_view)
    system = (
        "You are a JSON-only warehouse semantic-state annotator. Use only the "
        "supplied observable temporal view. Do not prescribe behavior, paths, "
        "task assignments, priority changes, or rewards."
    )
    user = (
        "Return exactly one JSON object with keys scores and evidence. Both "
        "objects must contain exactly mission_urgency, "
        "concession_appropriateness, interaction_risk, and deadlock_severity. "
        "Each score is a finite number in [0,1]. Each evidence value is a list "
        "of one to four unique exact leaf paths from the supplied view. Emit no "
        "free-form explanation. Paths for frame values must include the full "
        "history[index]. prefix, for example history[0].self.task_phase, "
        "history[0].goal.goal_type, history[0].peers[0].valid, or "
        "history[0].local.non_peer_free_degree; frame-relative shorthand is invalid. "
        "Aggregate paths begin with summary, for example summary.blocked_count. "
        "Every cited value must support the direction and magnitude of its score, "
        "not merely exist in the input. Free space alone does not justify concession; "
        "concession_appropriateness must be supported by observable coordination "
        "context. interaction_risk requires an observable interaction object, such "
        "as a valid peer or peer-occupied local cell. "
        "Anchors for every dimension are 0.0 absent or "
        "opposite, 0.25 weak, 0.5 mixed or moderate, 0.75 strong, and 1.0 "
        "extreme with direct observable support; interpolation is allowed. "
        "mission_urgency describes focal task-progress urgency for the team. "
        "concession_appropriateness describes the observable coordination case "
        "for temporarily releasing local space. interaction_risk describes "
        "short-horizon direct local interference risk. deadlock_severity "
        "describes persistent mutual blocking, oscillation, or ineffective "
        "progress. These are state descriptions, never decisions.\n"
        "TEMPORAL_SEMANTIC_VIEW="
        + canonical
    )
    return LabelPrompt(
        prompt_version=LLM_LABEL_PROMPT_V3,
        system_text=system,
        user_text=user,
        system_sha256=_digest(system),
        user_sha256=_digest(user),
        semantic_view_sha256=_digest(canonical),
        score_dimensions=_DIMENSIONS,
        available_evidence_paths=frozenset(_leaf_paths(semantic_view)),
    )


def _validate_temporal_view(view: Mapping[str, object]) -> None:
    try:
        _expect_keys(
            view,
            {"schema_version", "observation_schema", "history", "summary"},
        )
        if view["schema_version"] != "temporal-semantic-view-v1":
            raise ValueError
        if view["observation_schema"] != "direct-goal-observation-v2":
            raise ValueError
        history = view["history"]
        if not isinstance(history, list) or not history:
            raise ValueError
        for frame in history:
            _validate_frame(frame)
        _validate_summary(view["summary"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Prompt requires the exact TemporalSemanticView-v1.") from error


def _validate_frame(frame: object) -> None:
    _expect_keys(frame, _FRAME_KEYS)
    _expect_keys(frame["self"], _SELF_KEYS)
    _expect_keys(frame["goal"], _GOAL_KEYS)
    peers = frame["peers"]
    if not isinstance(peers, list) or len(peers) != 3:
        raise ValueError
    for peer in peers:
        expected = _PEER_KEYS if peer.get("valid") is True else {"valid"}
        _expect_keys(peer, expected)
    local = frame["local"]
    _expect_keys(local, _LOCAL_KEYS)
    cells = local["cells"]
    if not isinstance(cells, list) or len(cells) != 25:
        raise ValueError
    for cell in cells:
        _expect_keys(cell, _CELL_KEYS)
    cardinals = {"forward", "right", "back", "left"}
    _expect_keys(local["static_or_shelf_blocked"], cardinals)
    _expect_keys(local["peer_occupied"], cardinals)


def _validate_summary(summary: object) -> None:
    _expect_keys(summary, _SUMMARY_KEYS)
    _expect_keys(
        summary["action_counts"],
        {"NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE_LOAD"},
    )
    _expect_keys(
        summary["goal_distance_trend"],
        {
            "first",
            "final",
            "net_decrease",
            "segment_length",
            "target_identity_available",
        },
    )
    _expect_keys(
        summary["task_phase_age_lower_bound"], {"frames", "left_censored"}
    )
    _expect_keys(
        summary["visible_peer_cell_statistics"],
        {"current", "minimum", "maximum", "mean"},
    )
    _expect_keys(
        summary["anonymous_nearest_peer_distance"],
        {"first", "final", "net_decrease", "peer_identity_available"},
    )


def _expect_keys(value: object, expected: set[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError


def _leaf_paths(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        paths = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(child, child_prefix))
        return paths
    if isinstance(value, list):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_leaf_paths(child, f"{prefix}[{index}]"))
        return paths
    return [prefix]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()
