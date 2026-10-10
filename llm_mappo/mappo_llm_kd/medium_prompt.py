"""Medium dataset prompt versions with exact observable evidence paths."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Mapping

from .prompt import LabelPrompt, _leaf_paths, build_label_prompt


MEDIUM_LABEL_PROMPT_V4 = "llm-label-prompt-v4-medium-evidence-allowlist"
MEDIUM_LABEL_PROMPT_V5 = "llm-label-prompt-v5-medium-json-and-current-state"
MEDIUM_LABEL_PROMPT_V6 = "llm-label-prompt-v6-compact-cells-evidence-support"
MEDIUM_LABEL_PROMPT_V7 = "llm-label-prompt-v7-observable-coordination-v2"
SEMANTIC_CONTRACT_V2 = "observable-coordination-four-score-v2"
_DISTANCE_CONSTANT_TOLERANCE = 1e-6
LOCAL_CELL_COLUMNS = (
    "forward", "right", "static_obstacle", "shelf", "workstation",
    "charger", "other_agv",
)


def compact_medium_view(semantic_view: Mapping[str, object]) -> dict[str, object]:
    """Encode only cell dictionaries as fixed-column rows, without data loss."""

    compact = deepcopy(semantic_view)
    for frame in compact["history"]:
        frame["local"]["cells"] = [
            [cell[name] for name in LOCAL_CELL_COLUMNS]
            for cell in frame["local"]["cells"]
        ]
    return compact


def expand_medium_view(compact_view: Mapping[str, object]) -> dict[str, object]:
    """Recover the exact original semantic view from the compact cell rows."""

    expanded = deepcopy(compact_view)
    for frame in expanded["history"]:
        rows = frame["local"]["cells"]
        if len(rows) != 25 or any(len(row) != len(LOCAL_CELL_COLUMNS)
                                  for row in rows):
            raise ValueError("Compact local cells must contain 25 seven-field rows.")
        frame["local"]["cells"] = [
            dict(zip(LOCAL_CELL_COLUMNS, row)) for row in rows
        ]
    return expanded


def validate_medium_v6_semantic_support(scores, evidence, semantic_view):
    """Reject positive local-interaction claims without cited occupancy."""

    current = semantic_view["history"][7]["local"]
    statistics = semantic_view["summary"]["visible_peer_cell_statistics"]
    current_paths = {
        "history[7].local.visible_peer_cell_count": (
            current["visible_peer_cell_count"] > 0
        ),
        "summary.visible_peer_cell_statistics.current": (
            statistics["current"] > 0
        ),
        **{
            f"history[7].local.peer_occupied.{name}": bool(value)
            for name, value in current["peer_occupied"].items()
        },
    }
    recent_path = "summary.visible_peer_cell_statistics.maximum"
    for dimension in ("concession_appropriateness", "interaction_risk"):
        score = scores[dimension]
        if score <= 0:
            continue
        cited = evidence[dimension]
        direct = any(current_paths.get(path, False) for path in cited)
        recent = recent_path in cited and statistics["maximum"] > 0
        if not (direct or recent):
            raise ValueError(
                f"{dimension} positive score lacks cited positive local "
                "peer occupancy."
            )
        if not direct and score > 0.25:
            raise ValueError(
                f"{dimension} past-only occupancy supports at most 0.25."
            )


def observable_coordination_evidence(semantic_view):
    """Audit current observable sustained blocking or turn oscillation.

    The final same-phase, same-goal-type segment is used. The first frame's
    previous action is excluded because it can belong to the preceding phase
    or goal. A blocked-forward witness needs three
    blocked FORWARD requests including the current one. An oscillation witness
    needs four LEFT/RIGHT requests, three alternations, constant observed goal
    distance within 1e-6, and no FORWARD request lacking a blocked witness in
    that phase segment. This is a conservative observable proxy; a false
    previous_blocked flag does not prove a FORWARD request moved successfully.
    Both require an active current phase and goal and positive battery.
    """

    history = semantic_view["history"]
    if len(history) != 8:
        raise ValueError("Observable coordination requires eight frames.")
    current = history[-1]
    phase = current["self"]["task_phase"]
    goal_type = current["goal"]["goal_type"]
    active = (
        phase != "idle" and goal_type != "idle"
        and current["self"]["battery"] > 0
    )
    start = 7
    while (
        start > 0
        and history[start - 1]["self"]["task_phase"] == phase
        and history[start - 1]["goal"]["goal_type"] == goal_type
    ):
        start -= 1
    segment = history[start:]
    action_start = start + 1
    blocked_indices = [
        index for index in range(action_start, 8)
        if history[index]["self"]["previous_requested_action"] == "FORWARD"
        and history[index]["self"]["previous_blocked"] is True
    ]
    current_blocked = 7 in blocked_indices
    turns = [
        (index, history[index]["self"]["previous_requested_action"])
        for index in range(action_start, 8)
        if history[index]["self"]["previous_requested_action"]
        in {"LEFT", "RIGHT"}
    ]
    alternations = sum(
        before[1] != after[1]
        for before, after in zip(turns, turns[1:])
    )
    distances = [frame["goal"]["normalized_distance"] for frame in segment]
    constant_distance = (
        max(distances) - min(distances) <= _DISTANCE_CONSTANT_TOLERANCE
    )
    no_unmarked_forward_request = not any(
        history[index]["self"]["previous_requested_action"] == "FORWARD"
        and history[index]["self"]["previous_blocked"] is False
        for index in range(action_start, 8)
    )
    blocked_supported = active and current_blocked and len(blocked_indices) >= 3
    oscillation_supported = (
        active and len(turns) >= 4 and alternations >= 3
        and constant_distance and no_unmarked_forward_request
    )
    return {
        "semantic_contract_version": SEMANTIC_CONTRACT_V2,
        "current_active": active,
        "phase_segment_start": start,
        "phase_segment_length": len(segment),
        "action_observation_start": action_start,
        "blocked_forward_indices": blocked_indices,
        "current_blocked_forward": current_blocked,
        "blocked_forward_supported": blocked_supported,
        "turn_indices_and_actions": turns,
        "turn_alternations": alternations,
        "constant_goal_distance": constant_distance,
        "no_unmarked_forward_request": no_unmarked_forward_request,
        "oscillation_supported": oscillation_supported,
        "positive_supported": blocked_supported or oscillation_supported,
    }


def validate_medium_v7_semantic_support(scores, evidence, semantic_view):
    """Reject unsupported v7 positives without rewriting any teacher value."""

    validate_medium_v6_semantic_support(scores, evidence, semantic_view)
    current = semantic_view["history"][7]
    if (
        (current["self"]["task_phase"] == "idle"
         or current["goal"]["goal_type"] == "idle")
        and scores["mission_urgency"] > 0
    ):
        raise ValueError("Idle current phase and goal require zero mission_urgency.")
    if scores["deadlock_severity"] <= 0:
        return
    witness = observable_coordination_evidence(semantic_view)
    if not witness["positive_supported"]:
        raise ValueError(
            "deadlock_severity positive score lacks observable sustained "
            "blocking or oscillation witness."
        )
    cited = set(evidence["deadlock_severity"])
    blocked_citation = witness["blocked_forward_supported"] and {
        "history[7].self.previous_requested_action",
        "history[7].self.previous_blocked",
        "summary.blocked_count",
    }.issubset(cited) and semantic_view["summary"]["blocked_count"] >= 3
    cited_turn_values = {
        action for index, action in witness["turn_indices_and_actions"]
        if f"history[{index}].self.previous_requested_action" in cited
    }
    cited_goal_distances = [
        semantic_view["history"][index]["goal"]["normalized_distance"]
        for index in range(witness["phase_segment_start"], 8)
        if f"history[{index}].goal.normalized_distance" in cited
    ]
    goal_citation = (
        ("summary.goal_distance_trend.net_decrease" in cited
         and abs(semantic_view["summary"]["goal_distance_trend"][
             "net_decrease"
         ]) <= _DISTANCE_CONSTANT_TOLERANCE)
        or len(cited_goal_distances) >= 2
    )
    oscillation_citation = (
        witness["oscillation_supported"]
        and {"LEFT", "RIGHT"}.issubset(cited_turn_values)
        and goal_citation
    )
    if not (blocked_citation or oscillation_citation):
        raise ValueError(
            "deadlock_severity positive score lacks mode-specific cited "
            "observable evidence."
        )


def build_medium_label_prompt(
    semantic_view: Mapping[str, object], *, version: str
) -> LabelPrompt:
    """Preserve v3 score semantics while restricting v4 evidence citations."""

    original = build_label_prompt(semantic_view)
    if version == original.prompt_version:
        return original
    if version == MEDIUM_LABEL_PROMPT_V7:
        previous = build_medium_label_prompt(
            semantic_view, version=MEDIUM_LABEL_PROMPT_V6
        )
        extra_paths = {
            f"history[{index}].{suffix}"
            for index in range(7)
            for suffix in (
                "self.previous_requested_action", "self.previous_blocked",
                "self.task_phase", "goal.goal_type",
                "goal.normalized_distance",
            )
        }
        allowed = frozenset(previous.available_evidence_paths | extra_paths)
        original_allowlist = json.dumps(
            tuple(sorted(previous.available_evidence_paths)),
            ensure_ascii=False, separators=(",", ":"),
        )
        new_allowlist = json.dumps(
            tuple(sorted(allowed)), ensure_ascii=False, separators=(",", ":")
        )
        user = _replace_once(
            previous.user_text,
            "EVIDENCE_PATH_ALLOWLIST=" + original_allowlist,
            "EVIDENCE_PATH_ALLOWLIST=" + new_allowlist,
        )
        user = _replace_once(
            user,
            "deadlock_severity describes persistent mutual blocking, "
            "oscillation, or ineffective progress.",
            "deadlock_severity is a legacy JSON field name for an observable "
            "sustained blocking or oscillation indicator; it never asserts "
            "confirmed or global deadlock.",
        )
        user = _replace_once(
            user,
            "Idle NOOP, including at zero battery, is not mutual deadlock "
            "without persistent mutual blocking, oscillation, or ineffective "
            "progress. An isolated turn or three earlier phase-change frames "
            "do not establish persistent mutual deadlock. When this evidence "
            "is absent, deadlock_severity is exactly 0.",
            "An idle NOOP, depleted battery, isolated turn, or prior phase "
            "change does not establish a sustained observable coordination "
            "indicator. Without the exact v7 witness below, "
            "deadlock_severity is 0.",
        )
        user = _replace_once(
            user,
            "For deadlock_severity, persistent ineffective task progress "
            "remains observable even without a nearby peer: an active phase "
            "over the observed history, repeated blocked steps, and zero or "
            "negative goal-distance progress can support weak or moderate "
            "severity, but target identity is unavailable and distance trend "
            "alone is insufficient. Do not infer deadlock from an idle NOOP "
            "or a single blocked step.",
            "For the v7 deadlock_severity transport field, positive values "
            "require the precise observable-coordination witness below. "
            "Distance trend, direction_change_count, and ordinary goal "
            "approach alone never suffice.",
        )
        contract = (
            "\nSEMANTIC_CONTRACT_VERSION=" + SEMANTIC_CONTRACT_V2 + "\n"
            "The fourth JSON key stays deadlock_severity for transport "
            "compatibility. Interpret it only as an observable sustained "
            "blocking/oscillation indicator, never a confirmed/global "
            "deadlock. Its positive score requires current phase and goal "
            "both active, battery > 0, and one of these witnesses within the "
            "final same-task-phase and same-goal-type segment including "
            "history[7]. Ignore the first frame's previous action in each "
            "segment because it may belong to the old phase or goal: "
            "(A) at least three blocked FORWARD requests including a current "
            "blocked FORWARD; cite history[7].self.previous_requested_action, "
            "history[7].self.previous_blocked and summary.blocked_count. "
            "(B) at least four LEFT/RIGHT turn requests with at least three "
            "alternations, goal.normalized_distance constant within 1e-6 "
            "through that segment, and no FORWARD request lacking a recorded "
            "blocked witness; this is an observable proxy, not proof of "
            "movement. Cite two exact "
            "opposite turn-action history paths and "
            "summary.goal_distance_trend.net_decrease (when that aggregate "
            "is zero) or two same-segment history goal distance paths. "
            "Any unsupported positive is invalid. A zero score states only "
            "that this observable witness is absent, not that no global "
            "deadlock exists. Current idle phase or idle goal requires "
            "mission_urgency=0. Nearby peer occupancy alone is a necessary "
            "citation for positive concession or risk, not proof of strong "
            "local competition; normal motion with space available may "
            "support only weak or zero scores. non_peer_free_degree and "
            "static_or_shelf_blocked are shelf/static geometric proxies, "
            "not proven traversability: an unloaded AGV may enter a shelf. "
            "No hidden state or action "
            "instruction.\n"
        )
        contract += (
            "OBSERVABLE_COORDINATION_SUPPORT="
            + json.dumps(observable_coordination_evidence(semantic_view),
                         ensure_ascii=False, separators=(",", ":"))
            + "\nThis derived audit only summarizes the supplied view; its keys "
            "are not valid evidence citations. Cite original allowlist paths.\n"
        )
        user = _replace_once(
            user, "\nOUTPUT_STRUCTURE_ONLY",
            contract + "OUTPUT_STRUCTURE_ONLY",
        )
        return LabelPrompt(
            prompt_version=MEDIUM_LABEL_PROMPT_V7,
            system_text=previous.system_text,
            user_text=user,
            system_sha256=previous.system_sha256,
            user_sha256=sha256(user.encode("utf-8")).hexdigest(),
            semantic_view_sha256=previous.semantic_view_sha256,
            score_dimensions=previous.score_dimensions,
            available_evidence_paths=allowed,
        )
    if version == MEDIUM_LABEL_PROMPT_V6:
        previous = build_medium_label_prompt(
            semantic_view, version=MEDIUM_LABEL_PROMPT_V5
        )
        compact = compact_medium_view(semantic_view)
        if expand_medium_view(compact) != semantic_view:
            raise ValueError("Compact medium view failed lossless round trip.")
        original_view_json = original.user_text.split(
            "TEMPORAL_SEMANTIC_VIEW=", 1
        )[1]
        compact_view_json = json.dumps(
            compact, ensure_ascii=False, separators=(",", ":")
        )
        user = previous.user_text.replace(
            "TEMPORAL_SEMANTIC_VIEW=" + original_view_json,
            "LOCAL_CELL_COLUMNS=" + json.dumps(LOCAL_CELL_COLUMNS) + "\n"
            "Each history[*].local.cells row follows those seven columns in "
            "the same 5x5 offset order as the original view. All other view "
            "fields retain their original names and values. Evidence paths "
            "still refer to the original semantic view and exact allowlist.\n"
            "TEMPORAL_SEMANTIC_VIEW=" + compact_view_json,
            1,
        )
        guidance = (
            "\nDIMENSION_EVIDENCE_SUPPORT: A positive "
            "concession_appropriateness or interaction_risk requires citing "
            "a positive current history[7].local.visible_peer_cell_count "
            "or true cardinal history[7].local.peer_occupied path. If current "
            "local occupancy is zero but recent 8-frame occupancy is positive, "
            "cite positive summary.visible_peer_cell_statistics.maximum; "
            "past-only occupancy supports at most weak 0.25. Current free "
            "degree, static blockage, or a remote valid peer alone cannot "
            "support positive concession or short-horizon local risk. If "
            "current and recent local occupancy are all zero, those two "
            "dimensions are 0. For deadlock_severity, persistent ineffective "
            "task progress remains observable even without a nearby peer: "
            "an active phase over the observed history, repeated blocked "
            "steps, and zero or negative goal-distance progress can support "
            "weak or moderate severity, but target identity is unavailable "
            "and distance trend alone is insufficient. Do not infer deadlock "
            "from an idle "
            "NOOP or a single blocked step. Cite dimension-specific values "
            "whose actual values support the stated score; these are state "
            "descriptions, never actions.\n"
        )
        user = user.replace(
            "\nOUTPUT_STRUCTURE_ONLY",
            guidance + "OUTPUT_STRUCTURE_ONLY", 1,
        )
        return LabelPrompt(
            prompt_version=MEDIUM_LABEL_PROMPT_V6,
            system_text=previous.system_text,
            user_text=user,
            system_sha256=previous.system_sha256,
            user_sha256=sha256(user.encode("utf-8")).hexdigest(),
            semantic_view_sha256=previous.semantic_view_sha256,
            score_dimensions=previous.score_dimensions,
            available_evidence_paths=previous.available_evidence_paths,
        )
    if version == MEDIUM_LABEL_PROMPT_V5:
        previous = build_medium_label_prompt(
            semantic_view, version=MEDIUM_LABEL_PROMPT_V4
        )
        guidance = (
            "\nCURRENT_STATE_CHECKS: Describe the current history[7] state using "
            "earlier frames only as context. A task-phase change resets the "
            "meaning of an earlier goal-distance trend. If the current phase "
            "and goal are idle with no active goal, mission_urgency is 0. "
            "Free space or distant observable peers alone do not establish a "
            "local coordination case; when local coordination evidence is "
            "absent, concession_appropriateness is exactly 0. A valid peer "
            "slot alone is not direct local interference; check local peer "
            "occupancy before assigning interaction risk. Current local "
            "occupancy includes visible_peer_cell_count in the local view; "
            "peer_occupied marks only four cardinal cells, so four false "
            "flags do not exclude a nearby diagonal peer. If current local "
            "peer occupancy is absent, recent occupancy may support only "
            "weak risk when summary.visible_peer_cell_statistics confirms it; "
            "remote valid peers alone do not. "
            "Idle NOOP, including at zero battery, is not mutual deadlock "
            "without persistent mutual blocking, oscillation, or ineffective "
            "progress. An isolated turn or three earlier phase-change frames "
            "do not establish persistent mutual deadlock. When this evidence "
            "is absent, deadlock_severity is exactly 0. These remain "
            "descriptions, never actions.\n"
            "OUTPUT_STRUCTURE_ONLY (replace all example numbers and paths "
            "using the view; do not copy example values):\n"
            '{"scores":{"mission_urgency":0.0,'
            '"concession_appropriateness":0.0,"interaction_risk":0.0,'
            '"deadlock_severity":0.0},"evidence":{'
            '"mission_urgency":["history[7].self.task_phase"],'
            '"concession_appropriateness":'
            '["history[7].local.peer_occupied.forward"],'
            '"interaction_risk":'
            '["history[7].local.visible_peer_cell_count"],'
            '"deadlock_severity":["summary.blocked_count"]}}\n'
            "REQUIRED_OUTPUT: one JSON object, exactly scores and evidence; "
            "each is an object with exactly the same four dimension keys. "
            "Each evidence value is an array of 1-4 exact allowlist strings. "
            "No array may replace the evidence object. No prose or extra key."
        )
        user = previous.user_text + guidance
        return LabelPrompt(
            prompt_version=MEDIUM_LABEL_PROMPT_V5,
            system_text=previous.system_text,
            user_text=user,
            system_sha256=previous.system_sha256,
            user_sha256=sha256(user.encode("utf-8")).hexdigest(),
            semantic_view_sha256=previous.semantic_view_sha256,
            score_dimensions=previous.score_dimensions,
            available_evidence_paths=previous.available_evidence_paths,
        )
    if version != MEDIUM_LABEL_PROMPT_V4:
        raise ValueError("Unsupported medium label prompt version.")
    history = semantic_view["history"]
    if len(history) != 8:
        raise ValueError("Medium v4 requires a complete eight-frame history.")
    current = history[7]
    paths = set(_leaf_paths(semantic_view["summary"], "summary"))
    for name in ("self", "goal", "peers"):
        paths.update(_leaf_paths(current[name], f"history[7].{name}"))
    local = current["local"]
    for name in ("non_peer_free_degree", "visible_peer_cell_count",
                 "static_or_shelf_blocked", "peer_occupied"):
        paths.update(_leaf_paths(local[name], f"history[7].local.{name}"))
    allowed = tuple(sorted(paths))
    prefix, canonical = original.user_text.split("TEMPORAL_SEMANTIC_VIEW=", 1)
    prefix = prefix.replace(
        "from the supplied view.", "from EVIDENCE_PATH_ALLOWLIST below."
    ).replace("history[0].", "history[7].")
    user = (
        prefix
        + "Copy each evidence path character-for-character from "
        "EVIDENCE_PATH_ALLOWLIST. The list contains only actual leaf paths "
        "present in this view. Do not invent indices, aliases, optional peer "
        "fields, or paths outside the list. An invalid peer has only its "
        "valid=false leaf; its absent optional fields cannot be cited. "
        "Peer and goal relative_forward/relative_right and anonymous peer "
        "distance are normalized by the map diagonal, whereas local cell "
        "offsets use grid cells. A peer valid=true means an observable peer "
        "slot, not adjacency or collision; use local.peer_occupied and "
        "visible_peer_cell_count for direct local occupancy. "
        "When direct support is absent, give a low score and cite the most "
        "relevant available observable path.\n"
        "EVIDENCE_PATH_ALLOWLIST="
        + json.dumps(allowed, ensure_ascii=False, separators=(",", ":"))
        + "\nTEMPORAL_SEMANTIC_VIEW=" + canonical
    )
    return LabelPrompt(
        prompt_version=MEDIUM_LABEL_PROMPT_V4,
        system_text=original.system_text,
        user_text=user,
        system_sha256=original.system_sha256,
        user_sha256=sha256(user.encode("utf-8")).hexdigest(),
        semantic_view_sha256=original.semantic_view_sha256,
        score_dimensions=original.score_dimensions,
        available_evidence_paths=frozenset(allowed),
    )


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Medium v7 parent prompt text changed unexpectedly.")
    return source.replace(old, new, 1)
