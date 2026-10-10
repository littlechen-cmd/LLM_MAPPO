"""Observable v3 semantics; reference anchors never quantize LLM labels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import hypot, isclose, isfinite
from typing import Mapping


SEMANTIC_CONTRACT_V3 = "observable-priority-coordination-four-score-v3"
SEMANTIC_DIMENSIONS_V3 = (
    "goal-priority", "yielding-preference", "local-constraint", "motion-stagnation"
)
LLM_DIMENSIONS_V3 = SEMANTIC_DIMENSIONS_V3[:3]
_YIELD_REASONS = frozenset((
    "ambiguous-peer-association", "missing-peer-attributes", "symmetric-priority",
    "conflicting-peer-evidence", "hard-service-lock", "ambiguous-service-state",
))
_REASONS = {
    "goal-priority": frozenset(),
    "yielding-preference": _YIELD_REASONS,
    "local-constraint": frozenset(("ambiguous-shelf-mobility",)),
}
_CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_ORIENTATIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0),
                 "RIGHT": (1, 0)}
_PHASES = {"idle", "task", "delivery", "charging"}
_SELF_KEYS = {
    "orientation", "carrying", "battery", "task_phase", "priority_weight",
    "previous_requested_action", "previous_blocked",
}
_GOAL_KEYS = {
    "relative_forward", "relative_right", "normalized_distance", "goal_type",
}
_PEER_KEYS = _SELF_KEYS | {"valid", "relative_forward", "relative_right"}
_CELL_KEYS = {"forward", "right", "static_obstacle", "shelf", "workstation",
              "charger", "other_agv"}
_ACTIONS = {"NOOP", "FORWARD", "LEFT", "RIGHT", "TOGGLE_LOAD"}


@dataclass(frozen=True)
class ParsedSemanticV3:
    scores: dict[str, float | None]
    evidence: dict[str, tuple[str, ...]]
    unknown_reasons: dict[str, str | None]


def _numeric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be finite numeric data.")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite numeric data.")
    return float(value)


def _validate_agent_fields(agent: dict) -> None:
    if "task_phase" in agent and agent["task_phase"] not in _PHASES:
        raise ValueError("Invalid task phase.")
    if "orientation" in agent and agent["orientation"] not in _ORIENTATIONS:
        raise ValueError("Invalid orientation.")
    if ("previous_requested_action" in agent
            and agent["previous_requested_action"] not in _ACTIONS):
        raise ValueError("Invalid previous action.")
    for key in ("carrying", "previous_blocked"):
        if key in agent and type(agent[key]) is not bool:
            raise ValueError("Raw boolean expected.")
    if "battery" in agent and not 0 <= _numeric(agent["battery"], "battery") <= 1:
        raise ValueError("battery out of range.")
    if "priority_weight" in agent:
        _numeric(agent["priority_weight"], "priority_weight")


def _validate_peers(peers: object) -> None:
    if not isinstance(peers, list) or len(peers) != 3:
        raise ValueError("Exactly three anonymous peer slots required.")
    for peer in peers:
        if not isinstance(peer, dict) or type(peer.get("valid")) is not bool:
            raise ValueError("Invalid peer validity field.")
        if not set(peer) <= _PEER_KEYS:
            raise ValueError("Unobservable peer field.")
        if not peer["valid"] and set(peer) != {"valid"}:
            raise ValueError("Invalid peer cannot carry attributes.")
        _validate_agent_fields(peer)
        for key in ("relative_forward", "relative_right"):
            if key in peer:
                _numeric(peer[key], key)


def _validate_cells(cells: object) -> None:
    if not isinstance(cells, list) or len(cells) != 25:
        raise ValueError("Exactly 25 local cells required.")
    expected = {(f, r) for f in range(-2, 3) for r in range(-2, 3)}
    actual = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != _CELL_KEYS:
            raise ValueError("Invalid local cell fields.")
        if type(cell["forward"]) is not int or type(cell["right"]) is not int:
            raise ValueError("Cell offsets must be integers.")
        actual.add((cell["forward"], cell["right"]))
        if any(type(cell[key]) is not bool for key in _CELL_KEYS - {"forward", "right"}):
            raise ValueError("Cell channels must be boolean.")
    if actual != expected:
        raise ValueError("Local cell offsets are incomplete or duplicated.")


def _validate_frame(frame: dict) -> None:
    if not isinstance(frame, dict) or set(frame) != {"self", "goal", "peers", "local"}:
        raise ValueError("Invalid raw frame groups.")
    own, goal = frame["self"], frame["goal"]
    if not isinstance(own, dict) or set(own) != _SELF_KEYS:
        raise ValueError("Invalid self fields.")
    if not isinstance(goal, dict) or set(goal) != _GOAL_KEYS:
        raise ValueError("Invalid goal fields.")
    _validate_agent_fields(own)
    if goal["goal_type"] not in _PHASES:
        raise ValueError("Invalid goal type.")
    for key in _GOAL_KEYS - {"goal_type"}:
        _numeric(goal[key], key)
    if goal["normalized_distance"] < 0:
        raise ValueError("Negative goal distance.")
    _validate_peers(frame["peers"])
    _validate_cells(frame["local"].get("cells"))


def _frames(view: Mapping[str, object]) -> list[dict]:
    frames = view.get("history")
    if not isinstance(frames, list) or len(frames) != 8:
        raise ValueError("v3 requires eight observable frames.")
    for frame in frames:
        _validate_frame(frame)
    return frames


def _leaf_paths(value: object, prefix: str) -> set[str]:
    paths = {prefix}
    if isinstance(value, dict):
        for name, item in value.items():
            paths.update(_leaf_paths(item, f"{prefix}.{name}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.update(_leaf_paths(item, f"{prefix}[{index}]"))
    return paths


def raw_evidence_paths_v3(view: Mapping[str, object]) -> frozenset[str]:
    """Allow only actual raw fields/containers, never summaries or proxies."""
    frames = _frames(view)
    result = set()
    for index, frame in enumerate(frames):
        for name in ("self", "goal", "peers"):
            result.update(_leaf_paths(frame[name], f"history[{index}].{name}"))
        result.update(_leaf_paths(frame["local"]["cells"],
                                  f"history[{index}].local.cells"))
    return frozenset(result)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("LLM JSON contains a duplicate key.")
        result[key] = value
    return result


def parse_semantic_v3_response(
    content: str, semantic_view: Mapping[str, object]
) -> ParsedSemanticV3:
    """Strict schema only. Evidence support is a separate semantic gate."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty LLM response.")
    try:
        payload = json.loads(content, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("Response must be strict JSON.") from error
    if not isinstance(payload, dict) or set(payload) != {
        "scores", "evidence", "unknown_reasons"
    }:
        raise ValueError("Response requires scores, evidence, unknown_reasons only.")
    for name in payload:
        if not isinstance(payload[name], dict) or set(payload[name]) != set(
            LLM_DIMENSIONS_V3
        ):
            raise ValueError("Each response object requires exactly three dimensions.")
    allowed = raw_evidence_paths_v3(semantic_view)
    scores, evidence, reasons = {}, {}, {}
    for dimension in LLM_DIMENSIONS_V3:
        scores[dimension], reasons[dimension] = _parse_score_reason(
            payload["scores"][dimension], payload["unknown_reasons"][dimension],
            dimension,
        )
        evidence[dimension] = _parse_evidence(payload["evidence"][dimension], allowed)
    return ParsedSemanticV3(scores, evidence, reasons)


def _parse_score_reason(value: object, reason: object, dimension: str) -> tuple:
    if value is None:
        if not isinstance(reason, str) or reason not in _REASONS[dimension]:
            raise ValueError("Invalid unknown reason for dimension.")
    else:
        value = _numeric(value, dimension)
        if not 0 <= value <= 1:
            raise ValueError("Score outside [0, 1].")
        if reason is not None:
            raise ValueError("Known score cannot have an unknown reason.")
    return value, reason


def _parse_evidence(paths: object, allowed: frozenset) -> tuple[str, ...]:
    if not isinstance(paths, list) or not 1 <= len(paths) <= 4:
        raise ValueError("Each LLM dimension requires one to four raw paths.")
    if any(not isinstance(path, str) or path not in allowed for path in paths):
        raise ValueError("Unavailable raw evidence path.")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate evidence path.")
    return tuple(paths)


def _cells(frame: dict) -> dict[tuple[int, int], dict]:
    return {(cell["forward"], cell["right"]): cell
            for cell in frame["local"]["cells"]}


def _service(frame: dict) -> str | None:
    own, goal = frame["self"], frame["goal"]
    zero = all(abs(goal[key]) <= 1e-6 for key in (
        "relative_forward", "relative_right", "normalized_distance"
    ))
    center = _cells(frame)[(0, 0)]
    if zero and own["task_phase"] == goal["goal_type"] == "charging":
        if center["charger"] and 0 < own["battery"] < 1:
            return "hard-service-lock"
    if zero and own["task_phase"] in {"task", "delivery"} and center["workstation"]:
        return "ambiguous-service-state"
    return None


def _priority(own: dict, goal_type: str | None = None) -> float:
    phase = own["task_phase"]
    if phase not in _PHASES or (goal_type is not None and phase != goal_type):
        raise ValueError("Phase and goal type do not match.")
    battery = _numeric(own["battery"], "battery")
    if not 0 <= battery <= 1:
        raise ValueError("battery out of range.")
    if phase == "idle" or battery == 0:
        return 0.0
    if phase == "charging":
        if battery >= 1:
            raise ValueError("Full battery still has charging goal.")
        if battery >= 0.30:
            return 0.25
        if battery >= 0.15:
            return 0.5
        return 1.0 if battery <= 0.05 else 0.75
    priority = _numeric(own["priority_weight"], "priority_weight")
    if priority not in (0.0, 0.5, 0.75, 1.0):
        raise ValueError("Undefined task priority.")
    return priority if priority else 0.25


def associate_current_peers_v3(
    semantic_view: Mapping[str, object]
) -> dict[tuple[int, int], int]:
    """Recover current scale from two observations, never stable identities."""
    frame = _frames(semantic_view)[-1]
    occupied = {offset for offset, cell in _cells(frame).items()
                if cell["other_agv"]}
    if not occupied:
        return {}
    if (0, 0) in occupied:
        raise ValueError("Other AGV cannot occupy own center.")
    valid = _ordered_peer_coordinates(frame["peers"])
    if not valid:
        raise ValueError("Local occupancy has no valid nearest peer.")
    if len(valid) < min(3, len(occupied)):
        raise ValueError("A required local peer slot is missing.")
    scale = min(hypot(*offset) for offset in occupied) / valid[0][3]
    mapped = {}
    for index, forward, right, _ in valid:
        position = (forward * scale, right * scale)
        rounded = tuple(round(number) for number in position)
        if any(abs(number - integer) > 1e-4
               for number, integer in zip(position, rounded)):
            raise ValueError("Peer mapping is not an integer grid location.")
        if rounded in _cells(frame):
            if rounded not in occupied or rounded in mapped:
                raise ValueError("Peer mapping conflicts with local occupancy.")
            mapped[rounded] = index
    # Up to three nearest peers are available; additional local AGVs may be
    # outside those slots. Such an AGV is unknown only if it is relevant.
    return mapped


def _ordered_peer_coordinates(peers: list[dict]) -> list[tuple]:
    valid = []
    invalid_seen = False
    for index, peer in enumerate(peers):
        if not peer["valid"]:
            invalid_seen = True
            continue
        if invalid_seen:
            raise ValueError("Valid peer slots must be contiguous.")
        forward = _numeric(peer.get("relative_forward"), "peer forward")
        right = _numeric(peer.get("relative_right"), "peer right")
        distance = hypot(forward, right)
        if distance == 0:
            raise ValueError("Zero peer distance.")
        valid.append((index, forward, right, distance))
    # Float32 coordinates perturb norms of equal grid distances in either
    # direction. Four unit roundoffs cover both rounded coordinate vectors.
    if any(before[3] > after[3]
           and not isclose(before[3], after[3], rel_tol=2**-22)
           for before, after in zip(valid, valid[1:])):
        raise ValueError("Peer distances are not sorted.")
    return valid


def _passability(frame: dict) -> dict[tuple[int, int], bool | None]:
    result = {}
    for offset, cell in _cells(frame).items():
        if cell["static_obstacle"]:
            result[offset] = False
        elif frame["self"]["carrying"] and cell["shelf"]:
            result[offset] = None if cell["other_agv"] else False
        else:
            result[offset] = True
    result[(0, 0)] = True
    return result


def _constraint_score(passable: dict[tuple[int, int], bool]) -> float:
    entrances = [offset for offset in _CARDINALS if passable[offset]]
    degree = len(entrances)
    if degree >= 3:
        return 0.0 if degree == 4 else 0.25
    if degree <= 1:
        return 1.0
    if tuple(-value for value in entrances[0]) != entrances[1]:
        return 0.5
    seen, frontier = {(0, 0)}, {(0, 0)}
    for _ in range(2):
        following = set()
        for f, r in frontier:
            for df, dr in _CARDINALS:
                target = (f + df, r + dr)
                if passable.get(target, False) and target not in seen:
                    following.add(target)
        seen.update(following)
        frontier = following
    for f, r in seen:
        degree = sum(passable.get((f + df, r + dr), False)
                     for df, dr in _CARDINALS)
        if degree >= 3:
            return 0.5
    return 0.75


def _constraint(frame: dict) -> tuple[float | None, str | None]:
    passability = _passability(frame)
    unknown = [offset for offset, value in passability.items() if value is None]
    # Monotonic structural opening can only reduce this anchor score. Checking
    # both extremes therefore covers all mobility assignments without 2**N work.
    scores = {_constraint_score({offset: default if value is None else value
                                for offset, value in passability.items()})
              for default in (False, True)}
    if unknown and len(scores) > 1:
        return None, "ambiguous-shelf-mobility"
    return scores.pop(), None


def _opposing_peer(frame: dict, peer: dict) -> bool:
    self_heading = _ORIENTATIONS[frame["self"]["orientation"]]
    peer_heading = _ORIENTATIONS.get(peer.get("orientation"))
    return peer_heading == tuple(-component for component in self_heading)


def _weak_relation(frame: dict, peer: dict) -> bool:
    if not (_opposing_peer(frame, peer)
            and peer["previous_requested_action"] == "FORWARD"
            and peer["previous_blocked"] is False):
        return False
    passability = _passability(frame)
    middle = _cells(frame)[(1, 0)]
    if passability[(1, 0)] is not True or middle["other_agv"]:
        return False
    if peer["carrying"] and middle["shelf"]:
        return False
    # Check the visible straight corridor only; beyond the view is unknown.
    return all(passability.get((f, r), False) is False
               for f in (0, 1, 2) for r in (-1, 1))


def _competition_candidate(frame: dict) -> tuple[tuple[int, int] | None, bool]:
    occupied = {offset for offset, cell in _cells(frame).items() if cell["other_agv"]}
    own = frame["self"]
    if ((1, 0) in occupied and own["previous_requested_action"] == "FORWARD"
            and own["previous_blocked"]):
        return (1, 0), True
    if ((2, 0) in occupied and own["previous_requested_action"] == "FORWARD"
            and not own["previous_blocked"]):
        return (2, 0), False
    return None, False


def _competing_peer(view: Mapping[str, object], target: tuple) -> tuple:
    try:
        mapped = associate_current_peers_v3(view)
    except ValueError:
        return None, "ambiguous-peer-association"
    if target not in mapped:
        return None, "ambiguous-peer-association"
    peer = view["history"][-1]["peers"][mapped[target]]
    required = {"battery", "task_phase", "priority_weight", "orientation",
                "previous_requested_action", "previous_blocked", "carrying"}
    if not required <= set(peer):
        return None, "missing-peer-attributes"
    return peer, None


def _yielding(
    view: Mapping[str, object], own_priority: float
) -> tuple[float | None, str | None]:
    frame = _frames(view)[-1]
    service = _service(frame)
    if service == "hard-service-lock":
        return None, service
    target, direct = _competition_candidate(frame)
    if target is None:
        return 0.0, None
    peer, reason = _competing_peer(view, target)
    if reason is not None:
        return None, reason
    if not direct and not _weak_relation(frame, peer):
        return 0.0, None
    if service == "ambiguous-service-state":
        return None, service
    peer_priority = _priority(peer)
    difference = peer_priority - own_priority
    if difference < 0:
        return 0.0, None
    if difference == 0:
        return None, "symmetric-priority"
    if not direct:
        return 0.25, None
    return min(1.0, difference + 0.25), None


def motion_stagnation_v3(semantic_view: Mapping[str, object]) -> dict[str, object]:
    """Deterministic contiguous-tail proxy; no inference of global deadlock."""
    frames = _frames(semantic_view)
    current = frames[-1]
    base_evidence = [f"history[{index}].{group}" for index in range(8)
                     for group in ("self", "goal")]
    base_evidence.append("history[7].local.cells")
    if (current["self"]["task_phase"] == "idle"
            or current["goal"]["goal_type"] == "idle"
            or current["self"]["battery"] == 0
            or _service(current) == "hard-service-lock"):
        return dict(score=0.0, evidence=tuple(base_evidence), unknown_reason=None)
    indices = _final_segment_indices(frames)
    blocked = _tail_indices(frames, indices, blocked=True)
    turns = _tail_indices(frames, indices, blocked=False)
    alternations = sum(
        frames[a]["self"]["previous_requested_action"]
        != frames[b]["self"]["previous_requested_action"]
        for a, b in zip(turns, turns[1:])
    )
    blocked_score = min(1.0, (len(blocked) - 2) * 0.25) if len(blocked) >= 3 else 0.0
    turn_score = _turn_score(frames, turns, alternations)
    score = max(blocked_score, turn_score)
    reason = None
    if score > 0 and _service(current) == "ambiguous-service-state":
        score, reason = None, "ambiguous-service-state"
    return dict(score=score, evidence=tuple(base_evidence), unknown_reason=reason)


def _final_segment_indices(frames: list[dict]) -> list[int]:
    current = frames[-1]
    phase, goal_type = current["self"]["task_phase"], current["goal"]["goal_type"]
    start = 7
    while (start > 0 and frames[start - 1]["self"]["task_phase"] == phase
           and frames[start - 1]["goal"]["goal_type"] == goal_type):
        start -= 1
    # Previous action at segment start precedes the segment itself.
    return list(range(start + 1, 8))


def _tail_indices(frames: list[dict], indices: list[int], *, blocked: bool) -> list[int]:
    tail = []
    for index in reversed(indices):
        own = frames[index]["self"]
        matches = (own["previous_requested_action"] == "FORWARD"
                   and own["previous_blocked"]) if blocked else (
                       own["previous_requested_action"] in {"LEFT", "RIGHT"})
        if not matches:
            break
        tail.append(index)
    return list(reversed(tail))


def _turn_score(frames: list[dict], turns: list[int], alternations: int) -> float:
    if len(turns) >= 4 and alternations >= 3:
        distances = [frames[index]["goal"]["normalized_distance"]
                     for index in range(turns[0] - 1, 8)]
        if max(distances) - min(distances) <= 1e-6:
            return min(1.0, (alternations - 2) * 0.25)
    return 0.0


def reference_semantics_v3(semantic_view: Mapping[str, object]) -> dict[str, dict]:
    """Reference anchors/unknowns for diagnostics, never LLM score replacement."""
    current = _frames(semantic_view)[-1]
    goal_priority = _priority(current["self"], current["goal"]["goal_type"])
    yielding, yielding_reason = _yielding(semantic_view, goal_priority)
    constraint, constraint_reason = _constraint(current)
    stagnation = motion_stagnation_v3(semantic_view)
    scores = dict(zip(SEMANTIC_DIMENSIONS_V3,
                      (goal_priority, yielding, constraint, stagnation["score"])))
    reasons = dict(zip(SEMANTIC_DIMENSIONS_V3,
                       (None, yielding_reason, constraint_reason,
                        stagnation["unknown_reason"])))
    evidence = {
        "goal-priority": ("history[7].self", "history[7].goal"),
        "yielding-preference": ("history[7].self", "history[7].goal",
                                "history[7].peers", "history[7].local.cells"),
        "local-constraint": ("history[7].self.carrying", "history[7].local.cells"),
        "motion-stagnation": stagnation["evidence"],
    }
    return {
        "scores": scores,
        "known_masks": {key: value is not None for key, value in scores.items()},
        "unknown_reasons": reasons,
        "evidence": evidence,
    }


def _covers(paths: tuple[str, ...], required: str) -> bool:
    return any(path == required or required.startswith(path + ".")
               or required.startswith(path + "[") for path in paths)


def _required_evidence(dimension: str, parsed: ParsedSemanticV3, frame: dict) -> tuple:
    if dimension == "goal-priority":
        return ("history[7].self.task_phase", "history[7].self.battery",
                "history[7].self.priority_weight", "history[7].goal.goal_type")
    if dimension == "local-constraint":
        return ("history[7].self.carrying", "history[7].local.cells")
    needs = ("history[7].local.cells",)
    reason = parsed.unknown_reasons[dimension]
    if reason in {"hard-service-lock", "ambiguous-service-state"}:
        return needs + ("history[7].self", "history[7].goal")
    target, _ = _competition_candidate(frame)
    if target is not None or reason is not None or (parsed.scores[dimension] or 0) > 0:
        needs += ("history[7].self", "history[7].peers")
    return needs


def validate_semantic_v3_support(
    parsed: ParsedSemanticV3, semantic_view: Mapping[str, object]
) -> None:
    """Check observable grounding and mandatory unknowns without quantization."""
    reference = reference_semantics_v3(semantic_view)
    current = semantic_view["history"][-1]
    for dimension in LLM_DIMENSIONS_V3:
        actual = parsed.unknown_reasons[dimension]
        required = reference["unknown_reasons"][dimension]
        if actual != required:
            raise ValueError(f"{dimension}: unknown condition does not match input.")
        paths = parsed.evidence[dimension]
        needs = _required_evidence(dimension, parsed, current)
        if any(not _covers(paths, path) for path in needs):
            raise ValueError(f"{dimension}: incomplete observable evidence.")
        # These checks prohibit claims with no semantic basis; values with
        # support are never compared to or rounded towards numeric anchors.
        score = parsed.scores[dimension]
        if (dimension == "goal-priority" and score is not None and score > 0
                and (current["self"]["task_phase"] == "idle"
                     or current["self"]["battery"] == 0)):
            raise ValueError("Inactive goal cannot support positive priority.")
        if (dimension == "yielding-preference" and score is not None and score > 0
                and reference["scores"][dimension] == 0):
            raise ValueError("No observable relative yielding support.")


def merge_semantic_v3_label(
    parsed: ParsedSemanticV3, semantic_view: Mapping[str, object]
) -> dict[str, dict]:
    """Join three continuous LLM targets and one independently derived target."""
    validate_semantic_v3_support(parsed, semantic_view)
    auxiliary = motion_stagnation_v3(semantic_view)
    scores = {**parsed.scores, "motion-stagnation": auxiliary["score"]}
    return dict(
        scores=scores,
        known_masks={dimension: score is not None
                     for dimension, score in scores.items()},
        sources={dimension: ("llm" if dimension in LLM_DIMENSIONS_V3
                             else "observable-rule-v3")
                 for dimension in SEMANTIC_DIMENSIONS_V3},
        evidence={**parsed.evidence, "motion-stagnation": auxiliary["evidence"]},
        unknown_reasons={**parsed.unknown_reasons,
                         "motion-stagnation": auxiliary["unknown_reason"]},
    )
