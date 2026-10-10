"""Observation-only semantics, strict continuous labels, and service exclusions."""

import copy
import json

import pytest

from llm_mappo.mappo_llm_kd import semantic_v3 as semantic
from llm_mappo.mappo_llm_kd.semantic_prompt_v3 import build_semantic_prompt_v3


def view():
    cells = [
        dict(forward=f, right=r, static_obstacle=False, shelf=False,
             workstation=False, charger=False, other_agv=False)
        for f in range(-2, 3) for r in range(-2, 3)
    ]
    frame = {
        "self": dict(orientation="UP", carrying=False, battery=0.8,
                     task_phase="task", priority_weight=0.5,
                     previous_requested_action="NOOP", previous_blocked=False),
        "goal": dict(relative_forward=0.1, relative_right=0.0,
                     normalized_distance=0.1, goal_type="task"),
        "peers": [{"valid": False} for _ in range(3)],
        "local": dict(cells=cells, static_or_shelf_blocked={}, peer_occupied={},
                      non_peer_free_degree=4, visible_peer_cell_count=0),
    }
    return dict(schema_version="temporal-semantic-view-v1",
                observation_schema="direct-goal-observation-v2",
                history=[copy.deepcopy(frame) for _ in range(8)], summary={})


def cell(frame, f, r):
    return next(c for c in frame["local"]["cells"]
                if (c["forward"], c["right"]) == (f, r))


def direct_competition(value, peer_priority=0.75):
    frame = value["history"][-1]
    frame["self"].update(previous_requested_action="FORWARD", previous_blocked=True)
    cell(frame, 1, 0)["other_agv"] = True
    frame["peers"][0] = dict(valid=True, relative_forward=0.02,
                             relative_right=0.0, orientation="DOWN", carrying=False,
                             battery=0.8, task_phase="task",
                             priority_weight=peer_priority,
                             previous_requested_action="FORWARD",
                             previous_blocked=False)
    return value


def response(scores=None):
    return {
        "scores": scores or {"goal-priority": 0.63, "yielding-preference": 0.37,
                              "local-constraint": 0.0},
        "evidence": {"goal-priority": ["history[7].self", "history[7].goal"],
                     "yielding-preference": ["history[7].self", "history[7].peers",
                                            "history[7].local.cells"],
                     "local-constraint": ["history[7].self.carrying",
                                          "history[7].local.cells"]},
        "unknown_reasons": {name: None for name in semantic.LLM_DIMENSIONS_V3},
    }


def test_continuous_scores_survive_strict_parse_support_and_merge():
    value = direct_competition(view())
    parsed = semantic.parse_semantic_v3_response(json.dumps(response()), value)
    semantic.validate_semantic_v3_support(parsed, value)
    merged = semantic.merge_semantic_v3_label(parsed, value)
    assert merged["scores"] == {"goal-priority": 0.63, "yielding-preference": 0.37,
                                "local-constraint": 0.0, "motion-stagnation": 0.0}
    assert all(merged["known_masks"].values())
    assert merged["sources"]["motion-stagnation"] == "observable-rule-v3"


@pytest.mark.parametrize("bad", [True, "0.37", -0.1, 1.01, float("nan"),
                               float("inf"), -float("inf")])
def test_invalid_score_does_not_enter_training(bad):
    payload = response()
    payload["scores"]["goal-priority"] = bad
    with pytest.raises(ValueError):
        semantic.parse_semantic_v3_response(json.dumps(payload), view())


@pytest.mark.parametrize("mutation", ["extra", "missing", "fourth", "fakepath",
                                    "summary", "duplicate", "knownreason"])
def test_schema_and_raw_evidence_paths_are_strict(mutation):
    payload = response()
    if mutation == "extra":
        payload["action"] = "NOOP"
    elif mutation == "missing":
        del payload["unknown_reasons"]
    elif mutation == "fourth":
        payload["scores"]["motion-stagnation"] = 0.0
    elif mutation == "fakepath":
        payload["evidence"]["goal-priority"] = ["history[7].self.secret"]
    elif mutation == "summary":
        payload["evidence"]["goal-priority"] = ["summary"]
    elif mutation == "duplicate":
        payload["evidence"]["goal-priority"] *= 2
        payload["evidence"]["goal-priority"][1] = "history[7].self"
    else:
        payload["unknown_reasons"]["goal-priority"] = "symmetric-priority"
    with pytest.raises(ValueError):
        semantic.parse_semantic_v3_response(json.dumps(payload), view())


def test_unknown_is_masked_and_cannot_hide_known_input():
    value = direct_competition(view(), peer_priority=0.5)
    payload = response()
    payload["scores"]["yielding-preference"] = None
    payload["unknown_reasons"]["yielding-preference"] = "symmetric-priority"
    merged = semantic.merge_semantic_v3_label(
        semantic.parse_semantic_v3_response(json.dumps(payload), value), value)
    assert merged["scores"]["yielding-preference"] is None
    assert merged["known_masks"]["yielding-preference"] is False
    payload["unknown_reasons"]["yielding-preference"] = "missing-peer-attributes"
    with pytest.raises(ValueError, match="unknown"):
        semantic.merge_semantic_v3_label(
            semantic.parse_semantic_v3_response(json.dumps(payload), value), value)


@pytest.mark.parametrize("battery,want", [(0.0, 0.0), (0.05, 1.0), (0.051, 0.75),
                                         (0.15, 0.5), (0.3, 0.25)])
def test_goal_priority_battery_reference_boundaries(battery, want):
    value = view()
    value["history"][-1]["self"].update(task_phase="charging", battery=battery)
    value["history"][-1]["goal"]["goal_type"] = "charging"
    assert semantic.reference_semantics_v3(value)["scores"]["goal-priority"] == want


def test_reference_dimensions_are_complementary_and_peer_slots_are_not_ids():
    value = direct_competition(view())
    # Same geometry: peer's higher priority creates yielding, not a wall.
    first = semantic.reference_semantics_v3(value)["scores"]
    assert (first["goal-priority"], first["yielding-preference"],
            first["local-constraint"]) == (0.5, 0.5, 0.0)
    value["history"][0]["peers"][0] = dict(valid=True, priority_weight=1.0)
    value["history"][-1]["peers"][0]["priority_weight"] = 0.25
    # Undefined priority is rejected, never inferred from a historical slot.
    with pytest.raises(ValueError):
        semantic.reference_semantics_v3(value)
    value["history"][-1]["peers"][0]["priority_weight"] = 0.0
    assert semantic.reference_semantics_v3(value)["scores"]["yielding-preference"] == 0


@pytest.mark.parametrize("carrying,want", [(False, 0.0), (True, 0.75)])
def test_empty_can_pass_shelves_but_loaded_straight_corridor_is_constrained(
    carrying, want
):
    value = view()
    frame = value["history"][-1]
    frame["self"]["carrying"] = carrying
    for c in frame["local"]["cells"]:
        c["shelf"] = c["right"] != 0
    assert semantic.reference_semantics_v3(value)["scores"]["local-constraint"] == want


def test_loaded_shelf_with_peer_preserves_mobility_uncertainty():
    value = view()
    frame = value["history"][-1]
    frame["self"]["carrying"] = True
    cell(frame, 1, 0).update(shelf=True, other_agv=True)
    result = semantic.reference_semantics_v3(value)
    assert result["scores"]["local-constraint"] is None
    assert result["unknown_reasons"]["local-constraint"] == "ambiguous-shelf-mobility"


@pytest.mark.parametrize("count,want", [(2, 0.0), (3, 0.25), (4, 0.5),
                                      (5, 0.75), (6, 1.0), (7, 1.0)])
def test_stagnation_uses_contiguous_blocked_tail(count, want):
    value = view()
    for frame in value["history"][-count:]:
        frame["self"].update(previous_requested_action="FORWARD", previous_blocked=True)
    assert semantic.motion_stagnation_v3(value)["score"] == want


def test_scattered_blocking_and_phase_boundary_do_not_establish_continuity():
    value = view()
    for i in (1, 3, 5, 7):
        value["history"][i]["self"].update(previous_requested_action="FORWARD",
                                          previous_blocked=True)
    assert semantic.motion_stagnation_v3(value)["score"] == 0
    for frame in value["history"]:
        frame["self"].update(previous_requested_action="FORWARD", previous_blocked=True)
    value["history"][5]["self"]["task_phase"] = "delivery"
    value["history"][5]["goal"]["goal_type"] = "delivery"
    assert semantic.motion_stagnation_v3(value)["score"] == 0


def test_turn_alternation_requires_four_contiguous_turns_and_constant_distance():
    value = view()
    for frame, action in zip(value["history"][-4:], ("LEFT", "RIGHT", "LEFT", "RIGHT")):
        frame["self"]["previous_requested_action"] = action
    assert semantic.motion_stagnation_v3(value)["score"] == 0.25
    value["history"][3]["goal"]["normalized_distance"] = 0.09
    assert semantic.motion_stagnation_v3(value)["score"] == 0


@pytest.mark.parametrize("service", ["charging", "picking", "idle", "dead"])
def test_services_idle_and_death_never_create_positive_stagnation(service):
    value = view()
    for frame in value["history"]:
        frame["self"].update(previous_requested_action="FORWARD", previous_blocked=True)
    frame = value["history"][-1]
    frame["goal"].update(relative_forward=0, relative_right=0, normalized_distance=0)
    if service == "charging":
        for f in value["history"]:
            f["self"]["task_phase"] = f["goal"]["goal_type"] = "charging"
        cell(frame, 0, 0)["charger"] = True
    elif service == "picking":
        cell(frame, 0, 0)["workstation"] = True
    elif service == "idle":
        frame["self"]["task_phase"] = frame["goal"]["goal_type"] = "idle"
    else:
        frame["self"]["battery"] = 0
    result = semantic.motion_stagnation_v3(value)
    assert result["score"] == (None if service == "picking" else 0.0)
    if service == "charging":
        assert semantic.reference_semantics_v3(value)["unknown_reasons"][
            "yielding-preference"] == "hard-service-lock"


def test_peer_mapping_reconstructs_scale_and_rejects_noninteger_locations():
    value = direct_competition(view())
    assert semantic.associate_current_peers_v3(value) == {(1, 0): 0}
    value["history"][-1]["peers"].append(dict(valid=True, relative_forward=0.031,
                                             relative_right=0.0))
    with pytest.raises(ValueError):
        semantic.associate_current_peers_v3(value)


def test_prompt_transmits_three_continuous_targets_without_auxiliary_answer():
    value = direct_competition(view())
    prompt = build_semantic_prompt_v3(value)
    assert prompt.score_dimensions == ("goal-priority", "yielding-preference",
                                        "local-constraint")
    assert "0.37" in prompt.user_text
    assert "motion-stagnation" not in prompt.user_text
    assert "summary" not in prompt.available_evidence_paths
    assert "history[7].local.cells" in prompt.available_evidence_paths


def test_duplicate_json_object_keys_are_not_silently_overwritten():
    content = json.dumps(response()).replace('"goal-priority": 0.63',
                                            '"goal-priority": 0.1, "goal-priority": 0.63')
    with pytest.raises(ValueError, match="duplicate"):
        semantic.parse_semantic_v3_response(content, view())


def test_hidden_peer_state_is_rejected_before_prompt_or_evidence_generation():
    value = direct_competition(view())
    value["history"][-1]["peers"][0]["agent_id"] = 42
    with pytest.raises(ValueError):
        build_semantic_prompt_v3(value)


def test_weak_relation_requires_both_recent_actions_and_corridor_geometry():
    value = view()
    frame = value["history"][-1]
    for c in frame["local"]["cells"]:
        c["static_obstacle"] = c["right"] != 0
    frame["self"]["previous_requested_action"] = "FORWARD"
    cell(frame, 2, 0)["other_agv"] = True
    frame["peers"][0] = dict(valid=True, relative_forward=0.04,
                             relative_right=0.0, orientation="DOWN", carrying=False,
                             battery=0.8, task_phase="task", priority_weight=0.75,
                             previous_requested_action="FORWARD", previous_blocked=False)
    assert semantic.reference_semantics_v3(value)["scores"]["yielding-preference"] == 0.25
    frame["peers"][0]["previous_requested_action"] = "LEFT"
    assert semantic.reference_semantics_v3(value)["scores"]["yielding-preference"] == 0.0


def test_zero_auxiliary_evidence_preserves_the_window_that_excluded_oscillation():
    value = view()
    for frame, action in zip(value["history"][-4:], ("LEFT", "RIGHT", "LEFT", "RIGHT")):
        frame["self"]["previous_requested_action"] = action
    value["history"][3]["goal"]["normalized_distance"] = 0.09
    result = semantic.motion_stagnation_v3(value)
    assert result["score"] == 0
    assert "history[3].goal" in result["evidence"]


def test_prompt_compacts_cells_without_losing_any_raw_value():
    value = direct_competition(view())
    prompt = build_semantic_prompt_v3(value)
    raw = json.loads(prompt.user_text.split("RAW_OBSERVABLE_VIEW=", 1)[1])
    columns = ("forward", "right", "static_obstacle", "shelf", "workstation",
               "charger", "other_agv")
    restored = [dict(zip(columns, row)) for row in raw["history"][7]["local"]["cells"]]
    assert restored == value["history"][7]["local"]["cells"]
    assert len(prompt.user_text) < 25000


def test_self_priority_win_requires_peer_evidence_even_for_zero_yielding():
    value = direct_competition(view(), peer_priority=0.0)
    payload = response()
    payload["scores"]["yielding-preference"] = 0.0
    payload["evidence"]["yielding-preference"] = ["history[7].local.cells"]
    with pytest.raises(ValueError, match="evidence"):
        semantic.merge_semantic_v3_label(
            semantic.parse_semantic_v3_response(json.dumps(payload), value), value)


def test_real_209d_serializer_enters_v3_without_privileged_fields():
    import numpy as np

    from llm_mappo.mappo_baseline.structured_observation import (
        DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
    )
    from llm_mappo.mappo_llm_kd.temporal_view import serialize_temporal_semantic_view

    ledger = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2).feature_names
    row = np.zeros(209, dtype=np.float32)
    index = {name: position for position, name in enumerate(ledger)}
    for name in ("self_orientation_up", "self_phase_task", "goal_type_task",
                 "self_previous_action_0"):
        row[index[name]] = 1
    row[index["self_battery"]] = .8
    row[index["self_priority_weight"]] = .75
    row[index["goal_relative_forward"]] = .1
    row[index["goal_normalized_distance"]] = .1
    value = serialize_temporal_semantic_view(
        np.stack([row] * 8), np.zeros(8, dtype=bool), ledger, 8)
    result = semantic.reference_semantics_v3(value)
    assert result["scores"] == {"goal-priority": .75, "yielding-preference": 0,
                                "local-constraint": 0, "motion-stagnation": 0}
    assert build_semantic_prompt_v3(value).semantic_view_sha256


def test_missing_local_peer_slot_cannot_support_a_partial_association():
    value = direct_competition(view())
    cell(value["history"][-1], 2, 0)["other_agv"] = True
    with pytest.raises(ValueError, match="missing"):
        semantic.associate_current_peers_v3(value)


def _real_209d_competition_with_equidistant_peers():
    from math import hypot

    import numpy as np

    from llm_mappo.mappo_baseline.structured_observation import (
        DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
    )
    from llm_mappo.mappo_llm_kd.temporal_view import serialize_temporal_semantic_view

    ledger = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2).feature_names
    index = {name: position for position, name in enumerate(ledger)}
    row = np.zeros(209, dtype=np.float32)
    for name in ("self_orientation_up", "self_phase_task", "goal_type_task",
                 "self_previous_action_1", "self_previous_blocked",
                 "map_f1_r0_other_agv"):
        row[index[name]] = 1
    row[index["self_battery"]] = .8
    row[index["self_priority_weight"]] = .5
    row[index["goal_relative_forward"]] = .1
    row[index["goal_normalized_distance"]] = .1
    scale = hypot(19, 23)
    # UP-facing world offsets (0,-1), (1,-7), (5,-5) follow native peer
    # sorting: nearest first, then increasing world x for the sqrt(50) tie.
    for slot, (forward, right) in enumerate(((1, 0), (7, 1), (5, 5))):
        prefix = f"peer_{slot}_"
        for suffix in ("valid", "orientation_down", "phase_task",
                       "previous_action_1"):
            row[index[prefix + suffix]] = 1
        row[index[prefix + "relative_forward"]] = forward / scale
        row[index[prefix + "relative_right"]] = right / scale
        row[index[prefix + "battery"]] = .8
        row[index[prefix + "priority_weight"]] = .75
    return serialize_temporal_semantic_view(
        np.stack([row] * 8), np.zeros(8, dtype=bool), ledger, 8)


def test_real_209d_equal_distance_peers_preserve_known_direct_competition():
    value = _real_209d_competition_with_equidistant_peers()
    assert semantic.associate_current_peers_v3(value) == {(1, 0): 0}
    parsed = semantic.parse_semantic_v3_response(json.dumps(response()), value)
    merged = semantic.merge_semantic_v3_label(parsed, value)
    assert merged["scores"]["yielding-preference"] == .37
    assert merged["known_masks"]["yielding-preference"] is True
    assert merged["unknown_reasons"]["yielding-preference"] is None


def test_real_209d_clearly_misordered_peers_still_reject_association():
    value = _real_209d_competition_with_equidistant_peers()
    peers = value["history"][-1]["peers"]
    peers[0], peers[1] = peers[1], peers[0]
    with pytest.raises(ValueError, match="sorted"):
        semantic.associate_current_peers_v3(value)
    assert semantic.reference_semantics_v3(value)["unknown_reasons"][
        "yielding-preference"] == "ambiguous-peer-association"
