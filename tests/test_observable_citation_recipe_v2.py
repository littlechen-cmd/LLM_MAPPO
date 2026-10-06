"""Offline per-head citation guidance leaves frozen scoring rules intact."""

from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
import json

import numpy as np
import pytest

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.labeling import parse_label_response
from llm_mappo.mappo_llm_kd.medium_prompt import (
    MEDIUM_LABEL_PROMPT_V7, build_medium_label_prompt,
    observable_coordination_evidence, validate_medium_v7_semantic_support,
)
from llm_mappo.mappo_llm_kd.observable_citation_recipe import (
    add_observable_citation_recipe,
)
from llm_mappo.mappo_llm_kd.observable_citation_recipe_v2 import (
    add_observable_citation_recipe_v2,
)
from llm_mappo.mappo_llm_kd.temporal_view import serialize_temporal_semantic_view


def _view(occupancy="absent"):
    ledger = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2,
    ).feature_names
    indices = {name: index for index, name in enumerate(ledger)}
    rows = np.zeros((8, 209), dtype=np.float32)
    for name in ("self_orientation_up", "self_phase_task", "goal_type_task"):
        rows[:, indices[name]] = 1
    rows[:, indices["self_battery"]] = 0.5
    rows[:, indices["goal_normalized_distance"]] = 0.5
    for index, action in enumerate([0, 2, 3, 2, 3, 0, 0, 0]):
        rows[index, indices[f"self_previous_action_{action}"]] = 1
    view = serialize_temporal_semantic_view(
        rows, np.zeros(8, dtype=np.bool_), ledger, 8,
    )
    if occupancy != "absent":
        current = 1 if occupancy == "current" else 0
        view["history"][-1]["local"]["visible_peer_cell_count"] = current
        view["history"][-1]["local"]["peer_occupied"]["forward"] = bool(current)
        view["summary"]["visible_peer_cell_statistics"].update(
            current=current, maximum=1, mean=0.625,
        )
    return view


def _prompt(view):
    return build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V7)


def _json_line(prompt, marker):
    return json.loads(prompt.user_text.split(marker, 1)[1].split("\n", 1)[0])


def _validate(view, prompt, local_score):
    recipes = _json_line(prompt, "HEAD_EVIDENCE_RECIPES=")
    scores = {name: 0.0 for name in prompt.score_dimensions}
    scores.update(mission_urgency=0.5, concession_appropriateness=local_score,
                  interaction_risk=local_score, deadlock_severity=0.5)
    evidence = {name: recipe["evidence"] for name, recipe in recipes.items()}
    evidence["deadlock_severity"] = _json_line(
        prompt, "\nEVIDENCE_RECIPES=",
    )["deadlock_severity"][0]["evidence"]
    parsed = parse_label_response(
        json.dumps({"scores": scores, "evidence": evidence}),
        prompt.available_evidence_paths,
    )
    validate_medium_v7_semantic_support(parsed.scores, parsed.evidence, view)


def test_absent_occupancy_requires_zero_and_current_count_citation():
    view = _view()
    prompt = add_observable_citation_recipe_v2(_prompt(view), view)
    recipes = _json_line(prompt, "HEAD_EVIDENCE_RECIPES=")
    for name in ("concession_appropriateness", "interaction_risk"):
        assert recipes[name]["mode"] == "absent"
        assert recipes[name]["evidence"] == [
            "history[7].local.visible_peer_cell_count",
        ]
        assert recipes[name]["score_constraint"] == "zero"
    _validate(view, prompt, 0)
    with pytest.raises(ValueError, match="positive local"):
        _validate(view, prompt, 0.25)


def test_past_only_occupancy_requires_maximum_and_zero_current_cites_with_cap():
    view = _view("past_only")
    prompt = add_observable_citation_recipe_v2(_prompt(view), view)
    recipes = _json_line(prompt, "HEAD_EVIDENCE_RECIPES=")
    for name in ("concession_appropriateness", "interaction_risk"):
        assert recipes[name]["mode"] == "past_only"
        assert recipes[name]["evidence"] == [
            "summary.visible_peer_cell_statistics.maximum",
            "history[7].local.visible_peer_cell_count",
        ]
        assert recipes[name]["score_constraint"] == "at_most_0.25"
    assert "remote peer valid or priority" in prompt.user_text
    _validate(view, prompt, 0.25)
    with pytest.raises(ValueError, match="at most 0.25"):
        _validate(view, prompt, 0.5)


def test_current_occupancy_uses_positive_count_and_only_true_optional_paths():
    view = _view("current")
    prompt = add_observable_citation_recipe_v2(_prompt(view), view)
    recipes = _json_line(prompt, "HEAD_EVIDENCE_RECIPES=")
    for name in ("concession_appropriateness", "interaction_risk"):
        assert recipes[name]["mode"] == "current"
        assert recipes[name]["evidence"] == [
            "history[7].local.visible_peer_cell_count",
        ]
        assert recipes[name]["optional_evidence"] == [
            "history[7].local.peer_occupied.forward",
        ]
        assert recipes[name]["score_constraint"] == "model_decides"
    _validate(view, prompt, 0.5)


@pytest.mark.parametrize("idle_path", ["self", "goal", None])
def test_urgency_cites_current_phase_goal_and_preserves_idle_zero_contract(idle_path):
    view = _view()
    if idle_path is not None:
        name = "task_phase" if idle_path == "self" else "goal_type"
        view["history"][-1][idle_path][name] = "idle"
    prompt = add_observable_citation_recipe_v2(_prompt(view), view)
    recipe = _json_line(prompt, "HEAD_EVIDENCE_RECIPES=")["mission_urgency"]
    assert recipe["evidence"] == [
        "history[7].self.task_phase", "history[7].goal.goal_type",
    ]
    assert recipe["score_constraint"] == (
        "model_decides" if idle_path is None else "zero"
    )
    if idle_path is None:
        assert recipe["optional_evidence"] == [
            "history[7].self.battery", "history[7].self.priority_weight",
        ]
    else:
        assert "mission_urgency=0" in prompt.user_text
        scores = {name: 0.0 for name in prompt.score_dimensions}
        evidence = {name: ["summary.blocked_count"] for name in scores}
        evidence["mission_urgency"] = recipe["evidence"]
        parsed = parse_label_response(
            json.dumps({"scores": scores, "evidence": evidence}),
            prompt.available_evidence_paths,
        )
        validate_medium_v7_semantic_support(parsed.scores, parsed.evidence, view)
        scores["mission_urgency"] = 0.25
        with pytest.raises(ValueError, match="zero mission_urgency"):
            validate_medium_v7_semantic_support(scores, evidence, view)


def test_v2_preserves_view_frozen_detector_v1_recipes_and_other_prompt_fields():
    view = _view("past_only")
    original_view = deepcopy(view)
    witness = observable_coordination_evidence(view)
    original = _prompt(view)
    v1 = add_observable_citation_recipe(original, view)
    updated = add_observable_citation_recipe_v2(original, view)
    assert view == original_view
    assert observable_coordination_evidence(view) == witness
    assert _json_line(updated, "\nEVIDENCE_RECIPES=") == _json_line(
        v1, "\nEVIDENCE_RECIPES=",
    )
    for field in fields(original):
        if field.name not in {"user_text", "user_sha256"}:
            assert getattr(updated, field.name) == getattr(original, field.name)
    assert updated.user_sha256 == sha256(updated.user_text.encode("utf-8")).hexdigest()
    assert updated.user_text.split("TEMPORAL_SEMANTIC_VIEW=", 1)[1] == (
        original.user_text.split("TEMPORAL_SEMANTIC_VIEW=", 1)[1]
    )
    assert updated.user_text.index("HEAD_EVIDENCE_RECIPES=") < (
        updated.user_text.index("TEMPORAL_SEMANTIC_VIEW=")
    )
    for recipe in _json_line(updated, "HEAD_EVIDENCE_RECIPES=").values():
        paths = recipe["evidence"] + recipe.get("optional_evidence", [])
        assert set(paths) <= original.available_evidence_paths
    assert "copy every evidence path from one complete" in updated.user_text


def test_v2_rejects_recipe_path_missing_from_original_allowlist():
    view = _view("past_only")
    prompt = _prompt(view)
    prompt = replace(prompt, available_evidence_paths=(
        prompt.available_evidence_paths - {"summary.visible_peer_cell_statistics.maximum"}
    ))
    with pytest.raises(ValueError, match="allowlist"):
        add_observable_citation_recipe_v2(prompt, view)
