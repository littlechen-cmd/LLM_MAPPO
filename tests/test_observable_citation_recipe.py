"""Offline citation recipes preserve the frozen observable contract."""

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
from llm_mappo.mappo_llm_kd.temporal_view import serialize_temporal_semantic_view


def _view(actions, blocked=()):
    ledger = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2,
    ).feature_names
    indices = {name: index for index, name in enumerate(ledger)}
    rows = np.zeros((8, 209), dtype=np.float32)
    for name in ("self_orientation_up", "self_phase_task", "goal_type_task"):
        rows[:, indices[name]] = 1
    rows[:, indices["self_battery"]] = 0.5
    rows[:, indices["goal_normalized_distance"]] = 0.5
    for index, action in enumerate(actions):
        rows[index, indices[f"self_previous_action_{action}"]] = 1
    for index in blocked:
        rows[index, indices["self_previous_blocked"]] = 1
    return serialize_temporal_semantic_view(
        rows, np.zeros(8, dtype=np.bool_), ledger, 8,
    )


def _prompt(view):
    return build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V7)


def _recipes(prompt):
    line = prompt.user_text.split("EVIDENCE_RECIPES=", 1)[1].split("\n", 1)[0]
    return json.loads(line)["deadlock_severity"]


def _validate_positive(view, prompt, recipe):
    scores = {name: 0.0 for name in prompt.score_dimensions}
    scores["deadlock_severity"] = 0.5
    evidence = {name: ["summary.blocked_count"] for name in scores}
    evidence["deadlock_severity"] = recipe["evidence"]
    parsed = parse_label_response(
        json.dumps({"scores": scores, "evidence": evidence}),
        prompt.available_evidence_paths,
    )
    validate_medium_v7_semantic_support(parsed.scores, parsed.evidence, view)


def test_blocked_recipe_is_exact_and_passes_positive_validator():
    view = _view([0, 0, 0, 0, 0, 1, 1, 1], blocked=(5, 6, 7))
    prompt = add_observable_citation_recipe(_prompt(view), view)
    assert _recipes(prompt) == [{
        "mode": "blocked_forward",
        "evidence": [
            "history[7].self.previous_requested_action",
            "history[7].self.previous_blocked",
            "summary.blocked_count",
        ],
    }]
    _validate_positive(view, prompt, _recipes(prompt)[0])


def test_oscillation_recipe_uses_final_segment_and_original_distances():
    view = _view([3, 3, 2, 2, 3, 2, 3, 0])
    for frame in view["history"][:2]:
        frame["self"]["task_phase"] = "delivery"
        frame["goal"]["normalized_distance"] = 0.8
    prompt = add_observable_citation_recipe(_prompt(view), view)
    witness = observable_coordination_evidence(view)
    assert witness["action_observation_start"] == 3
    recipe, = _recipes(prompt)
    assert recipe["mode"] == "oscillation"
    assert recipe["evidence"] == [
        "history[3].self.previous_requested_action",
        "history[4].self.previous_requested_action",
        "history[2].goal.normalized_distance",
        "history[7].goal.normalized_distance",
    ]
    _validate_positive(view, prompt, recipe)


def test_both_witnesses_offer_independently_valid_recipes():
    view = _view([0, 2, 3, 2, 3, 1, 1, 1], blocked=(5, 6, 7))
    prompt = add_observable_citation_recipe(_prompt(view), view)
    assert [recipe["mode"] for recipe in _recipes(prompt)] == [
        "blocked_forward", "oscillation",
    ]
    for recipe in _recipes(prompt):
        _validate_positive(view, prompt, recipe)


@pytest.mark.parametrize("case", [
    "normal", "idle", "depleted", "transition", "window_first",
    "unmarked_forward", "distance_progress",
])
def test_unsupported_cases_require_zero_and_have_no_positive_recipe(case):
    view = _view([0, 2, 3, 2, 3, 0, 0, 0])
    if case == "normal":
        view = _view([0, 1, 2, 1, 3, 1, 1, 0])
    elif case == "idle":
        view["history"][-1]["self"]["task_phase"] = "idle"
    elif case == "depleted":
        view["history"][-1]["self"]["battery"] = 0
    elif case == "transition":
        for frame in view["history"][:5]:
            frame["goal"]["goal_type"] = "charger"
    elif case == "window_first":
        view = _view([1, 0, 0, 0, 0, 0, 1, 1], blocked=(0, 6, 7))
    elif case == "unmarked_forward":
        view["history"][5]["self"]["previous_requested_action"] = "FORWARD"
    elif case == "distance_progress":
        view["history"][-1]["goal"]["normalized_distance"] = 0.4
    prompt = add_observable_citation_recipe(_prompt(view), view)
    assert not observable_coordination_evidence(view)["positive_supported"]
    assert _recipes(prompt) == []
    assert "deadlock_severity=0" in prompt.user_text


def test_recipe_changes_only_user_text_and_hash_without_mutating_view():
    view = _view([0, 2, 3, 2, 3, 0, 0, 0])
    original_view = deepcopy(view)
    original = _prompt(view)
    updated = add_observable_citation_recipe(original, view)
    assert view == original_view
    assert updated != original
    for field in fields(original):
        if field.name not in {"user_text", "user_sha256"}:
            assert getattr(updated, field.name) == getattr(original, field.name)
    assert updated.user_sha256 == sha256(updated.user_text.encode("utf-8")).hexdigest()
    assert updated.user_sha256 != original.user_sha256
    assert updated.user_text.split("TEMPORAL_SEMANTIC_VIEW=", 1)[1] == (
        original.user_text.split("TEMPORAL_SEMANTIC_VIEW=", 1)[1]
    )
    assert updated.user_text.index("EVIDENCE_RECIPES=") < (
        updated.user_text.index("TEMPORAL_SEMANTIC_VIEW=")
    )
    assert "You decide score strengths and all other dimensions" in updated.user_text
    for recipe in _recipes(updated):
        assert set(recipe["evidence"]) <= original.available_evidence_paths


def test_recipe_rejects_missing_original_allowlist_path():
    view = _view([0, 0, 0, 0, 0, 1, 1, 1], blocked=(5, 6, 7))
    prompt = _prompt(view)
    prompt = replace(prompt, available_evidence_paths=(
        prompt.available_evidence_paths - {"summary.blocked_count"}
    ))
    with pytest.raises(ValueError, match="allowlist"):
        add_observable_citation_recipe(prompt, view)


@pytest.mark.parametrize("marker", ["", "TEMPORAL_SEMANTIC_VIEW=" * 2])
def test_recipe_requires_one_temporal_view_marker(marker):
    view = _view([0] * 8)
    prompt = replace(_prompt(view), user_text=marker)
    with pytest.raises(ValueError, match="TEMPORAL_SEMANTIC_VIEW"):
        add_observable_citation_recipe(prompt, view)
