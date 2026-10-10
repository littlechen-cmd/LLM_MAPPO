"""Semantic boundaries and provenance for the approved observable contract."""

from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.medium_prompt import (
    MEDIUM_LABEL_PROMPT_V7, SEMANTIC_CONTRACT_V2,
    build_medium_label_prompt, observable_coordination_evidence,
    validate_medium_v7_semantic_support,
)
from llm_mappo.mappo_llm_kd.pretraining import validate_pretraining_identity
from llm_mappo.mappo_llm_kd.pretraining_run import _validate_label_artifact
from llm_mappo.mappo_llm_kd.representation_export import (
    export_actor_representation, import_actor_representation,
)
from llm_mappo.mappo_llm_kd.rule_semantic import rule_semantic_scores
from llm_mappo.mappo_llm_kd.labeling import AtomicLabelStore
from llm_mappo.mappo_llm_kd.medium_labeling import (
    _AttemptLedger, _label_one, _response_audit, load_medium_label_config,
)
from llm_mappo.mappo_llm_kd.provider import OpenAICompatibleProvider, ProviderConfig
from llm_mappo.mappo_llm_kd.temporal_view import serialize_temporal_semantic_view


def _view(actions, blocked=()):
    ledger = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    ).feature_names
    index = {name: i for i, name in enumerate(ledger)}
    rows = np.zeros((8, 209), dtype=np.float32)
    for name in ("self_orientation_up", "self_phase_task", "goal_type_task"):
        rows[:, index[name]] = 1
    rows[:, index["self_battery"]] = 0.5
    rows[:, index["goal_normalized_distance"]] = 0.5
    for i, action in enumerate(actions):
        rows[i, index[f"self_previous_action_{action}"]] = 1
    for i in blocked:
        rows[i, index["self_previous_blocked"]] = 1
    return serialize_temporal_semantic_view(
        rows, np.zeros(8, dtype=np.bool_), ledger, 8,
    )


def _identity(new=True):
    values = {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": 209,
        "ontology_schema": (
            "agent-centric-observable-coordination-four-score-v2" if new
            else "agent-centric-four-score-ontology-v1"
        ),
        "semantic_dimensions": (
            "mission_urgency", "concession_appropriateness",
            "interaction_risk", "deadlock_severity",
        ),
        "actor_architecture": "mappo-actor-encoder-gru-v1",
        "hidden_dim": 64, "dataset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    if new:
        values["semantic_contract_version"] = SEMANTIC_CONTRACT_V2
    return values


def test_contract_version_survives_export_and_rejects_legacy_import(tmp_path):
    identity = _identity()
    validate_pretraining_identity(identity)
    validate_pretraining_identity(_identity(False))
    path = tmp_path / "representation.pt"
    actor = MAPPOActor(209, 5, 64)
    export_actor_representation(path, actor, identity=identity)
    with pytest.raises(ValueError, match="identity"):
        import_actor_representation(
            path, actor, expected_identity=_identity(False),
            expected_artifact_sha256=sha256(path.read_bytes()).hexdigest(),
        )


def test_v2_ontology_cannot_omit_or_forge_contract_version():
    identity = _identity()
    identity.pop("semantic_contract_version")
    with pytest.raises(ValueError):
        validate_pretraining_identity(identity)
    forged = _identity()
    forged["semantic_contract_version"] = "old-unversioned-deadlock"
    with pytest.raises(ValueError):
        validate_pretraining_identity(forged)


def test_sustained_witness_excludes_old_phase_transition_and_window_first():
    view = _view([1, 0, 0, 0, 0, 1, 1, 1], blocked=(0, 5, 6, 7))
    assert observable_coordination_evidence(view)["positive_supported"]
    for frame in view["history"][:5]:
        frame["self"]["task_phase"] = "delivery"
    assert not observable_coordination_evidence(view)["positive_supported"]
    first_only = _view([1, 0, 0, 0, 0, 0, 1, 1], blocked=(0, 6, 7))
    assert not observable_coordination_evidence(first_only)["positive_supported"]


def test_alternating_turns_require_active_goal_and_no_unmarked_forward():
    view = _view([0, 2, 3, 2, 3, 0, 0, 0])
    assert observable_coordination_evidence(view)["oscillation_supported"]
    depleted = deepcopy(view)
    depleted["history"][-1]["self"]["battery"] = 0
    assert not observable_coordination_evidence(depleted)["positive_supported"]
    view["history"][5]["self"]["previous_requested_action"] = "FORWARD"
    assert not observable_coordination_evidence(view)["positive_supported"]


def test_normal_navigation_positive_fourth_is_rejected_not_rewritten():
    view = _view([0, 1, 2, 1, 3, 1, 1, 0])
    scores = {name: 0.0 for name in _identity()["semantic_dimensions"]}
    scores["deadlock_severity"] = 0.25
    evidence = {name: ["summary.blocked_count"] for name in scores}
    with pytest.raises(ValueError, match="witness"):
        validate_medium_v7_semantic_support(scores, evidence, view)
    assert scores["deadlock_severity"] == 0.25
    prompt = build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V7)
    assert SEMANTIC_CONTRACT_V2 in prompt.user_text


def test_rule_remote_valid_peer_is_not_local_interaction_evidence():
    view = _view([0] * 8)
    for frame in view["history"]:
        frame["peers"] = [{"valid": True}] * 3
    scores = rule_semantic_scores(view)
    assert scores["interaction_risk"] == 0
    assert scores["concession_appropriateness"] == 0


def test_rule_strength_uses_only_direct_witness_and_local_peer_counts():
    view = _view([0] + [1] * 7, blocked=tuple(range(1, 8)))
    assert rule_semantic_scores(view)["deadlock_severity"] == 0.75
    view["history"][-1]["self"]["battery"] = 0
    assert rule_semantic_scores(view)["deadlock_severity"] == 0
    normal = _view([0] * 8)
    normal["history"][-1]["local"]["visible_peer_cell_count"] = 1
    scores = rule_semantic_scores(normal)
    assert scores["interaction_risk"] == scores["concession_appropriateness"] == 0.25


def test_v2_loader_requires_each_label_receipt_contract(tmp_path):
    dataset = tmp_path / "dataset"
    labels = dataset / "labels"
    labels.mkdir(parents=True)
    (dataset / "manifest.json").write_text(json.dumps({
        "semantic_contract_version": SEMANTIC_CONTRACT_V2,
    }))
    raw = json.dumps({"status": "ok", "view_id": "train:fixture", "receipt": {}})
    (labels / "labels.jsonl").write_text(raw)
    (labels / "state.json").write_text('{"status":"completed"}')
    (labels / "manifest.json").write_text(json.dumps({
        "schema": "llm-dataset-label-run-v1",
        "dataset_manifest_sha256": sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "labels_sha256": sha256(raw.encode()).hexdigest(),
        "successful_records": 1, "pretraining_eligible": True,
        "semantic_contract_version": SEMANTIC_CONTRACT_V2,
    }))
    with pytest.raises(ValueError, match="contract"):
        _validate_label_artifact(dataset, labels)


def test_v7_record_receipt_and_cached_record_keep_explicit_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-fixture-key")
    ledger_names = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    ).feature_names
    row = np.zeros(209, dtype=np.float32)
    for name in ("self_orientation_up", "self_phase_task", "goal_type_task",
                 "self_previous_action_0"):
        row[ledger_names.index(name)] = 1
    row[ledger_names.index("self_battery")] = 0.5
    row[ledger_names.index("goal_normalized_distance")] = 0.5
    window = {
        "partition": "train", "episode_id": "fixture", "agent_id": 1,
        "start_timestep": 0, "end_timestep": 7, "stratum": "fixture",
        "observations": [row.tolist()] * 8, "reset_masks": [False] * 8,
    }
    config, _ = load_medium_label_config(
        "configs/llm_kd/medium-5ag-dataset-labeling-v7.yaml"
    )
    calls = []

    def transport(*args):
        calls.append(1)
        scores = {name: 0.0 for name in _identity()["semantic_dimensions"]}
        return {
            "model": "deepseek-flash", "system_fingerprint": "fixture",
            "choices": [{"message": {"content": json.dumps({
                "scores": scores,
                "evidence": {name: ["summary.blocked_count"] for name in scores},
            })}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100,
                      "total_tokens": 200},
        }

    provider = OpenAICompatibleProvider(
        ProviderConfig(**config["models"]["primary"]), transport=transport,
    )
    store = AtomicLabelStore(tmp_path / "labels.jsonl")
    attempts = _AttemptLedger(tmp_path / "attempts.jsonl", config)
    builder = lambda view: build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V7)
    record = _label_one(window, "primary", provider, store, attempts, builder)
    assert record["receipt"]["semantic_contract_version"] == SEMANTIC_CONTRACT_V2
    cached = _label_one(window, "primary", provider, store, attempts, builder)
    assert cached == record and len(calls) == 1


def test_invalid_response_audit_can_settle_oversized_integer_score():
    audit = _response_audit(json.dumps({
        "scores": {"mission_urgency": 10 ** 30},
        "evidence": {"mission_urgency": ["summary.blocked_count"]},
    }), {"summary.blocked_count"}, parse_error="score outside range")
    assert audit["parsed_scores"] == {}
    assert audit["cited_original_evidence"] == {
        "mission_urgency": ["summary.blocked_count"],
    }
