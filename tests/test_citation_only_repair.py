"""Owner-approved one-request repair changes citations, never teacher scores."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_medium_network_resume import run, _start  # noqa: F401
from test_medium_labeling import _reply, _window
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.citation_only_repair import (
    CITATION_REPAIR_ERROR, OWNER_APPROVAL_QUOTE,
    citation_repair_implementation_sha256, create_citation_only_repair_policy,
)
from llm_mappo.mappo_llm_kd.dataset_labeling import _atomic_json
from llm_mappo.mappo_llm_kd.labeling import AtomicLabelStore
from llm_mappo.mappo_llm_kd.medium_labeling import (
    MediumLabelingHalt, _label_one, _run_full_batches,
)
from llm_mappo.mappo_llm_kd.medium_prompt import (
    MEDIUM_LABEL_PROMPT_V7, build_medium_label_prompt,
)
from llm_mappo.mappo_llm_kd.network_resume import create_transient_network_authorizer
from llm_mappo.mappo_llm_kd.provider import OpenAICompatibleProvider, ProviderConfig


def _builder(view):
    return build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V7)


@pytest.fixture
def exhausted(run, monkeypatch):
    output, config, config_sha, ledger = run
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-fixture-only")
    window = _window()
    names = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2,
    ).feature_names
    rows = [list(row) for row in window["observations"]]
    for row, action in zip(rows, [1, 2, 2, 3, 0, 2, 3, 2]):
        for index in range(5):
            row[names.index(f"self_previous_action_{index}")] = float(index == action)
        row[names.index("self_battery")] = 0.5
    window["observations"] = rows
    payload = {
        "scores": {
            "mission_urgency": 0.5, "concession_appropriateness": 0.0,
            "interaction_risk": 0.0, "deadlock_severity": 0.5,
        },
        "evidence": {
            "mission_urgency": ["history[7].self.task_phase"],
            "concession_appropriateness": ["history[7].local.visible_peer_cell_count"],
            "interaction_risk": ["history[7].local.visible_peer_cell_count"],
            "deadlock_severity": [
                "history[1].self.previous_requested_action",
                "history[7].self.previous_requested_action",
                "history[0].goal.normalized_distance",
                "history[7].goal.normalized_distance",
            ],
        },
    }
    calls = []
    responses = [_reply(content=json.dumps(payload)) for _ in range(2)]

    def transport(endpoint, body, headers, timeout):
        del endpoint, headers, timeout
        calls.append(json.loads(body))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    provider = OpenAICompatibleProvider(
        ProviderConfig(**config["models"]["primary"]), transport=transport,
    )
    store = AtomicLabelStore(output / "labels.jsonl")
    with pytest.raises(MediumLabelingHalt, match="after_retry"):
        _label_one(window, "primary", provider, store, ledger, _builder)
    assert len(calls) == 2
    return (output, config, config_sha, ledger, window, payload, calls,
            responses, provider, store)


def _approval(exhausted):
    output, config, config_sha, ledger, *_ = exhausted
    raw_log = ledger.path.read_bytes()
    approval = {
        "schema": "medium-label-citation-only-repair-authorization-v1",
        "decision": "approved",
        "owner_approval_quote": OWNER_APPROVAL_QUOTE,
        "dataset_manifest_sha256": sha256(
            (output.parents[2] / "manifest.json").read_bytes()
        ).hexdigest(),
        "windows_sha256": sha256(
            (output.parents[2] / "windows.jsonl").read_bytes()
        ).hexdigest(),
        "label_config_sha256": config_sha,
        "identity_sha256": sha256((output / "identity.json").read_bytes()).hexdigest(),
        "attempts_sha256": sha256(raw_log).hexdigest(),
        "attempts_size_bytes": len(raw_log),
        "maximum_extra_requests_per_semantic_view": 1,
        "allowed_validation_error": CITATION_REPAIR_ERROR,
        "original_attempts_retained": True,
        "model_scores_rewritten": False,
        "cumulative_limits": config["limits"],
        "retry_rate_cap": 0.10,
        "implementation_sha256": citation_repair_implementation_sha256(),
    }
    path = output / "citation-repair-authorization.json"
    _atomic_json(path, approval)
    return path, sha256(path.read_bytes()).hexdigest()


def _policy(exhausted):
    path, digest = _approval(exhausted)
    return create_citation_only_repair_policy(
        exhausted[0], exhausted[1], path,
        authorization_sha256=digest, config_sha=exhausted[2],
    )


def _valid(payload):
    result = deepcopy(payload)
    result["scores"]["deadlock_severity"] = 0.4
    result["evidence"]["deadlock_severity"][1] = (
        "history[3].self.previous_requested_action"
    )
    return _reply(content=json.dumps(result))


def test_without_authorization_stays_exhausted(exhausted):
    _, _, _, ledger, window, _, calls, _, provider, store = exhausted
    with pytest.raises(MediumLabelingHalt, match="manual_reconciliation"):
        _label_one(window, "primary", provider, store, ledger, _builder)
    assert len(calls) == 2


def test_approved_one_repair_preserves_failures_cost_cache_and_fresh_scores(exhausted):
    output, _, _, ledger, window, payload, calls, responses, provider, store = exhausted
    original_log = ledger.path.read_bytes()
    original_scores = deepcopy(ledger.settled["2"]["response_audit"]["parsed_scores"])
    old_cost = ledger.summary()["conservative_cost_upper_usd"]
    policy = _policy(exhausted)
    responses.append(_valid(payload))
    record = _label_one(window, "primary", provider, store, ledger, _builder,
                        citation_repair_policy=policy)
    assert record["scores"]["deadlock_severity"] == 0.4
    assert ledger.settled["2"]["response_audit"]["parsed_scores"] == original_scores
    assert ledger.path.read_bytes().startswith(original_log)
    assert ledger.summary()["conservative_cost_upper_usd"] > old_cost
    assert ledger.prior_budget["http_attempts"]["primary"] == 59
    assert len(calls) == 3
    user = calls[-1]["messages"][1]["content"]
    assert "CITATION_ONLY_REPAIR_V1=" in user
    assert '"path":"history[1].self.previous_requested_action","value":"LEFT"' in user
    assert '"path":"history[3].self.previous_requested_action","value":"RIGHT"' in user
    assert ledger.starts["3"]["prompt_variant"] == "citation-only-repair-v1"
    assert (output / "citation-repair-attempt-3.json").is_file()
    cached_bytes = (output / "labels.jsonl").read_bytes()
    assert _label_one(window, "primary", provider, store, ledger, _builder,
                      citation_repair_policy=policy) == record
    assert len(calls) == 3
    assert (output / "labels.jsonl").read_bytes() == cached_bytes


def test_second_failure_exhausts_extra_request(exhausted):
    _, _, _, ledger, window, payload, calls, responses, provider, store = exhausted
    policy = _policy(exhausted)
    responses.append(_reply(content=json.dumps(payload)))
    with pytest.raises(MediumLabelingHalt, match="after_retry"):
        _label_one(window, "primary", provider, store, ledger, _builder,
                   citation_repair_policy=policy)
    with pytest.raises(MediumLabelingHalt):
        _label_one(window, "primary", provider, store, ledger, _builder,
                   citation_repair_policy=policy)
    assert len(calls) == 3


def test_extra_request_limit_is_per_view_across_scopes(exhausted):
    _, _, _, ledger, window, payload, calls, responses, provider, store = exhausted
    policy = _policy(exhausted)
    responses.append(_valid(payload))
    _label_one(window, "primary", provider, store, ledger, _builder,
               citation_repair_policy=policy)
    responses.extend([_reply(content=json.dumps(payload)) for _ in range(2)])
    with pytest.raises(MediumLabelingHalt, match="after_retry"):
        _label_one(window, "repeat", provider, store, ledger, _builder,
                   citation_repair_policy=policy)
    assert len(calls) == 5
    assert sum(start["prompt_variant"] == "citation-only-repair-v1"
               for start in ledger.starts.values()) == 1


def test_network_loss_of_extra_request_cannot_receive_network_retry(exhausted):
    output, config, _, ledger, window, _, calls, responses, provider, store = exhausted
    policy = _policy(exhausted)
    responses.append(TimeoutError("offline loss"))
    with pytest.raises(MediumLabelingHalt, match="provider_attempt"):
        _label_one(window, "primary", provider, store, ledger, _builder,
                   citation_repair_policy=policy)
    assert create_transient_network_authorizer(output, config)(ledger) is False
    assert len(calls) == 3
    assert not ledger.unknown_retry_authorizations


def test_repair_defers_until_all_sibling_requests_are_settled(exhausted):
    _, _, _, ledger, window, payload, calls, responses, provider, store = exhausted
    policy = _policy(exhausted)
    sibling = _start(ledger, "sibling-cache", "f" * 64)
    with pytest.raises(MediumLabelingHalt, match="requires_settled_batch"):
        _label_one(window, "primary", provider, store, ledger, _builder,
                   citation_repair_policy=policy)
    assert len(calls) == 2
    ledger.settle(sibling, "ok")
    responses.append(_valid(payload))
    assert _label_one(window, "primary", provider, store, ledger, _builder,
                      citation_repair_policy=policy)["status"] == "ok"
    assert len(calls) == 3


def test_worker_always_defers_extra_then_coordinator_performs_it(exhausted):
    _, _, _, ledger, window, payload, calls, responses, provider, store = exhausted
    policy = _policy(exhausted)
    original_starts = deepcopy(ledger.starts)
    with pytest.raises(MediumLabelingHalt, match="requires_settled_batch"):
        _label_one(window, "primary", provider, store, ledger, _builder,
                   citation_repair_policy=policy, allow_citation_repair=False)
    assert ledger.starts == original_starts
    assert len(calls) == 2
    responses.append(_valid(payload))
    _run_full_batches([("primary", window)], provider, store, ledger, 1,
                      _builder, citation_repair_policy=policy)
    assert len(calls) == 3


def test_malformed_authorization_is_refused(exhausted):
    path, _ = _approval(exhausted)
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(MediumLabelingHalt):
        create_citation_only_repair_policy(
            exhausted[0], exhausted[1], path,
            authorization_sha256=sha256(path.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize("tamper", ["digest", "owner", "identity", "limits", "code"])
def test_tampered_authorization_fails_closed(exhausted, tamper):
    path, digest = _approval(exhausted)
    body = json.loads(path.read_bytes())
    if tamper == "owner":
        body["owner_approval_quote"] = "not owner approval"
    elif tamper == "identity":
        body["identity_sha256"] = "0" * 64
    elif tamper == "limits":
        body["maximum_extra_requests_per_semantic_view"] = 2
    elif tamper == "code":
        body["implementation_sha256"] = {}
    _atomic_json(path, body)
    if tamper != "digest":
        digest = sha256(path.read_bytes()).hexdigest()
    else:
        digest = "0" * 64
    with pytest.raises(MediumLabelingHalt):
        create_citation_only_repair_policy(
            exhausted[0], exhausted[1], path, authorization_sha256=digest,
        )
    assert len(exhausted[6]) == 2


@pytest.mark.parametrize("wrong_error", [
    "deadlock_severity positive score lacks observable sustained blocking or oscillation witness.",
    "interaction_risk positive score lacks cited positive local peer occupancy.",
])
def test_other_semantic_errors_are_not_extra_repair_candidates(exhausted, wrong_error):
    _, _, _, ledger, window, _, calls, _, provider, store = exhausted
    event = ledger.settled["2"]
    event["response_audit"]["parse_error"] = wrong_error
    policy = _policy(exhausted)
    with pytest.raises(MediumLabelingHalt):
        _label_one(window, "primary", provider, store, ledger, _builder,
                   citation_repair_policy=policy)
    assert len(calls) == 2
