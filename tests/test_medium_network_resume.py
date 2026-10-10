"""Offline durable transient-network authorization and identity checks."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from llm_mappo.mappo_llm_kd.dataset_labeling import _atomic_json
from llm_mappo.mappo_llm_kd.medium_labeling import (
    MediumLabelingHalt, _AttemptLedger, load_medium_label_config,
)
from llm_mappo.mappo_llm_kd.network_resume import (
    create_transient_network_authorizer,
)
from llm_mappo.mappo_llm_kd.prompt import LabelPrompt
from llm_mappo.mappo_llm_kd.provider import OpenAICompatibleProvider, ProviderConfig


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def run(tmp_path):
    config, _ = load_medium_label_config(
        "configs/llm_kd/medium-5ag-dataset-labeling-v7.yaml",
    )
    dataset = tmp_path / "dataset"
    (dataset / "labels").mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    (dataset / "windows.jsonl").write_text("fixture\n", encoding="utf-8")
    approval = {
        "schema": "medium-5ag-observable-coordination-authorization-v1",
        "decision": "approved",
        "external_data_authorization": "Existing owner external approval fixture",
        "semantic_contract_version": config["semantic_contract_version"],
        "dataset_manifest_sha256": _digest(dataset / "manifest.json"),
        "windows_sha256": _digest(dataset / "windows.jsonl"),
    }
    approval_path = dataset / "labels" / "observable-coordination-v7-authorization.json"
    _atomic_json(approval_path, approval)
    config["authorization_sha256"] = _digest(approval_path)
    config_sha = sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    output = dataset / "labels" / config_sha[:12] / "deepseek-official-flash"
    output.mkdir(parents=True)
    prior = {
        "event": "historical_budget",
        "schema": "medium-label-historical-budget-v1",
        "prior_identity_sha256": "a" * 64,
        "prior_attempts_sha256": "b" * 64,
        "authorization_sha256": config["authorization_sha256"],
        "http_attempts": {"primary": 59},
        "input_tokens_upper": 1000,
        "output_tokens_upper": 100,
        "conservative_cost_upper_usd": 0.1,
    }
    identity = {
        "label_config_sha256": config_sha,
        "dataset_manifest_sha256": approval["dataset_manifest_sha256"],
        "windows_sha256": approval["windows_sha256"],
        "semantic_contract_version": config["semantic_contract_version"],
        "authorization_sha256": config["authorization_sha256"],
        "pilot_sampling_salt": config["pilot_sampling_salt"],
        "quality_policy_version": config["quality_policy_version"],
        "quality_policy_authorization_sha256": (
            config["quality_policy_authorization_sha256"]
        ),
        "primary_provider": ProviderConfig(**config["models"]["primary"])
        .public_identity(),
        "comparison_provider": ProviderConfig(**config["models"]["comparison"])
        .public_identity(),
        "prior_budget_source": {
            "identity_sha256": prior["prior_identity_sha256"],
            "attempts_sha256": prior["prior_attempts_sha256"],
        },
    }
    _atomic_json(output / "identity.json", identity)
    ledger = _AttemptLedger(output / "attempts.jsonl", config)
    ledger.bind_prior_budget(prior)
    return output, config, config_sha, ledger


def _start(ledger, cache_key="cache-one", view_sha="c" * 64, scope="repeat"):
    prompt = LabelPrompt(
        prompt_version=ledger.config["prompt_version"],
        system_text="fixture", user_text="fixture",
        system_sha256="d" * 64, user_sha256="e" * 64,
        semantic_view_sha256=view_sha,
        score_dimensions=(), available_evidence_paths=frozenset(),
    )
    provider = OpenAICompatibleProvider(ProviderConfig(**ledger.config["models"][
        "comparison" if scope == "comparison" else "primary"
    ]))
    return ledger.begin(scope, cache_key, prompt, provider)


def _fail(ledger, attempt_id, reason="SSLEOFError", status="outcome_unknown"):
    ledger.settle(attempt_id, status, response_audit={
        "provider_failure": {"exception_type": "URLError", "reason_type": reason},
    })


@pytest.mark.parametrize("reason", [
    "RemoteDisconnected", "SSLEOFError", "TimeoutError", "ConnectionResetError",
    "ConnectionAbortedError", "IncompleteRead",
])
def test_transient_authorization_is_durable_and_preserves_reservation(run, reason):
    output, config, config_sha, ledger = run
    attempt = _start(ledger)
    _fail(ledger, attempt, reason)
    before = deepcopy(ledger.summary())
    original_log = ledger.path.read_bytes()
    original_identity = (output / "identity.json").read_bytes()
    authorize = create_transient_network_authorizer(output, config, config_sha=config_sha)
    assert authorize(ledger) is True
    assert ledger.halted is True
    after = ledger.summary()
    for name in ("http_attempts", "input_tokens_upper", "output_tokens_upper",
                 "conservative_cost_upper_usd", "confirmed_not_sent_reconciliations"):
        assert after[name] == before[name]
    assert ledger.path.read_bytes().startswith(original_log)
    assert (output / "identity.json").read_bytes() == original_identity
    receipt = json.loads((output / f"unknown-retry-attempt-{attempt}.json").read_bytes())
    assert receipt["schema"] == "medium-label-unknown-retry-authorization-v1"
    assert receipt["reviewer"] == "Codex root"
    assert receipt["transient_failure_reason_type"] == reason
    assert receipt["original_reservation_retained"] is True
    assert receipt["actual_provider_bill_verified"] is False
    assert receipt["original_attempt_log_sha256"] == sha256(original_log).hexdigest()
    assert receipt["authorized_attempt_log_sha256"] == _digest(ledger.path)
    resumed = _AttemptLedger(ledger.path, config)
    assert resumed.authorized_unknown_retry(attempt)
    assert authorize(resumed) is False
    audit = json.loads((output / "network-resume-policy.json").read_bytes())
    assert audit["one_retry_per_semantic_view"] is True
    assert audit["possible_duplicate_charge"] is True
    assert audit["cumulative_limits"] == config["limits"]


def test_multiple_authorizations_bind_each_current_log_digest(run):
    output, config, _, ledger = run
    first = _start(ledger)
    second = _start(ledger, "cache-two", "f" * 64)
    _fail(ledger, first)
    _fail(ledger, second, "TimeoutError")
    original_sha = _digest(ledger.path)
    assert create_transient_network_authorizer(output, config)(ledger)
    first_receipt = json.loads((output / "unknown-retry-attempt-1.json").read_bytes())
    second_receipt = json.loads((output / "unknown-retry-attempt-2.json").read_bytes())
    assert first_receipt["original_attempt_log_sha256"] == original_sha
    assert second_receipt["original_attempt_log_sha256"] == (
        first_receipt["authorized_attempt_log_sha256"]
    )
    assert second_receipt["authorized_attempt_log_sha256"] == _digest(ledger.path)
    assert len(_AttemptLedger(ledger.path, config).unknown_retry_authorizations) == 2


@pytest.mark.parametrize("reason,status", [
    ("HTTPError", "outcome_unknown"), ("RuntimeError", "outcome_unknown"),
    ("PermissionError", "outcome_unknown"), ("TimeoutError", "invalid_response"),
    ("TimeoutError", "missing_usage"), ("HTTPError", "provider_error"),
    ("TimeoutError", "confirmed_not_sent"),
])
def test_other_failures_are_not_authorized(run, reason, status):
    output, config, _, ledger = run
    attempt = _start(ledger)
    _fail(ledger, attempt, reason, status)
    original_log = ledger.path.read_bytes()
    assert create_transient_network_authorizer(output, config)(ledger) is False
    assert ledger.path.read_bytes() == original_log
    assert not ledger.unknown_retry_authorizations


def test_nontransient_failure_prevents_partial_batch_authorization(run):
    output, config, _, ledger = run
    first = _start(ledger)
    second = _start(ledger, "cache-two", "f" * 64)
    _fail(ledger, first)
    _fail(ledger, second, "HTTPError")
    original_log = ledger.path.read_bytes()
    assert create_transient_network_authorizer(output, config)(ledger) is False
    assert ledger.path.read_bytes() == original_log


def test_inflight_attempt_refuses_authorization(run):
    output, config, _, ledger = run
    first = _start(ledger)
    _start(ledger, "cache-two", "f" * 64)
    _fail(ledger, first)
    original_log = ledger.path.read_bytes()
    with pytest.raises(MediumLabelingHalt, match="in_flight"):
        create_transient_network_authorizer(output, config)(ledger)
    assert ledger.path.read_bytes() == original_log


def test_second_authorization_for_same_view_across_scopes_is_refused(run):
    output, config, _, ledger = run
    first = _start(ledger)
    _fail(ledger, first)
    authorize = create_transient_network_authorizer(output, config)
    assert authorize(ledger)
    ledger.halted = False
    second = _start(ledger, "other-cache-same-view", scope="comparison")
    _fail(ledger, second)
    original_log = ledger.path.read_bytes()
    with pytest.raises(MediumLabelingHalt, match="same_semantic_view"):
        authorize(ledger)
    assert ledger.path.read_bytes() == original_log


@pytest.mark.parametrize("mismatch", [
    "config", "config_sha", "provider", "identity", "manifest", "windows",
    "approval", "prior", "policy",
])
def test_frozen_binding_mismatches_refuse_authorization(run, mismatch):
    output, config, config_sha, ledger = run
    attempt = _start(ledger)
    _fail(ledger, attempt)
    original_log = ledger.path.read_bytes()
    authorize = create_transient_network_authorizer(output, config)
    dataset = output.parents[2]
    if mismatch in {"config", "config_sha", "provider"}:
        config = deepcopy(config)
        if mismatch == "provider":
            config["models"]["primary"]["timeout_seconds"] += 1
        elif mismatch == "config":
            config["limits"]["conservative_usd"] += 1
        else:
            config_sha = "0" * 64
    elif mismatch == "identity":
        (output / "identity.json").write_bytes(
            (output / "identity.json").read_bytes() + b" "
        )
    elif mismatch in {"manifest", "windows"}:
        filename = "manifest.json" if mismatch == "manifest" else "windows.jsonl"
        (dataset / filename).write_text("changed\n", encoding="utf-8")
    elif mismatch == "approval":
        path = dataset / "labels" / "observable-coordination-v7-authorization.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mismatch == "prior":
        ledger.prior_budget["prior_identity_sha256"] = "0" * 64
    else:
        _atomic_json(output / "network-resume-policy.json", {"decision": "different"})
    with pytest.raises(MediumLabelingHalt):
        if mismatch in {"config", "config_sha", "provider"}:
            create_transient_network_authorizer(output, config, config_sha=config_sha)
        else:
            authorize(ledger)
    assert ledger.path.read_bytes() == original_log
