"""Durable root-authorized transient-network retries without budget refunds."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .dataset_labeling import _atomic_json
from .medium_labeling import MediumLabelingHalt, _AttemptLedger
from .medium_prompt import SEMANTIC_CONTRACT_V2
from .provider import ProviderConfig


TRANSIENT_NETWORK_REASONS = (
    "RemoteDisconnected", "SSLEOFError", "TimeoutError", "ConnectionResetError",
    "ConnectionAbortedError", "IncompleteRead",
)
_REVIEWER = "Codex root"
_POLICY_REVIEWER = "Codex root (explicit transient-network resume policy)"


def create_transient_network_authorizer(
    output: Path, config: Mapping[str, object], *, config_sha: str | None = None,
):
    """Return a callback for a settled batch; never clear the ledger halt flag.

    Runner startup must already have validated the recursive prior budget
    chain. This callback binds its immediate hashes to the unchanged identity.
    The caller resumes scheduling only after its own halt and budget checks.
    """

    output = Path(output).resolve()
    values = deepcopy(dict(config))
    computed_sha = sha256(json.dumps(
        values, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if config_sha is not None and config_sha != computed_sha:
        raise MediumLabelingHalt("network_resume_config_digest_differs")
    identity, identity_sha, approval = _validate_bindings(output, values, computed_sha)
    policy = {
        "schema": "medium-label-transient-network-resume-policy-v1",
        "reviewer": _REVIEWER,
        "policy_reviewer": _POLICY_REVIEWER,
        "identity_sha256": identity_sha,
        "label_config_sha256": computed_sha,
        "semantic_contract_version": values["semantic_contract_version"],
        "existing_external_approval": {
            "authorization_sha256": values["authorization_sha256"],
            "external_data_authorization": approval["external_data_authorization"],
        },
        "cumulative_limits": values["limits"],
        "provider_retry_limits": {
            scope: model["retry_limit"] for scope, model in values["models"].items()
        },
        "one_retry_per_cache_key": True,
        "one_retry_per_semantic_view": True,
        "possible_duplicate_charge": True,
        "original_reservation_retained": True,
        "actual_provider_bill_verified": False,
        "confirmed_not_sent_inferred": False,
        "transient_reason_whitelist": list(TRANSIENT_NETWORK_REASONS),
        "requires_all_futures_settled": True,
        "recursive_prior_chain_validated_by_runner_startup": True,
    }
    policy_path = output / "network-resume-policy.json"
    _bind_policy(policy_path, policy)

    def authorize(ledger):
        with ledger._lock:
            if set(ledger.starts) != set(ledger.settled):
                raise MediumLabelingHalt("network_resume_in_flight_requests")
            current_identity, current_sha, _ = _validate_bindings(
                output, values, computed_sha,
            )
            if current_sha != identity_sha or current_identity != identity:
                raise MediumLabelingHalt("network_resume_identity_changed")
            if (ledger.path.resolve() != output / "attempts.jsonl"
                    or ledger.config != values):
                raise MediumLabelingHalt("network_resume_ledger_binding_differs")
            _bind_policy(policy_path, policy)
            _validate_prior(ledger, identity, values)
            candidates = _eligible_candidates(ledger)
            if candidates is None:
                return False
            _check_unique_retry_views(ledger, candidates)
            for attempt_id, reason in candidates:
                original_sha = sha256(ledger.path.read_bytes()).hexdigest()
                ledger.authorize_unknown_retry(
                    attempt_id, reviewer=_REVIEWER,
                    original_attempt_log_sha256=original_sha,
                    identity_sha256=identity_sha,
                )
                receipt = {
                    "schema": "medium-label-unknown-retry-authorization-v1",
                    "attempt_id": str(attempt_id),
                    "reviewer": _REVIEWER,
                    "policy_reviewer": _POLICY_REVIEWER,
                    "decision": "one_retry_with_possible_duplicate_charge",
                    "original_attempt_log_sha256": original_sha,
                    "authorized_attempt_log_sha256": sha256(
                        ledger.path.read_bytes()
                    ).hexdigest(),
                    "identity_sha256": identity_sha,
                    "original_reservation_retained": True,
                    "actual_provider_bill_verified": False,
                    "transient_failure_reason_type": reason,
                    "transient_reason_whitelist": list(TRANSIENT_NETWORK_REASONS),
                }
                receipt_path = output / f"unknown-retry-attempt-{attempt_id}.json"
                _atomic_json(receipt_path, receipt)
            return bool(candidates)

    return authorize


def _validate_bindings(output, config, config_sha):
    try:
        dataset = output.parents[2]
        identity_bytes = (output / "identity.json").read_bytes()
        identity = json.loads(identity_bytes)
        approval_bytes = (
            dataset / "labels" / "observable-coordination-v7-authorization.json"
        ).read_bytes()
        approval = json.loads(approval_bytes)
        expected = {
            "label_config_sha256": config_sha,
            "dataset_manifest_sha256": sha256(
                (dataset / "manifest.json").read_bytes()
            ).hexdigest(),
            "windows_sha256": sha256(
                (dataset / "windows.jsonl").read_bytes()
            ).hexdigest(),
            "primary_provider": ProviderConfig(**config["models"]["primary"])
            .public_identity(),
            "comparison_provider": ProviderConfig(**config["models"]["comparison"])
            .public_identity(),
        }
        for key in ("semantic_contract_version", "authorization_sha256",
                    "pilot_sampling_salt", "quality_policy_version",
                    "quality_policy_authorization_sha256", "evidence_recipe_version",
                    "evidence_recipe_code_sha256", "evidence_recipe_dependency_sha256"):
            if key in config:
                expected[key] = config[key]
        valid = (
            output.name == "deepseek-official-flash"
            and output.parent.name == config_sha[:12]
            and all(identity.get(key) == value for key, value in expected.items())
            and config.get("semantic_contract_version") == SEMANTIC_CONTRACT_V2
            and sha256(approval_bytes).hexdigest() == config["authorization_sha256"]
            and approval.get("decision") == "approved"
            and approval.get("semantic_contract_version") == SEMANTIC_CONTRACT_V2
            and approval.get("dataset_manifest_sha256")
            == expected["dataset_manifest_sha256"]
            and approval.get("windows_sha256") == expected["windows_sha256"]
            and isinstance(approval.get("external_data_authorization"), str)
            and bool(approval["external_data_authorization"])
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise MediumLabelingHalt("network_resume_binding_unreadable") from error
    if not valid:
        raise MediumLabelingHalt("network_resume_frozen_identity_differs")
    return identity, sha256(identity_bytes).hexdigest(), approval


def _validate_prior(ledger, identity, config):
    prior = ledger.prior_budget
    if prior is None or identity.get("prior_budget_source") != {
        "identity_sha256": prior.get("prior_identity_sha256"),
        "attempts_sha256": prior.get("prior_attempts_sha256"),
    } or prior.get("authorization_sha256") != config["authorization_sha256"]:
        raise MediumLabelingHalt("network_resume_historical_budget_differs")
    durable = _AttemptLedger(ledger.path, config)
    for name in ("starts", "settled", "prior_budget", "reconciliations",
                 "unknown_retry_authorizations"):
        if getattr(durable, name) != getattr(ledger, name):
            raise MediumLabelingHalt("network_resume_durable_ledger_differs")


def _check_unique_retry_views(ledger, candidates):
    authorized = ledger.unknown_retry_authorizations
    views = {ledger.starts[attempt]["semantic_view_sha256"] for attempt in authorized}
    caches = {ledger.starts[attempt]["cache_key"] for attempt in authorized}
    for attempt, _ in candidates:
        start = ledger.starts[attempt]
        if start["semantic_view_sha256"] in views or start["cache_key"] in caches:
            raise MediumLabelingHalt(
                "network_resume_same_semantic_view_already_authorized"
            )
        views.add(start["semantic_view_sha256"])
        caches.add(start["cache_key"])


def _eligible_candidates(ledger):
    candidates = []
    for attempt_id, event in ledger.settled.items():
        status = event.get("status")
        if status in {"ok", "invalid_response"}:
            continue
        if status != "outcome_unknown" or attempt_id in ledger.reconciliations:
            return None
        if attempt_id in ledger.unknown_retry_authorizations:
            continue
        if ledger.starts[attempt_id].get("prompt_variant") == "citation-only-repair-v1":
            return None
        failure = event.get("response_audit", {}).get("provider_failure", {})
        reason = failure.get("reason_type")
        if reason not in TRANSIENT_NETWORK_REASONS:
            return None
        candidates.append((attempt_id, reason))
    return candidates


def _bind_policy(path, policy):
    if path.exists():
        try:
            previous = json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            raise MediumLabelingHalt("network_resume_policy_unreadable") from error
        if previous != policy:
            raise MediumLabelingHalt("network_resume_policy_changed")
    else:
        _atomic_json(path, policy)
