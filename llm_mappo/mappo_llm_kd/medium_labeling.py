"""Bounded, resumable DeepSeek labeling for the frozen medium-5ag dataset."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Mapping

import numpy as np
import yaml

from .dataset_labeling import (
    DATASET_LABEL_RUN_V1,
    _load_completed_dataset_manifest,
    _load_windows,
    _semantic_view,
    _window_id,
    _atomic_json,
)
from .diagnostics import SEMANTIC_DIMENSIONS, label_diagnostic_metrics
from .labeling import AtomicLabelStore, _cache_key, parse_label_response
from .medium_prompt import (
    MEDIUM_LABEL_PROMPT_V4,
    MEDIUM_LABEL_PROMPT_V5,
    MEDIUM_LABEL_PROMPT_V6,
    MEDIUM_LABEL_PROMPT_V7,
    SEMANTIC_CONTRACT_V2,
    build_medium_label_prompt,
    observable_coordination_evidence,
    validate_medium_v6_semantic_support,
    validate_medium_v7_semantic_support,
)
from .prompt import LLM_LABEL_PROMPT_V3
from .provider import OpenAICompatibleProvider, ProviderCallError, ProviderConfig


MEDIUM_LABEL_CONFIG = "configs/llm_kd/medium-5ag-dataset-labeling.yaml"
_SALT = "medium-5ag-llm-dataset-v1"
_V6_POLICY = "medium-5ag-semantic-coverage-quality-v2"
_V6_POLICY_AUTH_SHA = (
    "ffa8c16be9349c7e3096fef6ce6c528db2c8fbe7a9412774c767d295c01da9a3"
)
_V7_PILOT_SALT = "medium-5ag-llm-label-pilot-v3"
_V7_PILOT_POLICY = "medium-5ag-llm-label-pilot-balanced-v1"


class MediumLabelingHalt(RuntimeError):
    """A safe stop that never includes provider response or credential text."""


def load_medium_label_config(  # noqa: C901
    path: str | Path = MEDIUM_LABEL_CONFIG
):
    """Load the frozen public labeling contract and its canonical digest."""

    raw = Path(path).read_text(encoding="utf-8")
    values = yaml.safe_load(raw)
    if not isinstance(values, dict) or (
        values.get("schema_version"), values.get("prompt_version")
    ) not in {
        ("medium-5ag-dataset-labeling-v1", LLM_LABEL_PROMPT_V3),
        ("medium-5ag-dataset-labeling-v2", MEDIUM_LABEL_PROMPT_V4),
        ("medium-5ag-dataset-labeling-v3", MEDIUM_LABEL_PROMPT_V5),
        ("medium-5ag-dataset-labeling-v4", MEDIUM_LABEL_PROMPT_V6),
        ("medium-5ag-dataset-labeling-v5", MEDIUM_LABEL_PROMPT_V7),
    }:
        raise ValueError("Unsupported medium labeling config schema.")
    if values.get("temporal_view_schema") != (
        "temporal-semantic-view-v1"
    ) or values.get("sampling_salt") != _SALT:
        raise ValueError("Medium labeling prompt/view identity differs.")
    if values["prompt_version"] in {
        MEDIUM_LABEL_PROMPT_V5, MEDIUM_LABEL_PROMPT_V6,
        MEDIUM_LABEL_PROMPT_V7,
    } and values.get(
        "pilot_sampling_salt"
    ) != (
        _V7_PILOT_SALT if values["prompt_version"] == MEDIUM_LABEL_PROMPT_V7
        else "medium-5ag-llm-label-pilot-v2"
    ):
        raise ValueError("Medium pilot sampling salt differs.")
    if values["prompt_version"] not in {
        MEDIUM_LABEL_PROMPT_V5, MEDIUM_LABEL_PROMPT_V6,
        MEDIUM_LABEL_PROMPT_V7,
    } and (
        "pilot_sampling_salt" in values
    ):
        raise ValueError("Earlier medium pilot sampling must stay unchanged.")
    if values["prompt_version"] in {
        MEDIUM_LABEL_PROMPT_V6, MEDIUM_LABEL_PROMPT_V7
    }:
        if (values.get("quality_policy_version") != _V6_POLICY
                or values.get("quality_policy_authorization_sha256")
                != _V6_POLICY_AUTH_SHA):
            raise ValueError("Medium v6 approved quality policy differs.")
    elif ("quality_policy_version" in values
          or "quality_policy_authorization_sha256" in values):
        raise ValueError("Earlier medium quality policy must stay unchanged.")
    if values["prompt_version"] == MEDIUM_LABEL_PROMPT_V7:
        digest = values.get("authorization_sha256")
        if (values.get("semantic_contract_version") != SEMANTIC_CONTRACT_V2
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Medium v7 semantic authorization is not frozen.")
        if "evidence_recipe_version" in values:
            if values["evidence_recipe_version"] not in {
                "observable-witness-citations-v1", "observable-all-head-citations-v2",
            }:
                raise ValueError("Unknown observable evidence recipe.")
            for name in ("evidence_recipe_code_sha256", "revision_prior_identity_sha256",
                         "revision_prior_attempts_sha256"):
                value = values.get(name)
                if (not isinstance(value, str) or len(value) != 64
                        or any(char not in "0123456789abcdef" for char in value)):
                    raise ValueError("Evidence recipe revision is not frozen.")
            if values["evidence_recipe_version"] == "observable-all-head-citations-v2":
                value = values.get("evidence_recipe_dependency_sha256")
                if (not isinstance(value, str) or len(value) != 64
                        or any(char not in "0123456789abcdef" for char in value)):
                    raise ValueError("Evidence recipe dependency is not frozen.")
            if "revision_prior_config_path" in values:
                prior_path = Path(values["revision_prior_config_path"])
                if prior_path.resolve().parent != Path("configs/llm_kd").resolve():
                    raise ValueError("Recipe predecessor config is outside llm_kd.")
    elif ("authorization_sha256" in values
          or "semantic_contract_version" in values):
        raise ValueError("Earlier semantic contracts must stay unchanged.")
    models = values.get("models")
    if not isinstance(models, dict) or set(models) != {"primary", "comparison"}:
        raise ValueError("Medium labeling requires exactly two models.")
    primary = ProviderConfig(**models["primary"])
    comparison = ProviderConfig(**models["comparison"])
    if (
        primary.base_url != "https://api.deepseek.com"
        or comparison.base_url != primary.base_url
        or primary.requested_model != "deepseek-flash"
        or comparison.requested_model != "deepseek-v4-pro"
        or primary.api_key_environment_variable != "DEEPSEEK_API_KEY"
        or comparison.api_key_environment_variable != "DEEPSEEK_API_KEY"
        or primary.temperature != 0.0
        or comparison.temperature != 0.0
        or primary.timeout_seconds != (
            180.0 if values["prompt_version"] == MEDIUM_LABEL_PROMPT_V7
            else 60.0
        )
        or comparison.timeout_seconds != primary.timeout_seconds
        or primary.max_completion_tokens != 800
        or comparison.max_completion_tokens != 800
        or primary.retry_limit != 1
        or comparison.retry_limit != (
            1 if values["prompt_version"] in {
                MEDIUM_LABEL_PROMPT_V5, MEDIUM_LABEL_PROMPT_V6,
                MEDIUM_LABEL_PROMPT_V7,
            } else 0
        )
    ):
        raise ValueError("Medium labeling model contract differs.")
    limits = values.get("limits")
    expected = {
        "main_labels": 6000, "train_labels": 4200,
        "validation_labels": 900, "test_labels": 900,
        "pilot_labels": 120, "blind_review_labels": 60,
        "repeat_labels": 40, "comparison_labels": 40,
        "input_tokens": 100000000,
        "output_tokens": 5000000, "conservative_usd": 50.0,
    }
    expected.update(
        {"primary_http_attempts": 7000, "repeat_http_attempts": 200,
         "comparison_http_attempts": 128}
        if values["prompt_version"] == MEDIUM_LABEL_PROMPT_V7 else
        {"primary_http_attempts": 6500, "repeat_http_attempts": 160,
         "comparison_http_attempts": 88}
        if values["prompt_version"] == MEDIUM_LABEL_PROMPT_V6 else
        {"primary_http_attempts": 6500, "repeat_http_attempts": 100,
         "comparison_http_attempts": 48}
        if values["prompt_version"] == MEDIUM_LABEL_PROMPT_V5 else
        {"primary_http_attempts": 6300, "repeat_http_attempts": 40,
         "comparison_http_attempts": 40}
    )
    if limits != expected:
        raise ValueError("Medium labeling limits differ from the frozen contract.")
    prices = values.get("peak_uncached_usd_per_million_tokens")
    if prices != {
        "primary": {"input": 0.3, "output": 1.2},
        "comparison": {"input": 1.32, "output": 3.96},
    }:
        raise ValueError("Medium labeling peak price assumption differs.")
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return values, sha256(encoded.encode("utf-8")).hexdigest()


def reconcile_medium_presend_attempt(
    dataset_directory: str | Path,
    evidence_path: str | Path,
    *,
    config_path: str | Path = MEDIUM_LABEL_CONFIG,
) -> dict[str, object]:
    """Append an explicit review event for a proven unsent sandbox attempt."""

    _, config_sha = load_medium_label_config(config_path)
    dataset = Path(dataset_directory)
    output = dataset / "labels" / config_sha[:12] / "deepseek-official-flash"
    identity = json.loads((output / "identity.json").read_text("utf-8"))
    if (
        identity.get("label_config_sha256") != config_sha
        or identity.get("dataset_manifest_sha256") != sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest()
        or identity.get("windows_sha256") != sha256(
            (dataset / "windows.jsonl").read_bytes()
        ).hexdigest()
    ):
        raise MediumLabelingHalt("dataset_or_label_identity_changed")
    evidence_bytes = Path(evidence_path).read_bytes()
    evidence = json.loads(evidence_bytes)
    if not _valid_presend_evidence(evidence):
        raise ValueError("Presend evidence does not establish socket denial.")
    log_path = output / "attempts.jsonl"
    if sha256(log_path.read_bytes()).hexdigest() != evidence[
        "original_attempt_log_sha256"
    ]:
        raise MediumLabelingHalt("original_attempt_log_digest_changed")
    values, _ = load_medium_label_config(config_path)
    ledger = _AttemptLedger(log_path, values)
    ledger.reconcile_confirmed_not_sent(
        evidence["attempt_id"], evidence["reviewer"], evidence
    )
    receipt = {
        "schema": "medium-label-presend-reconciliation-v1",
        "attempt_id": evidence["attempt_id"],
        "decision": "confirmed_not_sent",
        "reviewer": evidence["reviewer"],
        "evidence_sha256": sha256(evidence_bytes).hexdigest(),
        "original_attempt_log_sha256": evidence["original_attempt_log_sha256"],
        "reconciled_attempt_log_sha256": sha256(log_path.read_bytes()).hexdigest(),
        "actual_provider_bill_verified": False,
        "conservative_original_reservation_retained": True,
    }
    _atomic_json(output / "network-reconciliation-receipt.json", receipt)
    return receipt


def authorize_medium_unknown_retry(
    dataset_directory: str | Path,
    attempt_id: str,
    original_attempt_log_sha256: str,
    *,
    config_path: str | Path = (
        "configs/llm_kd/medium-5ag-dataset-labeling-v5.yaml"
    ),
) -> dict[str, object]:
    """Record a reviewed, charge-conservative retry of one unknown result."""

    values, config_sha = load_medium_label_config(config_path)
    if values["prompt_version"] not in {
        MEDIUM_LABEL_PROMPT_V5, MEDIUM_LABEL_PROMPT_V6,
        MEDIUM_LABEL_PROMPT_V7,
    }:
        raise MediumLabelingHalt("unknown_retry_only_supported_for_v5_v6_v7")
    dataset = Path(dataset_directory)
    if values["prompt_version"] == MEDIUM_LABEL_PROMPT_V6:
        _require_v6_policy_authorization(
            dataset, sha256((dataset / "manifest.json").read_bytes()).hexdigest()
        )
    output = dataset / "labels" / config_sha[:12] / "deepseek-official-flash"
    identity_path = output / "identity.json"
    identity_bytes = identity_path.read_bytes()
    identity = json.loads(identity_bytes)
    if (
        identity.get("label_config_sha256") != config_sha
        or identity.get("dataset_manifest_sha256") != sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest()
        or identity.get("windows_sha256") != sha256(
            (dataset / "windows.jsonl").read_bytes()
        ).hexdigest()
        or identity.get("primary_provider") != ProviderConfig(
            **values["models"]["primary"]
        ).public_identity()
        or identity.get("comparison_provider") != ProviderConfig(
            **values["models"]["comparison"]
        ).public_identity()
        or identity.get("pilot_sampling_salt") != values["pilot_sampling_salt"]
        or (values["prompt_version"] == MEDIUM_LABEL_PROMPT_V6 and (
            identity.get("quality_policy_version") != _V6_POLICY
            or identity.get("quality_policy_authorization_sha256")
            != _V6_POLICY_AUTH_SHA
        ))
        or (values["prompt_version"] == MEDIUM_LABEL_PROMPT_V7 and (
            identity.get("quality_policy_version") != _V6_POLICY
            or identity.get("quality_policy_authorization_sha256")
            != _V6_POLICY_AUTH_SHA
            or identity.get("semantic_contract_version")
            != SEMANTIC_CONTRACT_V2
            or identity.get("authorization_sha256")
            != values["authorization_sha256"]
        ))
    ):
        raise MediumLabelingHalt("dataset_or_label_identity_changed")
    log_path = output / "attempts.jsonl"
    actual_digest = sha256(log_path.read_bytes()).hexdigest()
    if actual_digest != original_attempt_log_sha256:
        raise MediumLabelingHalt("original_attempt_log_digest_changed")
    ledger = _AttemptLedger(log_path, values)
    if ledger.prior_budget is None or identity.get("prior_budget_source") != {
        "identity_sha256": ledger.prior_budget["prior_identity_sha256"],
        "attempts_sha256": ledger.prior_budget["prior_attempts_sha256"],
    }:
        raise MediumLabelingHalt("historical_budget_source_changed")
    if values["prompt_version"] == MEDIUM_LABEL_PROMPT_V7:
        approval_path = (
            dataset / "labels" / "observable-coordination-v7-authorization.json"
        )
        approval_raw = approval_path.read_bytes()
        approval = json.loads(approval_raw)
        if (
            sha256(approval_raw).hexdigest() != values["authorization_sha256"]
            or ledger.prior_budget.get("authorization_sha256")
            != values["authorization_sha256"]
            or approval.get("dataset_manifest_sha256")
            != identity["dataset_manifest_sha256"]
            or approval.get("windows_sha256") != identity["windows_sha256"]
            or (not values.get("evidence_recipe_version")
                and approval.get("prior_budget_identity_sha256")
                != ledger.prior_budget["prior_identity_sha256"])
            or (not values.get("evidence_recipe_version")
                and approval.get("prior_budget_attempts_sha256")
                != ledger.prior_budget["prior_attempts_sha256"])
        ):
            raise MediumLabelingHalt("v7_unknown_retry_authorization_differs")
        if values.get("evidence_recipe_version"):
            prior_directory = (dataset / "labels" / load_medium_label_config(
                values.get("revision_prior_config_path",
                           "configs/llm_kd/medium-5ag-dataset-labeling-v7.yaml")
            )[1][:12] / "deepseek-official-flash")
            if ledger.prior_budget != _load_v7_recipe_prior_budget(
                dataset, identity["dataset_manifest_sha256"], prior_directory, values
            ):
                raise MediumLabelingHalt("recipe_unknown_retry_chain_differs")
    reviewer = "Codex root"
    ledger.authorize_unknown_retry(
        attempt_id, reviewer,
        original_attempt_log_sha256=original_attempt_log_sha256,
        identity_sha256=sha256(identity_bytes).hexdigest(),
    )
    receipt = {
        "schema": "medium-label-unknown-retry-authorization-v1",
        "attempt_id": str(attempt_id), "reviewer": reviewer,
        "decision": "one_retry_with_possible_duplicate_charge",
        "original_attempt_log_sha256": original_attempt_log_sha256,
        "authorized_attempt_log_sha256": sha256(log_path.read_bytes()).hexdigest(),
        "original_reservation_retained": True,
        "actual_provider_bill_verified": False,
    }
    _atomic_json(output / f"unknown-retry-attempt-{attempt_id}.json", receipt)
    return receipt


def _valid_presend_evidence(value):
    if not isinstance(value, dict):
        return False
    probe = value.get("unauthenticated_same_host_probe")
    approved = value.get("approved_network_probe")
    return (
        value.get("schema") == "medium-label-network-reconciliation-evidence-v1"
        and isinstance(value.get("attempt_id"), str)
        and value["attempt_id"].isdigit()
        and value.get("original_attempt_status") == "outcome_unknown"
        and isinstance(value.get("original_attempt_log_sha256"), str)
        and len(value["original_attempt_log_sha256"]) == 64
        and value.get("actual_provider_bill_verified") is False
        and value.get("sandbox_network_policy") == "restricted"
        and isinstance(value.get("reviewer"), str)
        and bool(value["reviewer"].strip())
        and isinstance(probe, dict)
        and probe.get("reason_type") == "PermissionError"
        and probe.get("errno") == 13
        and probe.get("winerror") == 10013
        and isinstance(approved, dict)
        and approved.get("connection_reached_server") is True
        and approved.get("credentials_sent") is False
        and approved.get("label_request_sent") is False
        and approved.get("http_status") == 401
    )


def _load_prior_budget(dataset, source_digest, version, directory):  # noqa: C901
    if version == LLM_LABEL_PROMPT_V3:
        if directory is not None:
            raise ValueError("v3 labeling cannot import a prior budget.")
        return None
    if directory is None:
        raise MediumLabelingHalt("v4_requires_prior_budget_directory")
    if version == MEDIUM_LABEL_PROMPT_V7:
        return _load_v7_prior_budget(dataset, source_digest, directory)
    old_config_path = (
        MEDIUM_LABEL_CONFIG if version == MEDIUM_LABEL_PROMPT_V4
        else "configs/llm_kd/medium-5ag-dataset-labeling-v4.yaml"
        if version == MEDIUM_LABEL_PROMPT_V5
        else "configs/llm_kd/medium-5ag-dataset-labeling-v5.yaml"
    )
    old_values, old_config_sha = load_medium_label_config(old_config_path)
    expected = dataset / "labels" / old_config_sha[:12] / "deepseek-official-flash"
    prior = Path(directory)
    if prior.resolve() != expected.resolve():
        raise MediumLabelingHalt("prior_budget_directory_is_not_frozen_v3_output")
    identity_path, attempts_path = prior / "identity.json", prior / "attempts.jsonl"
    try:
        identity_bytes = identity_path.read_bytes()
        attempts_bytes = attempts_path.read_bytes()
        old_identity = json.loads(identity_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise MediumLabelingHalt("prior_budget_artifact_unreadable") from error
    if (
        old_identity.get("label_config_sha256") != old_config_sha
        or old_identity.get("dataset_manifest_sha256") != source_digest
        or old_identity.get("windows_sha256") != sha256(
            (dataset / "windows.jsonl").read_bytes()
        ).hexdigest()
    ):
        raise MediumLabelingHalt("prior_budget_source_identity_differs")
    old_ledger = _AttemptLedger(attempts_path, old_values)
    summary = old_ledger.summary()
    labels = AtomicLabelStore(prior / "labels.jsonl").records()
    if version == MEDIUM_LABEL_PROMPT_V4:
        valid_snapshot = (
            summary["http_attempts"] == {"primary": 66}
            and summary["attempt_status_counts"] == {
                "ok": 63, "invalid_response": 2, "outcome_unknown": 1
            }
            and summary["confirmed_not_sent_reconciliations"] == 1
            and summary["input_tokens_upper"] == 829724
            and summary["output_tokens_upper"] == 15876
            and abs(summary["conservative_cost_upper_usd"] - 0.2679684) < 1e-9
            and len(labels) == 63
        )
    else:
        chain = old_ledger.prior_budget or {}
        prior_config_path = (
            MEDIUM_LABEL_CONFIG if version == MEDIUM_LABEL_PROMPT_V5
            else "configs/llm_kd/medium-5ag-dataset-labeling-v4.yaml"
        )
        _, prior_sha = load_medium_label_config(prior_config_path)
        prior_directory = (
            dataset / "labels" / prior_sha[:12] / "deepseek-official-flash"
        )
        expected_chain = _load_prior_budget(
            dataset, source_digest,
            MEDIUM_LABEL_PROMPT_V4 if version == MEDIUM_LABEL_PROMPT_V5
            else MEDIUM_LABEL_PROMPT_V5,
            prior_directory,
        )
        valid_snapshot = (
            chain == expected_chain
            and old_identity.get("prior_budget_source") == {
                "identity_sha256": chain.get("prior_identity_sha256"),
                "attempts_sha256": chain.get("prior_attempts_sha256"),
            }
            and summary["http_attempts"].get("primary", 0) >= 66
            and old_ledger._counts["primary"] >= 120
            and summary["attempt_status_counts"].get("ok", 0) >= 120
            and len(labels) == 120
            and (version != MEDIUM_LABEL_PROMPT_V6 or (
                len(AtomicLabelStore(prior / "repeat.jsonl").records()) == 40
                and len(AtomicLabelStore(prior / "comparison.jsonl").records())
                == 40
            ))
        )
    if not valid_snapshot or any(row.get("status") != "ok" for row in labels):
        raise MediumLabelingHalt("prior_budget_frozen_chain_differs")
    budget_event = {
        "event": "historical_budget",
        "schema": "medium-label-historical-budget-v1",
        "prior_label_config_sha256": old_config_sha,
        "prior_identity_sha256": sha256(identity_bytes).hexdigest(),
        "prior_attempts_sha256": sha256(attempts_bytes).hexdigest(),
        "prior_dataset_manifest_sha256": source_digest,
        "http_attempts": summary["http_attempts"],
        "attempt_status_counts": summary["attempt_status_counts"],
        "input_tokens_upper": summary["input_tokens_upper"],
        "output_tokens_upper": summary["output_tokens_upper"],
        "conservative_cost_upper_usd": summary[
            "conservative_cost_upper_usd"
        ],
        "confirmed_not_sent_reconciliations": 1,
    }
    if old_ledger.prior_budget:
        budget_event["prior_historical_budget_sha256"] = (
            sha256(json.dumps(old_ledger.prior_budget, sort_keys=True,
                              separators=(",", ":")).encode("utf-8")).hexdigest()
        )
    return budget_event


def _load_v7_authorization(dataset, source_digest, prior, values):
    path = dataset / "labels" / "observable-coordination-v7-authorization.json"
    try:
        raw = path.read_bytes()
        approval = json.loads(raw)
        parent = prior.resolve().parents[2]
        parent_manifest_sha = sha256((parent / "manifest.json").read_bytes())
        parent_windows_sha = sha256((parent / "windows.jsonl").read_bytes())
        prior_identity_sha = sha256((prior / "identity.json").read_bytes())
        prior_attempts_sha = sha256((prior / "attempts.jsonl").read_bytes())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise MediumLabelingHalt("v7_authorization_or_parent_unreadable") from error
    expected = {
        "schema": "medium-5ag-observable-coordination-authorization-v1",
        "decision": "approved",
        "owner_approval_quote": "可以按照你推荐的方案进行",
        "semantic_contract_version": SEMANTIC_CONTRACT_V2,
        "quality_policy_version": _V6_POLICY,
        "dataset_manifest_sha256": source_digest,
        "windows_sha256": sha256((dataset / "windows.jsonl").read_bytes())
        .hexdigest(),
        "parent_dataset_manifest_sha256": parent_manifest_sha.hexdigest(),
        "parent_windows_sha256": parent_windows_sha.hexdigest(),
        "prior_budget_identity_sha256": prior_identity_sha.hexdigest(),
        "prior_budget_attempts_sha256": prior_attempts_sha.hexdigest(),
        "previous_quality_authorization_sha256": _V6_POLICY_AUTH_SHA,
        "conservative_usd": 50.0,
        "input_tokens": 100000000,
        "output_tokens": 5000000,
        "main_labels": 6000,
        "repeat_labels": 40,
        "comparison_labels": 40,
        "allowed_http_attempts": {
            "primary": 7000, "repeat": 200, "comparison": 128,
        },
    }
    if (sha256(raw).hexdigest() != values["authorization_sha256"]
            or not isinstance(approval, dict)
            or any(approval.get(key) != value for key, value in expected.items())):
        raise MediumLabelingHalt("v7_authorization_identity_differs")
    return approval, parent


def _load_v7_prior_budget(dataset, source_digest, directory):
    prior = Path(directory)
    values, _ = load_medium_label_config(
        "configs/llm_kd/medium-5ag-dataset-labeling-v7.yaml"
    )
    approval, parent = _load_v7_authorization(
        dataset, source_digest, prior, values
    )
    old_values, old_config_sha = load_medium_label_config(
        "configs/llm_kd/medium-5ag-dataset-labeling-v6.yaml"
    )
    expected = (
        parent / "labels" / old_config_sha[:12] / "deepseek-official-flash"
    )
    if prior.resolve() != expected.resolve():
        raise MediumLabelingHalt("v7_prior_budget_directory_differs")
    identity_bytes = (prior / "identity.json").read_bytes()
    attempts_bytes = (prior / "attempts.jsonl").read_bytes()
    identity = json.loads(identity_bytes)
    if (
        identity.get("dataset_manifest_sha256")
        != approval["parent_dataset_manifest_sha256"]
        or identity.get("windows_sha256") != approval["parent_windows_sha256"]
        or identity.get("label_config_sha256") != old_config_sha
        or identity.get("quality_policy_version") != _V6_POLICY
        or identity.get("quality_policy_authorization_sha256")
        != _V6_POLICY_AUTH_SHA
    ):
        raise MediumLabelingHalt("v7_prior_budget_identity_differs")
    ledger = _AttemptLedger(prior / "attempts.jsonl", old_values)
    _, v5_sha = load_medium_label_config(
        "configs/llm_kd/medium-5ag-dataset-labeling-v5.yaml"
    )
    expected_chain = _load_prior_budget(
        parent, approval["parent_dataset_manifest_sha256"],
        MEDIUM_LABEL_PROMPT_V6,
        parent / "labels" / v5_sha[:12] / "deepseek-official-flash",
    )
    labels = AtomicLabelStore(prior / "labels.jsonl").records()
    if (
        ledger.prior_budget != expected_chain
        or identity.get("prior_budget_source") != {
            "identity_sha256": expected_chain["prior_identity_sha256"],
            "attempts_sha256": expected_chain["prior_attempts_sha256"],
        }
        or len(labels) != sum(
            row["status"] == "ok" and ledger.starts[row["attempt_id"]][
                "scope"
            ] == "primary"
            for row in ledger.settled.values()
        )
        or any(row.get("status") != "ok" for row in labels)
    ):
        raise MediumLabelingHalt("v7_prior_budget_frozen_chain_differs")
    summary = ledger.summary()
    return {
        "event": "historical_budget",
        "schema": "medium-label-historical-budget-v1",
        "prior_label_config_sha256": old_config_sha,
        "prior_identity_sha256": sha256(identity_bytes).hexdigest(),
        "prior_attempts_sha256": sha256(attempts_bytes).hexdigest(),
        "prior_dataset_manifest_sha256": (
            approval["parent_dataset_manifest_sha256"]
        ),
        "prior_windows_sha256": approval["parent_windows_sha256"],
        "child_dataset_manifest_sha256": source_digest,
        "child_windows_sha256": approval["windows_sha256"],
        "authorization_sha256": values["authorization_sha256"],
        "http_attempts": summary["http_attempts"],
        "attempt_status_counts": summary["attempt_status_counts"],
        "input_tokens_upper": summary["input_tokens_upper"],
        "output_tokens_upper": summary["output_tokens_upper"],
        "conservative_cost_upper_usd": summary[
            "conservative_cost_upper_usd"
        ],
        "confirmed_not_sent_reconciliations": summary[
            "confirmed_not_sent_reconciliations"
        ],
        "authorized_unknown_retries": summary["authorized_unknown_retries"],
        "prior_historical_budget_sha256": sha256(json.dumps(
            ledger.prior_budget, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }


def _load_v7_recipe_prior_budget(
    dataset, source_digest, directory, values, _depth=0,
):
    """Freeze the failed citation pilot and retain its complete budget chain."""
    if _depth >= 8:
        raise MediumLabelingHalt("recipe_budget_chain_too_deep")
    old_values, old_sha = load_medium_label_config(values.get(
        "revision_prior_config_path",
        "configs/llm_kd/medium-5ag-dataset-labeling-v7.yaml",
    ))
    if old_values["prompt_version"] != MEDIUM_LABEL_PROMPT_V7:
        raise MediumLabelingHalt("recipe_prior_contract_differs")
    expected = dataset / "labels" / old_sha[:12] / "deepseek-official-flash"
    if directory is None or Path(directory).resolve() != expected.resolve():
        raise MediumLabelingHalt("recipe_prior_directory_differs")
    identity_bytes = (expected / "identity.json").read_bytes()
    attempts_bytes = (expected / "attempts.jsonl").read_bytes()
    identity = json.loads(identity_bytes)
    if (sha256(identity_bytes).hexdigest() != values["revision_prior_identity_sha256"]
            or sha256(attempts_bytes).hexdigest()
            != values["revision_prior_attempts_sha256"]
            or identity.get("label_config_sha256") != old_sha
            or identity.get("dataset_manifest_sha256") != source_digest
            or identity.get("windows_sha256")
            != sha256((dataset / "windows.jsonl").read_bytes()).hexdigest()
            or identity.get("semantic_contract_version") != SEMANTIC_CONTRACT_V2):
        raise MediumLabelingHalt("recipe_prior_identity_differs")
    ledger = _AttemptLedger(expected / "attempts.jsonl", old_values)
    parent = dataset.parent / "main-c6d2df435f63"
    _, v6_sha = load_medium_label_config(
        "configs/llm_kd/medium-5ag-dataset-labeling-v6.yaml"
    )
    if old_values.get("evidence_recipe_version"):
        _, predecessor_sha = load_medium_label_config(old_values.get(
            "revision_prior_config_path",
            "configs/llm_kd/medium-5ag-dataset-labeling-v7.yaml",
        ))
        chain = _load_v7_recipe_prior_budget(
            dataset, source_digest,
            dataset / "labels" / predecessor_sha[:12] / "deepseek-official-flash",
            old_values, _depth + 1,
        )
    else:
        chain = _load_v7_prior_budget(
            dataset, source_digest,
            parent / "labels" / v6_sha[:12] / "deepseek-official-flash",
        )
    if (ledger.prior_budget != chain
            or identity.get("prior_budget_source") != {
                "identity_sha256": chain["prior_identity_sha256"],
                "attempts_sha256": chain["prior_attempts_sha256"],
            }
            or len(ledger.starts) != len(ledger.settled)):
        raise MediumLabelingHalt("recipe_prior_chain_unsettled_or_changed")
    summary = ledger.summary()
    return {
        "event": "historical_budget",
        "schema": "medium-label-historical-budget-v1",
        "prior_label_config_sha256": old_sha,
        "prior_identity_sha256": sha256(identity_bytes).hexdigest(),
        "prior_attempts_sha256": sha256(attempts_bytes).hexdigest(),
        "prior_dataset_manifest_sha256": source_digest,
        "prior_windows_sha256": identity["windows_sha256"],
        "authorization_sha256": values["authorization_sha256"],
        "http_attempts": summary["http_attempts"],
        "attempt_status_counts": summary["attempt_status_counts"],
        "input_tokens_upper": summary["input_tokens_upper"],
        "output_tokens_upper": summary["output_tokens_upper"],
        "conservative_cost_upper_usd": summary["conservative_cost_upper_usd"],
        "confirmed_not_sent_reconciliations": summary[
            "confirmed_not_sent_reconciliations"
        ],
        "authorized_unknown_retries": summary["authorized_unknown_retries"],
        "prior_historical_budget_sha256": sha256(json.dumps(
            chain, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
    }


def run_medium_dataset_labeling(  # noqa: C901
    dataset_directory: str | Path,
    *,
    stage: str,
    config_path: str | Path = MEDIUM_LABEL_CONFIG,
    prior_budget_directory: str | Path | None = None,
    pricing_checked_utc_date: str | None = None,
    concurrency: int = 4,
    provider_factory=OpenAICompatibleProvider,
    resume_transient_network: bool = False,
    citation_repair_authorization: str | Path | None = None,
    citation_repair_authorization_sha256: str | None = None,
) -> dict[str, object]:
    """Run smoke, pilot, or full labels without bypassing any earlier gate."""

    if stage not in {"smoke", "pilot", "full"}:
        raise ValueError("Stage must be smoke, pilot, or full.")
    if type(concurrency) is not int or not 1 <= concurrency <= 32:
        raise ValueError("Concurrency must be an integer from 1 to 32.")
    if type(resume_transient_network) is not bool:
        raise ValueError("Transient network resume must be an explicit boolean.")
    values, config_sha = load_medium_label_config(config_path)
    version = values["prompt_version"]

    def prompt_builder(view):
        prompt = build_medium_label_prompt(view, version=version)
        if values.get("evidence_recipe_version"):
            if values["evidence_recipe_version"] == "observable-all-head-citations-v2":
                from .observable_citation_recipe_v2 import (
                    add_observable_citation_recipe_v2,
                )
                prompt = add_observable_citation_recipe_v2(prompt, view)
            else:
                from .observable_citation_recipe import add_observable_citation_recipe
                prompt = add_observable_citation_recipe(prompt, view)
        return prompt

    dataset = Path(dataset_directory)
    source, windows, source_digest = _frozen_source(dataset, version=version)
    if version == MEDIUM_LABEL_PROMPT_V6:
        _require_v6_policy_authorization(dataset, source_digest)
    if values.get("evidence_recipe_version"):
        recipe_file = Path(__file__).with_name(
            "observable_citation_recipe_v2.py"
            if values["evidence_recipe_version"] == "observable-all-head-citations-v2"
            else "observable_citation_recipe.py"
        )
        if sha256(recipe_file.read_bytes()).hexdigest() != values[
            "evidence_recipe_code_sha256"
        ]:
            raise MediumLabelingHalt("evidence_recipe_code_changed")
        if values.get("evidence_recipe_dependency_sha256"):
            dependency = Path(__file__).with_name("observable_citation_recipe.py")
            if sha256(dependency.read_bytes()).hexdigest() != values[
                "evidence_recipe_dependency_sha256"
            ]:
                raise MediumLabelingHalt("evidence_recipe_dependency_changed")
        prior_budget = _load_v7_recipe_prior_budget(
            dataset, source_digest, prior_budget_directory, values
        )
    else:
        prior_budget = _load_prior_budget(
            dataset, source_digest, version, prior_budget_directory
        )
    train = sorted(
        (window for window in windows if window["partition"] == "train"),
        key=lambda window: _pilot_rank(
            window, values.get("pilot_sampling_salt", _SALT)
        ),
    )
    pilot_snapshot = None
    if version == MEDIUM_LABEL_PROMPT_V7:
        train, pilot_snapshot = select_medium_v7_pilot(
            train, source_digest,
            sha256((dataset / "windows.jsonl").read_bytes()).hexdigest(),
        )
    output = dataset / "labels" / config_sha[:12] / "deepseek-official-flash"
    identity = {
        "dataset_manifest_sha256": source_digest,
        "windows_sha256": sha256((dataset / "windows.jsonl").read_bytes()).hexdigest(),
        "label_config_sha256": config_sha,
        "primary_provider": ProviderConfig(**values["models"]["primary"])
        .public_identity(),
        "comparison_provider": ProviderConfig(**values["models"]["comparison"])
        .public_identity(),
    }
    if prior_budget is not None:
        identity["prior_budget_source"] = {
            "identity_sha256": prior_budget["prior_identity_sha256"],
            "attempts_sha256": prior_budget["prior_attempts_sha256"],
        }
    if version in {
        MEDIUM_LABEL_PROMPT_V5, MEDIUM_LABEL_PROMPT_V6,
        MEDIUM_LABEL_PROMPT_V7,
    }:
        identity["pilot_sampling_salt"] = values["pilot_sampling_salt"]
    if version in {MEDIUM_LABEL_PROMPT_V6, MEDIUM_LABEL_PROMPT_V7}:
        identity["quality_policy_version"] = _V6_POLICY
        identity["quality_policy_authorization_sha256"] = (
            _V6_POLICY_AUTH_SHA
        )
    if version == MEDIUM_LABEL_PROMPT_V7:
        identity["semantic_contract_version"] = SEMANTIC_CONTRACT_V2
        identity["authorization_sha256"] = values["authorization_sha256"]
        identity["pilot_selection_policy_version"] = _V7_PILOT_POLICY
        identity["pilot_selection_sha256"] = sha256((json.dumps(
            pilot_snapshot, ensure_ascii=False, separators=(",", ":"),
        ) + "\n").encode("utf-8")).hexdigest()
        if values.get("evidence_recipe_version"):
            identity["evidence_recipe_version"] = values["evidence_recipe_version"]
            identity["evidence_recipe_code_sha256"] = values[
                "evidence_recipe_code_sha256"
            ]
            if values.get("evidence_recipe_dependency_sha256"):
                identity["evidence_recipe_dependency_sha256"] = values[
                    "evidence_recipe_dependency_sha256"
                ]
    _bind_identity(output, identity)
    if pilot_snapshot is not None:
        _bind_v7_pilot_selection(output, pilot_snapshot)
    primary = provider_factory(ProviderConfig(**values["models"]["primary"]))
    comparison = provider_factory(ProviderConfig(**values["models"]["comparison"]))
    if not isinstance(primary, OpenAICompatibleProvider) or not isinstance(
        comparison, OpenAICompatibleProvider
    ):
        raise ValueError("Provider factory must return OpenAICompatibleProvider.")
    stores = {
        "primary": AtomicLabelStore(output / "labels.jsonl"),
        "repeat": AtomicLabelStore(output / "repeat.jsonl"),
        "comparison": AtomicLabelStore(output / "comparison.jsonl"),
    }
    ledger = _AttemptLedger(output / "attempts.jsonl", values)
    if prior_budget is not None:
        ledger.bind_prior_budget(prior_budget)
    citation_repair_policy = None
    if citation_repair_authorization is not None:
        if stage != "full" or version != MEDIUM_LABEL_PROMPT_V7:
            raise ValueError("Citation-only repair requires full stage and v7.")
        from .citation_only_repair import create_citation_only_repair_policy
        citation_repair_policy = create_citation_only_repair_policy(
            output, values, citation_repair_authorization,
            authorization_sha256=citation_repair_authorization_sha256,
            config_sha=config_sha,
        )
    elif citation_repair_authorization_sha256 is not None:
        raise ValueError("Citation repair digest requires its authorization artifact.")
    network_authorizer = None
    if resume_transient_network:
        if version != MEDIUM_LABEL_PROMPT_V7:
            raise ValueError("Transient network resume requires the v7 contract.")
        from .network_resume import create_transient_network_authorizer
        network_authorizer = create_transient_network_authorizer(
            output, values, config_sha=config_sha,
        )
        _resume_transient_network_failure(ledger, network_authorizer)
    tasks = [("primary", train[0])] if stage == "smoke" else [
        ("primary", window) for window in (
            train[:120] if stage == "pilot" else windows
        )
    ]
    if stage == "pilot":
        tasks += [("repeat", window) for window in train[:40]]
        tasks += [("comparison", window) for window in train[:40]]
    if stage == "full":
        _require_pilot_and_review(output, train, stores, ledger, version)
    new_calls = any(
        not _cached(stores[scope], window, primary if scope != "comparison"
                    else comparison, scope, prompt_builder)
        for scope, window in tasks
    )
    if new_calls:
        _require_current_pricing_and_secret(pricing_checked_utc_date)
    _atomic_json(output / "state.json", {"status": "running", "stage": stage})
    try:
        if stage == "full":
            _run_full_batches(
                tasks, primary, stores["primary"], ledger, concurrency,
                prompt_builder, network_authorizer, citation_repair_policy,
            )
        elif stage == "pilot":
            _label_one(
                train[0], "primary", primary, stores["primary"], ledger,
                prompt_builder,
            )
            for scope, provider in (
                ("primary", primary), ("repeat", primary),
                ("comparison", comparison),
            ):
                group = [task for task in tasks if task[0] == scope]
                if scope == "primary":
                    group = group[1:]
                _run_full_batches(
                    group, provider, stores[scope], ledger, min(concurrency, 4),
                    prompt_builder, network_authorizer,
                )
        else:
            for scope, window in tasks:
                provider = comparison if scope == "comparison" else primary
                _label_one(
                    window, scope, provider, stores[scope], ledger, prompt_builder
                )
        if stage in {"pilot", "full"}:
            metrics = _quality_metrics(
                windows, train, stores, ledger, stage, version
            )
            _atomic_json(output / f"{stage}-metrics.json", metrics)
            if stage == "pilot":
                _write_blind_packet(
                    output, train[:60], stores["primary"], version
                )
            if not metrics["go"]:
                raise MediumLabelingHalt(f"{stage}_quality_no_go")
        receipt = _receipt(dataset, output, source, identity, stores, ledger, stage)
        if stage == "full":
            _atomic_json(output / "manifest.json", receipt)
            _atomic_json(output / "state.json", {"status": "completed"})
        else:
            _atomic_json(output / f"{stage}-receipt.json", receipt)
            _atomic_json(output / "state.json", {"status": "partial", "stage": stage})
        return receipt
    except Exception as error:
        _atomic_json(output / "state.json", {
            "status": "halted", "stage": stage,
            "reason": str(error) if isinstance(error, MediumLabelingHalt)
            else type(error).__name__,
        })
        raise


def _run_full_batches(tasks, provider, store, ledger, concurrency,  # noqa: C901
                      prompt_builder=None, network_authorizer=None,
                      citation_repair_policy=None):
    """Bound in-flight requests while all durable writes stay serialized."""

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for offset in range(0, len(tasks), concurrency):
            batch = tasks[offset:offset + concurrency]
            for _ in range(len(batch) + 1):
                futures = [
                    executor.submit(
                        _label_one, window, scope, provider, store, ledger,
                        prompt_builder, citation_repair_policy=citation_repair_policy,
                        allow_citation_repair=False,
                    ) for scope, window in batch
                ]
                failures = []
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        failures.append(error)
                if not failures:
                    break
                if citation_repair_policy is not None and all(
                    isinstance(error, MediumLabelingHalt)
                    and str(error) == "citation_repair_requires_settled_batch"
                    for error in failures
                ):
                    for scope, window in batch:
                        _label_one(
                            window, scope, provider, store, ledger, prompt_builder,
                            citation_repair_policy=citation_repair_policy,
                            allow_citation_repair=True,
                        )
                    break
                network_codes = {
                    "provider_attempt_requires_reconciliation",
                    "previous_attempt_requires_reconciliation",
                    "previous_attempt_requires_manual_reconciliation",
                }
                if citation_repair_policy is not None:
                    network_codes.add("citation_repair_requires_settled_batch")
                if any(not isinstance(error, MediumLabelingHalt)
                       or str(error) not in network_codes for error in failures):
                    raise failures[0]
                if not _resume_transient_network_failure(ledger, network_authorizer):
                    raise failures[0]
            else:
                raise MediumLabelingHalt("transient_batch_resume_bound_exceeded")


def _resume_transient_network_failure(ledger, authorizer):
    """Resume only settled, explicitly reviewed failures inside frozen caps."""
    if authorizer is None or not authorizer(ledger):
        return False
    with ledger._lock:
        if len(ledger.starts) != len(ledger.settled) or any(
            row["status"] not in {"ok", "invalid_response"}
            and not (row["status"] == "outcome_unknown"
                     and ledger.authorized_unknown_retry(row["attempt_id"]))
            for row in ledger.settled.values()
        ):
            raise MediumLabelingHalt("transient_resume_has_unreviewed_failure")
        counts, input_tokens, output_tokens, cost = ledger._totals()
        limits = ledger.config["limits"]
        if (input_tokens > limits["input_tokens"]
                or output_tokens > limits["output_tokens"]
                or cost > limits["conservative_usd"]
                or any(count > limits[scope + "_http_attempts"]
                       for scope, count in counts.items())):
            raise MediumLabelingHalt("transient_resume_exceeds_budget")
        ledger.halted = False
    return True


def _require_v6_policy_authorization(dataset, source_digest):
    approval = (
        dataset / "labels" / "semantic-coverage-quality-v2-authorization.json"
    )
    try:
        raw = approval.read_bytes()
        values = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise MediumLabelingHalt("v6_quality_policy_authorization_missing") from error
    if (
        sha256(raw).hexdigest() != _V6_POLICY_AUTH_SHA
        or values.get("schema") != "medium-5ag-quality-policy-authorization-v1"
        or values.get("decision") != "approved"
        or values.get("policy_version") != _V6_POLICY
        or values.get("dataset_manifest_sha256") != source_digest
        or values.get("windows_sha256") != sha256(
            (dataset / "windows.jsonl").read_bytes()
        ).hexdigest()
    ):
        raise MediumLabelingHalt("v6_quality_policy_authorization_changed")


def _frozen_source(dataset: Path, *, version=LLM_LABEL_PROMPT_V3):
    manifest = _load_completed_dataset_manifest(dataset)
    windows_path = dataset / "windows.jsonl"
    digest = sha256(windows_path.read_bytes()).hexdigest()
    if manifest.get("windows_sha256") != digest:
        raise ValueError("Frozen dataset window digest differs.")
    windows = _load_windows(windows_path)
    counts = Counter(window["partition"] for window in windows)
    if counts != {"train": 4200, "validation": 900, "test": 900}:
        raise ValueError("Dataset must contain exactly 4200/900/900 windows.")
    seen_ids, seen_views, episodes = set(), set(), {}
    for window in windows:
        if (
            len(window["observations"]) != 8
            or len(window["reset_masks"]) != 8
            or window["end_timestep"] - window["start_timestep"] != 7
            or window["agent_id"] not in range(1, 6)
        ):
            raise ValueError("Dataset requires complete 8-frame 5-AGV windows.")
        view_id = _window_id(window)
        prompt = build_medium_label_prompt(
            _semantic_view(window), version=version
        )
        if len(prompt.user_text) > 60000:
            raise ValueError("Frozen dataset contains an oversized LLM prompt.")
        view_hash = prompt.semantic_view_sha256
        if view_id in seen_ids or view_hash in seen_views:
            raise ValueError("Dataset has duplicate window or semantic view.")
        seen_ids.add(view_id)
        seen_views.add(view_hash)
        episode = window["episode_id"]
        if episode in episodes and episodes[episode] != window["partition"]:
            raise ValueError("Episode crosses a dataset partition.")
        episodes[episode] = window["partition"]
    manifest_digest = sha256((dataset / "manifest.json").read_bytes()).hexdigest()
    return manifest, windows, manifest_digest


def _pilot_rank(window, salt=_SALT):
    identity = (
        f"{salt}\0{window['stratum']}\0{window['episode_id']}"
        f"\0{window['agent_id']}\0{window['start_timestep']}"
        f"\0{window['end_timestep']}"
    )
    return sha256(identity.encode("utf-8")).digest(), identity


def select_medium_v7_pilot(train, source_digest, windows_sha256):  # noqa: C901
    """Choose 40 each of witness, local coordination, and diverse controls."""

    buckets = {"witness": [], "dynamic": []}
    controls = {name: [] for name in (
        "goal_approach_control", "idle_or_depleted",
        "empty_geometric_constraint", "other"
    )}
    for window in train:
        view = _semantic_view(window)
        witness = observable_coordination_evidence(view)
        current = view["history"][7]
        local = current["local"]
        recent = view["summary"]["visible_peer_cell_statistics"]["maximum"]
        if witness["positive_supported"]:
            buckets["witness"].append(window)
        elif local["visible_peer_cell_count"] > 0 or recent > 0:
            buckets["dynamic"].append(window)
        elif (current["self"]["task_phase"] == "idle"
              or current["goal"]["goal_type"] == "idle"
              or current["self"]["battery"] <= 0):
            controls["idle_or_depleted"].append(window)
        elif (
            current["self"]["previous_requested_action"] == "FORWARD"
            and not current["self"]["previous_blocked"]
            and current["self"]["task_phase"] != "idle"
            and current["goal"]["goal_type"] != "idle"
            and current["self"]["battery"] > 0
            and view["history"][6]["self"]["task_phase"]
            == current["self"]["task_phase"]
            and view["history"][6]["goal"]["goal_type"]
            == current["goal"]["goal_type"]
            and view["history"][6]["goal"]["normalized_distance"]
            - current["goal"]["normalized_distance"] > 1e-6
        ):
            controls["goal_approach_control"].append(window)
        elif local["non_peer_free_degree"] <= 2:
            controls["empty_geometric_constraint"].append(window)
        else:
            controls["other"].append(window)

    def ranked(rows, name):
        return sorted(rows, key=lambda window: _pilot_rank(
            window, _V7_PILOT_SALT + "|" + name
        ))

    if any(len(buckets[name]) < 40 for name in buckets):
        raise MediumLabelingHalt("v7_pilot_witness_or_dynamic_bucket_short")
    if any(len(controls[name]) < 10 for name in controls):
        raise MediumLabelingHalt("v7_pilot_control_subbucket_short")
    witness_rows = ranked(buckets["witness"], "witness")[:40]
    dynamic_rows = ranked(buckets["dynamic"], "dynamic")[:40]
    control_rows = {
        name: ranked(rows, "control|" + name)[:10]
        for name, rows in controls.items()
    }
    control_names = tuple(controls)
    selected = []
    assignments = []
    for index in range(40):
        control_name = control_names[index % 4]
        control = control_rows[control_name][index // 4]
        for bucket, window, subbucket in (
            ("witness", witness_rows[index], None),
            ("dynamic", dynamic_rows[index], None),
            ("control", control, control_name),
        ):
            selected.append(window)
            assignments.append({
                "ordinal": len(selected) - 1,
                "view_id": _window_id(window),
                "bucket": bucket,
                "control_subbucket": subbucket,
            })
    selected_ids = {_window_id(window) for window in selected}
    if len(selected_ids) != 120:
        raise MediumLabelingHalt("v7_pilot_selection_duplicate_window")
    remainder = sorted(
        (window for window in train if _window_id(window) not in selected_ids),
        key=lambda window: _pilot_rank(window, _V7_PILOT_SALT + "|remaining"),
    )
    snapshot = {
        "schema": _V7_PILOT_POLICY,
        "sampling_salt": _V7_PILOT_SALT,
        "dataset_manifest_sha256": source_digest,
        "windows_sha256": windows_sha256,
        "bucket_available_counts": {
            **{name: len(rows) for name, rows in buckets.items()},
            **{"control_" + name: len(rows) for name, rows in controls.items()},
        },
        "selected": assignments,
    }
    return selected + remainder, snapshot


def _bind_v7_pilot_selection(output, snapshot):
    path = output / "pilot-selection.json"
    if path.exists():
        if json.loads(path.read_text("utf-8")) != snapshot:
            raise MediumLabelingHalt("v7_pilot_selection_changed")
    else:
        _atomic_json(path, snapshot)


def _bind_identity(output: Path, identity: Mapping[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "identity.json"
    if path.exists():
        if json.loads(path.read_text("utf-8")) != identity:
            raise ValueError("Label output identity differs from frozen source.")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty label output lacks an identity file.")
        _atomic_json(path, identity)


def _require_current_pricing_and_secret(checked_date: str | None) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    if checked_date != today:
        raise MediumLabelingHalt("current_official_pricing_not_confirmed")
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise MediumLabelingHalt("DEEPSEEK_API_KEY_missing")


def _cached(store, window, provider, scope, prompt_builder=None):
    builder = prompt_builder or (
        lambda view: build_medium_label_prompt(view, version=LLM_LABEL_PROMPT_V3)
    )
    semantic_view = _semantic_view(window)
    prompt = builder(semantic_view)
    key = _cache_key(prompt.semantic_view_sha256, prompt.prompt_version,
                     provider, scope)
    return store.successful(key, prompt.available_evidence_paths)


def _label_one(  # noqa: C901
    window, scope, provider, store, ledger, prompt_builder=None, *,
    citation_repair_policy=None,
    allow_citation_repair=True,
):
    builder = prompt_builder or (
        lambda view: build_medium_label_prompt(view, version=LLM_LABEL_PROMPT_V3)
    )
    semantic_view = _semantic_view(window)
    prompt = builder(semantic_view)
    key = _cache_key(prompt.semantic_view_sha256, prompt.prompt_version,
                     provider, scope)
    cached = store.successful(key, prompt.available_evidence_paths)
    prior = ledger.history(key)
    if cached is not None:
        if not prior or prior[-1]["status"] != "ok":
            raise MediumLabelingHalt("label_log_and_attempt_log_disagree")
        if (prompt.prompt_version == MEDIUM_LABEL_PROMPT_V7
                and cached.get("receipt", {}).get("semantic_contract_version")
                != SEMANTIC_CONTRACT_V2):
            raise MediumLabelingHalt("cached_v7_contract_version_missing")
        if prompt.prompt_version in {
            MEDIUM_LABEL_PROMPT_V6, MEDIUM_LABEL_PROMPT_V7
        }:
            try:
                validator = (
                    validate_medium_v7_semantic_support
                    if prompt.prompt_version == MEDIUM_LABEL_PROMPT_V7
                    else validate_medium_v6_semantic_support
                )
                validator(cached["scores"], cached["evidence"], semantic_view)
            except ValueError as error:
                raise MediumLabelingHalt(
                    "cached_v7_semantic_support_invalid"
                    if prompt.prompt_version == MEDIUM_LABEL_PROMPT_V7
                    else "cached_v6_semantic_support_invalid"
                ) from error
        return cached
    invalid_count = sum(row["status"] == "invalid_response" for row in prior)
    extra_limit = 1 if citation_repair_policy is not None else 0
    if invalid_count > provider.config.retry_limit + extra_limit or any(
        row["status"] not in {"invalid_response", "presend_socket_denied"}
        and not (row["status"] == "outcome_unknown"
                 and (ledger.reconciled_not_sent(row["attempt_id"])
                      or ledger.authorized_unknown_retry(row["attempt_id"])))
        for row in prior
    ):
        raise MediumLabelingHalt("previous_attempt_requires_manual_reconciliation")
    for retry_index in range(
        invalid_count, provider.config.retry_limit + 1 + extra_limit,
    ):
        invalid = [row for row in ledger.history(key)
                   if row["status"] == "invalid_response"]
        repair = bool(invalid and prompt.prompt_version in {
            MEDIUM_LABEL_PROMPT_V4, MEDIUM_LABEL_PROMPT_V5,
            MEDIUM_LABEL_PROMPT_V6, MEDIUM_LABEL_PROMPT_V7,
        })
        citation_extra = retry_index > provider.config.retry_limit
        if citation_extra:
            extra = citation_repair_policy.begin(
                ledger, key, prompt, semantic_view, provider, scope,
                allow_request=allow_citation_repair,
            )
            if extra is None:
                raise MediumLabelingHalt("response_validation_failed_after_retry")
            active_prompt, attempt_id, _ = extra
            variant = "citation-only-repair-v1"
        else:
            active_prompt = _repair_prompt(prompt, invalid[-1]) if repair else prompt
            variant = "validation-repair-v1" if repair else "base"
            if len(active_prompt.user_text) > provider.config.max_prompt_characters:
                raise MediumLabelingHalt("repair_prompt_size_limit_exceeded")
            attempt_id = ledger.begin(
                scope, key, active_prompt, provider, prompt_variant=variant,
                base_user_prompt_sha256=prompt.user_sha256,
            )
        try:
            completion = provider.complete(active_prompt)
        except ProviderCallError as error:
            status = (
                "presend_socket_denied" if error.confirmed_not_sent
                else "outcome_unknown" if error.outcome_unknown else error.code
            )
            ledger.settle(
                attempt_id, status,
                response_audit={"provider_failure": error.failure_metadata},
            )
            raise MediumLabelingHalt(
                "provider_attempt_requires_reconciliation"
            ) from None
        usage = completion.receipt.get("usage", {})
        if not all(type(usage.get(name)) is int for name in
                   ("prompt_tokens", "completion_tokens", "total_tokens")):
            ledger.settle(attempt_id, "missing_usage", completion.receipt)
            raise MediumLabelingHalt("provider_usage_missing")
        if usage["prompt_tokens"] + usage["completion_tokens"] > usage[
            "total_tokens"
        ]:
            ledger.settle(attempt_id, "invalid_usage", completion.receipt)
            raise MediumLabelingHalt("provider_usage_invalid")
        try:
            parsed = parse_label_response(
                completion.content, active_prompt.available_evidence_paths
            )
            if prompt.prompt_version in {
                MEDIUM_LABEL_PROMPT_V6, MEDIUM_LABEL_PROMPT_V7
            }:
                validator = (
                    validate_medium_v7_semantic_support
                    if prompt.prompt_version == MEDIUM_LABEL_PROMPT_V7
                    else validate_medium_v6_semantic_support
                )
                validator(parsed.scores, parsed.evidence, semantic_view)
        except ValueError as error:
            ledger.settle(
                attempt_id, "invalid_response", completion.receipt,
                response_audit=_response_audit(
                    completion.content, prompt.available_evidence_paths,
                    parse_error=str(error),
                ),
            )
            continue
        evidence_count = sum(len(paths) for paths in parsed.evidence.values())
        ledger.settle(
            attempt_id, "ok", completion.receipt,
            response_audit={
                "forbidden_content": False,
                "evidence_path_count": evidence_count,
                "valid_evidence_path_count": evidence_count,
            },
        )
        retries = sum(
            event["status"] == "invalid_response" for event in ledger.history(key)
        )
        unknown_retries = sum(
            event["status"] == "outcome_unknown"
            and ledger.authorized_unknown_retry(event["attempt_id"])
            for event in ledger.history(key)
        )
        receipt = {
            **completion.receipt,
            "retry_history": ["response_validation_failed"] * retries,
            "authorized_unknown_retry_count": unknown_retries,
            "prompt_variant": variant,
            "base_user_prompt_sha256": prompt.user_sha256,
            "repair_basis": invalid[-1]["response_audit"] if repair else None,
        }
        if prompt.prompt_version == MEDIUM_LABEL_PROMPT_V7:
            receipt["semantic_contract_version"] = SEMANTIC_CONTRACT_V2
        if citation_extra:
            receipt["citation_only_repair_authorization_sha256"] = (
                citation_repair_policy.authorization_sha256
            )
        record = {
            "schema_version": "llm-semantic-label-record-v1",
            "status": "ok", "view_id": _window_id(window),
            "cache_key": key, "cache_namespace": scope,
            "scores": parsed.scores,
            "evidence": {name: list(paths) for name, paths in parsed.evidence.items()},
            "receipt": receipt,
        }
        store.append(record)
        return record
    raise MediumLabelingHalt("response_validation_failed_after_retry")


def _repair_prompt(base, invalid_event):
    audit = invalid_event.get("response_audit", {})
    error = audit.get("parse_error", "response_validation_failed")
    unavailable = audit.get("unavailable_evidence_paths", [])
    prefix, canonical = base.user_text.split("TEMPORAL_SEMANTIC_VIEW=", 1)
    repair = (
        "VALIDATION_REPAIR_V1: The previous response failed strict validation. "
        "Validation error: " + str(error) + ". Unavailable evidence paths: "
        + json.dumps(unavailable, ensure_ascii=False, separators=(",", ":"))
        + ". Return a fresh complete JSON object with all four scores and "
        "all four evidence lists. Copy evidence strings only from "
        "EVIDENCE_PATH_ALLOWLIST; do not cite unavailable paths or invent "
        "optional peer fields.\n"
    )
    user = prefix + repair + "TEMPORAL_SEMANTIC_VIEW=" + canonical
    return replace(
        base, user_text=user,
        user_sha256=sha256(user.encode("utf-8")).hexdigest(),
    )


def _response_audit(content, allowed, *, parse_error=None):
    """Count visible evidence paths without preserving untrusted model text."""

    forbidden_names = (
        "recommended_action", "priority_winner", "agent_assignment",
        "next_action", "route_command", "waypoint_command",
    )
    forbidden = any(name in content.lower() for name in forbidden_names)
    total = valid = 0
    unavailable = set()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        forbidden = forbidden or bool(set(payload) - {"scores", "evidence"})
        evidence = payload.get("evidence")
        if isinstance(evidence, dict):
            for paths in evidence.values():
                if isinstance(paths, list):
                    total += len(paths)
                    valid += sum(
                        isinstance(path, str) and path in allowed for path in paths
                    )
                    unavailable.update(
                        path for path in paths
                        if isinstance(path, str) and path not in allowed
                        and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,120}", path)
                    )
    evidence_value = payload.get("evidence") if isinstance(payload, dict) else None
    scores_value = payload.get("scores") if isinstance(payload, dict) else None

    def safe_keys(value):
        if not isinstance(value, dict):
            return []
        return sorted(
            key for key in value if isinstance(key, str)
            and re.fullmatch(r"[A-Za-z0-9_]{1,80}", key)
        )[:16]

    return {
        "forbidden_content": forbidden,
        "evidence_path_count": total,
        "valid_evidence_path_count": valid,
        "unavailable_evidence_paths": sorted(unavailable)[:16],
        "parse_error": parse_error,
        "top_level_json_type": type(payload).__name__,
        "top_level_keys": safe_keys(payload),
        "evidence_json_type": type(evidence_value).__name__,
        "evidence_keys": safe_keys(evidence_value),
        "scores_json_type": type(scores_value).__name__,
        "scores_keys": safe_keys(scores_value),
        "parsed_scores": {
            key: value for key, value in (
                scores_value.items() if isinstance(scores_value, dict) else ()
            )
            if key in SEMANTIC_DIMENSIONS and type(value) in {int, float}
            and 0 <= value <= 1 and np.isfinite(value)
        },
        "cited_original_evidence": {
            key: [path for path in paths if isinstance(path, str) and path in allowed]
            for key, paths in (evidence_value.items()
                               if isinstance(evidence_value, dict) else ())
            if key in SEMANTIC_DIMENSIONS and isinstance(paths, list) and len(paths) <= 4
        },
    }


class _AttemptLedger:
    def __init__(self, path: Path, config: Mapping[str, object]):
        self.path = path
        self.config = config
        self._lock = RLock()
        self.halted = False
        self.starts = {}
        self.settled = {}
        self.by_key = {}
        self.models = {}
        self.fingerprints = {}
        self.reconciliations = {}
        self.unknown_retry_authorizations = {}
        self.prior_budget = None
        self._counts = Counter()
        self._input_total = 0
        self._output_total = 0
        self._cost_total = 0.0
        if path.exists():
            running_digest = sha256()
            for raw_line in path.read_bytes().splitlines(keepends=True):
                event = json.loads(raw_line)
                if (event.get("event") == "unknown_retry_authorization"
                        and event.get("original_attempt_log_sha256")
                        != running_digest.hexdigest()):
                    raise MediumLabelingHalt(
                        "unknown_retry_original_log_digest_changed"
                    )
                self._apply(event)
                running_digest.update(raw_line)
        if set(self.starts) != set(self.settled):
            raise MediumLabelingHalt("unresolved_sent_request_requires_reconciliation")

    def _apply(self, event):  # noqa: C901
        attempt_id = event.get("attempt_id")
        if event.get("event") == "historical_budget":
            if (
                self.prior_budget is not None or self.starts
                or event.get("schema") != "medium-label-historical-budget-v1"
                or event.get("http_attempts", {}).get("primary", -1) < 1
                or event.get("input_tokens_upper", -1) < 0
                or event.get("output_tokens_upper", -1) < 0
                or event.get("conservative_cost_upper_usd", -1) < 0
            ):
                raise ValueError("Invalid historical budget event.")
            self.prior_budget = event
        elif event.get("event") == "start":
            if attempt_id in self.starts:
                raise ValueError("Duplicate attempt identity.")
            self.starts[attempt_id] = event
            scope = event["scope"]
            self._counts[scope] += 1
            self._input_total += event["input_reserve"]
            self._output_total += event["output_reserve"]
            rate = self._rate(scope)
            self._cost_total += (
                event["input_reserve"] * rate["input"]
                + event["output_reserve"] * rate["output"]
            ) / 1000000
        elif event.get("event") == "settled":
            if attempt_id not in self.starts or attempt_id in self.settled:
                raise ValueError("Invalid attempt settlement.")
            self.settled[attempt_id] = event
            start = self.starts[attempt_id]
            self.by_key.setdefault(start["cache_key"], []).append(event)
            usage = event.get("usage", {})
            if event.get("status") in {"ok", "invalid_response"} and usage:
                input_delta = usage["prompt_tokens"] - start["input_reserve"]
                output_delta = usage["completion_tokens"] - start["output_reserve"]
                self._input_total += input_delta
                self._output_total += output_delta
                rate = self._rate(start["scope"])
                self._cost_total += (
                    input_delta * rate["input"]
                    + output_delta * rate["output"]
                ) / 1000000
            if event.get("returned_model"):
                scope = start["scope"]
                model = event["returned_model"]
                fingerprint = event.get("system_fingerprint")
                if scope in self.models and self.models[scope] != model:
                    raise MediumLabelingHalt("returned_model_changed")
                if (scope in self.fingerprints
                        and self.fingerprints[scope] != fingerprint):
                    raise MediumLabelingHalt("system_fingerprint_changed")
                self.models[scope] = model
                self.fingerprints[scope] = fingerprint
                if (scope == "comparison" and self.models.get("primary") == model):
                    raise MediumLabelingHalt("comparison_model_not_distinct")
        elif event.get("event") == "reconciliation":
            original = self.settled.get(attempt_id, {})
            if (
                original.get("status") != "outcome_unknown"
                or attempt_id in self.reconciliations
                or event.get("decision") != "confirmed_not_sent"
                or not isinstance(event.get("reviewer"), str)
                or not event["reviewer"]
                or not _valid_presend_evidence(event.get("evidence"))
            ):
                raise ValueError("Invalid presend reconciliation event.")
            self.reconciliations[attempt_id] = event
        elif event.get("event") == "unknown_retry_authorization":
            self._validate_unknown_retry_authorization(event)
            self.unknown_retry_authorizations[attempt_id] = event
        else:
            raise ValueError("Unknown attempt log event.")

    def _validate_unknown_retry_authorization(self, event):
        attempt_id = event.get("attempt_id")
        start = self.starts.get(attempt_id, {})
        same_view_authorized = any(
            self.starts[old_id]["semantic_view_sha256"]
            == start.get("semantic_view_sha256")
            for old_id in self.unknown_retry_authorizations
        )
        identity_path = self.path.parent / "identity.json"
        if (
            event.get("schema") != "medium-label-unknown-retry-authorization-v1"
            or self.settled.get(attempt_id, {}).get("status")
            != "outcome_unknown"
            or attempt_id in self.reconciliations
            or attempt_id in self.unknown_retry_authorizations
            or same_view_authorized
            or event.get("reviewer") != "Codex root"
            or event.get("reason") != (
                "Request outcome unavailable; accepted possible duplicate "
                "charge under existing USD50 authorization"
            )
            or event.get("decision") != "one_retry_with_possible_duplicate_charge"
            or event.get("original_reservation_retained") is not True
            or event.get("actual_provider_bill_verified") is not False
            or not identity_path.exists()
            or event.get("identity_sha256") != sha256(
                identity_path.read_bytes()
            ).hexdigest()
        ):
            raise MediumLabelingHalt("invalid_unknown_retry_authorization")

    def _append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            stream.write((json.dumps(event, sort_keys=True, separators=(",", ":"))
                          + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        self._apply(event)

    def history(self, cache_key):
        with self._lock:
            return list(self.by_key.get(cache_key, []))

    def reconciled_not_sent(self, attempt_id):
        with self._lock:
            return attempt_id in self.reconciliations

    def authorized_unknown_retry(self, attempt_id):
        with self._lock:
            return attempt_id in self.unknown_retry_authorizations

    def authorize_unknown_retry(self, attempt_id, reviewer, *,
                                original_attempt_log_sha256,
                                identity_sha256):
        with self._lock:
            attempt_id = str(attempt_id)
            if sha256(self.path.read_bytes()).hexdigest() != (
                original_attempt_log_sha256
            ):
                raise MediumLabelingHalt("original_attempt_log_digest_changed")
            event = {
                "event": "unknown_retry_authorization",
                "schema": "medium-label-unknown-retry-authorization-v1",
                "attempt_id": attempt_id,
                "reviewer": reviewer,
                "reason": (
                    "Request outcome unavailable; accepted possible duplicate "
                    "charge under existing USD50 authorization"
                ),
                "decision": "one_retry_with_possible_duplicate_charge",
                "original_attempt_log_sha256": original_attempt_log_sha256,
                "identity_sha256": identity_sha256,
                "original_reservation_retained": True,
                "actual_provider_bill_verified": False,
            }
            self._validate_unknown_retry_authorization(event)
            self._append(event)

    def bind_prior_budget(self, event):
        with self._lock:
            if self.prior_budget is not None:
                if self.prior_budget != event:
                    raise MediumLabelingHalt("historical_budget_source_changed")
                return
            if self.starts:
                raise MediumLabelingHalt("historical_budget_missing_before_attempts")
            self._append(event)

    def reconcile_confirmed_not_sent(self, attempt_id, reviewer, evidence):
        with self._lock:
            event = {
                "event": "reconciliation", "attempt_id": str(attempt_id),
                "decision": "confirmed_not_sent", "reviewer": reviewer,
                "evidence": evidence,
            }
            if not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError("Reconciliation reviewer is required.")
            if not _valid_presend_evidence(evidence):
                raise ValueError("Presend reconciliation evidence is invalid.")
            self._append(event)

    def _totals(self):
        prior = self.prior_budget or {}
        counts = self._counts.copy()
        counts.update(prior.get("http_attempts", {}))
        return (
            counts,
            self._input_total + prior.get("input_tokens_upper", 0),
            self._output_total + prior.get("output_tokens_upper", 0),
            self._cost_total + prior.get("conservative_cost_upper_usd", 0.0),
        )

    def _rate(self, scope):
        model = "comparison" if scope == "comparison" else "primary"
        return self.config["peak_uncached_usd_per_million_tokens"][model]

    def begin(self, scope, cache_key, prompt, provider, *,
              prompt_variant="base", base_user_prompt_sha256=None):
        with self._lock:
            return self._begin_unlocked(
                scope, cache_key, prompt, provider, prompt_variant,
                base_user_prompt_sha256,
            )

    def _begin_unlocked(self, scope, cache_key, prompt, provider,
                        prompt_variant, base_user_prompt_sha256):
        if self.halted:
            raise MediumLabelingHalt("previous_attempt_requires_reconciliation")
        limits = self.config["limits"]
        counts, input_tokens, output_tokens, cost = self._totals()
        cap = {
            "primary": limits["primary_http_attempts"],
            "repeat": limits["repeat_http_attempts"],
            "comparison": limits["comparison_http_attempts"],
        }[scope]
        input_reserve = 2 * len((prompt.system_text + prompt.user_text)
                                .encode("utf-8")) + 4096
        output_reserve = provider.config.max_completion_tokens
        rate = self.config["peak_uncached_usd_per_million_tokens"][
            "comparison" if scope == "comparison" else "primary"
        ]
        reserve_cost = (input_reserve * rate["input"]
                        + output_reserve * rate["output"]) / 1000000
        if (counts[scope] >= cap or input_tokens + input_reserve > limits["input_tokens"]
                or output_tokens + output_reserve > limits["output_tokens"]
                or cost + reserve_cost > limits["conservative_usd"]):
            raise MediumLabelingHalt("labeling_budget_exhausted")
        attempt_id = str(len(self.starts) + 1)
        self._append({
            "event": "start", "attempt_id": attempt_id, "scope": scope,
            "cache_key": cache_key, "requested_model": provider.config.requested_model,
            "prompt_version": prompt.prompt_version,
            "prompt_variant": prompt_variant,
            "user_prompt_sha256": prompt.user_sha256,
            "base_user_prompt_sha256": base_user_prompt_sha256,
            "semantic_view_sha256": prompt.semantic_view_sha256,
            "input_reserve": input_reserve, "output_reserve": output_reserve,
        })
        return attempt_id

    def settle(self, attempt_id, status, receipt=None, response_audit=None):
        with self._lock:
            self._settle_unlocked(attempt_id, status, receipt, response_audit)

    def _settle_unlocked(self, attempt_id, status, receipt=None,
                         response_audit=None):
        receipt = receipt or {}
        event = {
            "event": "settled", "attempt_id": attempt_id, "status": status,
            "returned_model": receipt.get("returned_model"),
            "system_fingerprint": receipt.get("system_fingerprint"),
            "response_sha256": receipt.get("response_sha256"),
            "usage": receipt.get("usage", {}),
            "response_audit": response_audit or {},
        }
        try:
            self._append(event)
        except Exception:
            self.halted = True
            raise
        if status not in {"ok", "invalid_response"}:
            self.halted = True
        _, input_tokens, output_tokens, cost = self._totals()
        limits = self.config["limits"]
        if (input_tokens > limits["input_tokens"]
                or output_tokens > limits["output_tokens"]
                or cost > limits["conservative_usd"]):
            self.halted = True
            raise MediumLabelingHalt("settled_usage_exceeds_budget")

    def summary(self):
        with self._lock:
            counts, input_tokens, output_tokens, cost = self._totals()
            models = dict(self.models)
            fingerprints = dict(self.fingerprints)
            statuses = dict(Counter(
                event["status"] for event in self.settled.values()
            ))
            reconciliation_count = len(self.reconciliations)
            unknown_authorization_count = len(self.unknown_retry_authorizations)
            prior = dict(self.prior_budget) if self.prior_budget else None
        return {
            "http_attempts": dict(counts), "input_tokens_upper": input_tokens,
            "output_tokens_upper": output_tokens,
            "conservative_cost_upper_usd": cost,
            "cost_is_actual_bill": False,
            "attempt_status_counts": statuses,
            "confirmed_not_sent_reconciliations": reconciliation_count,
            "authorized_unknown_retries": unknown_authorization_count,
            "historical_budget": prior,
            "returned_models": models,
            "system_fingerprints": fingerprints,
            "fingerprint_absence_limits_backend_verification": any(
                value is None for value in fingerprints.values()
            ),
        }


def _quality_metrics(  # noqa: C901
    windows, train, stores, ledger, stage, version=LLM_LABEL_PROMPT_V3
):
    primary = {row["view_id"]: row for row in stores["primary"].records()
               if row.get("status") == "ok"}
    chosen = train[:120] if stage == "pilot" else windows
    expected = [_window_id(window) for window in chosen]
    if any(view_id not in primary for view_id in expected):
        raise MediumLabelingHalt("primary_label_coverage_incomplete")
    scores = np.array([[primary[view_id]["scores"][name]
                        for name in SEMANTIC_DIMENSIONS] for view_id in expected])
    repeats = {row["view_id"]: row for row in stores["repeat"].records()
               if row.get("status") == "ok"}
    comparisons = {row["view_id"]: row for row in stores["comparison"].records()
                   if row.get("status") == "ok"}
    audit_ids = [_window_id(window) for window in train[:40]]
    if any(view_id not in repeats or view_id not in comparisons
           for view_id in audit_ids):
        raise MediumLabelingHalt("pilot_audit_coverage_incomplete")

    def matrix(rows):
        return np.array([[rows[view_id]["scores"][name]
                          for name in SEMANTIC_DIMENSIONS] for view_id in audit_ids])
    attempts = [ledger.history(primary[view_id]["cache_key"]) for view_id in expected]
    sent_attempts = [
        [event for event in history
         if event["status"] != "presend_socket_denied"
         and not (event["status"] == "outcome_unknown"
                  and ledger.reconciled_not_sent(event["attempt_id"]))]
        for history in attempts
    ]
    first_pass = sum(bool(history and history[0]["status"] == "ok")
                     for history in sent_attempts)
    retry_count = sum(max(0, len(history) - 1) for history in sent_attempts)
    audits = [event.get("response_audit", {}) for history in attempts
              for event in history]
    forbidden_count = sum(bool(audit.get("forbidden_content")) for audit in audits)
    evidence_total = sum(audit.get("evidence_path_count", 0) for audit in audits)
    evidence_valid = sum(audit.get("valid_evidence_path_count", 0)
                         for audit in audits)
    metrics = label_diagnostic_metrics(
        scores=scores, requested_samples=len(expected), parsed_samples=len(expected),
        retry_count=retry_count, forbidden_response_count=forbidden_count,
        valid_evidence_path_count=evidence_valid,
        evidence_path_count=evidence_total,
        repeated_call_pairs=(matrix(primary), matrix(repeats)),
        cross_model_pairs=(matrix(primary), matrix(comparisons)),
        boundary_epsilon=0.05,
    )
    metrics["first_attempt_parse_rate"] = first_pass / len(expected)
    metrics["parse_rate"] = metrics["first_attempt_parse_rate"]
    metrics["final_success_coverage"] = len(expected) / len(expected)
    reasons = []
    if metrics["first_attempt_parse_rate"] < 0.98:
        reasons.append("first_attempt_parse_rate")
    if metrics["retry_rate"] > 0.10:
        reasons.append("retry_rate")
    if metrics["forbidden_content_rate"] != 0.0:
        reasons.append("forbidden_content")
    if (metrics["evidence_path_valid_rate"] is None
            or metrics["evidence_path_valid_rate"] < 0.98):
        reasons.append("invalid_evidence_paths")
    if metrics["repeated_call_mean_absolute_disagreement"] > 0.15:
        reasons.append("repeat_disagreement")
    if metrics["cross_model_mean_absolute_disagreement"] > 0.20:
        reasons.append("cross_model_disagreement")
    boundary_warnings = []
    for name, dimension in metrics["per_dimension"].items():
        if dimension["standard_deviation"] < 0.08:
            reasons.append(name + "_collapse")
        if dimension["boundary_saturation_rate"] > 0.60:
            if version in {MEDIUM_LABEL_PROMPT_V6, MEDIUM_LABEL_PROMPT_V7}:
                boundary_warnings.append(name + "_saturation")
            else:
                reasons.append(name + "_saturation")
    if version in {MEDIUM_LABEL_PROMPT_V6, MEDIUM_LABEL_PROMPT_V7}:
        metrics["quality_policy_version"] = _V6_POLICY
        metrics["quality_policy_authorization_sha256"] = (
            _V6_POLICY_AUTH_SHA
        )
        metrics["boundary_saturation_diagnostic_warnings"] = boundary_warnings
        metrics["score_ge_0_25_counts"] = {
            name: int(np.count_nonzero(scores[:, index] >= 0.25))
            for index, name in enumerate(SEMANTIC_DIMENSIONS)
        }
        if version == MEDIUM_LABEL_PROMPT_V7:
            metrics["semantic_contract_version"] = SEMANTIC_CONTRACT_V2
        if stage == "full":
            train_values = np.array([
                [primary[_window_id(window)]["scores"][name]
                 for name in SEMANTIC_DIMENSIONS]
                for window in train
            ])
            train_counts = {
                name: int(np.count_nonzero(train_values[:, index] >= 0.25))
                for index, name in enumerate(SEMANTIC_DIMENSIONS)
            }
            metrics["train_score_ge_0_25_counts"] = train_counts
            for name, count in train_counts.items():
                if count < 120:
                    reasons.append(name + "_insufficient_train_positive_coverage")
    metrics["go"] = not reasons
    metrics["no_go_reasons"] = reasons
    return metrics


def _write_blind_packet(
    output, windows, store, version=LLM_LABEL_PROMPT_V3
):
    records = {row["view_id"]: row for row in store.records()
               if row.get("status") == "ok"}
    path = output / "blind-review-packet.jsonl"
    if version == MEDIUM_LABEL_PROMPT_V7:
        encoded = b"".join(
            (json.dumps({
                "view_id": _window_id(window),
                "semantic_view": _semantic_view(window),
                "scores": records[_window_id(window)]["scores"],
                "evidence": records[_window_id(window)]["evidence"],
            }, separators=(",", ":")) + "\n").encode("utf-8")
            for window in windows
        )
        if path.exists():
            if path.read_bytes() != encoded:
                raise MediumLabelingHalt("v7_blind_packet_changed")
            return
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for window in windows:
            view_id = _window_id(window)
            row = records[view_id]
            stream.write(json.dumps({
                "view_id": view_id, "semantic_view": _semantic_view(window),
                "scores": row["scores"], "evidence": row["evidence"],
            }, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _require_pilot_and_review(output, train, stores, ledger, version):
    path = output / "pilot-metrics.json"
    if not path.exists() or json.loads(path.read_text("utf-8")).get("go") is not True:
        raise MediumLabelingHalt("pilot_quality_not_go")
    current = _quality_metrics(train, train, stores, ledger, "pilot", version)
    if current != json.loads(path.read_text("utf-8")) or not current["go"]:
        raise MediumLabelingHalt("pilot_metrics_changed")
    if version == MEDIUM_LABEL_PROMPT_V7:
        _write_blind_packet(
            output, train[:60], stores["primary"], version
        )
    review_path = output / "blind-review.json"
    if not review_path.exists():
        raise MediumLabelingHalt("blind_review_missing")
    review = json.loads(review_path.read_text("utf-8"))
    expected = {_window_id(window) for window in train[:60]}
    assessments = review.get("assessments", []) if isinstance(review, dict) else []
    if (review.get("schema") != "medium-5ag-blind-review-v1"
            or review.get("reviewer_type") not in {"codex", "human"}
            or not isinstance(review.get("reviewer"), str)
            or not review["reviewer"] or not isinstance(assessments, list)
            or {row.get("view_id") for row in assessments} != expected
            or len(assessments) != 60
            or any(type(row.get("critical_error")) is not bool
                   or type(row.get("substantive_error")) is not bool
                   for row in assessments)):
        raise MediumLabelingHalt("blind_review_invalid")
    if version == MEDIUM_LABEL_PROMPT_V7 and (
        review.get("packet_sha256") != sha256(
            (output / "blind-review-packet.jsonl").read_bytes()
        ).hexdigest()
        or review.get("label_identity_sha256") != sha256(
            (output / "identity.json").read_bytes()
        ).hexdigest()
        or review.get("semantic_contract_version") != SEMANTIC_CONTRACT_V2
    ):
        raise MediumLabelingHalt("v7_blind_review_binding_invalid")
    if (sum(row["critical_error"] for row in assessments) != 0
            or sum(row["substantive_error"] for row in assessments) > 3):
        raise MediumLabelingHalt("blind_review_no_go")


def _receipt(dataset, output, source, identity, stores, ledger, stage):
    primary = [row for row in stores["primary"].records()
               if row.get("status") == "ok"]
    counts = Counter(row["view_id"].split(":", 1)[0] for row in primary)
    if stage == "full" and counts != {
        "train": 4200, "validation": 900, "test": 900
    }:
        raise MediumLabelingHalt("full_primary_coverage_incomplete")
    receipt = {
        "schema": DATASET_LABEL_RUN_V1,
        "stage": stage, "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
        "dataset_distribution": source["distribution"],
        "label_config_sha256": identity["label_config_sha256"],
        "windows_sha256": identity["windows_sha256"],
        "successful_records": len(primary),
        "partition_counts": dict(counts),
        "labels_sha256": sha256((output / "labels.jsonl").read_bytes()).hexdigest(),
        "attempts_sha256": sha256((output / "attempts.jsonl").read_bytes()).hexdigest(),
        "repeat_records": sum(
            row.get("status") == "ok" for row in stores["repeat"].records()
        ) if "repeat" in stores else 0,
        "comparison_records": sum(
            row.get("status") == "ok" for row in stores["comparison"].records()
        ) if "comparison" in stores else 0,
        "selected_strata_counts": source.get("selected_strata_counts"),
        "source_coverage": source.get("coverage"),
        "source_hashes": {
            name: source.get(name) for name in (
                "source_actor_sha256", "source_checkpoint_sha256",
                "source_training_config_sha256", "environment_sha256",
                "config_sha256", "collection_identity_sha256",
                "selection_code_sha256", "episode_receipts_sha256",
            )
        },
        "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/",
        "usage_and_budget": ledger.summary(),
        "blind_review_is_independent_human": False,
        "pretraining_eligible": stage == "full",
    }
    if "quality_policy_version" in identity:
        receipt["quality_policy_version"] = identity["quality_policy_version"]
        receipt["quality_policy_authorization_sha256"] = identity[
            "quality_policy_authorization_sha256"
        ]
    if "semantic_contract_version" in identity:
        for name in (
            "semantic_contract_version", "authorization_sha256",
            "pilot_selection_policy_version", "pilot_selection_sha256",
        ):
            receipt[name] = identity[name]
    if "evidence_recipe_version" in identity:
        for name in ("evidence_recipe_version", "evidence_recipe_code_sha256"):
            receipt[name] = identity[name]
        if "evidence_recipe_dependency_sha256" in identity:
            receipt["evidence_recipe_dependency_sha256"] = identity[
                "evidence_recipe_dependency_sha256"
            ]
    if stage == "full":
        review = json.loads((output / "blind-review.json").read_text("utf-8"))
        receipt["blind_review_is_independent_human"] = (
            review["reviewer_type"] == "human"
        )
        receipt["blind_review_sha256"] = sha256(
            (output / "blind-review.json").read_bytes()
        ).hexdigest()
        receipt["quality_metrics"] = json.loads(
            (output / "full-metrics.json").read_text("utf-8")
        )
    return receipt
