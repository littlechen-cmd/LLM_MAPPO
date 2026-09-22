"""Tests for held-out representation-diagnostic access contracts."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

from llm_mappo.mappo_llm_kd import config as config_module
from llm_mappo.mappo_llm_kd.config import (
    ResolvedConfig,
    load_m1_config,
)
from llm_mappo.mappo_llm_kd.representation_diagnostics import (
    DIAGNOSTIC_RUN_V1,
    REPRESENTATION_VARIANTS,
    TestAccessLedger,
    create_diagnostic_run_identity,
    validate_diagnostic_run_identity,
)


M1_CONFIG = "configs/llm_kd/representation-diagnostics.yaml"
M1_APPROVAL = "configs/llm_kd/representation-diagnostics-approval.json"
SHA = "a" * 64


def _identity():
    return create_diagnostic_run_identity(
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        dataset_sha256=SHA,
        train_episode_ids=("train-001", "train-002"),
        validation_episode_ids=("validation-001",),
        test_episode_ids=("test-001",),
        representation_seeds=(7301, 7302, 7303),
    )


def _rehash_receipts(payload):
    previous = payload["run_identity"]["empty_ledger_anchor_sha256"]
    for sequence, receipt in enumerate(payload["receipts"], start=1):
        receipt["sequence"] = sequence
        receipt["previous_entry_hash"] = previous
        body = {key: value for key, value in receipt.items() if key != "entry_hash"}
        encoded = (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
        receipt["entry_hash"] = sha256(encoded.encode("utf-8")).hexdigest()
        previous = receipt["entry_hash"]


def test_run_identity_freezes_exact_variants_splits_seeds_and_m1_identity():
    m1 = load_m1_config(M1_CONFIG)
    identity = _identity()

    assert identity == {
        "schema_version": DIAGNOSTIC_RUN_V1,
        "metric_definition_version": "llm-kd-diagnostic-metrics-v1",
        "m1_config_schema_version": "phase3-m1-diagnostics-v1",
        "m1_config_sha256": m1.sha256,
        "dataset_sha256": SHA,
        "empty_ledger_anchor_sha256": identity["empty_ledger_anchor_sha256"],
        "representation_variants": list(REPRESENTATION_VARIANTS),
        "representation_seeds": [7301, 7302, 7303],
        "episode_partitions": {
            "train": ["train-001", "train-002"],
            "validation": ["validation-001"],
            "test": ["test-001"],
        },
    }
    assert REPRESENTATION_VARIANTS == (
        "normal_history",
        "shuffled_label",
        "shuffled_history",
        "single_frame",
        "random_init",
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.pop("dataset_sha256"), "fields"),
        (lambda value: value.update(extra=True), "fields"),
        (
            lambda value: value["representation_variants"].append("unknown"),
            "anchor",
        ),
        (
            lambda value: value["representation_seeds"].append(99),
            "anchor",
        ),
        (
            lambda value: value["episode_partitions"]["test"].append("train-001"),
            "anchor",
        ),
        (
            lambda value: value["episode_partitions"].update(test=[]),
            "anchor",
        ),
        (
            lambda value: value.update(m1_config_sha256="b" * 64),
            "M1 config",
        ),
    ],
)
def test_run_identity_rejects_missing_extra_overlap_and_tampering(mutation, match):
    identity = copy.deepcopy(_identity())
    mutation(identity)
    with pytest.raises(ValueError, match=match):
        validate_diagnostic_run_identity(identity, M1_CONFIG, M1_APPROVAL)


def test_run_identity_api_rejects_self_consistent_forged_resolved_config():
    values = copy.deepcopy(load_m1_config(M1_CONFIG).values)
    values["probe"]["seeds"] = [8101, 8102, 8103]
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    forged = ResolvedConfig(
        values=values, sha256=sha256(encoded).hexdigest()
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'm1_config'"):
        create_diagnostic_run_identity(
            m1_config=forged,
            dataset_sha256=SHA,
            train_episode_ids=("train",),
            validation_episode_ids=("validation",),
            test_episode_ids=("test",),
            representation_seeds=(7301, 7302, 7303),
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema_version", "llm-representation-diagnostics-v2", "schema version"),
        ("metric_definition_version", "metrics-v2", "metric definition"),
        ("probe.seeds", [8101, 8102, 8103], "approval hash"),
    ],
)
def test_modified_m1_cannot_bypass_strict_loader_or_owner_approval(
    tmp_path, field, value, match
):
    values = load_m1_config(M1_CONFIG).values
    if field == "probe.seeds":
        values["probe"]["seeds"] = value
        seeds = tuple(value)
    else:
        values[field] = value
        seeds = (7301, 7302, 7303)
    config_path = tmp_path / "m1.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        create_diagnostic_run_identity(
            m1_config_path=config_path,
            m1_approval_path=M1_APPROVAL,
            dataset_sha256=SHA,
            train_episode_ids=("train",),
            validation_episode_ids=("validation",),
            test_episode_ids=("test",),
            representation_seeds=seeds,
        )


def test_approved_m1_loader_uses_one_config_snapshot_across_validation(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "m1.yaml"
    approval_path = tmp_path / "m1-approval.json"
    original_bytes = Path(M1_CONFIG).read_bytes()
    config_path.write_bytes(original_bytes)
    approval_path.write_bytes(Path(M1_APPROVAL).read_bytes())
    approved_sha = load_m1_config(config_path).sha256
    original_validate = config_module._validate_m1_approval_record

    def replace_after_approval(approval, resolved):
        original_validate(approval, resolved)
        values = yaml.safe_load(original_bytes)
        values["probe"]["seeds"] = [8101, 8102, 8103]
        config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

    monkeypatch.setattr(
        config_module, "_validate_m1_approval_record", replace_after_approval
    )
    identity = create_diagnostic_run_identity(
        m1_config_path=config_path,
        m1_approval_path=approval_path,
        dataset_sha256=SHA,
        train_episode_ids=("train",),
        validation_episode_ids=("validation",),
        test_episode_ids=("test",),
        representation_seeds=(7301, 7302, 7303),
    )
    assert identity["m1_config_sha256"] == approved_sha
    assert identity["representation_seeds"] == [7301, 7302, 7303]

    monkeypatch.setattr(
        config_module, "_validate_m1_approval_record", original_validate
    )
    with pytest.raises(ValueError, match="approval hash"):
        create_diagnostic_run_identity(
            m1_config_path=config_path,
            m1_approval_path=approval_path,
            dataset_sha256=SHA,
            train_episode_ids=("train",),
            validation_episode_ids=("validation",),
            test_episode_ids=("test",),
            representation_seeds=(8101, 8102, 8103),
        )


def test_run_identity_rejects_tampered_m1_approval_hash(tmp_path):
    approval = json.loads(Path(M1_APPROVAL).read_text(encoding="utf-8"))
    approval["approved_config_sha256"] = "b" * 64
    approval_path = tmp_path / "m1-approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")

    with pytest.raises(ValueError, match="approval hash"):
        create_diagnostic_run_identity(
            m1_config_path=M1_CONFIG,
            m1_approval_path=approval_path,
            dataset_sha256=SHA,
            train_episode_ids=("train",),
            validation_episode_ids=("validation",),
            test_episode_ids=("test",),
            representation_seeds=(7301, 7302, 7303),
        )


def test_test_access_is_recorded_atomically_once_before_metrics(tmp_path):
    ledger_path = tmp_path / "test-access.json"
    ledger = TestAccessLedger(
        ledger_path, _identity(), M1_CONFIG, M1_APPROVAL
    )
    calls = []

    result = ledger.evaluate_once(
        representation_variant="normal_history",
        representation_seed=7301,
        model_selection_frozen=True,
        thresholds_frozen=True,
        expected_ledger_sha256=_identity()["empty_ledger_anchor_sha256"],
        metrics_callback=lambda: calls.append("metrics") or {"mae": 0.1},
    )

    assert result.metrics == {"mae": 0.1}
    assert len(result.ledger_sha256) == 64
    assert calls == ["metrics"]
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert payload["run_identity"] == _identity()
    assert payload["receipts"] == [
        {
            "schema_version": "representation-test-access-receipt-v1",
            "representation_variant": "normal_history",
            "representation_seed": 7301,
            "model_selection_frozen": True,
            "thresholds_frozen": True,
            "sequence": 1,
            "previous_entry_hash": _identity()["empty_ledger_anchor_sha256"],
            "entry_hash": result.receipt["entry_hash"],
        }
    ]
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(ValueError, match="already been accessed"):
        ledger.evaluate_once(
            representation_variant="normal_history",
            representation_seed=7301,
            model_selection_frozen=True,
            thresholds_frozen=True,
            expected_ledger_sha256=result.ledger_sha256,
            metrics_callback=lambda: calls.append("second"),
        )
    assert calls == ["metrics"]


@pytest.mark.parametrize(
    "model_frozen,thresholds_frozen,match",
    [
        (False, True, "model selection"),
        (True, False, "thresholds"),
    ],
)
def test_test_access_requires_both_freezes_before_metrics(
    tmp_path, model_frozen, thresholds_frozen, match
):
    calls = []
    ledger = TestAccessLedger(
        tmp_path / "access.json", _identity(), M1_CONFIG, M1_APPROVAL
    )
    with pytest.raises(ValueError, match=match):
        ledger.evaluate_once(
            representation_variant="random_init",
            representation_seed=7302,
            model_selection_frozen=model_frozen,
            thresholds_frozen=thresholds_frozen,
            expected_ledger_sha256=_identity()["empty_ledger_anchor_sha256"],
            metrics_callback=lambda: calls.append("metrics"),
        )
    assert calls == []


def test_test_access_rejects_unknown_pair_and_tampered_ledger_before_metrics(
    tmp_path,
):
    ledger_path = tmp_path / "access.json"
    identity = _identity()
    calls = []
    ledger = TestAccessLedger(ledger_path, identity, M1_CONFIG, M1_APPROVAL)
    with pytest.raises(ValueError, match="variant"):
        ledger.evaluate_once(
            representation_variant="constant_predictor",
            representation_seed=7301,
            model_selection_frozen=True,
            thresholds_frozen=True,
            expected_ledger_sha256=identity["empty_ledger_anchor_sha256"],
            metrics_callback=lambda: calls.append("unknown"),
        )

    first = ledger.evaluate_once(
        representation_variant="single_frame",
        representation_seed=7303,
        model_selection_frozen=True,
        thresholds_frozen=True,
        expected_ledger_sha256=identity["empty_ledger_anchor_sha256"],
        metrics_callback=lambda: None,
    )
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["run_identity"]["dataset_sha256"] = "b" * 64
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pinned ledger SHA"):
        ledger.evaluate_once(
            representation_variant="random_init",
            representation_seed=7301,
            model_selection_frozen=True,
            thresholds_frozen=True,
            expected_ledger_sha256=first.ledger_sha256,
            metrics_callback=lambda: calls.append("tampered"),
        )
    assert calls == []


@pytest.mark.parametrize("tamper", ["delete", "change_pair", "truncate", "reorder"])
def test_external_pinned_sha_rejects_rewritten_valid_internal_history(
    tmp_path, tamper
):
    path = tmp_path / "access.json"
    identity = _identity()
    ledger = TestAccessLedger(path, identity, M1_CONFIG, M1_APPROVAL)
    first = ledger.request_test_access(
        representation_variant="normal_history",
        representation_seed=7301,
        model_selection_frozen=True,
        thresholds_frozen=True,
        expected_ledger_sha256=identity["empty_ledger_anchor_sha256"],
    )
    second = ledger.request_test_access(
        representation_variant="random_init",
        representation_seed=7302,
        model_selection_frozen=True,
        thresholds_frozen=True,
        expected_ledger_sha256=first.ledger_sha256,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "delete":
        payload["receipts"] = []
    elif tamper == "change_pair":
        payload["receipts"][0]["representation_variant"] = "single_frame"
    elif tamper == "truncate":
        payload["receipts"] = payload["receipts"][:1]
    else:
        payload["receipts"].reverse()
    _rehash_receipts(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    calls = []
    with pytest.raises(ValueError, match="pinned ledger SHA"):
        ledger.evaluate_once(
            representation_variant="shuffled_history",
            representation_seed=7303,
            model_selection_frozen=True,
            thresholds_frozen=True,
            expected_ledger_sha256=second.ledger_sha256,
            metrics_callback=lambda: calls.append("metrics"),
        )
    assert calls == []


def test_pinned_current_bytes_are_still_rejected_when_internal_chain_is_broken(
    tmp_path,
):
    identity = _identity()
    path = tmp_path / "access.json"
    ledger = TestAccessLedger(path, identity, M1_CONFIG, M1_APPROVAL)
    ledger.request_test_access(
        representation_variant="normal_history",
        representation_seed=7301,
        model_selection_frozen=True,
        thresholds_frozen=True,
        expected_ledger_sha256=identity["empty_ledger_anchor_sha256"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["receipts"][0]["sequence"] = 9
    encoded = json.dumps(payload).encode("utf-8")
    path.write_bytes(encoded)

    with pytest.raises(ValueError, match="sequence"):
        ledger.request_test_access(
            representation_variant="random_init",
            representation_seed=7302,
            model_selection_frozen=True,
            thresholds_frozen=True,
            expected_ledger_sha256=sha256(encoded).hexdigest(),
        )


def test_callback_failure_still_consumes_the_access(tmp_path):
    identity = _identity()
    ledger = TestAccessLedger(
        tmp_path / "access.json", identity, M1_CONFIG, M1_APPROVAL
    )
    with pytest.raises(RuntimeError, match="metric failed"):
        ledger.evaluate_once(
            representation_variant="shuffled_label",
            representation_seed=7301,
            model_selection_frozen=True,
            thresholds_frozen=True,
            expected_ledger_sha256=identity["empty_ledger_anchor_sha256"],
            metrics_callback=lambda: (_ for _ in ()).throw(
                RuntimeError("metric failed")
            ),
        )
    payload = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert len(payload["receipts"]) == 1


def test_same_old_anchor_allows_only_one_cooperating_concurrent_writer(tmp_path):
    identity = _identity()
    path = tmp_path / "access.json"

    def reserve(variant):
        ledger = TestAccessLedger(path, identity, M1_CONFIG, M1_APPROVAL)
        return ledger.request_test_access(
            representation_variant=variant,
            representation_seed=7301,
            model_selection_frozen=True,
            thresholds_frozen=True,
            expected_ledger_sha256=identity["empty_ledger_anchor_sha256"],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve, name) for name in REPRESENTATION_VARIANTS[:2]]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except ValueError:
            outcomes.append(None)
    assert sum(item is not None for item in outcomes) == 1
    assert len(json.loads(path.read_text(encoding="utf-8"))["receipts"]) == 1


def test_existing_initial_destination_is_never_overwritten(tmp_path):
    identity = _identity()
    path = tmp_path / "access.json"
    original = b"race-created-destination\n"
    path.write_bytes(original)
    ledger = TestAccessLedger(path, identity, M1_CONFIG, M1_APPROVAL)

    with pytest.raises(ValueError, match="pinned ledger SHA"):
        ledger.request_test_access(
            representation_variant="normal_history",
            representation_seed=7301,
            model_selection_frozen=True,
            thresholds_frozen=True,
            expected_ledger_sha256=identity["empty_ledger_anchor_sha256"],
        )
    assert path.read_bytes() == original
