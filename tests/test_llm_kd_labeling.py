"""Strict, secret-safe Phase 3 LLM labeling contracts."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from urllib.error import HTTPError

import numpy as np
import pytest
import yaml

from llm_mappo.mappo_llm_kd.audit import (
    build_model_comparison_receipt,
    build_shared_audit_samples,
    run_label_audit,
    write_shared_audit_subset,
)
from llm_mappo.mappo_llm_kd.config import (
    load_l1_config,
    verify_l1_approval,
)
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.labeling import (
    AtomicLabelStore,
    LabelingFailure,
    label_view,
    parse_label_response,
)
from llm_mappo.mappo_llm_kd.prompt import (
    LLM_LABEL_PROMPT_V3,
    build_label_prompt,
)
from llm_mappo.mappo_llm_kd.provider import (
    OpenAICompatibleProvider,
    ProviderConfig,
)
from llm_mappo.mappo_llm_kd.temporal_view import (
    serialize_temporal_semantic_view,
)


DIMENSIONS = (
    "mission_urgency",
    "concession_appropriateness",
    "interaction_risk",
    "deadlock_severity",
)
L1_CONFIG = "configs/llm_kd/label-audit.yaml"
M1_CONFIG = "configs/llm_kd/representation-diagnostics.yaml"
M1_APPROVAL = "configs/llm_kd/representation-diagnostics-approval.json"
L1_APPROVAL = "configs/llm_kd/label-audit-approval.json"


def _view():
    schema = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    )
    index = {name: position for position, name in enumerate(schema.feature_names)}
    row = np.zeros(schema.dimension, dtype=np.float32)
    row[index["self_orientation_up"]] = 1.0
    row[index["self_phase_task"]] = 1.0
    row[index["self_previous_action_0"]] = 1.0
    row[index["goal_type_task"]] = 1.0
    row[index["self_battery"]] = 0.8
    row[index["self_priority_weight"]] = 1.0
    row[index["goal_normalized_distance"]] = 0.5
    return serialize_temporal_semantic_view(
        np.stack([row]), np.asarray([True]), schema.feature_names, 8
    )


def _response(score=0.5):
    return json.dumps(
        {
            "scores": {name: score for name in DIMENSIONS},
            "evidence": {
                name: ["summary.blocked_count"] for name in DIMENSIONS
            },
        }
    )


def _provider(monkeypatch, responses, *, retry_limit=1, secret="unit-secret"):
    monkeypatch.setenv("PHASE3_TEST_API_KEY", secret)
    calls = []

    def transport(endpoint, body, headers, timeout_seconds):
        calls.append((endpoint, body, headers, timeout_seconds))
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    config = ProviderConfig(
        provider_id="fixture-provider",
        requested_model="fixture-model",
        base_url="https://provider.invalid/v1",
        endpoint_path="/chat/completions",
        api_key_environment_variable="PHASE3_TEST_API_KEY",
        temperature=0.0,
        seed=None,
        timeout_seconds=5.0,
        retry_limit=retry_limit,
        max_completion_tokens=500,
        max_prompt_characters=60000,
        response_format="json_object",
        extra_body={},
    )
    return OpenAICompatibleProvider(config, transport=transport), calls, secret


def _completion(content, *, model="served-model"):
    return {
        "id": "chatcmpl-fixture",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


def test_prompt_contains_only_validated_temporal_view_and_four_score_contract():
    prompt = build_label_prompt(_view())
    assert prompt.prompt_version == LLM_LABEL_PROMPT_V3
    assert tuple(prompt.score_dimensions) == DIMENSIONS
    assert "summary.blocked_count" in prompt.available_evidence_paths
    assert "mission_urgency" in prompt.user_text
    assert "recommended_action" not in prompt.user_text
    assert "priority_winner" not in prompt.user_text
    assert "history[0].self.task_phase" in prompt.user_text
    assert "frame-relative shorthand is invalid" in prompt.user_text
    assert "Free space alone does not justify concession" in prompt.user_text
    assert "support the direction and magnitude" in prompt.user_text
    assert "interaction_risk requires an observable interaction object" in (
        prompt.user_text
    )
    assert "no visible peer means concession_appropriateness must be 0" not in (
        prompt.user_text
    )
    supplied = json.loads(prompt.user_text.split("TEMPORAL_SEMANTIC_VIEW=", 1)[1])
    assert supplied == _view()


def test_prompt_rejects_nested_extra_privileged_field():
    view = _view()
    view["history"][0]["self"]["reward"] = 99
    with pytest.raises(ValueError, match="exact TemporalSemanticView-v1"):
        build_label_prompt(view)


def test_response_parser_requires_exact_scores_and_valid_unique_evidence_paths():
    prompt = build_label_prompt(_view())
    parsed = parse_label_response(_response(0.25), prompt.available_evidence_paths)
    assert parsed.scores == {name: 0.25 for name in DIMENSIONS}
    assert parsed.evidence["mission_urgency"] == ("summary.blocked_count",)

    payload = json.loads(_response())
    payload["recommended_action"] = "FORWARD"
    with pytest.raises(ValueError, match="exactly scores and evidence"):
        parse_label_response(json.dumps(payload), prompt.available_evidence_paths)
    payload = json.loads(_response())
    payload["evidence"]["mission_urgency"] = ["reward"]
    with pytest.raises(ValueError, match="evidence path"):
        parse_label_response(json.dumps(payload), prompt.available_evidence_paths)
    payload = json.loads(_response())
    payload["scores"]["interaction_risk"] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        parse_label_response(json.dumps(payload), prompt.available_evidence_paths)


def test_provider_reads_secret_only_at_call_and_records_served_model(monkeypatch):
    provider, calls, secret = _provider(monkeypatch, [_completion(_response())])
    prompt = build_label_prompt(_view())
    completion = provider.complete(prompt)

    assert len(calls) == 1
    endpoint, body, headers, timeout = calls[0]
    assert endpoint == "https://provider.invalid/v1/chat/completions"
    assert json.loads(body)["model"] == "fixture-model"
    assert headers["Authorization"] == f"Bearer {secret}"
    assert timeout == 5.0
    assert completion.content == _response()
    assert completion.receipt["requested_model"] == "fixture-model"
    assert completion.receipt["returned_model"] == "served-model"
    assert secret not in json.dumps(completion.receipt)


def test_provider_missing_secret_names_variable_without_leaking_value(monkeypatch):
    provider, _, _ = _provider(monkeypatch, [_completion(_response())])
    monkeypatch.delenv("PHASE3_TEST_API_KEY")
    with pytest.raises(RuntimeError, match="PHASE3_TEST_API_KEY") as captured:
        provider.complete(build_label_prompt(_view()))
    assert "Bearer" not in str(captured.value)


@pytest.mark.parametrize(
    "changes",
    [
        {"api_key_environment_variable": "sk-not-an-env-name"},
        {"base_url": "https://embedded:credential@provider.invalid/v1"},
        {"extra_body": {"authorization": "Bearer embedded-secret"}},
        {"extra_body": {"apiKey": "embedded-credential-value"}},
        {"extra_body": {"x-api-key": "embedded-credential-value"}},
        {"extra_body": {"access_token": "embedded-credential-value"}},
        {"extra_body": {"nested": {"apiKey": "embedded-credential-value"}}},
    ],
)
def test_provider_config_rejects_any_embedded_credential_channel(changes):
    values = {
        "provider_id": "fixture-provider",
        "requested_model": "fixture-model",
        "base_url": "https://provider.invalid/v1",
        "endpoint_path": "/chat/completions",
        "api_key_environment_variable": "PHASE3_TEST_API_KEY",
        "temperature": 0.0,
        "seed": None,
        "timeout_seconds": 5.0,
        "retry_limit": 1,
        "max_completion_tokens": 500,
        "max_prompt_characters": 60000,
        "response_format": "json_object",
        "extra_body": {},
    }
    values.update(changes)
    with pytest.raises(ValueError, match="credential|environment variable"):
        ProviderConfig(**values)


def test_labeling_retries_malformed_response_and_atomically_resumes(monkeypatch, tmp_path):
    responses = [_completion("not-json"), _completion(_response(), model="actual")]
    provider, calls, _ = _provider(monkeypatch, responses, retry_limit=1)
    store = AtomicLabelStore(tmp_path / "labels.jsonl")

    first = label_view("view-1", _view(), provider, store)
    assert len(calls) == 2
    assert first["receipt"]["retry_history"] == ["response_validation_failed"]
    assert first["receipt"]["returned_model"] == "actual"
    assert not (tmp_path / "labels.jsonl.tmp").exists()

    resumed_provider, resumed_calls, _ = _provider(monkeypatch, [], retry_limit=1)
    resumed = label_view(
        "view-1", _view(), resumed_provider, AtomicLabelStore(store.path)
    )
    assert resumed == first
    assert resumed_calls == []

    repeat_provider, repeat_calls, _ = _provider(
        monkeypatch, [_completion(_response())], retry_limit=1
    )
    repeated = label_view(
        "view-1",
        _view(),
        repeat_provider,
        AtomicLabelStore(store.path),
        cache_namespace="repeat-1",
    )
    assert len(repeat_calls) == 1
    assert repeated["cache_namespace"] == "repeat-1"


@pytest.mark.parametrize(
    "mutation",
    ["legacy_schema", "missing_scores", "out_of_range_score", "credential_receipt"],
)
def test_labeling_rejects_a_forged_success_cache_before_any_provider_call(
    monkeypatch, tmp_path, mutation
):
    provider, calls, _ = _provider(monkeypatch, [_completion(_response())])
    store = AtomicLabelStore(tmp_path / "labels.jsonl")
    first = label_view("view-1", _view(), provider, store)
    tampered = dict(first)
    if mutation == "legacy_schema":
        tampered["schema_version"] = "llm-semantic-label-record-v0"
    elif mutation == "missing_scores":
        del tampered["scores"]
    elif mutation == "out_of_range_score":
        tampered["scores"] = {**tampered["scores"], "interaction_risk": 1.1}
    else:
        tampered["receipt"] = {**tampered["receipt"], "apiKey": "sk-forged"}
    store.path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")

    resumed_provider, resumed_calls, _ = _provider(
        monkeypatch, [_completion(_response())]
    )
    with pytest.raises(ValueError, match="cached label record"):
        label_view("view-1", _view(), resumed_provider, store)

    assert len(calls) == 1
    assert resumed_calls == []


def test_terminal_provider_failure_is_redacted_and_persisted(monkeypatch, tmp_path):
    provider, _, secret = _provider(
        monkeypatch,
        [RuntimeError("upstream exposed " + "unit-secret")],
        retry_limit=0,
    )
    store = AtomicLabelStore(tmp_path / "labels.jsonl")
    with pytest.raises(LabelingFailure, match="provider_request_failed") as captured:
        label_view("view-failure", _view(), provider, store)
    assert secret not in str(captured.value)
    serialized = store.path.read_text(encoding="utf-8")
    assert secret not in serialized
    assert json.loads(serialized)["status"] == "failed"


def test_http_status_is_preserved_as_a_safe_failure_code(monkeypatch, tmp_path):
    error = HTTPError(
        "https://provider.invalid/v1/chat/completions",
        403,
        "blocked with private upstream detail",
        {},
        BytesIO(b"private response body"),
    )
    provider, _, _ = _provider(monkeypatch, [error], retry_limit=0)
    store = AtomicLabelStore(tmp_path / "labels.jsonl")
    with pytest.raises(LabelingFailure, match="provider_http_403"):
        label_view("view-http-failure", _view(), provider, store)
    record = json.loads(store.path.read_text(encoding="utf-8"))
    assert record["error_code"] == "provider_http_403"
    assert "private" not in json.dumps(record)


def test_l1_config_is_hash_bound_explicit_and_contains_no_secret_values():
    resolved = load_l1_config(L1_CONFIG, M1_CONFIG, M1_APPROVAL)
    assert len(resolved.sha256) == 64
    assert resolved.values["logical_completion_max"] == 24
    assert resolved.values["http_attempt_max"] == 48
    models = resolved.values["models"]
    assert [model["requested_model"] for model in models] == [
        "deepseek-v4-flash",
        "gpt-5.6-terra",
    ]
    serialized = json.dumps(resolved.values)
    assert "sk-" not in serialized
    assert "DEEPSEEK_API_KEY" in serialized
    assert "GPT_TERRA_API_KEY" in serialized


def test_l1_approval_matches_the_exact_resolved_config_hash(tmp_path):
    resolved = load_l1_config(L1_CONFIG, M1_CONFIG, M1_APPROVAL)
    checked = verify_l1_approval(
        L1_CONFIG, M1_CONFIG, M1_APPROVAL, L1_APPROVAL
    )
    assert checked["approved_config_sha256"] == resolved.sha256

    approval = json.loads(Path(L1_APPROVAL).read_text(encoding="utf-8"))
    candidate = tmp_path / "approval.json"
    candidate.write_text(json.dumps(approval), encoding="utf-8")
    verified = verify_l1_approval(
        L1_CONFIG, M1_CONFIG, M1_APPROVAL, candidate
    )
    assert verified["gate"] == "L1"

    approval["approved_config_sha256"] = "0" * 64
    candidate.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        verify_l1_approval(L1_CONFIG, M1_CONFIG, M1_APPROVAL, candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prompt_version", "unreviewed-prompt"),
        ("temporal_view_schema", "unreviewed-view"),
        ("cache_policy", "no-cache"),
        ("concurrency", 2),
    ),
)
def test_l1_config_rejects_unsupported_runtime_contract_values(
    field, value, tmp_path
):
    config = yaml.safe_load(Path(L1_CONFIG).read_text(encoding="utf-8"))
    config[field] = value
    candidate = tmp_path / "l1.yaml"
    candidate.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported runtime contract"):
        load_l1_config(candidate, M1_CONFIG, M1_APPROVAL)


def test_shared_audit_subset_is_deterministic_unique_and_prompt_valid(tmp_path):
    first = build_shared_audit_samples()
    second = build_shared_audit_samples()
    assert first == second
    assert len(first) == 8
    assert len({sample["semantic_view_sha256"] for sample in first}) == 8
    assert {sample["stratum"] for sample in first} == {
        "idle_clear",
        "urgent_task",
        "loaded_delivery",
        "low_battery_charging",
        "close_peer",
        "persistent_blocking",
        "orientation_oscillation",
        "lower_priority_concession",
    }
    for sample in first:
        build_label_prompt(sample["temporal_semantic_view"])
    destination = tmp_path / "audit.jsonl"
    write_shared_audit_subset(destination, first)
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 8
    assert not (tmp_path / "audit.jsonl.tmp").exists()


def test_model_comparison_receipt_reports_disagreement_without_selecting_teacher():
    records = []
    for provider, primary, repeat in (
        ("deepseek", 0.2, 0.3),
        ("gpt", 0.4, 0.5),
    ):
        for namespace, score in (("primary", primary), ("repeat-1", repeat)):
            records.append(
                {
                    "status": "ok",
                    "view_id": "shared-000",
                    "cache_namespace": namespace,
                    "scores": {name: score for name in DIMENSIONS},
                    "receipt": {
                        "provider_id": provider,
                        "requested_model": provider + "-requested",
                        "returned_model": provider + "-served",
                    },
                }
            )
    receipt = build_model_comparison_receipt(
        records,
        required_provider_ids=("deepseek", "gpt"),
        required_primary_ids=("shared-000",),
        required_repeat_ids=("shared-000",),
    )
    assert receipt["teacher_selected"] is False
    assert receipt["selected_teacher"] is None
    assert receipt["cross_model_mean_absolute_disagreement"] == pytest.approx(0.2)
    assert receipt["repeated_call_mean_absolute_disagreement"] == {
        "deepseek": pytest.approx(0.1),
        "gpt": pytest.approx(0.1),
    }


@pytest.mark.parametrize("mutation", ["missing_primary", "missing_repeat", "duplicate"])
def test_model_comparison_stays_incomplete_without_exact_audit_coverage(mutation):
    records = []
    for provider in ("deepseek", "gpt"):
        for namespace in ("primary", "repeat-1"):
            records.append({
                "status": "ok", "view_id": "shared-000",
                "cache_namespace": namespace,
                "scores": {name: 0.2 for name in DIMENSIONS},
                "receipt": {"provider_id": provider, "requested_model": provider,
                            "returned_model": provider},
            })
    if mutation == "missing_primary":
        records.pop(0)
    elif mutation == "missing_repeat":
        records.pop(1)
    else:
        records.append(dict(records[0]))
    receipt = build_model_comparison_receipt(
        records, required_provider_ids=("deepseek", "gpt"),
        required_primary_ids=("shared-000",),
        required_repeat_ids=("shared-000",),
    )
    assert receipt["status"] == "incomplete"
    assert receipt["coverage"]["complete"] is False


def test_single_provider_receipt_reports_repeat_stability_but_stays_incomplete():
    records = []
    for namespace, score in (("primary", 0.2), ("repeat-1", 0.3)):
        records.append(
            {
                "status": "ok",
                "view_id": "shared-000",
                "cache_namespace": namespace,
                "scores": {name: score for name in DIMENSIONS},
                "receipt": {
                    "provider_id": "deepseek",
                    "requested_model": "deepseek-v4-flash",
                    "returned_model": "deepseek-flash",
                },
            }
        )
    receipt = build_model_comparison_receipt(records)
    assert receipt["status"] == "incomplete"
    assert receipt["reason"] == "insufficient_successful_provider_coverage"
    assert receipt["repeated_call_mean_absolute_disagreement"] == {
        "deepseek": pytest.approx(0.1)
    }
    assert receipt["cross_model_mean_absolute_disagreement"] is None
    assert receipt["teacher_selected"] is False


def test_audit_runner_uses_shared_subset_and_resumes_without_new_calls(
    monkeypatch, tmp_path
):
    config_path = Path(L1_CONFIG).resolve()
    m1_path = Path(M1_CONFIG).resolve()
    approval_path = Path(M1_APPROVAL).resolve()
    resolved = load_l1_config(config_path, m1_path, approval_path)
    call_count = {"value": 0}

    def factory(config):
        monkeypatch.setenv(config.api_key_environment_variable, "runner-secret")

        def transport(endpoint, body, headers, timeout_seconds):
            del endpoint, body, headers, timeout_seconds
            call_count["value"] += 1
            score = 0.2 if config.provider_id == "deepseek-official" else 0.4
            return _completion(_response(score), model=config.requested_model + "-served")

        return OpenAICompatibleProvider(config, transport=transport)

    monkeypatch.chdir(tmp_path)
    first = run_label_audit(resolved, mode="audit", provider_factory=factory)
    assert first["successful_records"] == 24
    assert first["failed_records"] == 0
    assert first["teacher_selected"] is False
    assert call_count["value"] == 24

    second = run_label_audit(resolved, mode="audit", provider_factory=factory)
    assert second == first
    assert call_count["value"] == 24
    output = (
        tmp_path
        / "artifacts"
        / "llm_kd"
        / "label_audit"
        / "l1"
        / resolved.sha256[:12]
    )
    assert (output / "model-comparison.json").exists()
    assert "runner-secret" not in (output / "run-receipt.json").read_text(
        encoding="utf-8"
    )


def test_audit_runner_records_incomplete_comparison_when_one_provider_fails(
    monkeypatch, tmp_path
):
    resolved = load_l1_config(L1_CONFIG, M1_CONFIG, M1_APPROVAL)

    def factory(config):
        monkeypatch.setenv(config.api_key_environment_variable, "runner-secret")

        def transport(endpoint, body, headers, timeout_seconds):
            del endpoint, body, headers, timeout_seconds
            if config.provider_id == "deepseek-official":
                raise RuntimeError("upstream details must not escape")
            return _completion(_response(0.4), model=config.requested_model)

        return OpenAICompatibleProvider(config, transport=transport)

    monkeypatch.chdir(tmp_path)
    receipt = run_label_audit(resolved, mode="smoke", provider_factory=factory)
    output = (
        tmp_path
        / "artifacts"
        / "llm_kd"
        / "label_audit"
        / "l1"
        / resolved.sha256[:12]
    )
    comparison = json.loads(
        (output / "model-comparison.json").read_text(encoding="utf-8")
    )

    assert receipt["comparison_status"] == "incomplete"
    assert receipt["successful_records"] == 1
    assert receipt["failed_records"] == 1
    assert comparison["status"] == "incomplete"
    assert comparison["reason"] == "insufficient_successful_provider_coverage"
    assert comparison["teacher_selected"] is False
    assert "upstream details" not in json.dumps(comparison)


def test_audit_runner_can_execute_an_explicit_approved_provider_subset(
    monkeypatch, tmp_path
):
    resolved = load_l1_config(L1_CONFIG, M1_CONFIG, M1_APPROVAL)
    created = []

    def factory(config):
        created.append(config.provider_id)
        monkeypatch.setenv(config.api_key_environment_variable, "runner-secret")

        def transport(endpoint, body, headers, timeout_seconds):
            del endpoint, body, headers, timeout_seconds
            return _completion(_response(0.2), model=config.requested_model)

        return OpenAICompatibleProvider(config, transport=transport)

    monkeypatch.chdir(tmp_path)
    receipt = run_label_audit(
        resolved,
        mode="smoke",
        provider_factory=factory,
        provider_ids=("deepseek-official",),
    )
    assert created == ["deepseek-official"]
    assert receipt["requested_provider_ids"] == ["deepseek-official"]
    assert receipt["successful_records"] == 1
    assert receipt["comparison_status"] == "incomplete"

    with pytest.raises(ValueError, match="configured provider"):
        run_label_audit(
            resolved,
            mode="smoke",
            provider_factory=factory,
            provider_ids=("not-approved",),
        )


def test_direct_owner_run_script_imports_the_current_worktree_package():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_llm_kd_label_audit.py"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--mode {smoke,audit}" in completed.stdout
