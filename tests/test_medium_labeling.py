"""Offline safety checks for the medium-5ag paid labeling boundary."""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.labeling import AtomicLabelStore
from llm_mappo.mappo_llm_kd.medium_prompt import (
    MEDIUM_LABEL_PROMPT_V4,
    MEDIUM_LABEL_PROMPT_V5,
    MEDIUM_LABEL_PROMPT_V6,
    build_medium_label_prompt,
    compact_medium_view,
    expand_medium_view,
    validate_medium_v6_semantic_support,
)
from llm_mappo.mappo_llm_kd.medium_labeling import (
    MediumLabelingHalt,
    authorize_medium_unknown_retry,
    _AttemptLedger,
    _label_one,
    _run_full_batches,
    _receipt,
    _repair_prompt,
    _response_audit,
    _pilot_rank,
    _quality_metrics,
    load_medium_label_config,
    reconcile_medium_presend_attempt,
    run_medium_dataset_labeling,
)
from llm_mappo.mappo_llm_kd.provider import (
    OpenAICompatibleProvider,
    ProviderConfig,
)


DIMENSIONS = (
    "mission_urgency", "concession_appropriateness",
    "interaction_risk", "deadlock_severity",
)


def _window(priority=0.5):
    schema = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    )
    names = {name: index for index, name in enumerate(schema.feature_names)}
    row = np.zeros(209, dtype=np.float32)
    for name in (
        "self_orientation_up", "self_phase_task", "self_previous_action_0",
        "goal_type_task",
    ):
        row[names[name]] = 1.0
    row[names["self_priority_weight"]] = priority
    row[names["goal_normalized_distance"]] = 0.5
    return {
        "partition": "train", "episode_id": "episode-1", "agent_id": 1,
        "start_timestep": 0, "end_timestep": 7,
        "observations": [row.tolist()] * 8,
        "reset_masks": [True] + [False] * 7,
        "stratum": "fixture",
    }


def _reply(*, model="deepseek-flash", fingerprint="fp-1", content=None):
    if content is None:
        content = json.dumps({
            "scores": {name: 0.5 for name in DIMENSIONS},
            "evidence": {name: ["summary.blocked_count"] for name in DIMENSIONS},
        })
    return {
        "id": "fixture-response", "model": model,
        "system_fingerprint": fingerprint,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40,
                  "total_tokens": 140},
    }


def _provider(monkeypatch, responses):
    values, _ = load_medium_label_config()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only-secret")
    calls = []

    def transport(endpoint, body, headers, timeout):
        del endpoint, body, headers, timeout
        calls.append(1)
        value = responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    provider = OpenAICompatibleProvider(
        ProviderConfig(**values["models"]["primary"]), transport=transport
    )
    return values, provider, calls


def test_official_medium_config_has_distinct_models_and_budget():
    values, digest = load_medium_label_config()
    assert values["models"]["primary"]["requested_model"] == "deepseek-flash"
    assert values["models"]["comparison"]["requested_model"] == (
        "deepseek-v4-pro"
    )
    assert values["limits"]["main_labels"] == 6000
    assert len(digest) == 64


def test_malformed_response_retries_once_and_cached_resume_is_free(
    monkeypatch, tmp_path
):
    values, provider, calls = _provider(
        monkeypatch, [_reply(content="not-json"), _reply()]
    )
    store = AtomicLabelStore(tmp_path / "labels.jsonl")
    ledger = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    record = _label_one(_window(), "primary", provider, store, ledger)
    assert record["receipt"]["retry_history"] == [
        "response_validation_failed"
    ]
    assert len(calls) == 2
    assert len(ledger.starts) == 2
    assert ledger.summary()["http_attempts"] == {"primary": 2}
    assert _label_one(_window(), "primary", provider, store, ledger) == record
    assert len(calls) == 2
    assert "fixture-only-secret" not in (
        tmp_path / "attempts.jsonl"
    ).read_text("utf-8")


def test_unknown_outcome_is_never_automatically_resent(monkeypatch, tmp_path):
    values, provider, calls = _provider(
        monkeypatch, [RuntimeError("private upstream detail")]
    )
    store = AtomicLabelStore(tmp_path / "labels.jsonl")
    ledger = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    with pytest.raises(MediumLabelingHalt, match="reconciliation"):
        _label_one(_window(), "primary", provider, store, ledger)
    resumed = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    with pytest.raises(MediumLabelingHalt, match="reconciliation"):
        _label_one(_window(), "primary", provider, store, resumed)
    assert len(calls) == 1
    assert "private upstream detail" not in (
        tmp_path / "attempts.jsonl"
    ).read_text("utf-8")
    assert ledger.settled["1"]["response_audit"]["provider_failure"] == {
        "exception_type": "RuntimeError", "reason_type": "RuntimeError",
    }


def test_reviewed_unknown_retry_keeps_original_charge_and_identity(
    monkeypatch, tmp_path
):
    from hashlib import sha256
    from llm_mappo.mappo_llm_kd.dataset_labeling import _atomic_json

    config = "configs/llm_kd/medium-5ag-dataset-labeling-v5.yaml"
    values, config_sha = load_medium_label_config(config)
    dataset = tmp_path / "dataset"
    output = dataset / "labels" / config_sha[:12] / "deepseek-official-flash"
    output.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    (dataset / "windows.jsonl").write_text("fixture\n", encoding="utf-8")
    prior = {
        "event": "historical_budget",
        "schema": "medium-label-historical-budget-v1",
        "prior_identity_sha256": "a" * 64,
        "prior_attempts_sha256": "b" * 64,
        "http_attempts": {"primary": 1},
        "input_tokens_upper": 100,
        "output_tokens_upper": 10,
        "conservative_cost_upper_usd": 0.001,
    }
    _atomic_json(output / "identity.json", {
        "dataset_manifest_sha256": sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "windows_sha256": sha256(
            (dataset / "windows.jsonl").read_bytes()
        ).hexdigest(),
        "label_config_sha256": config_sha,
        "primary_provider": ProviderConfig(
            **values["models"]["primary"]
        ).public_identity(),
        "comparison_provider": ProviderConfig(
            **values["models"]["comparison"]
        ).public_identity(),
        "pilot_sampling_salt": values["pilot_sampling_salt"],
        "prior_budget_source": {
            "identity_sha256": "a" * 64,
            "attempts_sha256": "b" * 64,
        },
    })
    identity_bytes = (output / "identity.json").read_bytes()
    log_path = output / "attempts.jsonl"
    ledger = _AttemptLedger(log_path, values)
    ledger.bind_prior_budget(prior)
    store = AtomicLabelStore(output / "labels.jsonl")
    prompt_builder = lambda view: build_medium_label_prompt(  # noqa: E731
        view, version=MEDIUM_LABEL_PROMPT_V5
    )

    def provider_for(response):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only-secret")

        def transport(endpoint, body, headers, timeout):
            del endpoint, body, headers, timeout
            if isinstance(response, Exception):
                raise response
            return response

        return OpenAICompatibleProvider(
            ProviderConfig(**values["models"]["primary"]),
            transport=transport,
        )

    failed = provider_for(TimeoutError("private timeout detail"))
    with pytest.raises(MediumLabelingHalt, match="reconciliation"):
        _label_one(_window(), "primary", failed, store, ledger, prompt_builder)
    original_log = log_path.read_bytes()
    assert ledger.summary()["http_attempts"]["primary"] == 2
    with pytest.raises(MediumLabelingHalt, match="original_attempt_log_digest"):
        authorize_medium_unknown_retry(
            dataset, "1", "0" * 64, config_path=config
        )
    receipt = authorize_medium_unknown_retry(
        dataset, "1", sha256(original_log).hexdigest(), config_path=config
    )
    assert receipt["original_reservation_retained"] is True
    assert log_path.read_bytes().startswith(original_log)
    assert len(log_path.read_text("utf-8").splitlines()) == 4
    with pytest.raises(MediumLabelingHalt, match="original_attempt_log_digest"):
        authorize_medium_unknown_retry(
            dataset, "1", sha256(original_log).hexdigest(), config_path=config
        )

    authorized_log = log_path.read_bytes()
    log_path.write_bytes(authorized_log.replace(b'"Codex root"', b'"Other"'))
    with pytest.raises(MediumLabelingHalt, match="invalid_unknown_retry"):
        _AttemptLedger(log_path, values)
    log_path.write_bytes(authorized_log)
    (output / "identity.json").write_bytes(identity_bytes + b" ")
    with pytest.raises(MediumLabelingHalt, match="invalid_unknown_retry"):
        _AttemptLedger(log_path, values)
    (output / "identity.json").write_bytes(identity_bytes)

    resumed = _AttemptLedger(log_path, values)
    successful = provider_for(_reply())
    record = _label_one(
        _window(), "primary", successful, store, resumed, prompt_builder
    )
    assert record["receipt"]["authorized_unknown_retry_count"] == 1
    assert resumed.summary()["http_attempts"]["primary"] == 3
    assert resumed.summary()["authorized_unknown_retries"] == 1
    assert resumed.settled["1"]["status"] == "outcome_unknown"
    assert "private timeout detail" not in log_path.read_text("utf-8")


def test_incomplete_attempt_log_requires_reconciliation(monkeypatch, tmp_path):
    values, provider, _ = _provider(monkeypatch, [])
    from llm_mappo.mappo_llm_kd.dataset_labeling import _semantic_view
    from llm_mappo.mappo_llm_kd.prompt import build_label_prompt

    ledger = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    ledger.begin("primary", "a" * 64,
                 build_label_prompt(_semantic_view(_window())), provider)
    with pytest.raises(MediumLabelingHalt, match="unresolved_sent_request"):
        _AttemptLedger(tmp_path / "attempts.jsonl", values)


def test_budget_rejects_before_sending_request(monkeypatch, tmp_path):
    values, provider, calls = _provider(monkeypatch, [_reply()])
    values["limits"]["input_tokens"] = 1
    ledger = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    with pytest.raises(MediumLabelingHalt, match="budget_exhausted"):
        _label_one(_window(), "primary", provider,
                   AtomicLabelStore(tmp_path / "labels.jsonl"), ledger)
    assert calls == []


def test_returned_model_change_stops_before_second_label(monkeypatch, tmp_path):
    values, provider, calls = _provider(
        monkeypatch, [_reply(), _reply(model="new-backend")]
    )
    store = AtomicLabelStore(tmp_path / "labels.jsonl")
    ledger = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    _label_one(_window(), "primary", provider, store, ledger)
    with pytest.raises(MediumLabelingHalt, match="returned_model_changed"):
        _label_one(_window(priority=0.7), "primary", provider, store, ledger)
    assert len(calls) == 2
    assert len(store.records()) == 1


def test_full_batch_stops_scheduling_after_unknown_result(monkeypatch, tmp_path):
    from threading import Lock

    values, _ = load_medium_label_config()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only-secret")
    calls = []
    lock = Lock()

    def transport(endpoint, body, headers, timeout):
        del endpoint, body, headers, timeout
        with lock:
            calls.append(1)
            number = len(calls)
        if number == 1:
            raise RuntimeError("upstream outcome unknown")
        return _reply()

    provider = OpenAICompatibleProvider(
        ProviderConfig(**values["models"]["primary"]), transport=transport
    )
    ledger = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    store = AtomicLabelStore(tmp_path / "labels.jsonl")
    tasks = [("primary", _window(priority=0.1 + index * 0.05))
             for index in range(8)]
    with pytest.raises(MediumLabelingHalt, match="reconciliation"):
        _run_full_batches(tasks, provider, store, ledger, 4)
    assert 1 <= len(calls) <= 4
    assert set(ledger.starts) == set(ledger.settled)


def test_smoke_requires_current_pricing_and_resumes_without_key(
    monkeypatch, tmp_path
):
    from datetime import datetime, timezone
    from llm_mappo.mappo_llm_kd import medium_labeling

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps({"distribution": "native-209d-collection-main-v1"}),
        encoding="utf-8",
    )
    (dataset / "windows.jsonl").write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(
        medium_labeling, "_frozen_source",
        lambda directory, **kwargs: (
            {"distribution": "native-209d-collection-main-v1"},
            [_window()],
            "a" * 64,
        ),
    )
    calls = []

    def factory(config):
        def transport(endpoint, body, headers, timeout):
            del endpoint, body, headers, timeout
            calls.append(1)
            return _reply(model=config.requested_model)

        return OpenAICompatibleProvider(config, transport=transport)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only-secret")
    with pytest.raises(MediumLabelingHalt, match="pricing_not_confirmed"):
        run_medium_dataset_labeling(
            dataset, stage="smoke", provider_factory=factory
        )
    assert calls == []
    result = run_medium_dataset_labeling(
        dataset, stage="smoke", provider_factory=factory,
        pricing_checked_utc_date=datetime.now(timezone.utc).date().isoformat(),
    )
    assert result["successful_records"] == 1
    assert len(calls) == 1
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    resumed = run_medium_dataset_labeling(
        dataset, stage="smoke", provider_factory=factory
    )
    assert resumed["successful_records"] == 1
    assert len(calls) == 1


def test_full_receipt_is_readable_by_current_pretraining_loader(tmp_path):
    from hashlib import sha256
    from llm_mappo.mappo_llm_kd.dataset_labeling import _atomic_json
    from llm_mappo.mappo_llm_kd.pretraining_run import (
        _load_scores,
        _validate_label_artifact,
    )

    dataset = tmp_path / "dataset"
    output = dataset / "labels" / "fixture"
    output.mkdir(parents=True)
    (dataset / "manifest.json").write_text(
        json.dumps({"schema": "llm-dataset-build-run-v1"}), encoding="utf-8"
    )
    (output / "full-metrics.json").write_text(
        json.dumps({"go": True}), encoding="utf-8"
    )
    (output / "blind-review.json").write_text(
        json.dumps({"reviewer_type": "codex"}), encoding="utf-8"
    )
    with (output / "labels.jsonl").open("w", encoding="utf-8") as stream:
        for partition, count in (("train", 4200), ("validation", 900),
                                 ("test", 900)):
            for index in range(count):
                stream.write(json.dumps({
                    "status": "ok", "view_id": f"{partition}:view-{index}",
                    "scores": {name: 0.5 for name in DIMENSIONS},
                }) + "\n")
    values, _ = load_medium_label_config()
    ledger = _AttemptLedger(output / "attempts.jsonl", values)
    (output / "attempts.jsonl").write_text("", encoding="utf-8")
    stores = {"primary": AtomicLabelStore(output / "labels.jsonl")}
    identity = {
        "dataset_manifest_sha256": sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "windows_sha256": "b" * 64,
        "label_config_sha256": "c" * 64,
    }
    receipt = _receipt(
        dataset, output,
        {"distribution": "native-209d-collection-main-v1"},
        identity, stores, ledger, "full",
    )
    _atomic_json(output / "manifest.json", receipt)
    _atomic_json(output / "state.json", {"status": "completed"})
    _validate_label_artifact(dataset, output)
    assert len(_load_scores(output / "labels.jsonl")) == 6000


def test_reviewed_presend_reconciliation_preserves_original_attempt_and_budget(
    monkeypatch, tmp_path
):
    from hashlib import sha256
    from llm_mappo.mappo_llm_kd.dataset_labeling import _atomic_json

    values, config_sha = load_medium_label_config()
    dataset = tmp_path / "dataset"
    output = dataset / "labels" / config_sha[:12] / "deepseek-official-flash"
    output.mkdir(parents=True)
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")
    (dataset / "windows.jsonl").write_text("fixture\n", encoding="utf-8")
    _atomic_json(output / "identity.json", {
        "dataset_manifest_sha256": sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "windows_sha256": sha256(
            (dataset / "windows.jsonl").read_bytes()
        ).hexdigest(),
        "label_config_sha256": config_sha,
    })
    _, failed_provider, failed_calls = _provider(
        monkeypatch, [RuntimeError("private network failure")]
    )
    log_path = output / "attempts.jsonl"
    ledger = _AttemptLedger(log_path, values)
    store = AtomicLabelStore(output / "labels.jsonl")
    with pytest.raises(MediumLabelingHalt):
        _label_one(_window(), "primary", failed_provider, store, ledger)
    original = log_path.read_bytes()
    evidence = {
        "schema": "medium-label-network-reconciliation-evidence-v1",
        "attempt_id": "1", "original_attempt_status": "outcome_unknown",
        "original_attempt_log_sha256": sha256(original).hexdigest(),
        "actual_provider_bill_verified": False,
        "sandbox_network_policy": "restricted", "reviewer": "Codex root",
        "unauthenticated_same_host_probe": {
            "reason_type": "PermissionError", "errno": 13,
            "winerror": 10013,
        },
        "approved_network_probe": {
            "connection_reached_server": True, "credentials_sent": False,
            "label_request_sent": False, "http_status": 401,
        },
    }
    evidence_path = dataset / "network-reconciliation-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    receipt = reconcile_medium_presend_attempt(dataset, evidence_path)
    assert receipt["conservative_original_reservation_retained"] is True
    assert log_path.read_bytes().startswith(original)
    _, successful_provider, successful_calls = _provider(
        monkeypatch, [_reply()]
    )
    resumed = _AttemptLedger(log_path, values)
    record = _label_one(_window(), "primary", successful_provider, store, resumed)
    assert record["status"] == "ok"
    assert len(failed_calls) == len(successful_calls) == 1
    assert resumed.summary()["http_attempts"]["primary"] == 2
    assert resumed.summary()["confirmed_not_sent_reconciliations"] == 1


def test_winerror_10013_is_presend_but_timeout_remains_unknown(monkeypatch):
    from urllib.error import URLError
    from llm_mappo.mappo_llm_kd.provider import ProviderCallError
    from llm_mappo.mappo_llm_kd.prompt import build_label_prompt
    from llm_mappo.mappo_llm_kd.dataset_labeling import _semantic_view

    denied = PermissionError(13, "sandbox socket denied")
    denied.winerror = 10013
    _, denied_provider, _ = _provider(monkeypatch, [URLError(denied)])
    with pytest.raises(ProviderCallError) as blocked:
        denied_provider.complete(build_label_prompt(_semantic_view(_window())))
    assert blocked.value.confirmed_not_sent is True
    assert blocked.value.outcome_unknown is False
    _, timed_out_provider, _ = _provider(monkeypatch, [TimeoutError("timed out")])
    with pytest.raises(ProviderCallError) as timeout:
        timed_out_provider.complete(build_label_prompt(_semantic_view(_window())))
    assert timeout.value.confirmed_not_sent is False
    assert timeout.value.outcome_unknown is True


def test_v6_compact_view_is_lossless_and_keeps_semantic_identity():
    from llm_mappo.mappo_llm_kd.dataset_labeling import _semantic_view

    view = _semantic_view(_window())
    compact = compact_medium_view(view)
    assert compact is not view
    assert expand_medium_view(compact) == view
    assert all(len(frame["local"]["cells"]) == 25 for frame in compact["history"])
    assert all(len(cell) == 7 for frame in compact["history"]
               for cell in frame["local"]["cells"])
    assert all(isinstance(cell, dict) for frame in view["history"]
               for cell in frame["local"]["cells"])
    v5 = build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V5)
    v6 = build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V6)
    assert v6.semantic_view_sha256 == v5.semantic_view_sha256
    assert v6.available_evidence_paths == v5.available_evidence_paths
    assert "LOCAL_CELL_COLUMNS=" in v6.user_text
    assert "DIMENSION_EVIDENCE_SUPPORT:" in v6.user_text
    assert v6.user_text.endswith(
        "No array may replace the evidence object. No prose or extra key."
    )
    assert len(v6.user_text) < len(v5.user_text)


def test_v6_positive_local_scores_require_value_support_and_cited_path():
    from copy import deepcopy
    from llm_mappo.mappo_llm_kd.dataset_labeling import _semantic_view

    view = _semantic_view(_window())
    scores = {name: 0.0 for name in DIMENSIONS}
    scores["concession_appropriateness"] = 0.25
    evidence = {
        name: ["summary.blocked_count"] for name in DIMENSIONS
    }
    evidence["concession_appropriateness"] = [
        "history[7].local.non_peer_free_degree"
    ]
    with pytest.raises(ValueError, match="lacks cited positive"):
        validate_medium_v6_semantic_support(scores, evidence, view)
    direct = deepcopy(view)
    direct["history"][7]["local"]["visible_peer_cell_count"] = 1
    direct["summary"]["visible_peer_cell_statistics"]["current"] = 1
    direct["summary"]["visible_peer_cell_statistics"]["maximum"] = 1
    with pytest.raises(ValueError, match="lacks cited positive"):
        validate_medium_v6_semantic_support(scores, evidence, direct)
    evidence["concession_appropriateness"] = [
        "history[7].local.visible_peer_cell_count"
    ]
    validate_medium_v6_semantic_support(scores, evidence, direct)
    scores["interaction_risk"] = 0.5
    evidence["interaction_risk"] = [
        "summary.visible_peer_cell_statistics.maximum"
    ]
    with pytest.raises(ValueError, match="past-only"):
        validate_medium_v6_semantic_support(scores, evidence, direct)
    scores["interaction_risk"] = 0.25
    validate_medium_v6_semantic_support(scores, evidence, direct)
    scores["deadlock_severity"] = 0.5
    validate_medium_v6_semantic_support(scores, evidence, direct)


def test_v6_quality_policy_warns_on_raw_boundary_and_gates_train_coverage():
    from llm_mappo.mappo_llm_kd.dataset_labeling import _window_id

    windows = []
    primary, repeat, comparison = [], [], []
    for index in range(120):
        window = _window()
        window["episode_id"] = f"episode-{index}"
        windows.append(window)
        view_id = _window_id(window)
        value = 0.0 if index < 84 else 0.5
        row = {
            "status": "ok", "view_id": view_id,
            "cache_key": f"key-{index}",
            "scores": {name: value for name in DIMENSIONS},
        }
        primary.append(row)
        if index < 40:
            repeat.append(row)
            comparison.append(row)

    class Store:
        def __init__(self, rows):
            self.rows = rows

        def records(self):
            return self.rows

    class Ledger:
        @staticmethod
        def history(key):
            del key
            return [{"status": "ok", "response_audit": {
                "forbidden_content": False,
                "evidence_path_count": 4,
                "valid_evidence_path_count": 4,
            }}]

        @staticmethod
        def reconciled_not_sent(attempt_id):
            del attempt_id
            return False

    stores = {"primary": Store(primary), "repeat": Store(repeat),
              "comparison": Store(comparison)}
    pilot = _quality_metrics(
        windows, windows, stores, Ledger(), "pilot", MEDIUM_LABEL_PROMPT_V6
    )
    assert pilot["go"] is True
    assert pilot["quality_policy_version"] == (
        "medium-5ag-semantic-coverage-quality-v2"
    )
    assert pilot["per_dimension"]["mission_urgency"]["boundary_saturation_rate"] == .7
    assert len(pilot["boundary_saturation_diagnostic_warnings"]) == 4
    assert pilot["score_ge_0_25_counts"] == {name: 36 for name in DIMENSIONS}
    assert "train_score_ge_0_25_counts" not in pilot
    full = _quality_metrics(
        windows, windows, stores, Ledger(), "full", MEDIUM_LABEL_PROMPT_V6
    )
    assert full["go"] is False
    assert full["train_score_ge_0_25_counts"] == {
        name: 36 for name in DIMENSIONS
    }
    assert "mission_urgency_insufficient_train_positive_coverage" in (
        full["no_go_reasons"]
    )
    assert not any(reason.endswith("_saturation")
                   for reason in full["no_go_reasons"])


def test_v4_evidence_allowlist_has_only_existing_compact_paths():
    from llm_mappo.mappo_llm_kd.dataset_labeling import _semantic_view
    from llm_mappo.mappo_llm_kd.prompt import build_label_prompt

    view = _semantic_view(_window())
    v3 = build_label_prompt(view)
    v4 = build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V4)
    assert v4.semantic_view_sha256 == v3.semantic_view_sha256
    assert v4.system_sha256 == v3.system_sha256
    assert v4.score_dimensions == v3.score_dimensions
    assert v4.user_sha256 != v3.user_sha256
    paths = v4.available_evidence_paths
    assert "summary.blocked_count" in paths
    assert "history[7].self.task_phase" in paths
    assert "history[7].goal.goal_type" in paths
    assert "history[7].peers[0].valid" in paths
    assert "history[7].peers[0].relative_forward" not in paths
    assert "history[7].local.non_peer_free_degree" in paths
    assert "history[7].local.peer_occupied.forward" in paths
    assert not any(".cells[" in path for path in paths)
    assert not any("history[0]" in path for path in paths)
    assert "normalized by the map diagonal" in v4.user_text
    assert "EVIDENCE_PATH_ALLOWLIST=" in v4.user_text


def test_v4_repair_uses_persisted_invalid_paths_and_new_user_hash(
    monkeypatch, tmp_path
):
    from llm_mappo.mappo_llm_kd.dataset_labeling import _semantic_view
    from llm_mappo.mappo_llm_kd.labeling import parse_label_response
    from llm_mappo.mappo_llm_kd.provider import ProviderConfig

    values, _ = load_medium_label_config(
        "configs/llm_kd/medium-5ag-dataset-labeling-v4.yaml"
    )
    view = _semantic_view(_window())
    prompt = build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V4)
    invalid_path = "history[7].peers[0].relative_forward"
    invalid_content = json.dumps({
        "scores": {name: 0.5 for name in DIMENSIONS},
        "evidence": {name: [invalid_path] for name in DIMENSIONS},
    })
    with pytest.raises(ValueError, match="unavailable evidence path"):
        parse_label_response(invalid_content, prompt.available_evidence_paths)
    requests = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-only-secret")

    def transport(endpoint, body, headers, timeout):
        del endpoint, headers, timeout
        requests.append(json.loads(body)["messages"][1]["content"])
        return _reply(content=invalid_content) if len(requests) == 1 else _reply()

    provider = OpenAICompatibleProvider(
        ProviderConfig(**values["models"]["primary"]), transport=transport
    )
    ledger = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    record = _label_one(
        _window(), "primary", provider,
        AtomicLabelStore(tmp_path / "labels.jsonl"), ledger,
        lambda item: build_medium_label_prompt(
            item, version=MEDIUM_LABEL_PROMPT_V4
        ),
    )
    assert len(requests) == 2
    assert "VALIDATION_REPAIR_V1" not in requests[0]
    assert "VALIDATION_REPAIR_V1" in requests[1]
    assert invalid_path in requests[1]
    assert ledger.history(record["cache_key"])[0]["response_audit"][
        "unavailable_evidence_paths"
    ] == [invalid_path]
    starts = list(ledger.starts.values())
    assert starts[0]["prompt_variant"] == "base"
    assert starts[1]["prompt_variant"] == "validation-repair-v1"
    assert starts[0]["user_prompt_sha256"] != starts[1]["user_prompt_sha256"]
    assert record["receipt"]["prompt_variant"] == "validation-repair-v1"
    assert requests[1] == _repair_prompt(
        prompt, ledger.history(record["cache_key"])[0]
    ).user_text


def test_v4_historical_budget_caps_new_requests_and_rejects_changed_source(
    monkeypatch, tmp_path
):
    values, provider, calls = _provider(monkeypatch, [_reply()])
    ledger = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    event = {
        "event": "historical_budget",
        "schema": "medium-label-historical-budget-v1",
        "prior_attempts_sha256": "a" * 64,
        "http_attempts": {"primary": 6300},
        "input_tokens_upper": 1000,
        "output_tokens_upper": 1000,
        "conservative_cost_upper_usd": 1.0,
    }
    ledger.bind_prior_budget(event)
    with pytest.raises(MediumLabelingHalt, match="budget_exhausted"):
        _label_one(_window(), "primary", provider,
                   AtomicLabelStore(tmp_path / "labels.jsonl"), ledger)
    assert calls == []
    resumed = _AttemptLedger(tmp_path / "attempts.jsonl", values)
    with pytest.raises(MediumLabelingHalt, match="historical_budget_source_changed"):
        resumed.bind_prior_budget({**event, "prior_attempts_sha256": "b" * 64})


def test_v5_prompt_keeps_allowlist_and_ends_with_exact_json_structure():
    from llm_mappo.mappo_llm_kd.dataset_labeling import _semantic_view

    view = _semantic_view(_window())
    v4 = build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V4)
    v5 = build_medium_label_prompt(view, version=MEDIUM_LABEL_PROMPT_V5)
    assert v5.available_evidence_paths == v4.available_evidence_paths
    assert v5.semantic_view_sha256 == v4.semantic_view_sha256
    assert v5.system_sha256 == v4.system_sha256
    assert v5.score_dimensions == v4.score_dimensions
    assert v5.user_sha256 != v4.user_sha256
    assert v5.user_text.endswith("No prose or extra key.")
    assert '"evidence":{"mission_urgency"' in v5.user_text
    assert "concession_appropriateness is exactly 0" in v5.user_text
    assert "deadlock_severity is exactly 0" in v5.user_text
    assert "mission_urgency is 0" in v5.user_text
    assert "remote valid peers alone do not" in v5.user_text


def test_v5_structural_audit_records_type_and_keys_without_raw_output():
    raw = json.dumps({
        "scores": {name: 0.3 for name in DIMENSIONS},
        "evidence": ["summary.blocked_count"],
    })
    audit = _response_audit(
        raw, {"summary.blocked_count"},
        parse_error="evidence must contain exactly the four semantic dimensions.",
    )
    assert audit["evidence_json_type"] == "list"
    assert audit["evidence_keys"] == []
    assert audit["scores_json_type"] == "dict"
    assert audit["scores_keys"] == sorted(DIMENSIONS)
    assert "summary.blocked_count" not in json.dumps(audit)


def test_v5_pilot_salt_is_independent_of_dataset_selection_salt():
    window = _window()
    old_rank = _pilot_rank(window, "medium-5ag-llm-dataset-v1")
    new_rank = _pilot_rank(window, "medium-5ag-llm-label-pilot-v2")
    assert old_rank != new_rank
    v5, _ = load_medium_label_config(
        "configs/llm_kd/medium-5ag-dataset-labeling-v5.yaml"
    )
    assert v5["pilot_sampling_salt"] == "medium-5ag-llm-label-pilot-v2"
    assert v5["limits"]["primary_http_attempts"] == 6500
    assert v5["limits"]["repeat_http_attempts"] == 100
    assert v5["limits"]["comparison_http_attempts"] == 48
