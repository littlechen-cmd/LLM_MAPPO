"""Bounded batch resume never repeats cached successes or clears unsafe halts."""
from threading import RLock
from types import SimpleNamespace

import pytest

from llm_mappo.mappo_llm_kd import medium_labeling as labeling


def _ledger(cost=0.0):
    return SimpleNamespace(
        _lock=RLock(), starts={}, settled={}, halted=True,
        config={"limits": {"input_tokens": 100, "output_tokens": 100,
                           "conservative_usd": 1.0, "primary_http_attempts": 10}},
        _totals=lambda: ({"primary": 1}, 10, 10, cost),
        authorized_unknown_retry=lambda identifier: True,
    )


def test_batch_resumes_after_all_workers_settle_and_caches_success(monkeypatch):
    ledger = _ledger()
    calls, cache, reviewed = [], set(), []

    def label(window, *args):
        if window in cache:
            return
        calls.append(window)
        if window == "interrupted" and calls.count(window) == 1:
            raise labeling.MediumLabelingHalt("provider_attempt_requires_reconciliation")
        cache.add(window)

    def authorize(value):
        assert "success" in cache
        reviewed.append(value)
        return True

    monkeypatch.setattr(labeling, "_label_one", label)
    labeling._run_full_batches(
        [("primary", "success"), ("primary", "interrupted")],
        None, None, ledger, 2, network_authorizer=authorize,
    )
    assert calls.count("success") == 1
    assert calls.count("interrupted") == 2
    assert len(reviewed) == 1 and not ledger.halted


def test_network_authorization_cannot_clear_budget_halt():
    ledger = _ledger(cost=2.0)
    with pytest.raises(labeling.MediumLabelingHalt, match="exceeds_budget"):
        labeling._resume_transient_network_failure(ledger, lambda value: True)
    assert ledger.halted


def test_network_authorization_cannot_clear_unsettled_or_other_failure():
    ledger = _ledger()
    ledger.starts = {"1": {}}
    with pytest.raises(labeling.MediumLabelingHalt, match="unreviewed_failure"):
        labeling._resume_transient_network_failure(ledger, lambda value: True)
    ledger.settled = {"1": {"status": "invalid_usage", "attempt_id": "1"}}
    with pytest.raises(labeling.MediumLabelingHalt, match="unreviewed_failure"):
        labeling._resume_transient_network_failure(ledger, lambda value: True)
    assert ledger.halted


def test_batch_does_not_retry_semantically_invalid_labels(monkeypatch):
    calls = []

    def label(*args):
        calls.append(1)
        raise labeling.MediumLabelingHalt("response_validation_failed_after_retry")

    monkeypatch.setattr(labeling, "_label_one", label)
    with pytest.raises(labeling.MediumLabelingHalt, match="response_validation"):
        labeling._run_full_batches(
            [("primary", "invalid")], None, None, _ledger(), 1,
            network_authorizer=lambda value: pytest.fail(
                "Semantic failure must stop before network authorization"
            ),
        )
    assert len(calls) == 1
