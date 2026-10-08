"""Conservative shared budget reservations survive interruption."""

import pytest

from llm_mappo.mappo_llm_kd.label_budget_v3 import LabelBudgetV3


def test_unsettled_reservations_count_against_phase_and_cannot_repeat(tmp_path):
    path = tmp_path / "phase.jsonl"
    first = LabelBudgetV3(path, job_id="first", job_cap_usd=30)
    first.reserve("a", 30.0)
    second = LabelBudgetV3(path, job_id="second", job_cap_usd=30)
    second.reserve("b", 30.0)
    third = LabelBudgetV3(path, job_id="third", job_cap_usd=30)
    third.reserve("c", 30.0)
    fourth = LabelBudgetV3(path, job_id="fourth", job_cap_usd=30)
    with pytest.raises(ValueError, match="phase"):
        fourth.reserve("d", 11.0)
    with pytest.raises(ValueError, match="already"):
        first.reserve("a", 1.0)
    with pytest.raises(ValueError, match="unsettled"):
        first.require_settled()
    first.settle("a", 1.0)
    fourth.reserve("d", 11.0)
    assert fourth.totals()["phase_reserved_or_spent_usd"] == 72.0
    assert first.totals()["job_reserved_or_spent_usd"] == 1.0


def test_ledger_refuses_broken_events_and_job_cap_increase(tmp_path):
    path = tmp_path / "phase.jsonl"
    ledger = LabelBudgetV3(path, job_id="job", job_cap_usd=3)
    ledger.reserve("first", 2.0)
    with pytest.raises(ValueError, match="job"):
        ledger.reserve("second", 1.01)
    with pytest.raises(ValueError, match="reservation"):
        ledger.settle("first", 2.01)
    changed = LabelBudgetV3(path, job_id="job", job_cap_usd=4)
    with pytest.raises(ValueError, match="changed"):
        changed.totals()
