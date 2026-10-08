"""V3 execution refuses unapproved sends and preserves raw attempts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from llm_mappo.mappo_llm_kd import labeling_run_v3 as run
from llm_mappo.mappo_llm_kd.label_budget_v3 import LabelBudgetV3
from llm_mappo.mappo_llm_kd.provider import (
    OpenAICompatibleProvider, ProviderCallError, ProviderCompletion, ProviderConfig,
)
from llm_mappo.mappo_llm_kd.semantic_prompt_v3 import build_semantic_prompt_v3


def test_execute_is_opt_in_even_if_an_authorization_path_exists(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        run.execute_labeling_v3(tmp_path, stage="trial", execute=False,
                               authorization_path=tmp_path / "approval.json")


def test_prepare_refuses_missing_builder_lists_without_accessing_provider(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(json.dumps({}), "utf-8")
    with pytest.raises(ValueError, match="completed|lists"):
        run.prepare_labeling_v3(
            dataset, tmp_path / "plan", environment="medium-5ag", providers={},
            pricing_evidence={}, phase_ledger_path=tmp_path / "phase.jsonl",
        )


def _view():
    cells = [dict(forward=f, right=r, static_obstacle=False, shelf=False,
                  workstation=False, charger=False, other_agv=(f == 1 and r == 0))
             for f in range(-2, 3) for r in range(-2, 3)]
    frame = {
        "self": dict(orientation="UP", carrying=False, battery=.8, task_phase="task",
                     priority_weight=.5, previous_requested_action="FORWARD",
                     previous_blocked=True),
        "goal": dict(relative_forward=.1, relative_right=0., normalized_distance=.1,
                     goal_type="task"),
        "peers": [dict(valid=True, relative_forward=.02, relative_right=0.,
                       orientation="DOWN", carrying=False, battery=.8, task_phase="task",
                       priority_weight=.75, previous_requested_action="FORWARD",
                       previous_blocked=False), {"valid": False}, {"valid": False}],
        "local": {"cells": cells},
    }
    return {"schema_version": "temporal-semantic-view-v1",
            "observation_schema": "direct-goal-observation-v2",
            "history": [frame for _ in range(8)], "summary": {}}


def _config():
    return ProviderConfig(
        provider_id="fixture-only", requested_model="deepseek-flash",
        base_url="https://api.deepseek.com", endpoint_path="/chat/completions",
        api_key_environment_variable="FIXTURE_API_KEY", temperature=0., seed=7,
        timeout_seconds=1., retry_limit=1, max_completion_tokens=256,
        max_prompt_characters=100000, response_format="json_object",
        extra_body={"thinking": {"type": "disabled"}},
    )


def _response():
    return json.dumps({
        "scores": {"goal-priority": .63, "yielding-preference": .37,
                   "local-constraint": 0.},
        "evidence": {"goal-priority": ["history[7].self", "history[7].goal"],
                     "yielding-preference": ["history[7].self", "history[7].peers",
                                            "history[7].local.cells"],
                     "local-constraint": ["history[7].self.carrying",
                                          "history[7].local.cells"]},
        "unknown_reasons": {name: None for name in run.SEMANTIC_DIMENSIONS_V3[:3]},
    })


class _FakeProvider:
    def __init__(self, outcomes):
        self.config = _config()
        self.outcomes = iter(outcomes)
        self.calls = 0

    def _request_body(self, prompt):
        return OpenAICompatibleProvider(self.config)._request_body(prompt)

    def complete(self, prompt):
        self.calls += 1
        result = next(self.outcomes)
        if isinstance(result, Exception):
            raise result
        content, model, fingerprint = result
        return ProviderCompletion(content=content, receipt={
            **self.config.public_identity(), "returned_model": model,
            "system_fingerprint": fingerprint,
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            "system_prompt_sha256": prompt.system_sha256,
            "user_prompt_sha256": prompt.user_sha256,
            "semantic_view_sha256": prompt.semantic_view_sha256,
            "prompt_version": prompt.prompt_version,
            "semantic_view_schema": "temporal-semantic-view-v1",
        })


def _attempt_fixture(tmp_path, outcomes):
    provider = _FakeProvider(outcomes)
    view = _view()
    prompt = build_semantic_prompt_v3(view)
    row = {"view_id": "train:episode-1:agent-1:start-0:end-7", "semantic_view": view,
           "semantic_view_sha256": prompt.semantic_view_sha256,
           "prompt_hashes": {"system": prompt.system_sha256, "user": prompt.user_sha256},
           "request_hashes": {"main": run._request_sha(provider, prompt)},
           "reservations": {"main": .1}}
    plan = {"pricing_evidence": {"rates": {
        "main": {"input_usd_per_million": .3, "output_usd_per_million": 1.2}
    }}}
    budget = LabelBudgetV3(tmp_path / "phase.jsonl", job_id="fixture", job_cap_usd=30)
    return provider, row, plan, budget


def test_raw_response_precedes_parse_and_one_retry_preserves_continuous_success(tmp_path):
    provider, row, plan, budget = _attempt_fixture(tmp_path, [
        ("invalid JSON", "Flash-real", "stable"), (_response(), "Flash-real", "stable"),
    ])
    record = run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    assert record["scores"]["goal-priority"] == .63
    assert record["scores"]["yielding-preference"] == .37
    assert record["receipt"]["request_sha256"] == row["request_hashes"]["main"]
    audit = run._audit(tmp_path, "main")
    assert [event["parse_ok"] for event in audit] == [False, True]
    assert [event["semantic_support_ok"] for event in audit] == [False, True]
    assert all(event["forbidden_content"] is False for event in audit)
    raw = run._store(str(tmp_path / "raw_responses.jsonl")).records()
    assert raw[0]["raw_content"] == "invalid JSON"
    assert len(raw) == 2
    budget.require_settled()
    run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    assert provider.calls == 2


def test_unknown_request_outcome_keeps_reservation_and_refuses_automatic_retry(tmp_path):
    provider, row, plan, budget = _attempt_fixture(tmp_path, [
        ProviderCallError("fixture-timeout", outcome_unknown=True),
    ])
    with pytest.raises(ValueError, match="Unknown"):
        run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    assert budget.totals()["job_reserved_or_spent_usd"] == .1
    with pytest.raises(ValueError, match="unsettled"):
        run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    assert provider.calls == 1


def test_model_identity_change_stops_and_retains_raw_response(tmp_path):
    provider, row, plan, budget = _attempt_fixture(tmp_path, [
        ("invalid JSON", "Flash-real", "first"), (_response(), "Flash-real", "changed"),
    ])
    with pytest.raises(ValueError, match="fingerprint changed"):
        run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    assert len(run._store(str(tmp_path / "raw_responses.jsonl")).records()) == 2
    with pytest.raises(ValueError, match="unsettled"):
        budget.require_settled()


def test_legal_action_evidence_is_not_forbidden_but_action_output_is():
    response = json.loads(_response())
    response["evidence"]["goal-priority"] = ["history[7].self.previous_requested_action"]
    assert run._forbidden(json.dumps(response)) is False
    response["action"] = "FORWARD"
    assert run._forbidden(json.dumps(response)) is True


def test_authorization_boolean_or_wrong_hash_is_rejected(tmp_path):
    plan = {"environment": "medium-5ag", "dataset_manifest_sha256": "a" * 64,
            "provider_plan_sha256": "b" * 64, "prompts_sha256": "c" * 64,
            "job_cap_usd": 30.0}
    (tmp_path / "plan.json").write_text(json.dumps(plan), "utf-8")
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps({"status": "approved", "approved": True}), "utf-8")
    with pytest.raises(ValueError, match="exact plan"):
        run._authorize(tmp_path, plan, path)


def test_stage_retry_allowance_stops_before_a_fifth_repeat_retry(tmp_path):
    provider, row, plan, budget = _attempt_fixture(tmp_path, [
        ("invalid JSON", "Flash-real", "stable"), (_response(), "Flash-real", "stable"),
    ])
    for index in range(4):
        run._store(str(tmp_path / "attempts.jsonl")).append({
            "event": "result", "group": "repeat", "view_id": f"other-{index}",
            "attempt": 1, "parse_ok": True, "forbidden_content": False,
            "semantic_support_ok": True,
        })
    with pytest.raises(ValueError, match="retry allowance"):
        run._label_one(tmp_path, plan, row, "repeat", provider, budget, 40)
    assert provider.calls == 1
    budget.require_settled()


def test_frozen_selection_binds_coordinates_and_same_forty_pair_windows():
    rows, index = [], {}
    for number in range(120):
        row = {"episode_id": f"episode-{number}", "agent_id": 1,
               "start_timestep": 0, "end_timestep": 7, "view_sha256": str(number)}
        key = run._window_id({**row, "partition": "train"})
        rows.append(row)
        index[key] = {"partition": "train", "semantic_view_sha256": str(number)}
    lists = {"trial": rows, "repeat": rows[:40], "pro": rows[:40], "review": rows[:60]}
    groups = run._frozen_groups(lists, index)
    assert groups["trial"][0] == "train:episode-0:agent-1:start-0:end-7"
    lists["pro"] = rows[1:41]
    with pytest.raises(ValueError, match="same frozen forty"):
        run._frozen_groups(lists, index)


def test_prepare_requests_uses_observations_and_freezes_hashes_without_secret(
    tmp_path, monkeypatch,
):
    schema = run.StructuredObservationSchema(
        observation_schema=run.DIRECT_GOAL_OBSERVATION_V2
    )
    positions = {name: i for i, name in enumerate(schema.feature_names)}
    row = np.zeros(209, dtype=np.float32)
    for name in ("self_orientation_up", "self_phase_task", "self_previous_action_0",
                 "goal_type_task"):
        row[positions[name]] = 1
    row[positions["self_priority_weight"]] = .5
    row[positions["self_battery"]] = .8
    row[positions["goal_normalized_distance"]] = .1
    windows = [{"partition": "train", "episode_id": "fixture-only", "agent_id": 1,
                "start_timestep": start, "end_timestep": start + 7,
                "observations": [row.tolist()] * 8,
                "reset_masks": [False] * 8, "stratum": "fixture"}
               for start in (0, 8)]
    (tmp_path / "windows.jsonl").write_text(
        "".join(json.dumps(window) + "\n" for window in windows), "utf-8",
    )
    monkeypatch.delenv("FIXTURE_API_KEY", raising=False)

    def forbidden_api(*_):
        raise AssertionError("prepare must not call an API")

    monkeypatch.setattr(OpenAICompatibleProvider, "complete", forbidden_api)
    providers = {"main": _config(), "pro": replace(
        _config(), requested_model="deepseek-v4-pro",
    )}
    pricing = {"rates": {
        role: {"input_usd_per_million": .3, "output_usd_per_million": 1.2}
        for role in providers
    }}
    output = tmp_path / "requests.jsonl"
    index = run._prepare_requests(tmp_path, output, providers, pricing)
    assert set(index) == {"train:fixture-only:agent-1:start-0:end-7",
                          "train:fixture-only:agent-1:start-8:end-15"}
    frozen = json.loads(output.read_text("utf-8").splitlines()[0])
    prompt = build_semantic_prompt_v3(frozen["semantic_view"])
    assert frozen["semantic_view_sha256"] == prompt.semantic_view_sha256
    assert frozen["request_hashes"]["main"] != frozen["request_hashes"]["pro"]
    assert frozen["reservations"]["main"] > 0


@pytest.mark.parametrize("field", ["returned_model", "plan_sha256", "request_sha256",
                                    "system_prompt_sha256", "semantic_view_sha256"])
def test_cached_success_rejects_tampered_provenance(tmp_path, field):
    provider, row, plan, budget = _attempt_fixture(
        tmp_path, [(_response(), "Flash-real", "stable")],
    )
    run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    path = tmp_path / "labels.jsonl"
    record = json.loads(path.read_text("utf-8"))
    record["receipt"][field] = "tampered"
    path.write_text(json.dumps(record) + "\n", "utf-8")
    with pytest.raises(ValueError, match="provenance|model"):
        run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    assert provider.calls == 1


def test_cached_success_must_reproduce_original_raw_response(tmp_path):
    provider, row, plan, budget = _attempt_fixture(
        tmp_path, [(_response(), "Flash-real", "stable")],
    )
    run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    path = tmp_path / "labels.jsonl"
    record = json.loads(path.read_text("utf-8"))
    record["scores"]["goal-priority"] = .64
    path.write_text(json.dumps(record) + "\n", "utf-8")
    with pytest.raises(ValueError, match="raw response"):
        run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    assert provider.calls == 1


def test_quality_binds_full_paired_receipts_and_raw_attempts(tmp_path, monkeypatch):
    provider, row, plan, budget = _attempt_fixture(
        tmp_path, [(_response(), "Flash-real", "stable")],
    )
    record = run._label_one(tmp_path, plan, row, "main", provider, budget, 120)
    row.update(partition="train", episode_id="episode-1")
    (tmp_path / "requests.jsonl").write_text(json.dumps(row) + "\n", "utf-8")
    plan.update(frozen_selection={"trial_ids": [row["view_id"]]},
                dataset_manifest_sha256="a" * 64)
    (tmp_path / "plan.json").write_text(json.dumps(plan), "utf-8")
    blind = tmp_path / "blind.json"
    blind.write_text("[]", "utf-8")
    run._records(tmp_path, "repeat").append(record)
    run._records(tmp_path, "pro").append(record)
    # Isolate report binding from score statistics: these statistics can remain
    # identical when model receipts are changed, which must still invalidate Go.
    monkeypatch.setattr(run, "evaluate_semantic_quality_v3", lambda **_: {
        "status": "go", "stage": "trial", "schema": "fixture-quality",
    })
    before = run._quality_payload(tmp_path, plan, "trial", blind)
    changed = dict(record)
    changed["receipt"] = {**record["receipt"], "returned_model": "tampered"}
    (tmp_path / "repeat.jsonl").write_text(json.dumps(changed) + "\n", "utf-8")
    after = run._quality_payload(tmp_path, plan, "trial", blind)
    assert before["bindings"] != after["bindings"]
    assert before["bindings"]["repeat_records_sha256"] != (
        after["bindings"]["repeat_records_sha256"]
    )
