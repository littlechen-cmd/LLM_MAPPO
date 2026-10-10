"""Stage boundaries and independent evidence gates for continuous v3 labels."""

import copy
from hashlib import sha256
import json

import pytest

from llm_mappo.mappo_llm_kd.semantic_quality_v3 import evaluate_semantic_quality_v3
from llm_mappo.mappo_llm_kd.semantic_v3 import (
    LLM_DIMENSIONS_V3, SEMANTIC_CONTRACT_V3, merge_semantic_v3_label,
    parse_semantic_v3_response,
)


def sample_view(kind):
    cells = [dict(forward=f, right=r, static_obstacle=False, shelf=False,
                  workstation=False, charger=False, other_agv=False)
             for f in range(-2, 3) for r in range(-2, 3)]
    frame = {
        "self": dict(orientation="UP", carrying=False, battery=.8, task_phase="task",
                     priority_weight=.5, previous_requested_action="NOOP",
                     previous_blocked=False),
        "goal": dict(relative_forward=.1, relative_right=0., normalized_distance=.1,
                     goal_type="task"),
        "peers": [{"valid": False} for _ in range(3)],
        "local": {"cells": cells},
    }
    if kind in (1, 2):
        frame["self"].update(previous_requested_action="FORWARD", previous_blocked=True)
        cells[17]["other_agv"] = True  # (forward=1,right=0)
        frame["peers"][0] = dict(valid=True, relative_forward=.02, relative_right=0.,
                                 orientation="DOWN", carrying=False, battery=.8,
                                 task_phase="task", priority_weight=.75,
                                 previous_requested_action="FORWARD",
                                 previous_blocked=False)
    if kind == 2:
        frame["self"]["priority_weight"] = 1.
        cells[11]["static_obstacle"] = True  # side entrance blocked; d=3
    if kind == 3:
        frame["self"].update(task_phase="charging", battery=.2)
        frame["goal"].update(goal_type="charging", relative_forward=0.,
                             relative_right=0., normalized_distance=0.)
        cells[12]["charger"] = True
    if kind in (4, 5):
        frame["self"]["priority_weight"] = .75
    if kind == 4:
        frame["self"].update(previous_requested_action="FORWARD", previous_blocked=True)
    if kind == 5:
        for cell in cells:
            cell["static_obstacle"] = cell["right"] != 0
    return {"history": [copy.deepcopy(frame) for _ in range(8)]}


def fixture():
    windows, records = {}, []
    for index in range(120):
        key, kind = f"v{index:03d}", index % 6
        view = sample_view(kind)
        scores = dict(zip(LLM_DIMENSIONS_V3, (
            (.23, .57, .91, .61, .74, .82)[kind],
            (0., .63, 0., None, 0., 0.)[kind],
            (0., 0., .37, 0., 0., .82)[kind],
        )))
        payload = dict(scores=scores, evidence={
            "goal-priority": ["history[7].self", "history[7].goal"],
            "yielding-preference": ["history[7].self", "history[7].goal",
                                   "history[7].peers", "history[7].local.cells"],
            "local-constraint": ["history[7].self.carrying", "history[7].local.cells"],
        }, unknown_reasons={name: ("hard-service-lock" if value is None else None)
                            for name, value in scores.items()})
        record = merge_semantic_v3_label(
            parse_semantic_v3_response(json.dumps(payload), view), view)
        record.update(view_id=key, status="ok", receipt={
            "semantic_contract_version": SEMANTIC_CONTRACT_V3,
            "semantic_view_sha256": sha256(json.dumps(
                view, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        })
        windows[key] = dict(partition="train", episode_id=f"e{index}", semantic_view=view)
        records.append(record)
    trial_ids = list(windows)
    paired = trial_ids[:40]
    frozen = dict(environment="medium-5ag", total_samples=3000,
                  frozen_before_labeling=True, trial_ids=trial_ids,
                  paired_ids=paired, review_ids=trial_ids[:60])
    attempts = lambda ids: [dict(view_id=key, attempt=0, parse_ok=True,
                                forbidden_content=False) for key in ids]
    return dict(windows=windows, records=records, frozen_selection=frozen, stage="trial",
                repeat_records=copy.deepcopy(records[:40]),
                pro_records=copy.deepcopy(records[:40]),
                blind_reviews=[dict(view_id=key, critical=0, substantive=0,
                                    reviewer_identity="Codex", blinded=True)
                               for key in trial_ids[:60]],
                request_audit={"main": attempts(trial_ids), "repeat": attempts(paired),
                               "pro": attempts(paired)})


def check(report, name):
    return next(row for row in report["checks"] if row["id"] == name)


def test_complete_trial_can_go_without_claiming_main_or_quantizing_scores():
    inputs = fixture()
    report = evaluate_semantic_quality_v3(**inputs)
    assert report["status"] == "go"
    assert report["main_pretraining_eligible"] is False
    assert report["diagnostics"]["partitions"]["train"]["goal-priority"]["minimum"] == .23
    assert not any(row["scope"] == "main-train-coverage" for row in report["checks"])


@pytest.mark.parametrize("missing", ["repeat_records", "pro_records", "blind_reviews",
                                   "request_audit"])
def test_missing_external_evidence_is_pending_never_go(missing):
    inputs = fixture()
    inputs[missing] = None
    assert evaluate_semantic_quality_v3(**inputs)["status"] == "pending"


def test_trial_labels_cannot_pass_full_main_quotas():
    inputs = fixture()
    inputs["stage"] = "main"
    report = evaluate_semantic_quality_v3(**inputs)
    assert report["status"] != "go"
    assert check(report, "main-partition-counts")["status"] == "pending"
    assert report["main_pretraining_eligible"] is False


def test_auxiliary_score_or_evidence_tampering_is_no_go():
    inputs = fixture()
    inputs["records"][4]["scores"]["motion-stagnation"] = .75
    report = evaluate_semantic_quality_v3(**inputs)
    assert report["status"] == "no-go"
    assert check(report, "main-record-integrity")["status"] == "no-go"


def test_parse_retry_and_forbidden_content_use_original_attempts():
    inputs = fixture()
    inputs["request_audit"]["main"][0]["parse_ok"] = False
    inputs["request_audit"]["main"].append(dict(
        view_id="v000", attempt=1, parse_ok=True, forbidden_content=True))
    report = evaluate_semantic_quality_v3(**inputs)
    assert check(report, "main-first-parse-rate")["observed"] == pytest.approx(119 / 120)
    assert check(report, "main-retry-rate")["observed"] == pytest.approx(1 / 120)
    assert check(report, "main-forbidden-content")["status"] == "no-go"


def test_known_agreement_is_separate_from_numeric_mae():
    inputs = fixture()
    # 7 of 40 become unknown; finite same-known values alone must not pass.
    for index in (0, 1, 2, 4, 5, 6, 7):
        record = inputs["repeat_records"][index]
        record["scores"]["yielding-preference"] = None
        record["known_masks"]["yielding-preference"] = False
        record["unknown_reasons"]["yielding-preference"] = "symmetric-priority"
    report = evaluate_semantic_quality_v3(**inputs)
    assert check(report, "repeat-known-agreement-yielding-preference")["status"] == "no-go"


def test_constant_or_few_common_pairs_do_not_become_zero_correlation():
    inputs = fixture()
    for record in inputs["records"]:
        record["scores"]["goal-priority"] = .63
    report = evaluate_semantic_quality_v3(**inputs)
    pair = report["diagnostics"]["redundancy"]["overall"][
        "goal-priority|yielding-preference"]
    assert pair["pearson"] is None
    assert pair["estimable"] is False
    assert check(report, "trial-std-goal-priority")["status"] == "no-go"


def test_review_failure_counts_and_identity_are_required():
    inputs = fixture()
    inputs["blind_reviews"][0]["critical"] = 1
    assert check(evaluate_semantic_quality_v3(**inputs), "blind-review-findings")[
        "status"] == "no-go"
    inputs["blind_reviews"][0]["critical"] = 0
    del inputs["blind_reviews"][0]["reviewer_identity"]
    assert check(evaluate_semantic_quality_v3(**inputs), "blind-review-completeness")[
        "status"] == "pending"


def test_missing_view_hash_is_pending_not_assumed_bound():
    inputs = fixture()
    del inputs["records"][0]["receipt"]["semantic_view_sha256"]
    assert check(evaluate_semantic_quality_v3(**inputs), "main-view-hash-binding")[
        "status"] == "pending"


def test_auxiliary_boolean_cannot_pass_numeric_rule_equality():
    inputs = fixture()
    inputs["records"][4]["scores"]["motion-stagnation"] = True
    assert check(evaluate_semantic_quality_v3(**inputs), "main-record-integrity")[
        "status"] == "no-go"


def full_fixture():
    inputs = fixture()
    base = copy.deepcopy(inputs["records"])
    for index in range(120, 1000):
        key = f"v{index:03d}"
        template = index % 120
        inputs["windows"][key] = copy.deepcopy(inputs["windows"][f"v{template:03d}"])
        inputs["windows"][key].update(episode_id=f"e{index}", partition=(
            "train" if index < 700 else "validation" if index < 850 else "test"))
        record = copy.deepcopy(base[template])
        record["view_id"] = key
        inputs["records"].append(record)
        inputs["request_audit"]["main"].append(dict(view_id=key, attempt=0,
                                                   parse_ok=True,
                                                   forbidden_content=False))
    inputs["frozen_selection"].update(environment="tiny-3ag", total_samples=1000)
    inputs["stage"] = "main"
    return inputs


def test_main_coverage_uses_train_and_heldout_only_frozen_acceptance():
    inputs = full_fixture()
    # Constant held-out labels cannot change semantic design/std/redundancy gates.
    for record in inputs["records"][700:]:
        record["scores"]["goal-priority"] = .63
        record["scores"]["local-constraint"] = 0.
    report = evaluate_semantic_quality_v3(**inputs)
    assert report["status"] == "go"
    assert report["main_pretraining_eligible"] is True
    assert check(report, "train-high-local-constraint")["observed"] == 116
    assert check(report, "validation-known-goal-priority")["required"] == 30
    assert check(report, "main-std-goal-priority")["observed"] > .08
    assert "population_std" not in report["diagnostics"]["partitions"][
        "test"]["goal-priority"]


def test_joint_zero_pairs_are_excluded_and_small_samples_are_unestimable():
    inputs = fixture()
    report = evaluate_semantic_quality_v3(**inputs)
    pair = report["diagnostics"]["redundancy"]["overall"][
        "yielding-preference|local-constraint"]
    assert pair["common_nonjointzero_count"] == 60
    subgroup = report["diagnostics"]["redundancy"]["phase=charging"][
        "yielding-preference|local-constraint"]
    assert subgroup["pearson"] is None
    assert subgroup["estimable"] is False


def test_formal_quality_identity_does_not_freeze_agent_count():
    inputs = fixture()
    inputs["frozen_selection"].update(environment="formal_environment", total_samples=8000)
    assert check(evaluate_semantic_quality_v3(**inputs), "selection-identity")[
        "status"] == "go"


def test_stagnant_open_space_does_not_fill_ordinary_review_stratum():
    inputs = fixture()
    keys = [key for index, key in enumerate(inputs["windows"]) if index % 6 != 0][:60]
    inputs["frozen_selection"]["review_ids"] = keys
    inputs["blind_reviews"] = [dict(view_id=key, critical=0, substantive=0,
                                    reviewer_identity="Codex", blinded=True) for key in keys]
    gate = check(evaluate_semantic_quality_v3(**inputs), "blind-review-strata")
    assert gate["observed"]["ordinary"] == 0
    assert gate["status"] == "no-go"


def test_repeat_and_pro_must_use_the_same_frozen_forty_ids():
    inputs = fixture()
    inputs["frozen_selection"]["pro_ids"] = list(inputs["windows"])[40:80]
    assert check(evaluate_semantic_quality_v3(**inputs), "frozen-paired-same-ids")[
        "status"] == "no-go"


@pytest.mark.parametrize("support_false", [False, True])
def test_accepted_label_requires_a_successful_attempt_for_its_own_id(support_false):
    inputs = fixture()
    if support_false:
        inputs["request_audit"]["main"][0]["semantic_support_ok"] = False
    else:
        inputs["request_audit"]["main"][0]["parse_ok"] = False
    gate = check(evaluate_semantic_quality_v3(**inputs), "main-accepted-attempt-binding")
    assert gate["status"] == "no-go"
    assert gate["observed"] == ["v000"]
