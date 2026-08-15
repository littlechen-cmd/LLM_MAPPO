"""Audit a frozen Phase 4 dual-semantic JSONL dataset without modifying it."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path
import random
import re

from llm_mappo.llm_teacher import OBSERVATION_VERSION, load_labelled_scenarios
from llm_mappo.phase4 import SCENARIO_TYPES


FORBIDDEN_REASON_PATTERNS = (
    re.compile(r"\bassign\s+(?:agv|agent)\b", re.IGNORECASE),
    re.compile(r"\b(?:turn|move)\s+(?:left|right|forward|backward)\b", re.IGNORECASE),
    re.compile(r"\bwaypoints?\s*[:=]", re.IGNORECASE),
    re.compile(r"\ba\*\s+path\s*[:=]", re.IGNORECASE),
)

FALSE_LOADED_PEER_PATTERNS = (
    re.compile(r"\bnearby agents? include(?:s)? a loaded\b", re.IGNORECASE),
    re.compile(r"\bnearby loaded\b", re.IGNORECASE),
    re.compile(r"\bnearby agent\s+\d+\s+is loaded\b", re.IGNORECASE),
    re.compile(r"\byield to nearby loaded\b", re.IGNORECASE),
    re.compile(r"\byield to (?:a |the )?loaded B(?: agent| AGV)?\b", re.IGNORECASE),
    re.compile(
        r"\bload rule and congestion management override\b", re.IGNORECASE
    ),
)

YIELD_REASON_PATTERN = re.compile(
    r"\bfocal(?: agent| agv)?(?:\s+\w+){0,3}\s+should yield\b", re.IGNORECASE
)

HYPOTHETICAL_LOADED_PEER_PATTERN = re.compile(
    r"\byield to (?:a |the )?loaded B(?: agent| AGV)?.{0,20}\bif\b",
    re.IGNORECASE,
)

PRIORITY_REVERSAL_PATTERNS = (
    re.compile(
        r"\b(?:B|agent\s+2\s*\(B\))\b.{0,20}\b(?:is|has)\b"
        r".{0,10}\bhigher[- ]priority\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:focal(?:\s+agent)?|agent\s+1|A)\b.{0,40}"
        r"\blower priority than\b.{0,30}\b(?:agent\s+2\s*)?\(?B\)?\b",
        re.IGNORECASE,
    ),
)


def _automatic_issues(record) -> list[str]:  # noqa: C901
    scenario = record.scenario
    label = record.label
    issues = []
    if scenario.observation_version != OBSERVATION_VERSION:
        issues.append("wrong_observation_version")
    if scenario.scenario_type == "low_battery_diversion":
        if label.task_commitment > 0.3:
            issues.append("low_battery_task_commitment_too_high")
    elif scenario.scenario_type == "priority_conflict":
        if label.local_assertiveness < 0.7:
            issues.append("priority_conflict_assertiveness_too_low")
    elif scenario.scenario_type == "narrow_corridor_yield":
        if label.local_assertiveness > 0.3:
            issues.append("narrow_corridor_assertiveness_too_high")
    elif scenario.scenario_type == "station_exit_congestion":
        if label.local_assertiveness > 0.4:
            issues.append("station_exit_assertiveness_too_high")
    reasons = f"{record.task_reason} {record.coordination_reason}"
    if any(pattern.search(reasons) for pattern in FORBIDDEN_REASON_PATTERNS):
        issues.append("forbidden_control_instruction")
    if scenario.priority_label == "A" and any(
        pattern.search(reasons) for pattern in PRIORITY_REVERSAL_PATTERNS
    ):
        issues.append("priority_order_reversed_in_reason")
    has_loaded_peer = any(
        bool(peer.get("loaded")) for peer in scenario.nearby_agents
    )
    if not has_loaded_peer and any(
        pattern.search(reasons) for pattern in FALSE_LOADED_PEER_PATTERNS
    ) and not HYPOTHETICAL_LOADED_PEER_PATTERN.search(reasons):
        issues.append("rationale_peer_load_state_mismatch")
    if (
        scenario.scenario_type == "priority_conflict"
        and label.local_assertiveness >= 0.7
        and YIELD_REASON_PATTERN.search(reasons)
    ):
        issues.append("score_reason_contradiction")
    return issues


def _nearby_summary(scenario) -> str:
    return "; ".join(
        (
            f"id={peer['agent_id']},distance={peer['distance']},"
            f"loaded={peer['loaded']},battery={peer['battery']},"
            f"priority={peer['priority_label']},target={peer.get('target_kind')},"
            f"at_charging_station={peer.get('at_charging_station')}"
        )
        for peer in scenario.nearby_agents
    )


def audit_dataset(
    dataset: str | Path,
    sample_csv: str | Path,
    summary_json: str | Path,
    sample_rate: float = 0.1,
    seed: int = 20260814,
    expected_total: int | None = None,
    quota_per_type: int | None = None,
) -> dict:
    if not 0.0 < sample_rate <= 1.0:
        raise ValueError("sample_rate must be within (0, 1].")
    records = load_labelled_scenarios(dataset)
    if expected_total is not None and len(records) != expected_total:
        raise ValueError(
            f"Expected {expected_total} records but found {len(records)}."
        )
    grouped = {
        name: [record for record in records if record.scenario.scenario_type == name]
        for name in SCENARIO_TYPES
    }
    if quota_per_type is not None:
        wrong = {
            name: len(items)
            for name, items in grouped.items()
            if len(items) != quota_per_type
        }
        if wrong:
            raise ValueError(f"Scenario quota mismatch: {wrong}")
    issue_rows = []
    for record in records:
        issues = _automatic_issues(record)
        if issues:
            issue_rows.append(
                {"scenario_id": record.scenario.scenario_id, "issues": issues}
            )
    generator = random.Random(seed)
    sample = []
    for scenario_type in SCENARIO_TYPES:
        candidates = grouped[scenario_type]
        count = min(len(candidates), math.ceil(len(candidates) * sample_rate))
        sample.extend(generator.sample(candidates, count))
    destination = Path(sample_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scenario_id",
        "scenario_type",
        "battery",
        "loaded",
        "priority_label",
        "target_kind",
        "nearby_agents",
        "task_commitment",
        "task_reason",
        "local_assertiveness",
        "coordination_reason",
        "automatic_issues",
        "human_verdict",
        "human_issue",
        "human_note",
    )
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in sample:
            scenario = record.scenario
            writer.writerow(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_type": scenario.scenario_type,
                    "battery": scenario.battery,
                    "loaded": scenario.loaded,
                    "priority_label": scenario.priority_label,
                    "target_kind": scenario.target_kind,
                    "nearby_agents": _nearby_summary(scenario),
                    "task_commitment": record.label.task_commitment,
                    "task_reason": record.task_reason,
                    "local_assertiveness": record.label.local_assertiveness,
                    "coordination_reason": record.coordination_reason,
                    "automatic_issues": ";".join(_automatic_issues(record)),
                    "human_verdict": "",
                    "human_issue": "",
                    "human_note": "",
                }
            )
    summary = {
        "dataset": str(dataset),
        "records": len(records),
        "observation_version": OBSERVATION_VERSION,
        "scenario_counts": {name: len(items) for name, items in grouped.items()},
        "sample_records": len(sample),
        "sample_csv": str(destination),
        "automatic_issue_count": len(issue_rows),
        "automatic_issues": issue_rows,
        "manual_review_required": True,
    }
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--quota-per-type", type=int)
    parser.add_argument("--fail-on-automatic-issues", action="store_true")
    args = parser.parse_args()
    summary = audit_dataset(
        args.dataset,
        args.sample_csv,
        args.summary_json,
        args.sample_rate,
        args.seed,
        args.expected_total,
        args.quota_per_type,
    )
    print(json.dumps(summary, indent=2))
    if args.fail_on_automatic_issues and summary["automatic_issue_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
