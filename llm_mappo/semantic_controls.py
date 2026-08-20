"""Deterministic semantic-control datasets for the G3-5 mechanism checks."""

from __future__ import annotations

import random
from pathlib import Path

from llm_mappo.llm_teacher import (
    LabelledScenario,
    load_labelled_scenarios,
    write_labelled_scenarios,
)
from llm_mappo.types import SemanticPreferenceLabel


RULE_KD_MODEL = "frozen-rule-kd-v1"
SHUFFLE_KD_MODEL = "shuffled-semantic-kd-v1"


def rule_label_record(record: LabelledScenario) -> LabelledScenario:
    """Return the preregistered non-LLM semantic control for one state."""
    scenario = record.scenario
    label = scenario.priority_label or ""
    if scenario.target_kind == "charging" or scenario.battery <= 0.30:
        commitment, task_reason = 0.1, "Low battery or charging safety target."
    elif scenario.loaded:
        commitment, task_reason = 0.9, "Loaded AGV should complete its delivery task."
    elif label.startswith("A"):
        commitment, task_reason = 0.8, "Highest-priority assigned task."
    elif label.startswith("B"):
        commitment, task_reason = 0.6, "Medium-priority assigned task."
    elif label.startswith("C"):
        commitment, task_reason = 0.4, "Lower-priority assigned task."
    else:
        commitment, task_reason = 0.5, "No ranked task is available."

    if scenario.scenario_type in {
        "narrow_corridor_yield",
        "station_exit_congestion",
        "low_battery_diversion",
    }:
        assertiveness, coordination_reason = 0.2, "Yield in the constrained scenario."
    elif scenario.scenario_type == "priority_conflict" and label.startswith("A"):
        assertiveness, coordination_reason = 0.8, "Assert the highest-priority task."
    else:
        assertiveness, coordination_reason = (
            0.6, "Proceed while respecting action masks."
        )
    return LabelledScenario(
        scenario=scenario,
        label=SemanticPreferenceLabel(
            scenario_id=scenario.scenario_id,
            observation_version=scenario.observation_version,
            task_commitment=commitment,
            local_assertiveness=assertiveness,
            model=RULE_KD_MODEL,
            created_at=record.label.created_at,
        ),
        task_reason=task_reason,
        coordination_reason=coordination_reason,
    )


def derive_rule_kd(records: list[LabelledScenario]) -> list[LabelledScenario]:
    """Derive one rule label per frozen LLM-labelled scenario."""
    return [rule_label_record(record) for record in records]


def derive_shuffle_kd(
    records: list[LabelledScenario], shuffle_seed: int
) -> list[LabelledScenario]:
    """Derange semantic labels while retaining their exact marginal distribution."""
    if len(records) < 2:
        raise ValueError("ShuffleKD requires at least two labelled scenarios.")
    donor_indices = list(range(len(records)))
    random.Random(shuffle_seed).shuffle(donor_indices)
    if any(index == donor for index, donor in enumerate(donor_indices)):
        donor_indices = donor_indices[1:] + donor_indices[:1]
    if any(index == donor for index, donor in enumerate(donor_indices)):
        raise RuntimeError("Unable to construct a deterministic label derangement.")
    derived = []
    for record, donor_index in zip(records, donor_indices):
        donor = records[donor_index]
        scenario = record.scenario
        derived.append(
            LabelledScenario(
                scenario=scenario,
                label=SemanticPreferenceLabel(
                    scenario_id=scenario.scenario_id,
                    observation_version=scenario.observation_version,
                    task_commitment=donor.label.task_commitment,
                    local_assertiveness=donor.label.local_assertiveness,
                    model=SHUFFLE_KD_MODEL,
                    created_at=donor.label.created_at,
                ),
                task_reason="Deterministically shuffled semantic-control label.",
                coordination_reason="Deterministically shuffled semantic-control label.",
            )
        )
    return derived


def derive_control_datasets(
    source_path: str | Path,
    rule_output_path: str | Path,
    shuffle_output_path: str | Path,
    shuffle_seed: int,
) -> dict[str, int]:
    """Create both controls atomically without querying an online teacher."""
    records = load_labelled_scenarios(source_path)
    rule_count = write_labelled_scenarios(rule_output_path, derive_rule_kd(records))
    shuffle_count = write_labelled_scenarios(
        shuffle_output_path, derive_shuffle_kd(records, shuffle_seed)
    )
    return {"source_records": len(records), "rule_records": rule_count,
            "shuffle_records": shuffle_count}
