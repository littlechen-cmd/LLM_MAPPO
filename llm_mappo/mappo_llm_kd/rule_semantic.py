"""Transparent four-score comparator on the exact observable v2 input."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)

from .medium_dataset_run import _file_sha256
from .medium_prompt import observable_coordination_evidence
from .prompt import _canonical_json
from .temporal_view import serialize_temporal_semantic_view


TEACHER_TYPE = "deterministic-observable-rule-v1"
SEMANTIC_CONTRACT = "observable-coordination-four-score-v2"
SOURCE_MANIFEST_SHA256 = (
    "ef273cbacfb9aa14cd17cce307b0d79180af3baf747f8a93329c12caacbeac61"
)
OUTPUT_NAME = "rule-observable-v2-c6d2df435f63"
_LEDGER = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
).feature_names


def rule_semantic_scores(view: dict) -> dict[str, float]:  # noqa: C901
    """Return deterministic proxy scores, not ground truth or action advice."""

    history = view["history"]
    current = history[-1]
    self_state = current["self"]
    goal = current["goal"]
    active = self_state["task_phase"] != "idle" and goal["goal_type"] != "idle"
    urgency = 0.0
    if active:
        urgency = 0.25
        if self_state["priority_weight"] >= 0.75:
            urgency += 0.25
        if self_state["task_phase"] == "charging" and self_state["battery"] < 0.25:
            urgency += 0.25

    def local_peer(frame: dict) -> bool:
        return frame["local"]["visible_peer_cell_count"] > 0

    peer_now = local_peer(current)
    peer_ever = any(local_peer(frame) for frame in history)
    cardinal_peer = any(current["local"]["peer_occupied"].values())
    constrained_proxy = current["local"]["non_peer_free_degree"] <= 2
    concession = 0.0
    risk = 0.0
    if peer_ever:
        concession = risk = 0.25
    if peer_now:
        concession = risk = 0.25
        if cardinal_peer or constrained_proxy:
            concession = 0.5
        if cardinal_peer and constrained_proxy:
            concession = 0.75
        multiple_peers = current["local"]["visible_peer_cell_count"] >= 2
        if cardinal_peer or multiple_peers:
            risk = 0.5
        if cardinal_peer and multiple_peers:
            risk = 0.75

    witness = observable_coordination_evidence(view)
    severity = 0.0
    if witness["positive_supported"]:
        severity = 0.25
        blocked_count = len(witness["blocked_forward_indices"])
        alternations = witness["turn_alternations"]
        if blocked_count >= 5 or alternations >= 5:
            severity = 0.5
        if blocked_count >= 7:
            severity = 0.75
    return {
        "mission_urgency": urgency,
        "concession_appropriateness": concession,
        "interaction_risk": risk,
        "deadlock_severity": severity,
    }


def export_rule_semantic_labels(source_root: str | Path,
                                output_root: str | Path) -> dict:
    """Write one immutable sibling artifact from the frozen 6000 windows."""

    source = Path(source_root).resolve()
    destination = Path(output_root).resolve() / OUTPUT_NAME
    if destination.exists():
        raise FileExistsError(f"Rule artifact already exists: {destination}")
    if _file_sha256(source / "manifest.json") != SOURCE_MANIFEST_SHA256:
        raise ValueError("Rule source manifest SHA-256 differs.")
    source_manifest = json.loads((source / "manifest.json").read_text(
        encoding="utf-8"
    ))
    if (
        source_manifest.get("semantic_contract_version") != SEMANTIC_CONTRACT
        or _file_sha256(source / "windows.jsonl")
        != source_manifest["windows_sha256"]
        or source_manifest.get("window_counts")
        != {"train": 4200, "validation": 900, "test": 900}
    ):
        raise ValueError("Rule source selection contract differs.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rule-observable-staging-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary)
        counts = Counter()
        positive = Counter()
        with (stage / "labels.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as output:
            with (source / "windows.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    window = json.loads(line)
                    view = serialize_temporal_semantic_view(
                        np.asarray(window["observations"], dtype=np.float32),
                        np.asarray(window["reset_masks"], dtype=np.bool_),
                        _LEDGER, 8,
                    )
                    scores = rule_semantic_scores(view)
                    partition = window["partition"]
                    counts[partition] += 1
                    for dimension, score in scores.items():
                        if score >= 0.25:
                            positive[(partition, dimension)] += 1
                    record = {
                        "schema_version": "semantic-label-record-v2",
                        "status": "ok",
                        "view_id": (
                            f"{partition}:{window['episode_id']}:"
                            f"agent-{window['agent_id']}:"
                            f"start-{window['start_timestep']}:"
                            f"end-{window['end_timestep']}"
                        ),
                        "teacher_type": TEACHER_TYPE,
                        "semantic_contract_version": SEMANTIC_CONTRACT,
                        "receipt": {
                            "teacher_type": TEACHER_TYPE,
                            "semantic_contract_version": SEMANTIC_CONTRACT,
                        },
                        "partition": partition,
                        "episode_id": window["episode_id"],
                        "agent_id": window["agent_id"],
                        "start_timestep": window["start_timestep"],
                        "end_timestep": window["end_timestep"],
                        "semantic_view_sha256": sha256(
                            _canonical_json(view).encode("utf-8")
                        ).hexdigest(),
                        "scores": scores,
                    }
                    output.write(json.dumps(
                        record, sort_keys=True, separators=(",", ":")
                    ) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if dict(counts) != source_manifest["window_counts"]:
            raise ValueError("Rule labels do not cover all source windows.")
        manifest = {
            "schema": "deterministic-semantic-rule-labels-v1",
            "teacher_type": TEACHER_TYPE,
            "semantic_contract_version": SEMANTIC_CONTRACT,
            "source_dataset_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "dataset_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_windows_sha256": source_manifest["windows_sha256"],
            "rule_code_sha256": _file_sha256(Path(__file__)),
            "predicate_code_sha256": _file_sha256(
                Path(__file__).with_name("medium_prompt.py")
            ),
            "labels_sha256": _file_sha256(stage / "labels.jsonl"),
            "label_counts": dict(counts),
            "successful_records": sum(counts.values()),
            "pretraining_eligible": True,
            "positive_counts": {
                partition: {
                    dimension: positive[(partition, dimension)]
                    for dimension in (
                        "mission_urgency", "concession_appropriateness",
                        "interaction_risk", "deadlock_severity",
                    )
                } for partition in counts
            },
            "rubric": {
                "mission_urgency": (
                    "active .25; priority>=.75 +.25; charging battery<.25 +.25"
                ),
                "concession_appropriateness": (
                    "local peer history .25; current peer .25; cardinal occupancy "
                    "or shelf-inclusive constraint proxy .5; both .75"
                ),
                "interaction_risk": (
                    "local peer history .25; current peer .25; cardinal peer "
                    "or >=2 current local peer cells .5; both .75"
                ),
                "deadlock_severity": (
                    "pure v2 witness .25; >=5 blocked requests or alternations "
                    ".5; >=7 blocked requests .75; no witness 0"
                ),
            },
            "claim_boundary": "deterministic_comparator_not_llm_gold_or_deadlock_truth",
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        (stage / "state.json").write_text(
            '{"status":"completed"}\n', encoding="utf-8", newline="\n",
        )
        os.replace(stage, destination)
    return {"run_directory": str(destination), "manifest": manifest}
