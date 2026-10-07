"""Audit frozen training labels offline without modifying source artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_mappo.mappo_baseline.structured_observation import (  # noqa: E402
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.medium_prompt import (  # noqa: E402
    observable_coordination_evidence,
    validate_medium_v7_semantic_support,
)
from llm_mappo.mappo_llm_kd.temporal_view import (  # noqa: E402
    serialize_temporal_semantic_view,
)
from llm_mappo.mappo_llm_kd.rule_semantic import (  # noqa: E402
    rule_semantic_scores,
)


DIMENSIONS = (
    "mission_urgency", "concession_appropriateness", "interaction_risk",
    "deadlock_severity",
)


def file_digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_id(window):
    return (
        f"{window['partition']}:{window['episode_id']}:"
        f"agent-{window['agent_id']}:start-{window['start_timestep']}:"
        f"end-{window['end_timestep']}"
    )


def audit(dataset, labels):  # noqa: C901
    records = {}
    with labels.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["view_id"] in records:
                raise ValueError("Duplicate label view_id.")
            records[record["view_id"]] = record
    ledger = StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2
    ).feature_names
    strata = defaultdict(Counter)
    values = []
    rule_values = []
    issues = []
    candidates = defaultdict(list)
    episodes = defaultdict(set)
    seen = set()
    counts = Counter()
    with (dataset / "windows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            window = json.loads(line)
            identifier = window_id(window)
            if identifier in seen:
                raise ValueError("Duplicate selected window.")
            seen.add(identifier)
            partition = window["partition"]
            counts[partition] += 1
            episodes[partition].add(window["episode_id"])
            record = records[identifier]
            if partition != "train":
                continue  # Held-out semantic content never informs this review.
            scores = record["scores"]
            if record["status"] != "ok" or set(scores) != set(DIMENSIONS):
                raise ValueError("Invalid label schema or status.")
            vector = [float(scores[name]) for name in DIMENSIONS]
            if not np.isfinite(vector).all() or not all(
                0 <= value <= 1 for value in vector
            ):
                raise ValueError("Invalid semantic score.")
            values.append(vector)
            view = serialize_temporal_semantic_view(
                np.asarray(window["observations"], dtype=np.float32),
                np.asarray(window["reset_masks"], dtype=bool), ledger, 8,
            )
            try:
                validate_medium_v7_semantic_support(
                    scores, record["evidence"], view
                )
            except ValueError as error:
                issues.append({"view_id": identifier, "error": str(error)})
            current = view["history"][-1]
            rule_values.append([
                rule_semantic_scores(view)[name] for name in DIMENSIONS
            ])
            witness = observable_coordination_evidence(view)
            phase = current["self"]["task_phase"]
            peer_present = current["local"]["visible_peer_cell_count"] > 0
            positive = vector[1] > 0 or vector[2] > 0
            equal = vector[1] == vector[2]
            evidence_equal = set(record["evidence"][DIMENSIONS[1]]) == set(
                record["evidence"][DIMENSIONS[2]]
            )
            for name in (
                "all", f"phase={phase}", f"peer_present={peer_present}",
                f"either_coordination_score_positive={positive}",
                f"sustained_witness={witness['positive_supported']}",
            ):
                strata[name].update({
                    "count": 1, "equal_scores": int(equal),
                    "equal_evidence_sets": int(evidence_equal),
                    "either_positive": int(positive),
                })
            at_charger = any(
                cell["forward"] == 0 and cell["right"] == 0
                and cell["charger"] for cell in current["local"]["cells"]
            )
            if phase == "charging":
                strata["charging_details"].update({
                    "count": 1, "currently_on_station": int(at_charger),
                    "previous_noop": int(
                        current["self"]["previous_requested_action"] == "NOOP"
                    ),
                    "battery_at_least_0_8": int(current["self"]["battery"] >= .8),
                    "observable_sustained_witness": int(
                        witness["positive_supported"]
                    ),
                })
            group = (
                f"{phase}|peer={peer_present}|witness={witness['positive_supported']}"
            )
            candidates[group].append({
                "view_id": identifier,
                "rank": sha256(identifier.encode()).hexdigest(),
                "scores": scores, "evidence": record["evidence"],
                "current": current["self"], "goal": current["goal"],
                "on_station": bool(at_charger),
                "visible_peer_cell_count": current["local"]["visible_peer_cell_count"],
                "witness": witness,
            })
    if set(records) != seen:
        raise ValueError("Labels and selected windows must join exactly.")
    partitions = sorted(episodes)
    for index, left in enumerate(partitions):
        for right in partitions[index + 1:]:
            if episodes[left] & episodes[right]:
                raise ValueError("Episode leakage across partitions.")
    matrix = np.asarray(values, dtype=np.float64)
    rule_matrix = np.asarray(rule_values, dtype=np.float64)
    if not len(matrix):
        raise ValueError("No training labels.")
    selected = []
    ordered = {name: sorted(rows, key=lambda row: row["rank"])
               for name, rows in candidates.items()}
    offset = 0
    while len(selected) < min(60, len(matrix)):
        for group in sorted(ordered):
            if offset < len(ordered[group]) and len(selected) < 60:
                selected.append({"group": group, **ordered[group][offset]})
        offset += 1
    return {
        "schema": "medium5-training-semantic-redundancy-audit-v1",
        "analysis_partition": "train",
        "held_out_content_used_for_semantic_design": False,
        "dataset_manifest_sha256": file_digest(dataset / "manifest.json"),
        "windows_sha256": file_digest(dataset / "windows.jsonl"),
        "labels_sha256": file_digest(labels),
        "analysis_script_sha256": file_digest(Path(__file__)),
        "partition_counts": dict(counts),
        "episode_counts": {key: len(value) for key, value in episodes.items()},
        "strata": {key: dict(value) for key, value in sorted(strata.items())},
        "dimensions": list(DIMENSIONS),
        "training_score_std": matrix.std(axis=0).tolist(),
        "training_score_histograms": {
            name: dict(sorted(Counter(matrix[:, index]).items()))
            for index, name in enumerate(DIMENSIONS)
        },
        "rule_concession_risk_equal_count": int(np.sum(
            rule_matrix[:, 1] == rule_matrix[:, 2]
        )),
        "llm_rule_per_dimension_mae": np.mean(
            np.abs(matrix - rule_matrix), axis=0
        ).tolist(),
        "training_correlation": np.corrcoef(matrix, rowvar=False).tolist(),
        "concession_risk_training_mae": float(
            np.mean(np.abs(matrix[:, 1] - matrix[:, 2]))
        ),
        "v7_support_failures": issues,
        "review_candidates": selected,
        "review_candidates_are_human_review": False,
        "source_modified": False,
        "api_requests": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output path to preserve audit versions.")
    source = args.dataset.resolve()
    if args.output.resolve().is_relative_to(source):
        raise ValueError("Write analysis outside the frozen source dataset.")
    report = audit(args.dataset, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "training_windows": report["strata"]["all"],
        "v7_support_failure_count": len(report["v7_support_failures"]),
    }))


if __name__ == "__main__":
    main()
