"""Episode-isolated, observable-coordination selection from frozen C1 rollouts."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)

from .dataset import _observable_stratum
from .medium_dataset_run import _complete_episode_receipts, _file_sha256
from .prompt import _canonical_json
from .temporal_view import serialize_temporal_semantic_view


SELECTION_SALT = "medium-5ag-coordination-selection-v2"
PARENT_MANIFEST_SHA256 = (
    "30827fe634f734b78c9ceb271f4e95a4c48f27d31ad4467a974732a7caddd9f1"
)
OUTPUT_NAME = "coordination-v2-c6d2df435f63"
WINDOW_BUDGETS = {"train": 4200, "validation": 900, "test": 900}
TRAIN_QUOTAS = {
    "blocked_forward": 420, "oscillation": 420, "dynamic_context": 1260,
    "goal_approach_control": 120, "empty_constrained": 120,
}
_LEDGER = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
).feature_names
_INDEX = {name: index for index, name in enumerate(_LEDGER)}
_PEERS = [
    index for index, name in enumerate(_LEDGER)
    if name.startswith("map_") and name.endswith("_other_agv")
    and name != "map_f0_r0_other_agv"
]
_CARDINAL = ("map_f1_r0", "map_f-1_r0", "map_f0_r1", "map_f0_r-1")
_CARDINAL_SOLID = [
    [_INDEX[prefix + "_static_obstacle"], _INDEX[prefix + "_shelf"]]
    for prefix in _CARDINAL
]
_CARDINAL_PEERS = [_INDEX[prefix + "_other_agv"] for prefix in _CARDINAL]


@dataclass(frozen=True)
class Candidate:
    """Small index record; full 209D windows stay in the episode NPZ files."""

    rank: bytes
    partition: str
    episode_id: str
    agent_id: int
    end: int
    blocked_forward: bool
    oscillation: bool
    dynamic_context: bool
    goal_approach_control: bool
    empty_constrained: bool
    current_peer: bool
    constrained: bool

    @property
    def hard(self) -> bool:
        return self.blocked_forward or self.oscillation


def classify_trajectory(observations: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized exact counterpart of the v2 observable witness predicate."""

    rows = np.asarray(observations, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 209 or len(rows) < 8:
        raise ValueError("A trajectory needs at least eight native 209D frames.")
    if not np.isfinite(rows).all():
        raise ValueError("Trajectory contains nonfinite observations.")
    end = np.arange(7, len(rows))
    index = end[:, None] - np.arange(7, -1, -1)
    phase = np.argmax(rows[:, 6:10], axis=1)
    goal = np.argmax(rows[:, 20:24], axis=1)
    action = np.argmax(rows[:, 11:16], axis=1)[index]
    blocked = rows[:, 16][index] > 0.5
    same = (phase[index] == phase[end, None]) & (
        goal[index] == goal[end, None]
    )
    suffix = np.logical_and.accumulate(same[:, ::-1], axis=1)[:, ::-1]
    action_valid = np.zeros_like(suffix)
    action_valid[:, 1:] = suffix[:, 1:] & suffix[:, :-1]
    active = (phase[end] != 0) & (goal[end] != 0) & (rows[end, 5] > 0)
    blocked_actions = (action == 1) & blocked & action_valid
    blocked_forward = (
        active & blocked_actions[:, -1]
        & (np.count_nonzero(blocked_actions, axis=1) >= 3)
    )
    turns = ((action == 2) | (action == 3)) & action_valid
    alternations = np.zeros(len(end), dtype=np.int16)
    previous_turn = np.zeros(len(end), dtype=np.int8)
    for column in range(8):
        current = action[:, column]
        present = turns[:, column]
        alternations += present & (previous_turn != 0) & (
            previous_turn != current
        )
        previous_turn = np.where(present, current, previous_turn)
    distances = rows[:, 19][index]
    minimum = np.min(np.where(suffix, distances, np.inf), axis=1)
    maximum = np.max(np.where(suffix, distances, -np.inf), axis=1)
    unmarked_forward = (action == 1) & ~blocked & action_valid
    oscillation = (
        active & (np.count_nonzero(turns, axis=1) >= 4)
        & (alternations >= 3) & (maximum - minimum <= 1e-6)
        & ~np.any(unmarked_forward, axis=1)
    )
    hard = blocked_forward | oscillation
    peer_counts = np.count_nonzero(rows[end][:, _PEERS] > 0.5, axis=1)
    current_peer = peer_counts > 0
    peer_history = np.any(rows[:, _PEERS][index] > 0.5, axis=(1, 2))
    free_degree = 4 - sum(
        np.any(rows[end][:, pair] > 0.5, axis=1)
        for pair in _CARDINAL_SOLID
    )
    constrained = free_degree <= 2
    cardinal_peer = np.any(rows[end][:, _CARDINAL_PEERS] > 0.5, axis=1)
    dynamic_context = (
        active & ~hard & (
            (current_peer & constrained) | (peer_counts >= 2) | cardinal_peer
        )
    )
    segment_start = np.argmax(suffix, axis=1)
    observed_goal_approach = (
        distances[np.arange(len(end)), segment_start] - distances[:, -1]
        > 1e-6
    )
    goal_approach_control = (
        active & ~hard & ~peer_history
        & np.any(unmarked_forward, axis=1) & observed_goal_approach
    )
    empty_constrained = active & ~hard & ~peer_history & constrained
    return {
        "end": end, "blocked_forward": blocked_forward,
        "oscillation": oscillation, "dynamic_context": dynamic_context,
        "goal_approach_control": goal_approach_control,
        "empty_constrained": empty_constrained,
        "current_peer": current_peer, "constrained": constrained,
    }


def _rank(partition: str, episode_id: str, agent_id: int, end: int) -> bytes:
    identity = (
        f"{SELECTION_SALT}\0{partition}\0{episode_id}"
        f"\0{agent_id}\0{end}"
    )
    return sha256(identity.encode("utf-8")).digest()


def _read_source(source: Path) -> tuple[dict, dict, dict, dict]:
    if _file_sha256(source / "manifest.json") != PARENT_MANIFEST_SHA256:
        raise ValueError("Frozen parent manifest SHA-256 differs.")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    identity = json.loads((source / "collection_identity.json").read_text(
        encoding="utf-8"
    ))
    partitions = json.loads((source / "episode_partitions.json").read_text(
        encoding="utf-8"
    ))
    if (
        manifest.get("schema") != "llm-dataset-build-run-v1"
        or manifest.get("main_pretraining_eligible") is not True
        or manifest.get("distribution") != "native-209d-collection-main-v1"
        or manifest.get("window_counts") != WINDOW_BUDGETS
        or manifest.get("source_actor_sha256") != identity["actor_sha256"]
        or _file_sha256(source / "windows.jsonl") != manifest["windows_sha256"]
        or set(partitions) != set(WINDOW_BUDGETS)
        or {name: len(partitions[name]) for name in partitions}
        != {"train": 84, "validation": 18, "test": 18}
    ):
        raise ValueError("Frozen parent dataset contract differs.")
    receipts = _complete_episode_receipts(source, identity)
    all_ids = [episode for ids in partitions.values() for episode in ids]
    if len(all_ids) != 120 or set(all_ids) != set(receipts):
        raise ValueError("Parent episode partitions differ from receipts.")
    assigned = {
        episode_id: name for name, episode_ids in partitions.items()
        for episode_id in episode_ids
    }
    window_counts = Counter()
    with (source / "windows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = row.get("partition")
            if assigned.get(row.get("episode_id")) != name:
                raise ValueError("Parent window differs from frozen episode split.")
            window_counts[name] += 1
    if dict(window_counts) != WINDOW_BUDGETS:
        raise ValueError("Parent window partition sizes differ.")
    return manifest, identity, partitions, receipts


def _index_candidates(source: Path, partitions: dict) -> dict[str, list[Candidate]]:
    output = {name: [] for name in WINDOW_BUDGETS}
    for partition, episode_ids in partitions.items():
        for episode_id in sorted(episode_ids):
            seed = int(episode_id.split("-")[2])
            with np.load(
                source / "episodes" / f"seed_{seed:05d}.npz", allow_pickle=False
            ) as payload:
                observations = payload["observations"]
            for agent_index in range(5):
                features = classify_trajectory(observations[agent_index])
                for position, end in enumerate(features["end"]):
                    agent_id = agent_index + 1
                    output[partition].append(Candidate(
                        _rank(partition, episode_id, agent_id, int(end)),
                        partition, episode_id, agent_id, int(end),
                        *(bool(features[key][position]) for key in (
                            "blocked_forward", "oscillation", "dynamic_context",
                            "goal_approach_control", "empty_constrained",
                            "current_peer",
                            "constrained",
                        )),
                    ))
    for candidates in output.values():
        candidates.sort(key=lambda row: (
            row.rank, row.episode_id, row.agent_id, row.end
        ))
    return output


def _select(  # noqa: C901
    source: Path, candidates: dict[str, list[Candidate]]
) -> tuple[dict, dict]:
    selected = {name: [] for name in WINDOW_BUDGETS}
    taken = defaultdict(list)
    seen_views = set()
    cache = {}
    rejected = Counter()
    stage_counts = Counter()

    def take(row: Candidate, stage: str) -> bool:
        key = row.episode_id, row.agent_id
        if any(abs(row.end - previous) < 8 for previous in taken[key]):
            rejected["spacing"] += 1
            return False
        if row.episode_id not in cache:
            seed = int(row.episode_id.split("-")[2])
            with np.load(
                source / "episodes" / f"seed_{seed:05d}.npz", allow_pickle=False
            ) as payload:
                cache[row.episode_id] = (
                    payload["observations"], payload["reset_masks"]
                )
        observations, masks = cache[row.episode_id]
        start = row.end - 7
        view = serialize_temporal_semantic_view(
            observations[row.agent_id - 1, start:row.end + 1],
            masks[row.agent_id - 1, start:row.end + 1], _LEDGER, 8,
        )
        digest = sha256(_canonical_json(view).encode("utf-8")).hexdigest()
        if digest in seen_views:
            rejected["semantic_duplicate"] += 1
            return False
        seen_views.add(digest)
        taken[key].append(row.end)
        selected[row.partition].append((row, digest, stage))
        stage_counts[stage] += 1
        return True

    def fill(partition: str, stage: str, limit: int, predicate) -> None:
        for row in candidates[partition]:
            if stage_counts[stage] >= limit:
                break
            if predicate(row):
                take(row, stage)
        if stage_counts[stage] != limit:
            raise ValueError(f"Insufficient {stage}: {stage_counts[stage]}/{limit}.")

    # Held-out partitions are selected first so train enrichment cannot bias them.
    for partition in ("validation", "test"):
        fill(partition, partition + "_natural", WINDOW_BUDGETS[partition],
             lambda row: True)
    fill("train", "blocked_forward", TRAIN_QUOTAS["blocked_forward"],
         lambda row: row.blocked_forward)
    fill("train", "oscillation", TRAIN_QUOTAS["oscillation"],
         lambda row: row.oscillation)
    fill("train", "dynamic_context", TRAIN_QUOTAS["dynamic_context"],
         lambda row: row.dynamic_context)
    fill("train", "goal_approach_control",
         TRAIN_QUOTAS["goal_approach_control"],
         lambda row: row.goal_approach_control)
    fill("train", "empty_constrained", TRAIN_QUOTAS["empty_constrained"],
         lambda row: row.empty_constrained)
    fill("train", "representative", 1860, lambda row: True)
    if {name: len(rows) for name, rows in selected.items()} != WINDOW_BUDGETS:
        raise AssertionError("Selected partition sizes differ from budget.")
    categories = (
        "blocked_forward", "oscillation", "hard", "dynamic_context",
        "goal_approach_control", "empty_constrained", "current_peer",
        "constrained",
    )
    candidate_coverage = {
        name: {
            category: sum(bool(getattr(row, category)) for row in rows)
            for category in categories
        } for name, rows in candidates.items()
    }
    return selected, {
        "candidate_counts": {name: len(rows) for name, rows in candidates.items()},
        "candidate_coverage": candidate_coverage,
        "stage_counts": dict(stage_counts), "rejections": dict(rejected),
    }


def _write_windows(path: Path, source: Path, selected: dict) -> dict:
    counts = {}
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for partition in ("train", "validation", "test"):
            counts[partition] = Counter()
            grouped = defaultdict(list)
            for row, digest, stage in selected[partition]:
                grouped[row.episode_id].append((row, digest, stage))
            for episode_id, values in sorted(grouped.items()):
                seed = int(episode_id.split("-")[2])
                with np.load(
                    source / "episodes" / f"seed_{seed:05d}.npz", allow_pickle=False
                ) as payload:
                    observations = payload["observations"]
                    masks = payload["reset_masks"]
                for row, _digest, _stage in values:
                    start = row.end - 7
                    segment = observations[row.agent_id - 1, start:row.end + 1]
                    stratum = _observable_stratum(segment)
                    record = {
                        "partition": partition, "episode_id": episode_id,
                        "agent_id": row.agent_id, "start_timestep": start,
                        "end_timestep": row.end, "observations": segment.tolist(),
                        "reset_masks": masks[
                            row.agent_id - 1, start:row.end + 1
                        ].tolist(), "stratum": stratum,
                    }
                    stream.write(json.dumps(
                        record, sort_keys=True, separators=(",", ":")
                    ) + "\n")
                    counts[partition][stratum] += 1
        stream.flush()
        os.fsync(stream.fileno())
    return {name: dict(sorted(values.items())) for name, values in counts.items()}


def select_coordination_dataset(source_root: str | Path,
                                output_root: str | Path) -> dict:
    """Build one new immutable derivative; never modify the frozen parent."""

    source = Path(source_root).resolve()
    destination = Path(output_root).resolve() / OUTPUT_NAME
    if destination.exists():
        raise FileExistsError(f"Derivative destination already exists: {destination}")
    manifest, identity, partitions, receipts = _read_source(source)
    candidates = _index_candidates(source, partitions)
    selected, audit = _select(source, candidates)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".coordination-v2-staging-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary)
        (stage / "episodes").mkdir()
        for receipt in receipts.values():
            name = f"seed_{receipt['seed']:05d}"
            for suffix in (".json", ".npz"):
                shutil.copy2(
                    source / "episodes" / (name + suffix),
                    stage / "episodes" / (name + suffix),
                )
        shutil.copy2(source / "collection_identity.json",
                     stage / "collection_identity.json")
        shutil.copy2(source / "episode_partitions.json",
                     stage / "episode_partitions.json")
        strata = _write_windows(stage / "windows.jsonl", source, selected)
        selected_coverage = {
            name: {
                "blocked_forward": sum(row.blocked_forward for row, _, _ in values),
                "oscillation": sum(row.oscillation for row, _, _ in values),
                "hard": sum(row.hard for row, _, _ in values),
                "dynamic_context": sum(row.dynamic_context for row, _, _ in values),
                "goal_approach_control": sum(
                    row.goal_approach_control for row, _, _ in values
                ),
                "empty_constrained": sum(
                    row.empty_constrained for row, _, _ in values
                ),
                "current_peer": sum(row.current_peer for row, _, _ in values),
                "constrained": sum(row.constrained for row, _, _ in values),
            } for name, values in selected.items()
        }
        natural_drift = {
            name: {
                category: round(
                    selected_coverage[name][category] / WINDOW_BUDGETS[name]
                    - count / audit["candidate_counts"][name], 8
                ) for category, count in audit["candidate_coverage"][name].items()
            } for name in ("validation", "test")
        }
        selection_identity = {
            "schema": "medium5-coordination-selection-v2",
            "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
            "parent_windows_sha256": manifest["windows_sha256"],
            "parent_episode_partitions_sha256": _file_sha256(
                source / "episode_partitions.json"
            ),
            "selection_salt": SELECTION_SALT,
            "selection_code_sha256": _file_sha256(Path(__file__)),
            "semantic_predicate_code_sha256": _file_sha256(
                Path(__file__).with_name("medium_prompt.py")
            ),
            "train_quotas": TRAIN_QUOTAS,
            "validation_test_policy": "uniform_hash_all_eligible_endpoints",
            "global_semantic_view_deduplication": True,
            "minimum_same_agv_endpoint_gap": 8,
            "target_identity_available": False,
            "geometric_constraint_is_shelf_inclusive_proxy": True,
            "unmarked_forward_is_not_confirmed_success": True,
        }
        (stage / "selection_identity.json").write_text(
            json.dumps(selection_identity, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        derived = {
            **manifest,
            "schema": "llm-dataset-build-run-v1",
            "source_dataset_manifest_sha256": PARENT_MANIFEST_SHA256,
            "source_dataset_windows_sha256": manifest["windows_sha256"],
            "source_selection_code_sha256": manifest[
                "selection_code_sha256"
            ],
            "selection_code_sha256": {
                "coordination_dataset.py": _file_sha256(Path(__file__)),
                "medium_prompt.py": _file_sha256(
                    Path(__file__).with_name("medium_prompt.py")
                ),
            },
            "selection_identity_sha256": _file_sha256(
                stage / "selection_identity.json"
            ),
            "selection_policy": "observable-coordination-v2",
            "semantic_contract_version": "observable-coordination-four-score-v2",
            "candidate_counts": audit["candidate_counts"],
            "candidate_coordination_coverage": audit["candidate_coverage"],
            "selection_stage_counts": audit["stage_counts"],
            "selection_rejections": audit["rejections"],
            "selected_coordination_coverage": selected_coverage,
            "natural_selection_rate_deltas": natural_drift,
            "selected_strata_counts": strata,
            "coverage": selected_coverage,
            "windows_sha256": _file_sha256(stage / "windows.jsonl"),
            "window_counts": WINDOW_BUDGETS,
            "main_pretraining_eligible": True,
            "formal_performance_evidence": False,
        }
        (stage / "manifest.json").write_text(
            json.dumps(derived, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        (stage / "state.json").write_text(
            '{"status":"completed"}\n', encoding="utf-8", newline="\n",
        )
        os.replace(stage, destination)
    return {"run_directory": str(destination), "manifest": derived}
