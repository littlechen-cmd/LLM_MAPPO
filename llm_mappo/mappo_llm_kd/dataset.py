"""Episode-safe dataset partitions for Phase 3 LLM KD diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Mapping, Sequence

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.trajectory import ActorObservationTrajectory


MAIN_BASELINE_DISTRIBUTION = "native-209d-collection-main-v1"
ASTAR_DIAGNOSTIC_DISTRIBUTION = "phase2-astar-diagnostic-only-v1"
_PARTITIONS = ("train", "validation", "test")
_LEDGER = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
).feature_names
_INDEX = {name: index for index, name in enumerate(_LEDGER)}


@dataclass(frozen=True)
class DistributionTrajectory:
    """Bind a raw trajectory to its behavior-distribution role."""

    distribution: str
    trajectory: ActorObservationTrajectory

    def __post_init__(self) -> None:
        if self.distribution not in {
            MAIN_BASELINE_DISTRIBUTION,
            ASTAR_DIAGNOSTIC_DISTRIBUTION,
        }:
            raise ValueError("Unknown behavior distribution.")


@dataclass(frozen=True)
class TemporalWindow:
    """One history window whose frames share an episode and focal AGV."""

    episode_id: str
    agent_id: int
    start_timestep: int
    end_timestep: int
    observations: np.ndarray
    reset_masks: np.ndarray
    stratum: str


def split_complete_episodes(
    records: Sequence[DistributionTrajectory],
    split_ratios: Mapping[str, float],
    *,
    salt: str,
) -> dict[str, list[DistributionTrajectory]]:
    """Deterministically assign whole baseline episodes to canonical splits."""

    ratios = _validate_split_ratios(split_ratios)
    if not isinstance(salt, str) or not salt:
        raise ValueError("split salt must be a non-empty string.")
    if any(item.distribution != MAIN_BASELINE_DISTRIBUTION for item in records):
        raise ValueError("A* trajectories cannot enter the main dataset split.")
    _validate_complete_five_agent_episodes(records)
    episode_ids = sorted({item.trajectory.episode_id for item in records})
    ranked = sorted(
        episode_ids,
        key=lambda episode_id: (
            sha256(f"{salt}\0{episode_id}".encode("utf-8")).digest(),
            episode_id,
        ),
    )
    counts = _partition_counts(len(ranked), ratios)
    assignments = {}
    cursor = 0
    for name in _PARTITIONS:
        for episode_id in ranked[cursor:cursor + counts[name]]:
            assignments[episode_id] = name
        cursor += counts[name]
    result = {name: [] for name in _PARTITIONS}
    for item in sorted(
        records,
        key=lambda value: (
            value.trajectory.episode_id,
            value.trajectory.agent_id,
        ),
    ):
        result[assignments[item.trajectory.episode_id]].append(item)
    _assert_disjoint_episode_splits(result)
    return result


def _validate_complete_five_agent_episodes(records) -> None:
    grouped = {}
    for item in records:
        grouped.setdefault(item.trajectory.episode_id, []).append(item.trajectory)
    for trajectories in grouped.values():
        if sorted(value.agent_id for value in trajectories) != [1, 2, 3, 4, 5]:
            raise ValueError("Each complete episode requires agents 1..5 exactly once.")
        first = trajectories[0]
        provenance = (
            first.behavior_policy_id, first.environment_sha256,
            first.config_sha256, first.observation_schema,
        )
        expected_steps = np.arange(len(first.timesteps), dtype=np.int64)
        for trajectory in trajectories:
            if (
                trajectory.behavior_policy_id, trajectory.environment_sha256,
                trajectory.config_sha256, trajectory.observation_schema,
            ) != provenance:
                raise ValueError("Episode agents must have identical provenance.")
            if not np.array_equal(trajectory.timesteps, expected_steps):
                raise ValueError("Episode agents require aligned contiguous timesteps.")
            expected_reset = np.zeros(len(expected_steps), dtype=np.bool_)
            expected_reset[0] = True
            if not np.array_equal(trajectory.reset_masks, expected_reset):
                raise ValueError("Episode agents require aligned reset boundaries.")


def build_partition_windows(
    partitions: Mapping[str, Sequence[DistributionTrajectory]],
    *,
    history_length: int,
) -> dict[str, list[TemporalWindow]]:
    """Create overlapping windows only after episode-level partitioning."""

    if set(partitions) != set(_PARTITIONS):
        raise ValueError("Dataset partitions must be train/validation/test.")
    if isinstance(history_length, bool) or not isinstance(history_length, int):
        raise ValueError("history_length must be a positive integer.")
    if history_length < 1:
        raise ValueError("history_length must be a positive integer.")
    _assert_disjoint_episode_splits(partitions)
    output = {name: [] for name in _PARTITIONS}
    for partition in _PARTITIONS:
        for item in partitions[partition]:
            if item.distribution != MAIN_BASELINE_DISTRIBUTION:
                raise ValueError("A* trajectories cannot enter pretraining windows.")
            trajectory = item.trajectory
            for end in range(len(trajectory.timesteps)):
                start = max(0, end - history_length + 1)
                observations = trajectory.observations[start:end + 1].copy()
                reset_masks = trajectory.reset_masks[start:end + 1].copy()
                output[partition].append(
                    TemporalWindow(
                        episode_id=trajectory.episode_id,
                        agent_id=int(trajectory.agent_id),
                        start_timestep=int(trajectory.timesteps[start]),
                        end_timestep=int(trajectory.timesteps[end]),
                        observations=observations,
                        reset_masks=reset_masks,
                        stratum=_observable_stratum(observations),
                    )
                )
    return output


def deterministic_stratified_sample(
    windows: Sequence[TemporalWindow], *, sample_budget: int, salt: str
) -> list[TemporalWindow]:
    """Round-robin observable strata with stable within-stratum ordering."""

    if (
        isinstance(sample_budget, bool)
        or not isinstance(sample_budget, int)
        or sample_budget < 1
        or sample_budget > len(windows)
    ):
        raise ValueError("sample_budget must be within the available windows.")
    if not isinstance(salt, str) or not salt:
        raise ValueError("sampling salt must be a non-empty string.")
    grouped: dict[str, list[TemporalWindow]] = {}
    for window in windows:
        grouped.setdefault(window.stratum, []).append(window)
    for stratum, values in grouped.items():
        values.sort(key=lambda value: _window_rank(value, salt, stratum))
    selected = []
    strata = sorted(grouped)
    cursor = 0
    while len(selected) < sample_budget:
        stratum = strata[cursor % len(strata)]
        values = grouped[stratum]
        if values:
            selected.append(values.pop(0))
        if not values:
            strata.remove(stratum)
            cursor = 0
        else:
            cursor += 1
    return selected


def _window_rank(window: TemporalWindow, salt: str, stratum: str):
    identity = (
        f"{salt}\0{stratum}\0{window.episode_id}\0{window.agent_id}"
        f"\0{window.start_timestep}\0{window.end_timestep}"
    )
    return sha256(identity.encode("utf-8")).digest(), identity


def _validate_split_ratios(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(_PARTITIONS):
        raise ValueError("split ratios must contain train/validation/test exactly.")
    if any(type(values[name]) is not float for name in _PARTITIONS):
        raise ValueError("split ratios must be explicit floating-point values.")
    ratios = {name: float(values[name]) for name in _PARTITIONS}
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios.values()):
        raise ValueError("split ratios must be finite and positive.")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-12):
        raise ValueError("split ratios must sum to one.")
    return ratios


def _partition_counts(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in _PARTITIONS}
    counts = {name: int(math.floor(raw[name])) for name in _PARTITIONS}
    remainder = total - sum(counts.values())
    order = sorted(
        _PARTITIONS,
        key=lambda name: (-(raw[name] - counts[name]), _PARTITIONS.index(name)),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _assert_disjoint_episode_splits(partitions) -> None:
    episode_sets = {
        name: {item.trajectory.episode_id for item in partitions[name]}
        for name in _PARTITIONS
    }
    for index, left in enumerate(_PARTITIONS):
        for right in _PARTITIONS[index + 1:]:
            if not episode_sets[left].isdisjoint(episode_sets[right]):
                raise ValueError("Dataset episode identities are not disjoint.")


def _observable_stratum(observations: np.ndarray) -> str:
    final = observations[-1]
    phase = next(
        name
        for name in ("idle", "task", "delivery", "charging")
        if final[_INDEX[f"self_phase_{name}"]] > 0.5
    )
    priority = float(final[_INDEX["self_priority_weight"]])
    priority_bin = "low" if priority <= 0.25 else (
        "medium" if priority <= 0.75 else "high"
    )
    blocked = bool(final[_INDEX["self_previous_blocked"]] > 0.5)
    distances = observations[:, _INDEX["goal_normalized_distance"]]
    no_progress = len(distances) > 1 and float(distances[0] - distances[-1]) <= 1e-6
    congested = any(
        final[index] > 0.5
        for name, index in _INDEX.items()
        if name.startswith("map_") and name.endswith("_other_agv")
    )
    return (
        f"phase={phase}|priority={priority_bin}|"
        f"blocked_or_no_progress={str(blocked or no_progress).lower()}|"
        f"congestion={'present' if congested else 'absent'}"
    )
