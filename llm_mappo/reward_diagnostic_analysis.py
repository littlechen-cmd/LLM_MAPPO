"""Pure, deterministic aggregation for completed Group 2 diagnostic runs."""

from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


EVENT_FIELDS = (
    "completed_tasks",
    "picked_tasks",
    "progress_positive_events",
    "progress_negative_events",
    "blocked_forwards",
    "collisions",
    "deadlocked",
    "low_battery_triggers",
    "charged_events",
    "energy_deaths",
)
LEDGER_FIELDS = (
    "reward_completion",
    "reward_pickup",
    "reward_progress",
    "reward_blocked_forward",
    "reward_collision",
    "reward_charging",
    "reward_deadlock",
    "reward_energy",
    "reward_step_cost",
)
EVALUATION_TOTAL_FIELDS = (
    "completed_tasks",
    "energy_deaths",
    "collisions",
    "blocked_forwards",
)
ACTION_PHASE_FIELDS = tuple(
    f"{phase}_{action}_count"
    for phase in ("idle", "task", "delivery", "charging")
    for action in ("noop", "forward", "left", "right", "toggle_load")
)


def analyze_completed_run(run_directory: str | Path) -> dict[str, Any]:
    """Read one completed diagnostic run without modifying its artifacts."""

    directory = Path(run_directory).resolve()
    state = _read_json(directory / "state.json")
    if state.get("status") != "complete":
        raise ValueError(f"Diagnostic run is not complete: {directory}")
    manifest = _read_json(directory / "run_manifest.json")
    summary = _read_json(directory / "summary.json")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"Diagnostic run has no identity: {directory}")
    episodes = _read_csv(directory / "episodes.csv")
    updates = _read_csv(directory / "updates.csv")
    evaluation = _read_json(directory / "evaluation" / "evaluation.json")
    checkpoint = directory / "checkpoint_final.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Diagnostic run has no final checkpoint: {directory}")
    ordered_episodes = sorted(
        episodes,
        key=lambda row: (
            _integer(row, "global_environment_steps"),
            _integer(row, "worker_index"),
            _integer(row, "episode_index"),
        ),
    )
    logged_episode_steps = sum(
        _number(row, "steps") for row in ordered_episodes
    )
    training_environment_steps = int(summary["global_environment_steps"])
    completed_tasks = sum(
        _number(row, "completed_tasks") for row in ordered_episodes
    )
    return {
        "artifact_path": str(directory),
        "identity": identity,
        "evaluation_seeds": list(manifest.get("evaluation", {}).get("seeds", [])),
        "evaluation_episode_seeds": _evaluation_episode_seeds(evaluation),
        "fixed_evaluation": _fixed_evaluation_summary(evaluation),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "episode_count": len(ordered_episodes),
        "update_count": len(updates),
        "training_environment_steps": training_environment_steps,
        "logged_completed_episode_steps": logged_episode_steps,
        "completed_tasks": completed_tasks,
        "events": _rate_summary(
            ordered_episodes, EVENT_FIELDS, logged_episode_steps, completed_tasks
        ),
        "ledger": _rate_summary(
            ordered_episodes, LEDGER_FIELDS, logged_episode_steps, completed_tasks
        ),
        "trend": _episode_trend(ordered_episodes),
        "raw_distributions": _raw_distribution_summary(updates),
        "numerical_health": _numerical_health(ordered_episodes, updates),
    }


def build_diagnostic_report(
    run_directories: Iterable[str | Path],
) -> dict[str, Any]:
    """Build a stable report over completed runs, preserving each training seed."""

    runs = [analyze_completed_run(directory) for directory in run_directories]
    if not runs:
        raise ValueError("At least one completed diagnostic run is required.")
    seeds = [int(run["identity"].get("seed", -1)) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Diagnostic report contains duplicate training seeds.")
    runs.sort(key=lambda run: int(run["identity"].get("seed", -1)))
    return {
        "schema": "reward-diagnostic-analysis-v1",
        "training_seeds": [int(run["identity"]["seed"]) for run in runs],
        "runs": runs,
        "evidence_complete": all(
            bool(run["trend"]["evidence_complete"])
            and bool(run["numerical_health"]["finite"])
            for run in runs
        ),
    }


def _rate_summary(
    rows: list[Mapping[str, str]],
    fields: tuple[str, ...],
    total_steps: float,
    completed_tasks: float,
) -> dict[str, dict[str, float | None]]:
    summary = {}
    for field in fields:
        total = sum(_number(row, field) for row in rows)
        summary[field] = {
            "total": total,
            "per_1000_environment_steps": _divide_or_null(total * 1000.0, total_steps),
            "per_completed_task": _divide_or_null(total, completed_tasks),
        }
    return summary


def _episode_trend(rows: list[Mapping[str, str]]) -> dict[str, Any]:
    count = len(rows)
    if count < 2:
        return {
            "evidence_complete": False,
            "reduced_window": True,
            "window_size": 0,
            "early": None,
            "late": None,
        }
    if count >= 200:
        window_size = 100
        reduced_window = False
    else:
        window_size = count // 2
        reduced_window = True
    return {
        "evidence_complete": True,
        "reduced_window": reduced_window,
        "window_size": window_size,
        "early": _window_summary(rows[:window_size]),
        "late": _window_summary(rows[-window_size:]),
    }


def _window_summary(rows: list[Mapping[str, str]]) -> dict[str, float | None]:
    return {
        "mean_completed_tasks": _mean([_number(row, "completed_tasks") for row in rows]),
        "mean_steps_per_task": _mean_non_null(
            [_optional_number(row, "steps_per_task") for row in rows]
        ),
        "mean_reward": _mean([_number(row, "reward") for row in rows]),
    }


def _raw_distribution_summary(rows: list[Mapping[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for prefix in ("raw_advantage", "raw_return"):
        groups = []
        for row in rows:
            count = _optional_number(row, f"{prefix}_count")
            mean = _optional_number(row, f"{prefix}_mean")
            std = _optional_number(row, f"{prefix}_std")
            if count is None or mean is None or std is None or count <= 0.0:
                continue
            groups.append((row, count, mean, std))
        if not groups:
            summary[prefix] = None
            continue
        count = sum(group[1] for group in groups)
        pooled_mean = sum(group[1] * group[2] for group in groups) / count
        pooled_variance = sum(
            group_count
            * (group_std ** 2 + (group_mean - pooled_mean) ** 2)
            for _, group_count, group_mean, group_std in groups
        ) / count
        distribution = {
            "sample_count": int(count),
            "pooled_mean": pooled_mean,
            "pooled_std": pooled_variance ** 0.5,
            "observed_min": min(
                _number(row, f"{prefix}_min") for row, *_ in groups
            ),
            "observed_max": max(
                _number(row, f"{prefix}_max") for row, *_ in groups
            ),
            "observed_max_abs": max(
                _number(row, f"{prefix}_max_abs") for row, *_ in groups
            ),
        }
        for quantile in ("p05", "p50", "p95"):
            distribution[f"update_weighted_{quantile}_estimate"] = sum(
                group_count * _number(row, f"{prefix}_{quantile}")
                for row, group_count, _, _ in groups
            ) / count
        summary[prefix] = distribution
    return summary


def _numerical_health(
    episodes: list[Mapping[str, str]], updates: list[Mapping[str, str]]
) -> dict[str, Any]:
    values = [
        value
        for row in [*episodes, *updates]
        for value in (_optional_number(row, field) for field in row)
        if value is not None
    ]
    non_finite = sum(not _is_finite(value) for value in values)
    return {"finite": non_finite == 0, "non_finite_value_count": non_finite}


def _evaluation_episode_seeds(evaluation: Mapping[str, Any]) -> list[int]:
    episodes = evaluation.get("episodes", [])
    if not isinstance(episodes, list):
        return []
    return sorted(
        int(episode["seed"])
        for episode in episodes
        if isinstance(episode, dict) and "seed" in episode
    )


def _fixed_evaluation_summary(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    episodes = evaluation.get("episodes", [])
    if not isinstance(episodes, list):
        episodes = []
    ordered = sorted(
        (dict(episode) for episode in episodes if isinstance(episode, dict)),
        key=lambda episode: int(episode.get("seed", -1)),
    )
    totals = {
        field: sum(float(episode.get(field, 0.0)) for episode in ordered)
        for field in EVALUATION_TOTAL_FIELDS
    }
    return {
        "episode_count": len(ordered),
        "total_completed_tasks": totals["completed_tasks"],
        "mean_completed_tasks": _divide_or_null(
            totals["completed_tasks"], float(len(ordered))
        ),
        "zero_completion_episode_count": sum(
            float(episode.get("completed_tasks", 0.0)) == 0.0
            for episode in ordered
        ),
        "deadlocked_episode_count": sum(
            bool(episode.get("deadlocked", False)) for episode in ordered
        ),
        "totals": totals,
        "action_phase_totals": {
            field: sum(float(episode.get(field, 0.0)) for episode in ordered)
            for field in ACTION_PHASE_FIELDS
        },
        "episodes": ordered,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Diagnostic artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Diagnostic JSON must be a mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Diagnostic artifact is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, str], field: str) -> float:
    value = _optional_number(row, field)
    return 0.0 if value is None else value


def _optional_number(row: Mapping[str, str], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(row: Mapping[str, str], field: str) -> int:
    return int(_number(row, field))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_non_null(values: list[float | None]) -> float | None:
    return _mean([value for value in values if value is not None])


def _divide_or_null(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_finite(value: float) -> bool:
    return value not in (float("inf"), float("-inf")) and value == value
