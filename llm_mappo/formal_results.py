"""Aggregation and paired statistics for the preregistered core experiment."""

from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml


METRICS: Mapping[str, str] = {
    "task_completion_rate": "higher",
    "completed_tasks_per_1000_steps": "higher",
    "mean_reward_per_episode": "higher",
    "mean_steps_per_episode": "lower",
    "mean_collisions_per_episode": "lower",
    "mean_energy_deaths_per_episode": "lower",
}

PRIMARY_EVALUATION_METRICS: tuple[str, ...] = (
    "completed_tasks_per_1000_steps",
)

LEARNING_CURVE_METRICS: Mapping[str, str] = {
    "completed_tasks_per_1000_steps_auc": "higher",
}

FACTORIAL_CONTRASTS: Mapping[str, Mapping[str, float]] = {
    "full_vs_baseline": {
        "MAPPO-WP": -1.0,
        "MAPPO-WP+A*KD+LLMKD": 1.0,
    },
    "astar_kd_main_effect": {
        "MAPPO-WP": -0.5,
        "MAPPO-WP+A*KD": 0.5,
        "MAPPO-WP+LLMKD": -0.5,
        "MAPPO-WP+A*KD+LLMKD": 0.5,
    },
    "llm_kd_main_effect": {
        "MAPPO-WP": -0.5,
        "MAPPO-WP+A*KD": -0.5,
        "MAPPO-WP+LLMKD": 0.5,
        "MAPPO-WP+A*KD+LLMKD": 0.5,
    },
    "astar_llm_interaction": {
        "MAPPO-WP": 1.0,
        "MAPPO-WP+A*KD": -1.0,
        "MAPPO-WP+LLMKD": -1.0,
        "MAPPO-WP+A*KD+LLMKD": 1.0,
    },
}


def load_manifest(path: str | Path) -> Dict[str, object]:
    """Load the formal experiment manifest."""
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("Experiment manifest must contain a YAML mapping")
    return manifest


def evaluation_path(root: Path, group: Mapping[str, object], seed: int) -> Path:
    """Return the preregistered evaluation path for one trained policy."""
    slug = str(group["artifact_slug"])
    return root / slug / f"seed_{seed:03d}" / "evaluation.json"


def collect_seed_rows(
    manifest: Mapping[str, object],
    evaluation_root: str | Path,
    *,
    group_section: str = "core_groups",
) -> List[Dict[str, object]]:
    """Validate and flatten all group-by-training-seed evaluation results."""
    root = Path(evaluation_root)
    groups = _mapping(manifest, group_section)
    training = _mapping(manifest, "training")
    evaluation = _mapping(manifest, "evaluation")
    policy_seeds = _integer_list(training, "policy_initialization_seeds")
    held_out_seeds = _integer_list(evaluation, "held_out_seeds")
    episodes_per_seed = int(evaluation["episodes_per_seed"])
    rows: List[Dict[str, object]] = []
    missing: List[str] = []

    for group_name, group_value in groups.items():
        group = _as_mapping(group_value, f"{group_section}.{group_name}")
        for policy_seed in policy_seeds:
            source = evaluation_path(root, group, policy_seed)
            if not source.is_file():
                missing.append(str(source))
                continue
            result = json.loads(source.read_text(encoding="utf-8"))
            _validate_evaluation(
                result,
                held_out_seeds=held_out_seeds,
                episodes_per_seed=episodes_per_seed,
                source=source,
            )
            for metric, direction in METRICS.items():
                if metric not in result:
                    raise ValueError(f"{source}: missing metric {metric!r}")
                rows.append(
                    {
                        "group": group_name,
                        "artifact_slug": group["artifact_slug"],
                        "training_seed": policy_seed,
                        "metric": metric,
                        "direction": direction,
                        "value": float(result[metric]),
                        "evaluation_path": str(source),
                    }
                )
    if missing:
        preview = "\n".join(missing[:8])
        remainder = len(missing) - min(len(missing), 8)
        suffix = f"\n... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            "Formal evaluation matrix is incomplete:\n" + preview + suffix
        )
    return rows


def collect_learning_curve_auc_rows(
    manifest: Mapping[str, object],
    training_root: str | Path,
    *,
    group_section: str = "core_groups",
) -> List[Dict[str, object]]:
    """Collect normalized throughput AUC values with training seed as the unit."""
    root = Path(training_root)
    groups = _mapping(manifest, group_section)
    training = _mapping(manifest, "training")
    policy_seeds = _integer_list(training, "policy_initialization_seeds")
    rows: List[Dict[str, object]] = []
    missing: List[str] = []

    for group_name, group_value in groups.items():
        group = _as_mapping(group_value, f"{group_section}.{group_name}")
        for policy_seed in policy_seeds:
            source = learning_curve_path(root, group, policy_seed)
            if not source.is_file():
                missing.append(str(source))
                continue
            rows.append(
                {
                    "group": group_name,
                    "artifact_slug": group["artifact_slug"],
                    "training_seed": policy_seed,
                    "metric": "completed_tasks_per_1000_steps_auc",
                    "direction": "higher",
                    "value": learning_curve_auc(source),
                    "training_path": str(source),
                }
            )
    if missing:
        preview = "\n".join(missing[:8])
        remainder = len(missing) - min(len(missing), 8)
        suffix = f"\n... and {remainder} more" if remainder else ""
        raise FileNotFoundError(
            "Formal training matrix is incomplete:\n" + preview + suffix
        )
    return rows


def learning_curve_path(root: Path, group: Mapping[str, object], seed: int) -> Path:
    """Return the preregistered training-log path for one policy seed."""
    slug = str(group["artifact_slug"])
    return root / slug / f"seed_{seed:03d}" / "episodes.csv"


def learning_curve_auc(path: str | Path) -> float:
    """Return step-normalized AUC of cumulative episode throughput samples.

    Episode records that share an environment-step value are pooled before the
    throughput is calculated, which avoids treating simultaneous vectorized
    environment completions as a sequence of independent observations.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    required = {"environment_steps", "completed_tasks", "steps"}
    observed = set(records[0]) if records else set()
    missing = sorted(required.difference(observed))
    if missing:
        raise ValueError(f"{source}: missing learning-curve columns {missing}")

    pooled: Dict[float, List[float]] = {}
    for record in records:
        step = float(record["environment_steps"])
        completed = float(record["completed_tasks"])
        episode_steps = float(record["steps"])
        if step <= 0.0 or episode_steps <= 0.0:
            raise ValueError(f"{source}: environment and episode steps must be positive")
        totals = pooled.setdefault(step, [0.0, 0.0])
        totals[0] += completed
        totals[1] += episode_steps
    if not pooled:
        raise ValueError(f"{source}: episodes.csv contains no completed episodes")

    points = [(0.0, 0.0)]
    for step, (completed, episode_steps) in sorted(pooled.items()):
        points.append((step, 1000.0 * completed / episode_steps))
    if len(points) < 2 or points[-1][0] <= 0.0:
        raise ValueError(f"{source}: no positive environment-step range for AUC")

    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in zip(
            points, points[1:]
        )
    )
    return area / points[-1][0]


def summarize_groups(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int = 10000,
    random_seed: int = 20260817,
) -> List[Dict[str, object]]:
    """Summarize each metric at the independent training-seed level."""
    grouped: Dict[Tuple[str, str], List[float]] = {}
    directions: Dict[str, str] = {}
    for row in rows:
        key = (str(row["group"]), str(row["metric"]))
        grouped.setdefault(key, []).append(float(row["value"]))
        directions[str(row["metric"])] = str(row["direction"])

    output: List[Dict[str, object]] = []
    for index, ((group, metric), values) in enumerate(sorted(grouped.items())):
        low, high = bootstrap_mean_ci(
            values,
            samples=bootstrap_samples,
            seed=random_seed + index,
        )
        output.append(
            {
                "group": group,
                "metric": metric,
                "direction": directions[metric],
                "n_training_seeds": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return output


def compare_full_to_baseline(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline: str = "MAPPO-WP",
    full: str = "MAPPO-WP+A*KD+LLMKD",
    bootstrap_samples: int = 10000,
    random_seed: int = 20260817,
) -> List[Dict[str, object]]:
    """Compute preregistered paired comparisons and Holm-adjusted p-values."""
    values: Dict[Tuple[str, str], Dict[int, float]] = {}
    directions: Dict[str, str] = {}
    for row in rows:
        key = (str(row["group"]), str(row["metric"]))
        values.setdefault(key, {})[int(row["training_seed"])] = float(row["value"])
        directions[str(row["metric"])] = str(row["direction"])

    output: List[Dict[str, object]] = []
    for index, metric in enumerate(METRICS):
        baseline_values = values.get((baseline, metric), {})
        full_values = values.get((full, metric), {})
        if set(baseline_values) != set(full_values) or not baseline_values:
            raise ValueError(
                f"Unmatched training seeds for {metric}: "
                f"baseline={sorted(baseline_values)}, full={sorted(full_values)}"
            )
        seeds = sorted(baseline_values)
        differences = [
            full_values[seed] - baseline_values[seed] for seed in seeds
        ]
        low, high = bootstrap_mean_ci(
            differences,
            samples=bootstrap_samples,
            seed=random_seed + index,
        )
        difference_std = (
            statistics.stdev(differences) if len(differences) > 1 else 0.0
        )
        output.append(
            {
                "metric": metric,
                "direction": directions[metric],
                "baseline": baseline,
                "comparison": full,
                "n_pairs": len(differences),
                "mean_paired_difference": statistics.fmean(differences),
                "ci95_low": low,
                "ci95_high": high,
                "paired_effect_dz": (
                    statistics.fmean(differences) / difference_std
                    if difference_std > 0.0
                    else math.nan
                ),
                "p_value": exact_sign_flip_pvalue(differences),
            }
        )
    adjusted = holm_adjust([float(row["p_value"]) for row in output])
    for row, p_adjusted in zip(output, adjusted):
        row["p_value_holm"] = p_adjusted
    return output


def compare_factorial_effects(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_samples: int = 10000,
    random_seed: int = 20260817,
) -> List[Dict[str, object]]:
    """Estimate the preregistered paired 2x2 contrasts by training seed."""
    values: Dict[Tuple[str, str], Dict[int, float]] = {}
    directions: Dict[str, str] = {}
    for row in rows:
        group = str(row["group"])
        metric = str(row["metric"])
        values.setdefault((group, metric), {})[
            int(row["training_seed"])
        ] = float(row["value"])
        directions[metric] = str(row["direction"])

    output: List[Dict[str, object]] = []
    for contrast_index, (contrast, coefficients) in enumerate(
        FACTORIAL_CONTRASTS.items()
    ):
        contrast_rows: List[Dict[str, object]] = []
        for metric_index, metric in enumerate(METRICS):
            seed_sets = [set(values.get((group, metric), {})) for group in coefficients]
            if not seed_sets or not seed_sets[0] or any(
                seeds != seed_sets[0] for seeds in seed_sets[1:]
            ):
                raise ValueError(
                    f"Unmatched training seeds for {contrast} and {metric}"
                )
            seeds = sorted(seed_sets[0])
            differences = [
                sum(
                    coefficient * values[(group, metric)][seed]
                    for group, coefficient in coefficients.items()
                )
                for seed in seeds
            ]
            low, high = bootstrap_mean_ci(
                differences,
                samples=bootstrap_samples,
                seed=random_seed + contrast_index * len(METRICS) + metric_index,
            )
            mean_difference = statistics.fmean(differences)
            difference_std = (
                statistics.stdev(differences) if len(differences) > 1 else 0.0
            )
            direction_multiplier = -1.0 if directions[metric] == "lower" else 1.0
            contrast_rows.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "direction": directions[metric],
                    "n_pairs": len(differences),
                    "mean_contrast": mean_difference,
                    "benefit_oriented_contrast": (
                        direction_multiplier * mean_difference
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_effect_dz": (
                        mean_difference / difference_std
                        if difference_std > 0.0
                        else math.nan
                    ),
                    "p_value": exact_sign_flip_pvalue(differences),
                }
            )
        adjusted = holm_adjust(
            [float(row["p_value"]) for row in contrast_rows]
        )
        for row, p_adjusted in zip(contrast_rows, adjusted):
            row["p_value_holm_within_contrast"] = p_adjusted
        output.extend(contrast_rows)
    return output


def bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> Tuple[float, float]:
    """Return a deterministic percentile bootstrap confidence interval."""
    if not values:
        raise ValueError("Cannot bootstrap an empty sample")
    if samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Two-sided exact paired randomization test under exchangeability."""
    if not differences:
        raise ValueError("Paired test requires at least one difference")
    observed = abs(statistics.fmean(differences))
    null_values = (
        abs(statistics.fmean(sign * value for sign, value in zip(signs, differences)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    )
    extreme = sum(value >= observed - 1e-15 for value in null_values)
    return extreme / (2 ** len(differences))


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Return Holm step-down family-wise-error adjusted p-values."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write a non-empty collection of dictionaries as UTF-8 CSV."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("Refusing to write an empty formal result table")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def _validate_evaluation(
    result: Mapping[str, object],
    *,
    held_out_seeds: Sequence[int],
    episodes_per_seed: int,
    source: Path,
) -> None:
    per_seed = result.get("seeds")
    if not isinstance(per_seed, list):
        raise ValueError(f"{source}: evaluation must contain a seeds list")
    observed = [int(_as_mapping(row, "seeds entry")["seed"]) for row in per_seed]
    if observed != list(held_out_seeds):
        raise ValueError(
            f"{source}: held-out seeds {observed} do not match {held_out_seeds}"
        )
    bad_counts = [
        int(row["seed"])
        for row in per_seed
        if int(row.get("episodes", -1)) != episodes_per_seed
    ]
    if bad_counts:
        raise ValueError(
            f"{source}: seeds {bad_counts} do not have {episodes_per_seed} episodes"
        )


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _as_mapping(parent.get(key), key)


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _integer_list(parent: Mapping[str, object], key: str) -> List[int]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list")
    return [int(item) for item in value]
