"""Aggregate the receipted six-run O2 calibration Go/No-Go evidence."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from llm_mappo.o2_contract import O2ExperimentConfig, expand_o2_matrix


_GRID = list(range(0, 150001, 10000))


def normalized_throughput_auc(rows: list[Mapping[str, Any]]) -> float:
    """Integrate throughput over the pre-registered 0..150k normalized grid."""
    ordered = sorted(rows, key=lambda row: int(row["real_env_steps"]))
    if [int(row["real_env_steps"]) for row in ordered] != _GRID:
        raise ValueError("Throughput grid must exactly equal 0..150000 by 10000.")
    values = [float(row["throughput"]) for row in ordered]
    if not all(value >= 0.0 for value in values):
        raise ValueError("Throughput values must be finite non-negative numbers.")
    return sum(
        (values[index] + values[index + 1]) * 0.5 * (_GRID[index + 1] - _GRID[index])
        / 150000.0
        for index in range(len(values) - 1)
    )


def aggregate_o2(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the per-seed coverage and paired median-AUC O2 criteria."""
    rc = {int(run["seed"]): run for run in runs if run["group"] == "RC-AStarKD"}
    baseline = {int(run["seed"]): run for run in runs if run["group"] == "MAPPO-DG"}
    reasons: list[str] = []
    coverage = {seed: item.get("coverage") for seed, item in rc.items()}
    if any(value is None or float(value) < 0.25 for value in coverage.values()):
        reasons.append("coverage")
    seeds = sorted(set(rc) & set(baseline))
    if not seeds:
        reasons.append("paired_runs_missing")
    degradations: dict[int, float] = {}
    for seed in seeds:
        denominator = float(baseline[seed]["auc"])
        if denominator <= 0.0:
            reasons.append("baseline_auc_nonpositive")
            continue
        degradations[seed] = (denominator - float(rc[seed]["auc"])) / denominator
    median_degradation = median(degradations.values()) if degradations else None
    if median_degradation is None or median_degradation > 0.10:
        reasons.append("median_auc_degradation")
    return {
        "gate_pass": not reasons,
        "reasons": sorted(set(reasons)),
        "rc_coverage": coverage,
        "paired_relative_degradation": degradations,
        "median_relative_degradation": median_degradation,
    }


def _read_run(directory: Path, config: O2ExperimentConfig) -> dict[str, Any]:
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("diagnostic_only"):
        raise ValueError("Diagnostic O2 artifact cannot enter the formal Gate.")
    identity = manifest.get("identity", {})
    if identity.get("config_sha256") != config.sha256():
        raise ValueError(
            "O2 run configuration hash does not match the frozen contract."
        )
    grid_path = directory / "throughput_grid.csv"
    with grid_path.open(newline="", encoding="utf-8") as handle:
        auc = normalized_throughput_auc(list(csv.DictReader(handle)))
    group, seed = identity.get("group"), identity.get("seed")
    coverage = None
    if group == "RC-AStarKD":
        denominator = int(summary.get("calibration_selected_agent_slots", 0))
        numerator = int(summary.get("valid_teacher_selected_slots", 0))
        coverage = None if denominator == 0 else numerator / denominator
    return {"group": group, "seed": seed, "coverage": coverage, "auc": auc}


def analyze(
    config: O2ExperimentConfig, runs_root: str | Path
) -> dict[str, Any]:
    root = Path(runs_root)
    expected = {(item.group, item.seed) for item in expand_o2_matrix(config)}
    runs = []
    errors = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        try:
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            if state.get("status") != "complete":
                raise ValueError("run is not complete")
            runs.append(_read_run(directory, config))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"{directory.name}:{type(error).__name__}")
    actual = {(run["group"], run["seed"]) for run in runs}
    result = aggregate_o2(runs)
    if actual != expected:
        result["gate_pass"] = False
        result["reasons"] = sorted(set(result["reasons"] + ["formal_matrix_missing"]))
    if errors:
        result["gate_pass"] = False
        result["reasons"] = sorted(set(result["reasons"] + ["corrupt_run"]))
    result["runs"] = runs
    result["errors"] = errors
    return result


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    result = analyze(
        O2ExperimentConfig.from_yaml(arguments.config), arguments.runs_root
    )
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
