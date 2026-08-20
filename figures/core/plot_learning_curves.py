"""Plot matched-seed learning curves from formal training logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GROUP_ORDER = (
    "MAPPO-WP",
    "MAPPO-WP+A*KD",
    "MAPPO-WP+LLMKD",
    "MAPPO-WP+A*KD+LLMKD",
)
COLORS = ("#4477AA", "#EE6677", "#228833", "#CCBB44")
PANELS = (
    ("task_completion_rate", "Task completion rate"),
    ("reward", "Team reward per episode"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="configs/g3_experiment_manifest.yaml",
    )
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument(
        "--output-stem",
        default="figures/core/core_learning_curves",
    )
    return parser.parse_args()


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("rolling_window must be positive")
    if len(values) < window:
        raise ValueError(
            f"A run has {len(values)} episodes, fewer than window {window}"
        )
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def load_curves(manifest_path: str | Path, window: int):
    manifest_path = Path(manifest_path)
    root = manifest_path.resolve().parents[1]
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    seeds = manifest["training"]["policy_initialization_seeds"]
    curves = {}
    missing = []
    for group_name in GROUP_ORDER:
        group = manifest["core_groups"][group_name]
        config_path = root / group["config"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        run_root = root / config["training"]["output_dir"]
        group_curves = []
        for seed in seeds:
            source = run_root / f"seed_{int(seed):03d}" / "episodes.csv"
            if not source.is_file():
                missing.append(str(source))
                continue
            rows = _read_csv(source)
            required = {"environment_steps", *(metric for metric, _ in PANELS)}
            absent = required.difference(rows[0] if rows else {})
            if absent:
                raise ValueError(f"{source}: missing columns {sorted(absent)}")
            step_values = np.asarray(
                [float(row["environment_steps"]) for row in rows], dtype=float
            )
            x = step_values[window - 1:]
            metrics = {
                metric: _rolling(
                    np.asarray([float(row[metric]) for row in rows]), window
                )
                for metric, _ in PANELS
            }
            group_curves.append((x, metrics))
        curves[group_name] = group_curves
    if missing:
        preview = "\n".join(missing[:8])
        raise FileNotFoundError(f"Formal training matrix is incomplete:\n{preview}")
    return curves


def plot(curves, output_stem: str | Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for axis, (metric, label) in zip(axes, PANELS):
        for group_name, color in zip(GROUP_ORDER, COLORS):
            runs = curves[group_name]
            start = max(run[0][0] for run in runs)
            stop = min(run[0][-1] for run in runs)
            if stop <= start:
                raise ValueError(f"No shared step range for {group_name}")
            grid = np.linspace(start, stop, 200)
            values = np.stack(
                [np.interp(grid, x, metrics[metric]) for x, metrics in runs]
            )
            mean = values.mean(axis=0)
            standard_deviation = values.std(axis=0, ddof=1)
            axis.plot(grid, mean, color=color, linewidth=1.7, label=group_name)
            axis.fill_between(
                grid,
                mean - standard_deviation,
                mean + standard_deviation,
                color=color,
                alpha=0.14,
                linewidth=0,
            )
        axis.set_xlabel("Environment steps")
        axis.set_ylabel(label)
        axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    output = Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    curves = load_curves(args.manifest, args.rolling_window)
    plot(curves, args.output_stem)
    print(f"Wrote {args.output_stem}.png and {args.output_stem}.svg")


if __name__ == "__main__":
    main()
