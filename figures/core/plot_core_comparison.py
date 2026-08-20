"""Plot preregistered core metrics from the real-data aggregate table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GROUP_ORDER = (
    "MAPPO-WP",
    "MAPPO-WP+A*KD",
    "MAPPO-WP+LLMKD",
    "MAPPO-WP+A*KD+LLMKD",
)
METRICS = (
    ("task_completion_rate", "Task completion rate"),
    ("completed_tasks_per_1000_steps", "Tasks per 1,000 steps"),
    ("mean_reward_per_episode", "Team reward per episode"),
    ("mean_collisions_per_episode", "Collisions per episode"),
)
COLORS = ("#4477AA", "#EE6677", "#228833", "#CCBB44")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="artifacts/formal_aggregate/group_summary.csv",
    )
    parser.add_argument(
        "--output-stem",
        default="figures/core/core_comparison",
    )
    return parser.parse_args()


def load_rows(path: str | Path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot(rows, output_stem: str | Path) -> None:
    lookup = {(row["group"], row["metric"]): row for row in rows}
    missing = [
        (group, metric)
        for group in GROUP_ORDER
        for metric, _ in METRICS
        if (group, metric) not in lookup
    ]
    if missing:
        raise ValueError(f"Aggregate table is incomplete; missing {missing}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    short_names = ("MAPPO-WP", "+A*KD", "+LLMKD", "+Both")
    for axis, (metric, label) in zip(axes.flat, METRICS):
        metric_rows = [lookup[(group, metric)] for group in GROUP_ORDER]
        means = [float(row["mean"]) for row in metric_rows]
        lower = [mean - float(row["ci95_low"]) for mean, row in zip(means, metric_rows)]
        upper = [float(row["ci95_high"]) - mean for mean, row in zip(means, metric_rows)]
        axis.bar(
            range(len(GROUP_ORDER)),
            means,
            yerr=[lower, upper],
            color=COLORS,
            edgecolor="#222222",
            linewidth=0.6,
            capsize=3,
        )
        axis.set_xticks(range(len(GROUP_ORDER)), short_names, rotation=18, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.set_axisbelow(True)
    output = Path(output_stem)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=450, bbox_inches="tight")
    figure.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plot(load_rows(args.input), args.output_stem)
    print(f"Wrote {args.output_stem}.png and {args.output_stem}.svg")


if __name__ == "__main__":
    main()
