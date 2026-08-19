"""Validate and aggregate the preregistered four-group evaluation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_mappo.formal_results import (
    collect_seed_rows,
    compare_factorial_effects,
    compare_full_to_baseline,
    load_manifest,
    summarize_groups,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="configs/g3_experiment_manifest.yaml",
    )
    parser.add_argument(
        "--evaluation-root",
        default="artifacts/formal_evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/formal_aggregate",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    rows = collect_seed_rows(manifest, args.evaluation_root)
    summaries = summarize_groups(
        rows,
        bootstrap_samples=args.bootstrap_samples,
    )
    comparisons = compare_full_to_baseline(
        rows,
        bootstrap_samples=args.bootstrap_samples,
    )
    factorial_effects = compare_factorial_effects(
        rows,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.output_dir)
    write_csv(output / "per_training_seed.csv", rows)
    write_csv(output / "group_summary.csv", summaries)
    write_csv(output / "paired_full_vs_baseline.csv", comparisons)
    write_csv(output / "paired_factorial_effects.csv", factorial_effects)
    print(
        f"Validated {len(rows)} metric rows and wrote formal tables to {output}"
    )


if __name__ == "__main__":
    main()
