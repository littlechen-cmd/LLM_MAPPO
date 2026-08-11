"""Render Phase 3 figures from a completed or partial training run.

Usage (from the repository root)::

    python scripts/plot_phase3.py --run-dir artifacts/phase3a_dual_head/seed_007
    python scripts/plot_phase3.py --run-dir ... --output-dir figures/phase3a
    python scripts/plot_phase3.py --run-dir ... --eval-json artifacts/phase3a_eval.json
    python scripts/plot_phase3.py --run-dir ... \
        --engagement-csv artifacts/engagement_samples.csv

The script reads ``episodes.csv``, ``updates.csv`` and ``summary.json`` from the
run directory and writes PNG files to ``figures/`` by default.  TensorBoard
already provides the live-training view; this script is for offline reports.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from llm_mappo.plotting import (
    EvalResult,
    render_all_figures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing episodes.csv / updates.csv / summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Where to write PNG files (default: figures)",
    )
    parser.add_argument(
        "--eval-json",
        default=None,
        help="Optional JSON written by eval/evaluate_phase3.py for comparison plots",
    )
    parser.add_argument(
        "--engagement-csv",
        default=None,
        help="Optional CSV with (label, engagement) rows for the diagnostic plot",
    )
    return parser.parse_args()


def _load_eval_result(path: str) -> Optional[EvalResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalResult(
        label="Phase 3a",
        completion=float(data.get("task_completion_rate", 0.0)),
        collisions=float(data.get("mean_collisions_per_episode", 0.0)),
        deadlock=float(data.get("deadlock_rate", 0.0)),
    )


def _load_engagement_samples(path: str) -> List[Tuple[str, float]]:
    samples: List[Tuple[str, float]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            label = row.get("label", "none")
            value = float(row.get("engagement", 0.0))
            samples.append((label, value))
    return samples


def main() -> None:
    args = parse_args()
    eval_result: Optional[EvalResult] = None
    if args.eval_json:
        eval_result = _load_eval_result(args.eval_json)
    engagement_samples: Optional[Sequence[Tuple[str, float]]] = None
    if args.engagement_csv:
        engagement_samples = _load_engagement_samples(args.engagement_csv)
    written = render_all_figures(
        args.run_dir,
        args.output_dir,
        phase2_comparison=eval_result,
        engagement_samples=engagement_samples,
    )
    print(f"Wrote {len(written)} figures to {Path(args.output_dir).resolve()}:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
