"""Evaluate the frozen Heuristic-Dispatcher+A* non-learning baseline."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from evaluate_dynamic_ingress_astar import evaluate_dynamic_astar
from llm_mappo.phase3_training import Phase3TrainingConfig


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_dynamic_astar(
        Phase3TrainingConfig.from_yaml(args.config), args.seeds,
        args.episodes_per_seed,
    )
    result["algorithm"] = "Heuristic-Dispatcher+A*"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
