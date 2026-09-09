"""Evaluate a completed Phase 1 MAPPO-FP checkpoint without further training."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    experiment = load_baseline_experiment(arguments.config)
    run = BaselineRun(
        experiment,
        device=arguments.device,
        output_root=arguments.output_root,
        evaluation_only=True,
    )
    print(
        json.dumps(
            run.evaluate_checkpoint(arguments.checkpoint), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
