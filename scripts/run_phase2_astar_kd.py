"""Run one owner-approved Phase 2 MAPPO-FP-A*KD experiment."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_astar_kd.runner import AStarKDRun, load_astar_kd_experiment


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root")
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    experiment = load_astar_kd_experiment(arguments.config)
    run = AStarKDRun(
        experiment,
        device=arguments.device,
        output_root=arguments.output_root,
        resume=arguments.resume,
    )
    print(json.dumps(run.train(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
