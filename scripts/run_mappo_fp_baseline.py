"""Train the frozen teacher-free MAPPO-FP-DG capability baseline."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/optimization/r1_mappo_fp_dg_3ag_lowload.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root")
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    experiment = load_baseline_experiment(arguments.config)
    run = BaselineRun(
        experiment,
        device=arguments.device,
        output_root=arguments.output_root,
        resume=arguments.resume,
    )
    print(json.dumps(run.train(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
