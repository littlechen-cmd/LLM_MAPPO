"""Run one owner-authorized provisional Group 2 reward diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_mappo.mappo_baseline.runner import BaselineRun
from llm_mappo.reward_diagnostics import (
    load_reward_diagnostic_experiment,
    resolve_diagnostic_baseline_experiment,
    validate_diagnostic_invocation,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume")
    parser.add_argument(
        "--acknowledge-provisional-reward-diagnostic",
        action="store_true",
    )
    return parser, parser.parse_args()


def main() -> None:
    parser, arguments = _arguments()
    experiment = load_reward_diagnostic_experiment(arguments.config)
    try:
        validate_diagnostic_invocation(
            experiment,
            seed=arguments.seed,
            output_root=arguments.output_root,
            acknowledged=arguments.acknowledge_provisional_reward_diagnostic,
        )
    except ValueError as error:
        parser.error(str(error))
    baseline = resolve_diagnostic_baseline_experiment(experiment, seed=arguments.seed)
    run_arguments = {
        "device": arguments.device,
        "output_root": arguments.output_root,
        "resume": arguments.resume,
        "diagnostic_authorized": True,
        "diagnostic_reward_definition": experiment.reward_definition,
        "diagnostic_reward_definition_sha256": experiment.reward_definition_sha256,
    }
    if arguments.resume is None:
        run_arguments["run_directory_name"] = f"seed_{arguments.seed:04d}"
    run = BaselineRun(baseline, **run_arguments)
    print(json.dumps(run.train(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
