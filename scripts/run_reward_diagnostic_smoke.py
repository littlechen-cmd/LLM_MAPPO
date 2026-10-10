"""Run the non-comparable CPU lifecycle probe for Group 2 diagnostics."""

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
    make_local_smoke_experiment,
)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/reward_diagnostics/medium-5ag-reward-v2.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/reward_diagnostics/group2/local_smoke",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.device != "cpu":
        raise ValueError("The reward diagnostic smoke is CPU-only.")
    diagnostic = load_reward_diagnostic_experiment(arguments.config)
    experiment = make_local_smoke_experiment(diagnostic)
    run = BaselineRun(
        experiment,
        device="cpu",
        output_root=arguments.output_root,
        diagnostic_authorized=True,
        diagnostic_reward_definition=diagnostic.reward_definition,
        diagnostic_reward_definition_sha256=diagnostic.reward_definition_sha256,
    )
    print(json.dumps(run.train(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
