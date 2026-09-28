"""Run an explicitly confirmed, bounded seed-7 baseline scene diagnostic."""

from __future__ import annotations

import argparse
import json

from llm_mappo.formal_training import make_diagnostic_experiment
from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment-steps", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confirm-short-diagnostic", action="store_true")
    arguments = parser.parse_args()
    if not arguments.confirm_short_diagnostic:
        parser.error("Pass --confirm-short-diagnostic to run the bounded diagnostic.")
    source = load_baseline_experiment(arguments.config)
    experiment = make_diagnostic_experiment(
        source, environment_steps=arguments.environment_steps
    )
    run = BaselineRun(
        experiment, device=arguments.device, diagnostic_authorized=True
    )
    print(json.dumps(run.train(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
