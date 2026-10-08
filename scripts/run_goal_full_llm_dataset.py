"""Collect or select v3 goal-full data with separate source and rollout identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_mappo.mappo_llm_kd.goal_full_dataset_run import (  # noqa: E402
    run_goal_full_collection, select_goal_full_windows,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="Append complete bound episodes")
    collect.add_argument("--source-config", required=True,
                         help="Original approved C1 seed-7 YAML; never rollout YAML")
    collect.add_argument("--source-checkpoint", required=True)
    collect.add_argument("--source-manifest")
    collect.add_argument("--actor-artifact", required=True)
    collect.add_argument("--actor-sha256", required=True)
    collect.add_argument("--rollout-config", required=True,
                         help="Separately approved medium goal-full YAML")
    collect.add_argument("--output-root",
                         default="artifacts/llm_kd/phase_20261007/dataset")
    collect.add_argument("--episode-count", type=int, default=120,
                         help="Collect prefix of approved 120 episodes; default 120")
    collect.add_argument("--seed-start", type=int, default=10000)
    collect.add_argument("--resume", action="store_true")
    select = commands.add_parser("select", help="Freeze windows and report raw coverage")
    select.add_argument("--run-directory", required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    result = (run_goal_full_collection(**arguments) if command == "collect"
              else select_goal_full_windows(arguments["run_directory"]))
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
