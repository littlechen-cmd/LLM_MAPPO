"""Evaluate Phase 3 policy behavior groups without injecting special states."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from llm_mappo.behavior_evaluation import evaluate_behavior_groups
from llm_mappo.phase3_training import load_phase3_policy


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    policy, config, checkpoint = load_phase3_policy(args.checkpoint)
    result = evaluate_behavior_groups(
        policy, config, args.seeds, args.episodes_per_seed
    )
    result["checkpoint"] = str(Path(args.checkpoint))
    result["trained_episodes"] = checkpoint["episodes"]
    text = json.dumps(result, indent=2)
    print(text)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
