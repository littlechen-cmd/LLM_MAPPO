"""Evaluate a Phase 2 MAPPO checkpoint against the ten-seed gate."""

from argparse import ArgumentParser
import json
from pathlib import Path

from llm_mappo.phase2_training import evaluate_policy, load_policy


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    policy, config, checkpoint = load_policy(args.checkpoint)
    result = evaluate_policy(
        policy,
        n_agents=config["n_agents"],
        max_steps=config["max_steps"],
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
    )
    result["checkpoint"] = str(Path(args.checkpoint))
    result["trained_episodes"] = checkpoint["episodes"]
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
