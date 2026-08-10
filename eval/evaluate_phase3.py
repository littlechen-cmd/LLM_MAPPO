"""Evaluate a Phase 3a checkpoint on deterministic seeds."""

from argparse import ArgumentParser
import json
from pathlib import Path

from llm_mappo.phase3_training import evaluate_phase3, load_phase3_policy


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--output")
    args = parser.parse_args()
    policy, config, checkpoint = load_phase3_policy(args.checkpoint)
    result = evaluate_phase3(policy, config, args.seeds, args.episodes_per_seed)
    result["checkpoint"] = str(Path(args.checkpoint))
    result["trained_episodes"] = checkpoint["episodes"]
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
