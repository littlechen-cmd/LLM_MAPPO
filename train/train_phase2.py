"""Command-line entry point for the custom Phase 2 CTDE MAPPO baseline."""

from argparse import ArgumentParser
import json

from llm_mappo.phase2_training import Phase2TrainingConfig, train_phase2


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_mappo.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--agents", type=int, choices=(1, 3))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--waypoint-reward", type=float)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Phase2TrainingConfig.from_yaml(args.config)
    overrides = {
        "seed": args.seed,
        "n_agents": args.agents,
        "episodes": args.episodes,
        "waypoint_reward": args.waypoint_reward,
        "output_dir": args.output_dir,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    print(json.dumps(train_phase2(config), indent=2))


if __name__ == "__main__":
    main()
