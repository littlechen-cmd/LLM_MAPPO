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
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Phase2TrainingConfig.from_yaml(args.config)
    for name in ("seed", "n_agents", "episodes", "output_dir"):
        value = getattr(args, name if name != "n_agents" else "agents")
        if value is not None:
            setattr(config, name, value)
    print(json.dumps(train_phase2(config), indent=2))


if __name__ == "__main__":
    main()
