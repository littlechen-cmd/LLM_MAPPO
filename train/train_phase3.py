"""Command-line entry point for Phase 3a dual-head MAPPO."""

from argparse import ArgumentParser
import json

from llm_mappo.phase3_training import Phase3TrainingConfig, train_phase3


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/phase3a_dual_head.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = Phase3TrainingConfig.from_yaml(args.config)
    for name in ("seed", "episodes", "output_dir"):
        value = getattr(args, name.replace("-", "_"), None)
        if value is not None:
            setattr(config, name.replace("-", "_"), value)
    print(json.dumps(train_phase3(config), indent=2))


if __name__ == "__main__":
    main()
