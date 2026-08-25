"""CLI entry point for the O1 functional smoke only."""

import argparse

from llm_mappo.optimization_training import (
    OptimizationTrainingConfig,
    train_optimization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen O1 functional smoke.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = OptimizationTrainingConfig.from_yaml(args.config)
    summary = train_optimization(config, args.output)
    print(summary)


if __name__ == "__main__":
    main()
