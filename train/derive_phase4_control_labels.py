"""Derive frozen RuleKD and ShuffleKD caches from the offline LLM cache."""

from __future__ import annotations

from argparse import ArgumentParser
import json

from llm_mappo.semantic_controls import derive_control_datasets


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--rule-output", required=True)
    parser.add_argument("--shuffle-output", required=True)
    parser.add_argument("--shuffle-seed", type=int, default=20260820)
    args = parser.parse_args()
    print(json.dumps(derive_control_datasets(
        args.source, args.rule_output, args.shuffle_output, args.shuffle_seed
    ), indent=2))


if __name__ == "__main__":
    main()
