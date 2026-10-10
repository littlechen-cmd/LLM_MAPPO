"""Collect approved medium-5AG C1 episodes for the LLM dataset."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.medium_dataset_run import (
    run_medium5_episode_collection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--actor-artifact", required=True)
    parser.add_argument("--actor-sha256", required=True)
    parser.add_argument("--output-root", default="artifacts/llm_kd/dataset/medium-5ag")
    parser.add_argument("--episode-count", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    result = run_medium5_episode_collection(
        baseline_config=arguments.baseline_config,
        source_checkpoint=arguments.source_checkpoint,
        actor_artifact=arguments.actor_artifact,
        actor_sha256=arguments.actor_sha256,
        output_root=arguments.output_root,
        episode_count=arguments.episode_count,
        resume=arguments.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
