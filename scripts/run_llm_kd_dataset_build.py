"""Build one resumable, native-209D main LLM-KD dataset artifact."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.dataset_run import (
    load_dataset_build_config,
    run_dataset_build,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    config = load_dataset_build_config(arguments.config)
    result = run_dataset_build(
        config,
        source_checkpoint=arguments.source_checkpoint,
        device=arguments.device,
        output_root=arguments.output_root,
        run_id=arguments.run_id,
        resume=arguments.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
