"""Run a short or full Phase 3 Encoder+GRU semantic-pretraining job."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.pretraining_run import (
    load_pretraining_run_config,
    run_semantic_pretraining,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-id")
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    result = run_semantic_pretraining(
        load_pretraining_run_config(arguments.config),
        device=arguments.device,
        run_id=arguments.run_id,
        resume=arguments.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
