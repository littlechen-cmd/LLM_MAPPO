"""Select 6000 bounded, episode-disjoint windows from medium-5AG episodes."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.medium_dataset_run import select_medium5_windows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--source-run-manifest", required=True)
    arguments = parser.parse_args()
    result = select_medium5_windows(
        arguments.run_directory,
        source_run_manifest=arguments.source_run_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
