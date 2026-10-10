"""Select an immutable coordination-v2 derivative from frozen C1 episodes."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.coordination_dataset import (
    select_coordination_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = select_coordination_dataset(args.source_root, args.output_root)
    print(json.dumps({
        "run_directory": result["run_directory"],
        "windows_sha256": result["manifest"]["windows_sha256"],
        "selected_coordination_coverage": result["manifest"][
            "selected_coordination_coverage"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
