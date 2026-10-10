"""Authorize one reviewed retry of an unknown v5 label response."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.medium_labeling import (
    authorize_medium_unknown_retry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--original-attempt-log-sha256", required=True)
    parser.add_argument(
        "--config",
        default="configs/llm_kd/medium-5ag-dataset-labeling-v5.yaml",
    )
    options = parser.parse_args()
    receipt = authorize_medium_unknown_retry(
        options.dataset_directory,
        options.attempt_id,
        options.original_attempt_log_sha256,
        config_path=options.config,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
