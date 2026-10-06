"""Record reviewed evidence that a sandbox-blocked label request was never sent."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.medium_labeling import (
    MEDIUM_LABEL_CONFIG,
    reconcile_medium_presend_attempt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--config", default=MEDIUM_LABEL_CONFIG)
    options = parser.parse_args()
    receipt = reconcile_medium_presend_attempt(
        options.dataset_directory, options.evidence,
        config_path=options.config,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
