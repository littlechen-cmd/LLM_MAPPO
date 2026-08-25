"""Owner-only semantic label collection entry point.

O1 exposes dry-run validation only. Pilot/formal requests require a separately
approved owner-run workflow and are never imported by optimization training.
"""

import argparse
import json
from pathlib import Path

from llm_mappo.semantic_v3 import SemanticRecordV3


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate O0 semantic label records."
    )
    parser.add_argument("--dry-run-record", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    content = arguments.dry_run_record.read_text(encoding="utf-8")
    record = SemanticRecordV3.parse_response(content)
    payload = {
        "semantic_view_version": "semantic-view-v3",
        "validity": record.validity,
        "scores": record.scores.tolist(),
        "network_calls": 0,
    }
    if arguments.output is not None:
        arguments.output.write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
