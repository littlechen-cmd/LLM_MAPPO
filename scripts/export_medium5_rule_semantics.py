"""Export the same-window deterministic semantic comparator without an API."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.rule_semantic import export_rule_semantic_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = export_rule_semantic_labels(args.source_root, args.output_root)
    print(json.dumps({
        "run_directory": result["run_directory"],
        "label_counts": result["manifest"]["label_counts"],
        "labels_sha256": result["manifest"]["labels_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
