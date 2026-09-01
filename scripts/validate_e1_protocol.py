"""Validate the frozen E1 governance manifest and emit its formal run matrix."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from llm_mappo.e1_protocol import (
    expand_e1_formal_matrix,
    load_e1_governance_manifest,
    validate_e1_governance_manifest,
)


def run(manifest_path: str | Path) -> dict:
    manifest = load_e1_governance_manifest(manifest_path)
    validate_e1_governance_manifest(manifest)
    runs = expand_e1_formal_matrix(manifest)
    o3 = manifest["evaluation"]["o3_exploratory_matrix"]
    return {
        "schema": "e1-protocol-validation-v1",
        "gate_pass": True,
        "formal_run_count": len(runs),
        "formal_run_identities": [item.identity for item in runs],
        "o3_exploratory_total_episodes": o3["total_episodes"],
        "o3_confirmatory": o3["confirmatory"],
        "runs": [asdict(item) | {"identity": item.identity} for item in runs],
    }


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _arguments(argv)
    payload = run(arguments.manifest)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
