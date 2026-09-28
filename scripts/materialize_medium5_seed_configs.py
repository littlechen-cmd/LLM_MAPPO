"""Materialize immutable seed variants of one reviewed medium-5AG C1 config."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_mappo.mappo_baseline.runner import load_baseline_experiment
from llm_mappo.medium5_contract import CORE_MEDIUM5_C1_SCHEMA, TRAINING_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    arguments = parser.parse_args()

    source = Path(arguments.config).resolve()
    template = load_baseline_experiment(source)
    if template.schema != CORE_MEDIUM5_C1_SCHEMA:
        parser.error("Only the medium-5AG C1 contract can be materialized.")
    output = Path(arguments.output_directory).resolve()
    if output.exists():
        parser.error(
            "Output directory already exists; reviewed configs cannot be overwritten."
        )

    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for seed in TRAINING_SEEDS:
        variant = deepcopy(values)
        variant["training"]["seed"] = seed
        target = output / f"{source.stem}-seed_{seed:04d}.yaml"
        target.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        checked = load_baseline_experiment(target)
        manifest.append({
            "path": str(target),
            "seed": seed,
            "approval_status": checked.approval_status,
            "config_sha256": checked.config_sha256,
        })
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
