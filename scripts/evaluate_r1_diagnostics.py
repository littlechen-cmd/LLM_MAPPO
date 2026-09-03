"""Regenerate deterministic R1-C evaluation and replays without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_mappo.e1_training import load_e1_raw_semantic_evidence
from llm_mappo.r1_diagnostics import load_r1c_diagnostic
from llm_mappo.r1_evaluation import evaluate_r1c_checkpoint


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/optimization/r1_4agv_lowload.yaml")
    parser.add_argument("--records", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = _arguments()
    directory = Path(args.run_dir)
    manifest = json.loads(
        (directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("phase") != "R1-C":
        raise ValueError("The requested directory is not an R1-C artifact.")
    diagnostic = load_r1c_diagnostic(args.config, str(identity.get("arm")))
    if identity.get("environment") != dict(diagnostic.environment):
        raise ValueError("R1-C replay environment profile is incompatible.")
    if identity.get("training") != dict(diagnostic.training):
        raise ValueError("R1-C replay training profile is incompatible.")
    labels = load_e1_raw_semantic_evidence(args.records)
    if identity.get("raw_records_sha256") != labels.records_sha256:
        raise ValueError("R1-C replay raw-label identity is incompatible.")
    result = evaluate_r1c_checkpoint(
        directory=directory, environment=diagnostic.environment,
        run=diagnostic.run, dataset=labels.dataset, identity=identity,
        seeds=diagnostic.evaluation_seeds, device=args.device,
    )
    print(json.dumps({"run_directory": str(directory), "evaluation": result},
                     sort_keys=True))


if __name__ == "__main__":
    main()
