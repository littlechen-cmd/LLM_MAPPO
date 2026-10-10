"""Select an immutable train-only extension; no collection or API calls."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_mappo.mappo_llm_kd.medium_extension_dataset import (  # noqa: E402
    select_medium_train_extension, verify_medium_extension_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pool")
    parser.add_argument("--batch", action="append", default=[])
    parser.add_argument("--output-root")
    parser.add_argument("--verify", help="Verify an existing immutable derived version")
    parser.add_argument("--artifact-root", help="Relocated repository root")
    args = parser.parse_args()
    if args.verify:
        frozen = verify_medium_extension_dataset(
            args.verify, require_frozen=False, artifact_root=args.artifact_root)
        print(json.dumps(dict(verified=True, raw_selection_frozen=frozen is not None)))
        return
    if (not args.parent_pool or not args.batch or not args.output_root
            or args.artifact_root):
        parser.error("Selection requires --parent-pool, --batch and --output-root.")
    result = select_medium_train_extension(
        args.parent_pool, args.batch, args.output_root)
    print(json.dumps(dict(run_directory=result["run_directory"],
                          raw_eligible=result["manifest"]["main_pretraining_eligible"],
                          coverage_missing=result["manifest"]["coverage_missing"])))


if __name__ == "__main__":
    main()
