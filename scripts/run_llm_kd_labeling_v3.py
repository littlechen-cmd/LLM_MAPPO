"""Prepare a frozen v3 plan, export quality evidence, or explicitly execute."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_mappo.mappo_llm_kd.labeling_run_v3 import (  # noqa: E402
    execute_labeling_v3, prepare_labeling_v3, write_quality_report_v3,
)
from llm_mappo.mappo_llm_kd.provider import ProviderConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Freeze a public plan; sends no requests.")
    prepare.add_argument("--config", required=True, type=Path)
    quality = sub.add_parser("quality", help="Evaluate saved labels without an API.")
    quality.add_argument("--plan-directory", required=True, type=Path)
    quality.add_argument("--stage", choices=("trial", "main"), required=True)
    quality.add_argument("--blind-review", required=True, type=Path)
    quality.add_argument("--output", required=True, type=Path)
    execute = sub.add_parser("execute", help="Requires specific outbound authorization.")
    execute.add_argument("--plan-directory", required=True, type=Path)
    execute.add_argument("--stage", choices=("trial", "main"), required=True)
    execute.add_argument("--execute", action="store_true")
    execute.add_argument("--authorization", required=True, type=Path)
    execute.add_argument("--trial-quality", type=Path)
    execute.add_argument("--main-quality", type=Path)
    execute.add_argument("--blind-review", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        providers = {role: ProviderConfig(**value)
                     for role, value in config.pop("providers").items()}
        result = prepare_labeling_v3(providers=providers, **config)
    elif args.command == "quality":
        report = write_quality_report_v3(
            args.plan_directory, stage=args.stage, blind_review_path=args.blind_review,
            output_path=args.output,
        )
        result = {"status": report["status"], "stage": report["stage"]}
    else:
        result = execute_labeling_v3(
            args.plan_directory, stage=args.stage, execute=args.execute,
            authorization_path=args.authorization, trial_quality_path=args.trial_quality,
            main_quality_path=args.main_quality, blind_review_path=args.blind_review,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
