"""Run the bounded medium-5ag LLM dataset smoke, pilot, or full stage."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.medium_labeling import (
    MEDIUM_LABEL_CONFIG,
    run_medium_dataset_labeling,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True)
    parser.add_argument("--stage", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--config", default=MEDIUM_LABEL_CONFIG)
    parser.add_argument(
        "--prior-budget-directory",
        help="Required for v4; exact frozen v3 label output directory.",
    )
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Full stage in-flight requests, 1 to 32 (default 4).")
    parser.add_argument(
        "--resume-transient-network", action="store_true",
        help="Root-authorized one retry per observable view for audited network loss.",
    )
    parser.add_argument(
        "--pricing-checked-utc-date",
        help="Today's UTC date after checking official DeepSeek model and pricing page.",
    )
    parser.add_argument(
        "--citation-repair-authorization",
        help="Owner-approved bound artifact for one extra fourth-citation request.",
    )
    parser.add_argument(
        "--citation-repair-authorization-sha256",
        help="Exact SHA-256 of the owner-approved citation repair artifact.",
    )
    options = parser.parse_args()
    receipt = run_medium_dataset_labeling(
        options.dataset_directory,
        stage=options.stage,
        config_path=options.config,
        prior_budget_directory=options.prior_budget_directory,
        pricing_checked_utc_date=options.pricing_checked_utc_date,
        concurrency=options.concurrency,
        resume_transient_network=options.resume_transient_network,
        citation_repair_authorization=options.citation_repair_authorization,
        citation_repair_authorization_sha256=(
            options.citation_repair_authorization_sha256
        ),
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
