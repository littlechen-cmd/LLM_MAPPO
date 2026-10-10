"""Run or resume the owner-approved Phase 3 two-model label audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from llm_mappo.mappo_llm_kd.audit import run_label_audit
    from llm_mappo.mappo_llm_kd.config import (
        load_l1_config,
        verify_l1_approval,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/phase3/l1-label-audit.yaml"
    )
    parser.add_argument(
        "--m1-config", default="configs/phase3/m1-llm-kd-diagnostics.yaml"
    )
    parser.add_argument(
        "--m1-approval", default="configs/phase3/m1-approval.json"
    )
    parser.add_argument(
        "--l1-approval", default="configs/phase3/l1-approval.json"
    )
    parser.add_argument("--provider-id", action="append", dest="provider_ids")
    parser.add_argument("--mode", required=True, choices=("smoke", "audit"))
    args = parser.parse_args()
    verify_l1_approval(
        args.config, args.m1_config, args.m1_approval, args.l1_approval
    )
    resolved = load_l1_config(args.config, args.m1_config, args.m1_approval)
    receipt = run_label_audit(
        resolved, mode=args.mode, provider_ids=args.provider_ids
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
