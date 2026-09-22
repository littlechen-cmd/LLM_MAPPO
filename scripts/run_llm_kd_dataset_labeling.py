"""Label one completed main 209D dataset with an approved LLM provider."""

from __future__ import annotations

import argparse
import json

from llm_mappo.mappo_llm_kd.config import (
    load_l1_config,
    verify_l1_approval,
)
from llm_mappo.mappo_llm_kd.dataset_labeling import run_dataset_labeling
from llm_mappo.mappo_llm_kd.provider import ProviderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True)
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--l1-config", default="configs/phase3/l1-label-audit.yaml")
    parser.add_argument(
        "--m1-config", default="configs/phase3/m1-llm-kd-diagnostics.yaml"
    )
    parser.add_argument("--m1-approval", default="configs/phase3/m1-approval.json")
    parser.add_argument("--l1-approval", default="configs/phase3/l1-approval.json")
    arguments = parser.parse_args()
    resolved = load_l1_config(
        arguments.l1_config, arguments.m1_config, arguments.m1_approval
    )
    verify_l1_approval(
        arguments.l1_config,
        arguments.m1_config,
        arguments.m1_approval,
        arguments.l1_approval,
    )
    models = {
        model["provider_id"]: model
        for model in resolved.values["models"]
    }
    if arguments.provider_id not in models:
        raise ValueError("provider-id is not present in the approved L1 config.")
    result = run_dataset_labeling(
        arguments.dataset_directory,
        provider_config=ProviderConfig(**models[arguments.provider_id]),
        label_config_sha256=resolved.sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
