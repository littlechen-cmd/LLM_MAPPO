"""Create a cached Phase 4 semantic engagement dataset before training."""

from argparse import ArgumentParser
import json

import yaml

from llm_mappo.llm_teacher import DeepSeekTeacher, MockTeacher
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase3_training import Phase3TrainingConfig
from llm_mappo.phase4 import collect_offline_labels, collect_stratified_offline_labels


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_llm_distillation.yaml")
    parser.add_argument("--provider", choices=("mock", "deepseek"), default="mock")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--scenarios-per-seed", type=int, default=100)
    parser.add_argument(
        "--quota-per-type",
        type=int,
        help="Override every stratified quota; use 5 for the required v2 pilot.",
    )
    parser.add_argument(
        "--natural-only",
        action="store_true",
        help="Use natural A* rollouts only; not valid for the formal Phase 4 dataset.",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default=None)
    parser.add_argument(
        "--partial-output",
        default=None,
        help="Resumable JSONL checkpoint; defaults to '<output>.partial.jsonl'.",
    )
    args = parser.parse_args()
    config = Phase3TrainingConfig.from_yaml(args.config)
    with open(args.config, "r", encoding="utf-8") as stream:
        source = yaml.safe_load(stream) or {}
    provider = (
        MockTeacher()
        if args.provider == "mock"
        else DeepSeekTeacher(
            model=args.model,
            timeout_seconds=args.timeout_seconds,
            max_attempts=args.max_attempts,
            max_tokens=args.max_tokens,
            thinking_enabled=args.thinking,
            reasoning_effort=args.reasoning_effort,
        )
    )
    env = Phase2Warehouse(
        n_agents=config.n_agents,
        max_steps=config.max_steps,
        env_id=config.env_id,
        charge_threshold=config.charge_threshold,
        waypoint_reward=config.waypoint_reward,
        oracle_interaction_mask=config.oracle_interaction_mask,
        deadlock_steps=config.deadlock_steps,
        priority_schedule=config.priority_schedule,
        batch_interval=config.batch_interval,
        batch_size_range=config.batch_size_range,
        initial_priority_label=config.initial_priority_label,
        request_queue_size=config.request_queue_size,
        task_completion_target=config.task_completion_target,
        include_priority_features=True,
    )
    try:
        if args.natural_only:
            result = collect_offline_labels(
                env,
                provider,
                args.output,
                args.seeds,
                args.scenarios_per_seed,
                checkpoint_path=args.partial_output,
            )
        else:
            quotas = source.get("label_sampling")
            if not quotas:
                raise ValueError("The Phase 4 config must define label_sampling quotas.")
            if args.quota_per_type is not None:
                if args.quota_per_type < 1:
                    raise ValueError("--quota-per-type must be positive.")
                quotas = {name: args.quota_per_type for name in quotas}
            result = collect_stratified_offline_labels(
                env,
                provider,
                args.output,
                args.seeds,
                quotas,
                checkpoint_path=args.partial_output,
            )
    finally:
        env.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
