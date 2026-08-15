"""Re-label selected Phase 4 records into a new, resumable JSONL artifact."""

from argparse import ArgumentParser
import json
from pathlib import Path

from llm_mappo.llm_teacher import DeepSeekTeacher, MockTeacher
from llm_mappo.phase4 import repair_offline_labels


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--scenario-ids-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partial-output", default=None)
    parser.add_argument("--provider", choices=("mock", "deepseek"), default="mock")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("high", "max"), default=None)
    args = parser.parse_args()

    scenario_ids = Path(args.scenario_ids_file).read_text(
        encoding="utf-8"
    ).splitlines()
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
    result = repair_offline_labels(
        args.source,
        args.output,
        provider,
        scenario_ids,
        checkpoint_path=args.partial_output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
