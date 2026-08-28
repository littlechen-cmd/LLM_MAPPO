"""Read-only P1 server preflight with one-shot and bounded-wait modes."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import yaml

from llm_mappo.linux_server_runtime import (
    MachineSnapshot,
    PreflightResult,
    ServerPolicy,
    _default_command_runner,
    collect_machine_snapshot,
    evaluate_preflight,
    wait_for_resources,
    write_new_atomic_json,
)


def _artifact_path(output: Path, prefix: str, suffix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output / (prefix + "_" + timestamp + suffix)


def _load_policy(path: Path) -> ServerPolicy:
    return ServerPolicy.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))


def _default_collector() -> MachineSnapshot:
    return collect_machine_snapshot(
        _default_command_runner,
        {"repo": Path.cwd()},
    )


def _write_wait_sample(stream, result: PreflightResult) -> None:
    stream.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    stream.flush()


def run(
    arguments: Optional[Sequence[str]] = None,
    *,
    collector: Callable[[], MachineSnapshot] = _default_collector,
) -> int:
    """Run a read-only one-shot or bounded-wait P1 preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--once", action="store_true")
    modes.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/optimization/p1_linux_server"),
    )
    args = parser.parse_args(arguments)
    policy = _load_policy(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.once:
        try:
            result = evaluate_preflight(collector(), policy)
        except Exception:
            result = PreflightResult(False, ("inventory_collection_error",))
    else:
        wait_log = _artifact_path(args.output, "wait", ".jsonl")
        with wait_log.open("x", encoding="utf-8") as stream:
            result = wait_for_resources(
                sample=collector,
                policy=policy,
                clock=time.monotonic,
                sink=lambda item: _write_wait_sample(stream, item),
            )

    report_path = _artifact_path(args.output, "preflight", ".json")
    write_new_atomic_json(report_path, result.to_dict())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 2


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
