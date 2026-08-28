"""Owner-started wait-to-O1 launcher for the shared Linux optimization server."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import yaml

from llm_mappo.linux_server_runtime import (
    PreflightResult,
    ServerPolicy,
    _default_command_runner,
    collect_machine_snapshot,
    gpu_lease,
    wait_for_resources,
    write_new_atomic_json,
)


CANONICAL_LINUX_PYTHON = "/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python"
GPU_LEASE_PATH = Path("/tmp/llm-a-mappo-optimization-gpu-0.lock")


def build_child_environment(base: Mapping[str, str]) -> Mapping[str, str]:
    """Set the physical GPU visibility before the benchmark child imports Torch."""
    return {**base, "CUDA_VISIBLE_DEVICES": "0"}


def build_gate_command(
    *,
    python_executable: str,
    gate_config: Path,
    preflight_report: Path,
    environment_report: Path,
    output: Path,
    resume: Optional[Path] = None,
) -> Sequence[str]:
    """Build a no-shell normal Gate command; this function never names O2."""
    command = [
        python_executable,
        "scripts/benchmark_reward_calibration.py",
        "gate",
        "--config",
        str(gate_config),
        "--preflight-report",
        str(preflight_report),
        "--environment-report",
        str(environment_report),
        "--modes",
        "baseline",
        "h12",
        "--workers",
        "12",
        "--repeats",
        "5",
        "--warmup-vector-steps",
        "16",
        "--measure-vector-steps",
        "128",
        "--memory-warmup-windows",
        "2",
        "--memory-measure-windows",
        "10",
        "--output",
        str(output),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def route_gate_result(
    child_returncode: int, summary_path: Path, receipt_path: Path
) -> Mapping[str, str]:
    """Return the next phase without invoking it."""
    if child_returncode != 0:
        return {"next_required_phase": "O0", "reason": "benchmark_child_nonzero"}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"next_required_phase": "O0", "reason": "missing_gate_receipt"}
    if summary.get("gate_pass") and receipt.get("next_required_phase") == "O2":
        return {"next_required_phase": "O2", "reason": "o1_gate_go"}
    return {"next_required_phase": "O0", "reason": "o1_gate_no_go"}


def _load_policy(path: Path) -> ServerPolicy:
    return ServerPolicy.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))


def _run_id() -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short=8", "HEAD"], text=True
    ).strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return timestamp + "_" + commit


def _write_wait_sample(stream, result: PreflightResult) -> None:
    stream.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
    stream.flush()


def _ensure_canonical_interpreter() -> None:
    if Path(sys.executable) != Path(CANONICAL_LINUX_PYTHON):
        raise RuntimeError("Run this launcher with the P1 canonical Linux interpreter.")


def run(
    arguments: Optional[Sequence[str]] = None,
    *,
    collector: Optional[Callable[[], object]] = None,
    process_runner: Callable = subprocess.run,
    lease_factory: Callable = gpu_lease,
    canonical_check: Callable[[], None] = _ensure_canonical_interpreter,
    run_id_factory: Callable[[], str] = _run_id,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, str]:
    """Wait, run normal O1 once, and report the required next phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(arguments)
    canonical_check()
    policy = _load_policy(args.server_config)
    run_id = run_id_factory()
    p1_root = args.output_root / "p1_linux_server"
    p1_root.mkdir(parents=True, exist_ok=True)
    environment_report = p1_root / "environment_report.json"
    if not environment_report.is_file():
        raise FileNotFoundError("P1 environment report is required before O1 wait.")
    gate_output = args.resume or args.output_root / "o1_cuda_gate" / run_id
    preflight_report = p1_root / ("preflight_" + run_id + ".json")
    wait_log = p1_root / ("wait_" + run_id + ".jsonl")

    if collector is None:
        def collector():
            return collect_machine_snapshot(
                _default_command_runner, {"repo": Path.cwd()}
            )
    with lease_factory(GPU_LEASE_PATH):
        with wait_log.open("x", encoding="utf-8") as stream:
            preflight = wait_for_resources(
                sample=collector,
                policy=policy,
                clock=time.monotonic,
                sink=lambda item: _write_wait_sample(stream, item),
                sleep=sleep,
            )
        write_new_atomic_json(preflight_report, preflight.to_dict())
        if not preflight.passed:
            outcome = {"next_required_phase": "O0", "reason": "preflight_no_go"}
        else:
            command = build_gate_command(
                python_executable=sys.executable,
                gate_config=args.gate_config,
                preflight_report=preflight_report,
                environment_report=environment_report,
                output=gate_output,
                resume=args.resume,
            )
            child = process_runner(
                command,
                cwd=Path.cwd(),
                env=build_child_environment(os.environ),
                check=False,
            )
            outcome = route_gate_result(
                child.returncode,
                gate_output / "summary.json",
                gate_output / "o1_gate_receipt.json",
            )
            if child.returncode in {130, 143}:
                raise SystemExit(child.returncode)

    write_new_atomic_json(
        p1_root / ("o1_launcher_" + run_id + ".json"),
        {
            "run_id": run_id,
            "preflight_report": str(preflight_report),
            "gate_output": str(gate_output),
            **outcome,
        },
    )
    return outcome


def main() -> int:
    outcome = run()
    print(json.dumps(outcome, indent=2, sort_keys=True))
    return 0 if outcome["next_required_phase"] == "O2" else 2


if __name__ == "__main__":
    raise SystemExit(main())
