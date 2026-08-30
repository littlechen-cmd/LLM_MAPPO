"""Run the short deterministic Fixed/RC calibration-chain parity smoke."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from llm_mappo.o2_contract import O2ExperimentConfig, O2RunSpec
from llm_mappo.o2_training import O2Trainer


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def run(arguments: argparse.Namespace) -> dict:
    if arguments.steps < 1 or arguments.steps > 512:
        raise ValueError("Parity smoke steps must be in [1, 512].")
    config = O2ExperimentConfig.from_yaml(arguments.config)
    run_spec = O2RunSpec("RC-AStarKD", 107, config.real_env_steps)
    fixed = O2Trainer(
        experiment=config,
        run=run_spec,
        device=arguments.device,
        calibration_weight_mode="fixed",
    )
    try:
        fixed_result = fixed.run(max_steps=arguments.steps)
    finally:
        fixed.environment.close()
        fixed.student_shadow.close()
        fixed.teacher_shadow.close()
    calibrated = O2Trainer(
        experiment=config,
        run=run_spec,
        device=arguments.device,
        calibration_weight_mode="reward-calibrated",
    )
    try:
        calibrated_result = calibrated.run(max_steps=arguments.steps)
    finally:
        calibrated.environment.close()
        calibrated.student_shadow.close()
        calibrated.teacher_shadow.close()
    fields = ("teacher_queries", "calibration_selected", "shadow_calls", "ema_updates")
    mismatches = {
        name: [fixed_result[name], calibrated_result[name]]
        for name in fields
        if fixed_result[name] != calibrated_result[name]
    }
    selected_any = fixed_result["calibration_selected"] > 0
    result = {
        "diagnostic_only": True,
        "steps": arguments.steps,
        "fixed": fixed_result,
        "reward_calibrated": calibrated_result,
        "mismatches": mismatches,
        "selected_calibration_observed": selected_any,
        "parity_pass": not mismatches and selected_any,
    }
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    result = run(_arguments(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["parity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
