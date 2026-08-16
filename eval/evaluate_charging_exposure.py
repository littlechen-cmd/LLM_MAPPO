"""Calibrate energy pressure without retraining or changing A* behavior."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from llm_mappo.phase3_training import evaluate_phase3, load_phase3_policy


def _parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Evaluate one checkpoint under matched battery-cost scales. "
            "Results are G2 diagnostics, not formal policy comparisons."
        )
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.25, 1.5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--charge-threshold", type=float, default=0.2)
    parser.add_argument("--charge-release-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-exposure-rate", type=float, default=0.5)
    parser.add_argument("--minimum-completion-rate", type=float, default=0.95)
    return parser


def _validate_args(args) -> None:
    if any(scale <= 0.0 for scale in args.scales):
        raise ValueError("All battery-cost scales must be positive.")
    if not args.seeds:
        raise ValueError("Provide at least one seed.")
    if args.episodes_per_seed < 1:
        raise ValueError("episodes-per-seed must be positive.")
    if not 0.0 < args.charge_threshold < args.charge_release_threshold <= 1.0:
        raise ValueError("Charge thresholds must satisfy 0 < enter < release <= 1.")
    if not 0.0 <= args.minimum_exposure_rate <= 1.0:
        raise ValueError("minimum-exposure-rate must be within [0, 1].")
    if not 0.0 <= args.minimum_completion_rate <= 1.0:
        raise ValueError("minimum-completion-rate must be within [0, 1].")


def _passes_calibration(result: dict, args) -> bool:
    return bool(
        result["episodes_with_low_battery_rate"] >= args.minimum_exposure_rate
        and result["episodes_with_charging_rate"] >= args.minimum_exposure_rate
        and result["task_completion_rate"] >= args.minimum_completion_rate
        and result["mean_energy_deaths_per_episode"] == 0.0
    )


def run_calibration(checkpoint_path: str, args) -> dict:
    policy, base_config, checkpoint = load_phase3_policy(checkpoint_path)
    candidates = []
    for scale in sorted(set(args.scales)):
        config = dict(base_config)
        config["battery_cost_scale"] = scale
        config["charge_threshold"] = args.charge_threshold
        config["charge_release_threshold"] = args.charge_release_threshold
        result = evaluate_phase3(
            policy,
            config,
            seeds=args.seeds,
            episodes_per_seed=args.episodes_per_seed,
        )
        result["battery_cost_scale"] = scale
        result["passes_calibration"] = _passes_calibration(result, args)
        candidates.append(result)
    passing = [item for item in candidates if item["passes_calibration"]]
    return {
        "gate": "G2-charging-exposure-calibration",
        "checkpoint": str(Path(checkpoint_path)),
        "trained_episodes": int(checkpoint["episodes"]),
        "seeds": list(args.seeds),
        "episodes_per_seed": args.episodes_per_seed,
        "charge_threshold": args.charge_threshold,
        "charge_release_threshold": args.charge_release_threshold,
        "criteria": {
            "minimum_episode_low_battery_rate": args.minimum_exposure_rate,
            "minimum_episode_charging_rate": args.minimum_exposure_rate,
            "minimum_task_completion_rate": args.minimum_completion_rate,
            "maximum_mean_energy_deaths_per_episode": 0.0,
        },
        "selected_scale": passing[0]["battery_cost_scale"] if passing else None,
        "candidates": candidates,
        "interpretation": (
            "Diagnostic distribution-shift pilot only; retrain matched groups "
            "before making teacher-effect claims."
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    result = run_calibration(args.checkpoint, args)
    text = json.dumps(result, indent=2)
    print(text)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
