"""Run the dynamic-ingress A* feasibility gate before Phase 3 MAPPO training."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase2_expert import AStarExpert
from llm_mappo.phase3_training import Phase3TrainingConfig


def evaluate_dynamic_astar(
    config: Phase3TrainingConfig,
    seeds: Iterable[int],
    episodes_per_seed: int = 20,
    legacy_terminal_reservation: bool = False,
) -> dict:
    """Evaluate time-reserved A* in the exact dynamic Phase 3 environment."""
    if config.batch_interval is None:
        raise ValueError("Dynamic A* evaluation requires batch_interval.")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive.")
    env = Phase2Warehouse(
        n_agents=config.n_agents,
        max_steps=config.max_steps,
        env_id=config.env_id,
        charge_threshold=config.charge_threshold,
        charge_release_threshold=config.charge_release_threshold,
        battery_cost_scale=config.battery_cost_scale,
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
    per_seed = []
    all_planning_times_ms = []
    try:
        for seed in seeds:
            records = []
            expert_totals = _empty_expert_totals()
            planning_times_ms = []
            for offset in range(episodes_per_seed):
                expert = AStarExpert(
                    legacy_terminal_reservation=legacy_terminal_reservation
                )
                env.reset(seed=seed * 10_000 + offset)
                while True:
                    actions, _ = expert.act(env, env.action_masks())
                    transition = env.step(actions)
                    if (
                        transition.terminated
                        or transition.truncated
                        or transition.metrics.deadlocked
                    ):
                        records.append(transition.metrics.as_dict())
                        _accumulate_expert(expert_totals, expert.statistics())
                        planning_times_ms.extend(expert.planning_times_ms)
                        all_planning_times_ms.extend(expert.planning_times_ms)
                        break
            per_seed.append(
                _aggregate_seed(
                    seed, records, expert_totals, planning_times_ms
                )
            )
    finally:
        env.close()
    completion = np.asarray(
        [record["task_completion_rate"] for record in per_seed], dtype=float
    )
    collisions = np.asarray(
        [record["mean_collisions"] for record in per_seed], dtype=float
    )
    deadlocks = np.asarray([record["deadlock_rate"] for record in per_seed], dtype=float)
    all_expert_totals = _empty_expert_totals()
    for record in per_seed:
        _accumulate_expert(all_expert_totals, record["reservation_teacher"])
    result = {
        "reservation_mode": (
            "legacy_horizon" if legacy_terminal_reservation else "bounded_2_step"
        ),
        "seeds": per_seed,
        "episodes": int(sum(record["episodes"] for record in per_seed)),
        "task_completion_rate": float(completion.mean()) if len(completion) else 0.0,
        "mean_collisions_per_episode": (
            float(collisions.mean()) if len(collisions) else 0.0
        ),
        "deadlock_rate": float(deadlocks.mean()) if len(deadlocks) else 0.0,
        "mean_energy_deaths_per_episode": float(
            np.mean(
                [record["mean_energy_deaths"] for record in per_seed]
            )
        ) if per_seed else 0.0,
        "reservation_teacher": all_expert_totals,
    }
    result["reservation_teacher"]["planning_time_ms_p95"] = (
        float(np.percentile(all_planning_times_ms, 95))
        if all_planning_times_ms
        else 0.0
    )
    result["gate"] = {
        "completion_minimum": 0.95,
        "collisions_maximum": 2.0,
        "deadlock_rate_maximum": 0.05,
        "passed": (
            result["task_completion_rate"] >= 0.95
            and result["mean_collisions_per_episode"] <= 2.0
            and result["deadlock_rate"] <= 0.05
        ),
    }
    return result


def _aggregate_seed(
    seed: int,
    records: list[dict],
    expert_totals: dict,
    planning_times_ms: list[float],
) -> dict:
    expert_summary = dict(expert_totals)
    expert_summary["planning_time_ms_p95"] = (
        float(np.percentile(planning_times_ms, 95))
        if planning_times_ms
        else 0.0
    )
    return {
        "seed": seed,
        "episodes": len(records),
        "task_completion_rate": _mean(records, "task_completion_rate"),
        "mean_collisions": _mean(records, "collisions"),
        "deadlock_rate": _mean(records, "deadlocked"),
        "success_rate": _mean(records, "success"),
        "mean_energy_deaths": _mean(records, "energy_deaths"),
        "path_livelocks": expert_totals["path_livelocks"],
        "state_deadlocks": expert_totals["state_deadlocks"],
        "reservation_teacher": expert_summary,
    }


def _empty_expert_totals() -> dict:
    return {
        key: 0
        for key in (
            "path_livelocks",
            "state_deadlocks",
            "cache_hits",
            "cache_misses",
            "reached_goal_plans",
            "partial_paths",
            "terminal_conflicts",
            "reservation_false_no_paths",
            "explicit_waits",
            "replans",
            "expanded_nodes",
            "planning_time_count",
            "planning_time_ms_total",
        )
    }


def _accumulate_expert(target: dict, source: dict) -> None:
    for key in target:
        target[key] += source.get(key, 0)


def _mean(records: list[dict], key: str) -> float:
    return float(np.mean([record[key] for record in records])) if records else 0.0


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument(
        "--reservation-mode", choices=("fixed", "legacy"), default="fixed"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_dynamic_astar(
        Phase3TrainingConfig.from_yaml(args.config),
        args.seeds,
        args.episodes_per_seed,
        legacy_terminal_reservation=args.reservation_mode == "legacy",
    )
    result["config"] = str(Path(args.config))
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
