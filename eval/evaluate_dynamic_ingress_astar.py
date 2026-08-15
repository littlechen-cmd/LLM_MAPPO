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
    try:
        for seed in seeds:
            records = []
            path_livelocks = 0
            state_deadlocks = 0
            for offset in range(episodes_per_seed):
                expert = AStarExpert()
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
                        path_livelocks += expert.path_livelocks
                        state_deadlocks += expert.state_deadlocks
                        break
            per_seed.append(
                _aggregate_seed(seed, records, path_livelocks, state_deadlocks)
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
    result = {
        "seeds": per_seed,
        "episodes": int(sum(record["episodes"] for record in per_seed)),
        "task_completion_rate": float(completion.mean()) if len(completion) else 0.0,
        "mean_collisions_per_episode": (
            float(collisions.mean()) if len(collisions) else 0.0
        ),
        "deadlock_rate": float(deadlocks.mean()) if len(deadlocks) else 0.0,
    }
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
    path_livelocks: int,
    state_deadlocks: int,
) -> dict:
    return {
        "seed": seed,
        "episodes": len(records),
        "task_completion_rate": _mean(records, "task_completion_rate"),
        "mean_collisions": _mean(records, "collisions"),
        "deadlock_rate": _mean(records, "deadlocked"),
        "success_rate": _mean(records, "success"),
        "path_livelocks": path_livelocks,
        "state_deadlocks": state_deadlocks,
    }


def _mean(records: list[dict], key: str) -> float:
    return float(np.mean([record[key] for record in records])) if records else 0.0


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes-per-seed", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_dynamic_astar(
        Phase3TrainingConfig.from_yaml(args.config),
        args.seeds,
        args.episodes_per_seed,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
