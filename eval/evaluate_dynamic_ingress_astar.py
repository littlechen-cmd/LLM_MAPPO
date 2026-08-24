"""Run the dynamic-ingress A* feasibility gate before Phase 3 MAPPO training."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase2_expert import AStarExpert
from llm_mappo.phase3_training import Phase3TrainingConfig
from rware.warehouse import Action


STALL_REASON_CODES = (
    "planner_noop",
    "planner_turn",
    "coordinator_vertex_yield",
    "coordinator_edge_swap_yield",
    "coordinator_occupied_yield",
    "action_mask_block",
    "environment_blocked_forward",
    "interaction_lock",
    "charging_wait",
    "dead_or_inactive",
    "unknown_stationary",
)
TERMINATION_REASONS = (
    "target_reached",
    "deadlock",
    "time_limit",
    "energy_failure",
)
PIPELINE_STAGES = (
    "planner_to_mask",
    "mask_to_coordinator",
    "coordinator_to_executed",
)


def evaluate_dynamic_astar(  # noqa: C901
    config: Phase3TrainingConfig,
    seeds: Iterable[int],
    episodes_per_seed: int = 20,
    legacy_terminal_reservation: bool = False,
    coordinator_yield_action: str = "noop",
    collect_stall_diagnostics: bool = False,
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
        include_waypoint_features=config.include_waypoint_features,
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
                    legacy_terminal_reservation=legacy_terminal_reservation,
                    coordinator_yield_action=coordinator_yield_action,
                )
                env.reset(seed=seed * 10_000 + offset)
                stall_counts = Counter()
                pipeline_counts = Counter()
                while True:
                    positions_before = [
                        (agent.x, agent.y) for agent in env.env.agents
                    ]
                    actions, _ = expert.act(env, env.action_masks())
                    transition = env.step(actions)
                    if collect_stall_diagnostics:
                        _record_step_diagnostics(
                            env,
                            expert.last_action_pipeline,
                            positions_before,
                            actions,
                            transition,
                            stall_counts,
                            pipeline_counts,
                        )
                    if (
                        transition.terminated
                        or transition.truncated
                        or transition.metrics.deadlocked
                    ):
                        record = transition.metrics.as_dict()
                        if collect_stall_diagnostics:
                            record["stall_diagnostics"] = {
                                "termination_reason": _termination_reason(transition),
                                "stationary_reason_counts": dict(stall_counts),
                                "pipeline_counts": dict(pipeline_counts),
                                "agent_steps": transition.info["step"] * env.n_agents,
                                "stationary_agent_steps": sum(stall_counts.values()),
                            }
                        records.append(record)
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
        "coordinator_yield_action": coordinator_yield_action,
        "stall_diagnostics_enabled": collect_stall_diagnostics,
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
        "stall_diagnostics": _aggregate_stall_diagnostics(records),
    }


def _record_step_diagnostics(  # noqa: C901
    env, pipeline, positions_before, actions, transition, stall_counts, pipeline_counts
) -> None:
    """Classify each stationary agent-step without changing controller behavior."""
    for index, detail in enumerate(pipeline):
        planner = Action(detail["planner_action"])
        masked = Action(detail["masked_action"])
        coordinated = Action(detail["coordinated_action"])
        agent = env.env.agents[index]
        executed = _executed_action(
            agent,
            positions_before[index],
            coordinated,
        )
        _record_pipeline_stage(pipeline_counts, "planner_to_mask", planner, masked)
        _record_pipeline_stage(
            pipeline_counts,
            "mask_to_coordinator",
            masked,
            coordinated,
        )
        _record_pipeline_stage(
            pipeline_counts,
            "coordinator_to_executed",
            coordinated,
            executed,
        )
        if positions_before[index] != (agent.x, agent.y):
            continue
        if agent.dead:
            reason = "dead_or_inactive"
        elif agent.picking_lock_steps:
            reason = "interaction_lock"
        elif detail["coordinator_reason"]:
            reason = detail["coordinator_reason"]
        elif detail["planner_masked"]:
            reason = "action_mask_block"
        elif planner == Action.NOOP:
            reason = "planner_noop"
        elif planner in {Action.LEFT, Action.RIGHT}:
            reason = "planner_turn"
        elif any(
            event["type"] == "blocked_forward" and event["agent_id"] == agent.id
            for event in transition.info["events"]
        ):
            reason = "environment_blocked_forward"
        elif (agent.x, agent.y) in env.env.charging_stations:
            reason = "charging_wait"
        else:
            reason = "unknown_stationary"
        stall_counts[reason] += 1


def _executed_action(agent, position_before, coordinated: Action) -> Action:
    if coordinated == Action.FORWARD and position_before == (agent.x, agent.y):
        return Action.NOOP
    if agent.req_action is None:
        return coordinated
    return Action(agent.req_action)


def _record_pipeline_stage(
    counts: Counter,
    stage: str,
    action_before: Action,
    action_after: Action,
) -> None:
    suffix = "unchanged" if action_before == action_after else "overrides"
    counts[f"{stage}_{suffix}"] += 1


def _termination_reason(transition) -> str:
    if transition.metrics.energy_deaths:
        return "energy_failure"
    if transition.metrics.deadlocked:
        return "deadlock"
    if transition.metrics.completed_tasks >= transition.metrics.task_completion_target:
        return "target_reached"
    return "time_limit"


def _aggregate_stall_diagnostics(records: list[dict]) -> dict:  # noqa: C901
    counts = Counter()
    terminations = Counter()
    pipeline_counts = Counter()
    agent_steps = 0
    stationary_agent_steps = 0
    records_with_diagnostics = 0
    for record in records:
        diagnostics = record.get("stall_diagnostics", {})
        if not diagnostics:
            continue
        records_with_diagnostics += 1
        counts.update(diagnostics.get("stationary_reason_counts", {}))
        pipeline_counts.update(diagnostics.get("pipeline_counts", {}))
        termination = diagnostics.get("termination_reason")
        if termination:
            terminations[termination] += 1
        agent_steps += diagnostics.get("agent_steps", 0)
        stationary_agent_steps += diagnostics.get(
            "stationary_agent_steps",
            sum(diagnostics.get("stationary_reason_counts", {}).values()),
        )
    if not records_with_diagnostics:
        return {
            "schema_version": 1,
            "available": False,
            "unavailable_reason": "stall_diagnostics_not_collected",
            "records_with_diagnostics": 0,
        }
    conservation_errors = []
    if sum(counts.values()) != stationary_agent_steps:
        conservation_errors.append("stationary_reason_total")
    if sum(terminations.values()) != records_with_diagnostics:
        conservation_errors.append("termination_reason_total")
    for reason in STALL_REASON_CODES:
        counts.setdefault(reason, 0)
    for reason in TERMINATION_REASONS:
        terminations.setdefault(reason, 0)
    for stage in PIPELINE_STAGES:
        pipeline_counts.setdefault(f"{stage}_overrides", 0)
        pipeline_counts.setdefault(f"{stage}_unchanged", 0)
    for stage in PIPELINE_STAGES:
        stage_total = (
            pipeline_counts[f"{stage}_overrides"]
            + pipeline_counts[f"{stage}_unchanged"]
        )
        if records_with_diagnostics and stage_total != agent_steps:
            conservation_errors.append(stage)
    return {
        "schema_version": 1,
        "available": records_with_diagnostics > 0,
        "records_with_diagnostics": records_with_diagnostics,
        "stationary_reason_counts": dict(counts),
        "termination_reason_counts": dict(terminations),
        "pipeline_counts": dict(pipeline_counts),
        "agent_steps": agent_steps,
        "stationary_agent_steps": stationary_agent_steps,
        "stationary_reason_rates": {
            key: value / agent_steps if agent_steps else 0.0
            for key, value in counts.items()
        },
        "conservation_errors": conservation_errors,
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
    parser.add_argument(
        "--coordinator-yield-action", choices=("noop", "right"), default="noop"
    )
    parser.add_argument("--stall-diagnostics", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_dynamic_astar(
        Phase3TrainingConfig.from_yaml(args.config),
        args.seeds,
        args.episodes_per_seed,
        legacy_terminal_reservation=args.reservation_mode == "legacy",
        coordinator_yield_action=args.coordinator_yield_action,
        collect_stall_diagnostics=args.stall_diagnostics,
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
