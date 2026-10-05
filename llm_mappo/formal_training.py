"""Provisional three-scene experiment contract and diagnostic limits."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from llm_mappo.formal_layout import FORMAL_MAP_SHA256, load_formal_layout


SCHEMA = "three-scene-mappo-provisional-v1"
TRAIN_SEEDS = tuple(range(7, 78, 10))
EVALUATION_SEEDS = tuple(
    1000 * base + index for base in range(200, 210) for index in range(20)
)
DURABILITY_SEEDS = tuple(range(400, 410))
DIAGNOSTIC_SEEDS = (9300, 9301, 9302)
DURABILITY_COUNTER_FIELDS = (
    "completed_tasks", "created_tasks", "picked_tasks", "energy_deaths",
    "charger_arrivals", "charged_events", "task_recoveries", "collisions",
    "blocked_forwards",
)
METHODS = (
    "MAPPO-FP-DG",
    "MAPPO-FP-ASTARKD-NO-VALIDITY",
    "MAPPO-FP-DG-LLM-INIT",
    "MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT",
)

_formal_layout, FORMAL_PICKING, FORMAL_CHARGING = load_formal_layout()
SCENES = {
    "tiny-3ag": {
        "environment_id": "llm-mappo-tiny-3ag-v1",
        "n_agents": 3,
        "n_shelves": 32,
        "max_steps": 800,
        "charging_stations": ((0, 0), (9, 0), (9, 10)),
        "picking_stations": ((4, 10), (5, 10)),
    },
    "medium-5ag": {
        "environment_id": "llm-mappo-medium-3ag-v1",
        "n_agents": 5,
        "n_shelves": 144,
        "max_steps": 800,
        "charging_stations": ((0, 0), (19, 0), (0, 23), (19, 23), (9, 0)),
        "picking_stations": ((9, 21), (10, 21)),
    },
    "formal-8ag": {
        "environment_id": "llm-mappo-formal-8ag-v1",
        "n_agents": 8,
        "n_shelves": 504,
        "max_steps": 1200,
        "charging_stations": FORMAL_CHARGING,
        "picking_stations": FORMAL_PICKING,
    },
}

COMMON_ENVIRONMENT = {
    "battery_cost_scale": 1.10,
    "charge_threshold": 0.30,
    "charge_release_threshold": 0.80,
    "deadlock_steps": 180,
    "observation_schema": "direct-goal-observation-v2",
    "goal_interface": "direct_goal",
    "reward_version": "reward-v2",
    "reward_energy_death_aggregation": "local-mean",
    "reward_blocked_forward_penalty": 0.10,
    "reward_collision_penalty": 0.10,
    "charge_event_reward": 0.0,
    "reward_deadlock_penalty": 5.0,
    "oracle_interaction_mask": True,
}
COMMON_TRAINING = {
    "num_env_workers": 16,
    "rollout_length": 64,
    "ppo_epochs": 5,
    "num_minibatches": 1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_param": 0.20,
    "actor_learning_rate": 0.0003,
    "critic_learning_rate": 0.0003,
    "entropy_coefficient": 0.01,
    "value_loss_coefficient": 0.50,
    "max_gradient_norm": 0.50,
    "huber_delta": 10.0,
    "hidden_dimension": 64,
}

SCENE_TRAINING_BUDGETS = {
    "tiny-3ag": 450_000,
    "medium-5ag": 900_000,
    "formal-8ag": 1_800_000,
}


def authenticate_scene_representation(values: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate actual representation bytes without advancing training RNG."""

    import torch

    from llm_mappo.mappo_baseline.networks import MAPPOActor
    from llm_mappo.mappo_llm_kd.initialization import (
        initialize_final_actor, load_representation_initialization,
    )

    initialization = load_representation_initialization(values)
    artifact = Path(initialization.artifact_path)
    try:
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("Representation artifact could not be safely read.") from error
    artifact_identity = payload.get("identity") if isinstance(payload, Mapping) else None
    if json.dumps(artifact_identity) != json.dumps(initialization.identity):
        raise ValueError("Representation identity does not match expected identity.")
    canonical = {**values, "identity": artifact_identity}
    initialization = load_representation_initialization(canonical)
    with torch.random.fork_rng(devices=[]):
        actor = MAPPOActor(209, 5, 64)
        initialize_final_actor(actor, initialization)
    return canonical


def validate_approved_scene_artifacts(method: str, training: dict[str, Any]) -> None:
    """Allow training only with implemented teacher and real LLM representation."""

    if "ASTARKD" in method:
        from llm_mappo.mappo_astar_kd.teacher import TEACHER_VERSION

        if (training.get("astar_teacher_approval") != "approved"
                or training.get("astar_teacher_version") != TEACHER_VERSION):
            raise ValueError("Approved scene A* teacher identity is incompatible.")
    if method.endswith("LLM-INIT"):
        if training.get("representation_initialization") is None:
            raise ValueError(
                "Approved LLM scene requires a real representation artifact."
            )
        training["representation_initialization"] = authenticate_scene_representation(
            training["representation_initialization"]
        )


def make_durability_experiment(experiment):
    """Derive the formal durability evaluation without changing source identity."""

    if (experiment.schema != SCHEMA
            or experiment.environment["scene_id"] != "formal-8ag"):
        raise ValueError("Durability protocol requires the formal-8ag scene.")
    return replace(experiment, evaluation={
        **experiment.evaluation,
        "seeds": list(DURABILITY_SEEDS),
        "durability_max_steps": 5000,
        "environment_overrides": {"max_steps": 5000},
    })


def build_durability_windows(snapshots: list[Mapping[str, Any]]) -> list[dict]:
    """Difference boundary snapshots; unobserved windows are missing, not zero."""

    windows = []
    actual_end = int(snapshots[-1]["steps"])
    by_step = {int(snapshot["steps"]): snapshot for snapshot in snapshots}
    for start in range(0, 5000, 1000):
        end = min(start + 1000, actual_end)
        actual_steps = max(0, end - start)
        row = {
            "start_step": start, "planned_end_step": start + 1000,
            "actual_end_step": end if actual_steps else None,
            "actual_steps": actual_steps,
            "status": (
                "complete" if actual_steps == 1000
                else "partial" if actual_steps else "not_run"
            ),
        }
        for field in DURABILITY_COUNTER_FIELDS:
            row[field] = (
                int(by_step[end].get(field, 0)) - int(by_step[start].get(field, 0))
                if actual_steps else None
            )
        row["completed_tasks_per_1000_actual_steps"] = (
            1000.0 * row["completed_tasks"] / actual_steps if actual_steps else None
        )
        windows.append(row)
    return windows


def validate_scene_training_budget(scene_id: str, training: Mapping[str, Any]) -> None:
    """Keep all four methods on the owner's selected scene budget and schedule."""

    budget = SCENE_TRAINING_BUDGETS[scene_id]
    expected = {
        "environment_steps": budget,
        "optimization_schedule_steps": budget,
        "post_schedule_learning_rate": 0.0,
        "checkpoint_interval_steps": 25_000,
        "milestone_checkpoint_steps": [budget // 2, budget],
    }
    for field, value in expected.items():
        if training.get(field) != value:
            raise ValueError(f"Three-scene {field} differs from selected budget.")


def validate_provisional_environment(  # noqa: C901
    values: Mapping[str, Any]
) -> dict[str, Any]:
    """Check exact scene identity, then supply inert legacy constructor fields."""

    forbidden = {
        "task_target", "dynamic_ingress_interval", "batch_size_range",
        "initial_ingress_batches", "terminate_on_task_target",
    }
    if forbidden & values.keys():
        raise ValueError("Provisional scene contains legacy task or ingress fields.")
    scene_id = values.get("scene_id")
    if scene_id not in SCENES:
        raise ValueError("Unknown provisional warehouse scene.")
    scene = SCENES[scene_id]
    for field in ("environment_id", "n_agents", "max_steps"):
        if values.get(field) != scene[field]:
            raise ValueError(f"Provisional scene {field} differs from approved map.")
    if values.get("initial_active_tasks") != scene["n_agents"]:
        raise ValueError("Initial active tasks must equal the AGV count.")
    for field in ("charging_stations", "picking_stations"):
        coordinates = tuple(tuple(point) for point in values.get(field, ()))
        if coordinates != scene[field] or len(coordinates) != len(set(coordinates)):
            raise ValueError(f"Provisional scene {field} differs from approved map.")
    if scene_id == "formal-8ag" and values.get("layout_sha256") != (
        FORMAL_MAP_SHA256
    ):
        raise ValueError("Formal map identity differs from approved JSON.")
    if values.get("delivery_contract") != "balanced-random-v1":
        raise ValueError("Provisional scene requires balanced-random-v1.")
    if values.get("training_replenishment_mode") != "rware_delivery":
        raise ValueError("Provisional scene requires rware_delivery.")
    if values.get("priority_schedule") != ["A", "B", "C"]:
        raise ValueError("Provisional priority order differs.")
    if values.get("initial_priority_label") != "A":
        raise ValueError("Provisional initial priority differs.")
    for field, expected in COMMON_ENVIRONMENT.items():
        if values.get(field) != expected:
            raise ValueError(f"Provisional scene {field} differs across methods.")
    expanded = dict(values)
    expanded.update({
        "dynamic_ingress_interval": scene["max_steps"] + 1,
        "batch_size_range": [scene["n_agents"], scene["n_agents"]],
        "initial_ingress_batches": 1,
        "queue_size": scene["n_agents"],
        "task_target": scene["n_agents"] * 3,
        "terminate_on_task_target": False,
    })
    return expanded


def make_diagnostic_experiment(experiment, *, environment_steps: int):
    """Build one explicitly bounded baseline diagnostic from a provisional scene."""

    if (experiment.schema != SCHEMA or experiment.method != "MAPPO-FP-DG"
            or experiment.training["seed"] != 7
            or not 1 <= environment_steps <= 50_000):
        raise ValueError("Only seed-7 baseline diagnostics up to 50k are allowed.")
    scene = experiment.environment["scene_id"]
    derived = replace(
        experiment,
        artifact_root=(
            "artifacts/formal_environment_training/diagnostics/"
            f"{scene}/seed_0007"
        ),
        training={
            **experiment.training,
            "environment_steps": environment_steps,
            "optimization_schedule_steps": environment_steps,
            "milestone_checkpoint_steps": [],
        },
        evaluation={
            **experiment.evaluation,
            "seeds": list(DIAGNOSTIC_SEEDS),
            "render_replay": False,
        },
    )
    identity = {
        "source_config_sha256": experiment.config_sha256,
        "schema": derived.schema,
        "method": derived.method,
        "artifact_root": derived.artifact_root,
        "environment": derived.environment,
        "training": derived.training,
        "evaluation": derived.evaluation,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return replace(derived, config_sha256=sha256(encoded.encode("utf-8")).hexdigest())
