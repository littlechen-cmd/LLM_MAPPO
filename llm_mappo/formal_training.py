"""Provisional three-scene experiment contract and diagnostic limits."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from typing import Any, Mapping

from llm_mappo.formal_layout import FORMAL_MAP_SHA256, load_formal_layout


SCHEMA = "three-scene-mappo-provisional-v1"
TRAIN_SEEDS = tuple(range(7, 78, 10))
EVALUATION_SEEDS = tuple(
    1000 * base + index for base in range(200, 210) for index in range(20)
)
DURABILITY_SEEDS = tuple(range(400, 410))
DIAGNOSTIC_SEEDS = (9300, 9301, 9302)
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
