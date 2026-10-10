"""Frozen medium-5AG C1 contract, separate from historical delivery-v2 runs."""

from hashlib import sha256
import json
from typing import Any, Mapping


CORE_MEDIUM5_C1_SCHEMA = "core-mappo-209d-delivery-c1-v3"
COMPLETION_METRIC_CONTRACT = "task-target-v1"
TRAINING_SEEDS = (7, 17, 27)

REWARD_DEFINITION = {
    "version": "reward-v3-c1-energy-death-local-mean",
    "completion_reward": 5.0,
    "pickup_reward": 0.5,
    "progress_reward": 0.1,
    "blocked_forward_penalty": 0.1,
    "collision_penalty": 0.1,
    "step_cost": 0.005,
    "charge_event_reward": 0.0,
    "deadlock_penalty": 5.0,
    "priority_weights": {"A": 2.0, "B": 1.5, "C": 1.0},
    "energy_death_penalty": 20.0,
    "energy_death_aggregation": "local-mean",
}

ENVIRONMENT = {
    "profile_id": "medium-5ag-phase2-train",
    "environment_id": "llm-mappo-medium-3ag-v1",
    "n_agents": 5,
    "max_steps": 800,
    "dynamic_ingress_interval": 150,
    "batch_size_range": [5, 5],
    "initial_ingress_batches": 1,
    "queue_size": 5,
    "task_target": 18,
    "terminate_on_task_target": False,
    "training_replenishment_mode": "rware_delivery",
    "initial_priority_label": "A",
    "priority_schedule": ["A", "B", "C"],
    "delivery_contract": "balanced-random-v1",
    "charging_stations": [[0, 0], [19, 0], [0, 23], [19, 23], [9, 0]],
    "picking_stations": [[9, 21], [10, 21]],
    "battery_cost_scale": 1.10,
    "charge_threshold": 0.30,
    "charge_release_threshold": 0.80,
    "deadlock_steps": 180,
    "observation_schema": "direct-goal-observation-v2",
    "goal_interface": "direct_goal",
    "reward_version": "reward-v2",
    "reward_blocked_forward_penalty": 0.10,
    "reward_collision_penalty": 0.10,
    "charge_event_reward": 0.0,
    "reward_deadlock_penalty": 5.0,
    "reward_progress_coefficient": 0.1,
    "reward_energy_death_aggregation": "local-mean",
    "completion_metric_contract": COMPLETION_METRIC_CONTRACT,
    "oracle_interaction_mask": True,
}

TRAINING = {
    "environment_steps": 600000,
    "optimization_schedule_steps": 600000,
    "milestone_checkpoint_steps": [300000, 600000],
    "post_schedule_learning_rate": 0.0,
    "num_env_workers": 16,
    "learner_torch_threads": 4,
    "worker_numerical_threads": 1,
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
    "checkpoint_interval_steps": 25000,
}

EVALUATION = {
    "replenishment_mode": "rware_delivery",
    "seeds": list(range(9400, 9410)),
    "deterministic_policy": "argmax",
    "replay_frame_stride": 1,
    "replay_frame_duration_ms": 600,
    "environment_overrides": {
        "dynamic_ingress_interval": 100,
        "initial_ingress_batches": 2,
        "max_steps": 800,
        "task_target": 25,
        "terminate_on_task_target": True,
    },
}


def reward_definition_sha256() -> str:
    """Return the identity shared with the approved C1 diagnostic reward."""

    encoded = json.dumps(
        REWARD_DEFINITION, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def validate_medium5_c1(values: Mapping[str, Any]) -> None:
    """Reject mixed method contracts before any provisional run is opened."""

    if values.get("reward_definition") != REWARD_DEFINITION:
        raise ValueError("Medium-5AG C1 reward definition differs from frozen contract.")
    if values.get("environment") != ENVIRONMENT:
        raise ValueError("Medium-5AG C1 environment differs from frozen contract.")
    if values.get("evaluation") != EVALUATION:
        raise ValueError("Medium-5AG C1 evaluation differs from frozen contract.")
    training = dict(values.get("training", {}))
    if training.pop("seed", None) not in TRAINING_SEEDS:
        raise ValueError("Medium-5AG C1 training seed is not preregistered.")
    method = str(values.get("method", ""))
    if method.endswith("LLM-INIT"):
        if "representation_initialization" not in training:
            raise ValueError("Medium-5AG C1 LLM initialization is missing.")
        training.pop("representation_initialization")
    if "ASTARKD" in method:
        if training.pop("post_schedule_astar_lambda", None) != 0.0:
            raise ValueError("Medium-5AG C1 A*KD tail lambda differs.")
    if training != TRAINING:
        raise ValueError("Medium-5AG C1 training differs from frozen contract.")
