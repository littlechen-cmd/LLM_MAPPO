"""Owner-run training and evaluation for the teacher-free MAPPO-FP-DG baseline."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from llm_mappo.mappo_baseline.fp_state import FPStateSchema
from llm_mappo.formal_training import (
    COMMON_TRAINING, DIAGNOSTIC_SEEDS, DURABILITY_SEEDS, EVALUATION_SEEDS, METHODS,
    SCENES, SCHEMA, TRAIN_SEEDS, validate_provisional_environment,
    validate_scene_training_budget, validate_approved_scene_artifacts,
    build_durability_windows,
)
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V1,
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
    build_structured_observations,
)
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer
from llm_mappo.mappo_baseline.vector_env import (
    MAPPOVectorEnvironment,
    new_baseline_environment,
)
from llm_mappo.warehouse_environment import (
    EPISODE_ACTION_NAMES,
    EPISODE_DIAGNOSTIC_FIELDS,
    EPISODE_TARGET_PHASES,
)


@dataclass(frozen=True)
class BaselineExperiment:
    schema: str
    method: str
    artifact_root: str
    environment: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    config_sha256: str
    approval_status: str = "approved"


LEGACY_CORE_INTEGRATION_SCHEMA = "core-mappo-209d-v1"
CORE_INTEGRATION_SCHEMA = "core-mappo-209d-delivery-v2"
REWARD_DIAGNOSTIC_SCHEMA = "reward-diagnostic-mappo-209d-delivery-v1"
ASTAR_TAIL_DIAGNOSTIC_SCHEMA = "astar-tail-diagnostic-v1"
CONTINUOUS_DELIVERY_SCHEMAS = frozenset({SCHEMA, ASTAR_TAIL_DIAGNOSTIC_SCHEMA})
DIAGNOSTIC_CHECKPOINT_SCHEMAS = {
    ASTAR_TAIL_DIAGNOSTIC_SCHEMA: "mappo-fp-astar-tail-diagnostic-checkpoint-v1",
    REWARD_DIAGNOSTIC_SCHEMA: "mappo-fp-209d-delivery-reward-diagnostic-checkpoint-v1",
}
CORE_INTEGRATION_SCHEMAS = frozenset({
    LEGACY_CORE_INTEGRATION_SCHEMA,
    CORE_INTEGRATION_SCHEMA,
})
CORE_CHECKPOINT_SCHEMAS = {
    "MAPPO-FP-DG": "mappo-fp-direct-goal-209d-checkpoint-v1",
    "MAPPO-FP-DG-LLM-INIT": "mappo-fp-llm-init-209d-checkpoint-v1",
    "MAPPO-FP-ASTARKD-NO-VALIDITY": (
        "mappo-fp-astar-kd-209d-checkpoint-v1"
    ),
    "MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT": (
        "mappo-fp-astar-kd-llm-init-209d-checkpoint-v1"
    ),
}
DELIVERY_CHECKPOINT_SCHEMAS = {
    method: schema.replace("209d-checkpoint-v1", "209d-delivery-checkpoint-v2")
    for method, schema in CORE_CHECKPOINT_SCHEMAS.items()
}


def checkpoint_schema_for_method(method: str) -> str:
    """Return the frozen core-integration checkpoint schema for one method."""

    try:
        return CORE_CHECKPOINT_SCHEMAS[method]
    except KeyError as error:
        raise ValueError("Core-integration MAPPO method is incompatible.") from error


def _run_directory_name(
    requested: str | None, *, seed: int, timestamp: str
) -> str:
    """Accept one safe artifact leaf name or retain the ordinary timestamp layout."""

    if requested is None:
        return f"seed_{seed:04d}_{timestamp}"
    name = Path(requested)
    if name.is_absolute() or len(name.parts) != 1 or name.name in {"", ".", ".."}:
        raise ValueError("MAPPO artifact directory name must be one relative leaf.")
    return name.name


def checkpoint_schema_for_experiment(experiment: BaselineExperiment) -> str:
    """Resolve one experiment's strict checkpoint namespace."""

    if experiment.schema in DIAGNOSTIC_CHECKPOINT_SCHEMAS:
        return DIAGNOSTIC_CHECKPOINT_SCHEMAS[experiment.schema]
    if experiment.schema == SCHEMA:
        return f"{experiment.method.lower()}-{experiment.environment['scene_id']}-v1"
    if experiment.schema == CORE_INTEGRATION_SCHEMA:
        try:
            return DELIVERY_CHECKPOINT_SCHEMAS[experiment.method]
        except KeyError as error:
            raise ValueError("Core-integration MAPPO method is incompatible.") from error
    if experiment.schema == LEGACY_CORE_INTEGRATION_SCHEMA:
        return checkpoint_schema_for_method(experiment.method)
    if experiment.schema == "mappo-collection-209d-v1":
        return "mappo-fp-collection-209d-checkpoint-v1"
    if experiment.schema == "phase2-mappo-astar-kd-v1":
        return "mappo-fp-astar-kd-checkpoint-v1"
    if experiment.schema == "phase2-mappo-baseline-v1":
        return "mappo-fp-phase2-baseline-checkpoint-v1"
    return "mappo-fp-dg-checkpoint-v2"


def load_baseline_experiment(path: str | Path) -> BaselineExperiment:  # noqa: C901
    source = Path(path)
    raw = source.read_bytes()
    values = yaml.safe_load(raw)
    if values.get("schema") == ASTAR_TAIL_DIAGNOSTIC_SCHEMA:
        raise ValueError("A* tail configurations require the diagnostic runner.")
    if values.get("schema") == REWARD_DIAGNOSTIC_SCHEMA:
        raise ValueError(
            "Reward diagnostic configurations require the diagnostic runner."
        )
    required = {
        "schema", "method", "artifact_root", "environment", "training", "evaluation"
    }
    if values.get("schema") in CORE_INTEGRATION_SCHEMAS | {SCHEMA}:
        required.add("approval_status")
    if set(values) != required:
        raise ValueError("MAPPO baseline configuration has unexpected top-level keys.")
    provisional = values["schema"] == SCHEMA
    environment = (
        validate_provisional_environment(values["environment"])
        if provisional else dict(values["environment"])
    )
    training = dict(values["training"])
    evaluation = dict(values["evaluation"])
    if provisional:
        if (tuple(evaluation.get("seed_bases", ())) != tuple(range(200, 210))
                or evaluation.get("episodes_per_base") != 20
                or "seeds" in evaluation):
            raise ValueError("Three-scene evaluation seed derivation differs.")
        evaluation["seeds"] = list(EVALUATION_SEEDS)
        validate_scene_training_budget(environment["scene_id"], training)
    _validate_environment(environment, provisional=provisional)
    _validate_training(training, provisional=provisional)
    _validate_evaluation(evaluation, provisional=provisional)
    if provisional:
        if values["approval_status"] not in {"provisional", "approved"}:
            raise ValueError("Three-scene approval status is invalid.")
        if values["method"] not in METHODS:
            raise ValueError("Three-scene MAPPO method is incompatible.")
        if int(training["seed"]) not in TRAIN_SEEDS:
            raise ValueError("Three-scene training seed is not registered.")
        if tuple(training.get("paired_training_seeds", ())) != TRAIN_SEEDS:
            raise ValueError("Three-scene paired training seed list differs.")
        for field, expected in COMMON_TRAINING.items():
            if training.get(field) != expected:
                raise ValueError(f"Three-scene {field} differs across methods.")
        if ("representation_initialization" in training) != str(
            values["method"]
        ).endswith("LLM-INIT"):
            raise ValueError("LLM method and initialization must agree.")
        if values["approval_status"] == "approved":
            validate_approved_scene_artifacts(str(values["method"]), training)
        elif "representation_initialization" in training:
            if training["representation_initialization"] is not None:
                raise ValueError(
                    "Provisional LLM representation must remain pending approval."
                )
    if values["schema"] == "mappo-collection-209d-v1":
        if values["method"] != "MAPPO-FP-DG":
            raise ValueError("Collection baseline must use MAPPO-FP-DG.")
        if environment["observation_schema"] != DIRECT_GOAL_OBSERVATION_V2:
            raise ValueError("Collection baseline requires native 209D v2.")
    if values["schema"] in CORE_INTEGRATION_SCHEMAS:
        checkpoint_schema_for_method(str(values["method"]))
        if environment["observation_schema"] != DIRECT_GOAL_OBSERVATION_V2:
            raise ValueError("Core integration requires native 209D v2.")
        if values["approval_status"] not in {"provisional", "approved"}:
            raise ValueError("Core-integration approval status is invalid.")
        has_initialization = "representation_initialization" in training
        expects_initialization = str(values["method"]).endswith("LLM-INIT")
        if has_initialization != expects_initialization:
            raise ValueError("LLM method and representation initialization must agree.")
        if has_initialization:
            from llm_mappo.mappo_llm_kd.initialization import (
                load_representation_initialization,
            )

            load_representation_initialization(training["representation_initialization"])
    if values["schema"] == CORE_INTEGRATION_SCHEMA:
        if environment.get("delivery_contract") != "balanced-random-v1":
            raise ValueError("Delivery-v2 requires balanced-random-v1.")
        if environment["priority_schedule"] != ["A", "B", "C"]:
            raise ValueError("Delivery-v2 requires the A/B/C priority schedule.")
        if environment.get("training_replenishment_mode") != "rware_delivery":
            raise ValueError("Delivery-v2 training requires rware_delivery.")
        if values["evaluation"].get("replenishment_mode") != "rware_delivery":
            raise ValueError("Delivery-v2 evaluation requires rware_delivery.")
    if values["schema"] not in {
        "phase1-mappo-baseline-v1", "phase2-mappo-baseline-v1",
        "phase2-mappo-astar-kd-v1",
        "mappo-collection-209d-v1", *CORE_INTEGRATION_SCHEMAS, SCHEMA,
    }:
        raise ValueError("MAPPO baseline schema or method is incompatible.")
    if values["method"] not in {
        "MAPPO-FP-DG", "MAPPO-FP-WP", "MAPPO-FP-ASTARKD",
        "MAPPO-FP-ASTARKD-NO-VALIDITY", "MAPPO-FP-DG-LLM-INIT",
        "MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT",
    }:
        raise ValueError("MAPPO baseline schema or method is incompatible.")
    expected_interface = {
        "MAPPO-FP-DG": "direct_goal",
        "MAPPO-FP-WP": "waypoint",
        "MAPPO-FP-ASTARKD": "direct_goal",
        "MAPPO-FP-ASTARKD-NO-VALIDITY": "direct_goal",
        "MAPPO-FP-DG-LLM-INIT": "direct_goal",
        "MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT": "direct_goal",
    }[values["method"]]
    if environment.get("goal_interface") != expected_interface:
        raise ValueError("MAPPO method and goal_interface must agree.")
    return BaselineExperiment(
        schema=str(values["schema"]),
        method=str(values["method"]),
        artifact_root=str(values["artifact_root"]),
        environment=environment,
        training=training,
        evaluation=evaluation,
        config_sha256=sha256(raw).hexdigest(),
        approval_status=str(values.get("approval_status", "approved")),
    )


def _validate_environment(  # noqa: C901
    environment: Mapping[str, Any], *, provisional: bool = False
) -> None:  # noqa: C901
    required = {
        "profile_id", "environment_id", "n_agents", "max_steps",
        "dynamic_ingress_interval", "batch_size_range", "queue_size",
        "task_target", "initial_priority_label", "priority_schedule",
        "charging_stations", "picking_stations", "battery_cost_scale",
        "charge_threshold", "charge_release_threshold", "deadlock_steps",
        "observation_schema", "goal_interface", "reward_version",
        "oracle_interaction_mask",
    }
    missing = sorted(required - set(environment))
    if missing:
        raise ValueError(f"MAPPO baseline environment is missing: {', '.join(missing)}.")
    if environment["observation_schema"] not in {
        DIRECT_GOAL_OBSERVATION_V1,
        DIRECT_GOAL_OBSERVATION_V2,
    }:
        raise ValueError("MAPPO-FP-DG requires the DirectGoal observation schema.")
    if environment["goal_interface"] not in {"direct_goal", "waypoint"}:
        raise ValueError("MAPPO baseline goal_interface is incompatible.")
    agents = int(environment["n_agents"])
    if agents < 1 or (
        not provisional and len(environment["charging_stations"]) != agents
    ):
        raise ValueError("Phase 1 requires exactly one charging station per AGV.")
    if not provisional and len(environment["picking_stations"]) != agents // 2:
        raise ValueError("Phase 1 picking-station count must equal floor(AGV / 2).")
    if environment.get("delivery_contract", "legacy-v1") == "legacy-v1":
        if environment["priority_schedule"] != ["A", "B", "C", "D"]:
            raise ValueError("Legacy priority schedule must be A/B/C/D.")
    elif environment["delivery_contract"] != "balanced-random-v1":
        raise ValueError("MAPPO baseline delivery contract is invalid.")
    if int(environment["dynamic_ingress_interval"]) < 2:
        raise ValueError("MAPPO baseline dynamic ingress interval is invalid.")
    if len(environment["batch_size_range"]) != 2:
        raise ValueError("MAPPO baseline batch-size range is invalid.")
    if environment.get("training_replenishment_mode", "dynamic_ingress") not in {
        "dynamic_ingress", "rware_delivery"
    }:
        raise ValueError("MAPPO baseline training replenishment mode is invalid.")


def _validate_training(  # noqa: C901
    training: Mapping[str, Any], *, provisional: bool = False
) -> None:
    required = {
        "seed", "environment_steps", "num_env_workers", "learner_torch_threads",
        "worker_numerical_threads", "rollout_length", "ppo_epochs",
        "num_minibatches", "gamma", "gae_lambda", "clip_param",
        "actor_learning_rate", "critic_learning_rate", "entropy_coefficient",
        "value_loss_coefficient", "max_gradient_norm", "huber_delta",
        "hidden_dimension", "checkpoint_interval_steps",
    }
    missing = sorted(required - set(training))
    if missing:
        raise ValueError(f"MAPPO baseline training is missing: {', '.join(missing)}.")
    if int(training["num_env_workers"]) != (16 if provisional else 6):
        raise ValueError("Phase 1 local baseline requires six environment workers.")
    if int(training["learner_torch_threads"]) != 4:
        raise ValueError("Phase 1 local baseline requires four learner threads.")
    if int(training["worker_numerical_threads"]) != 1:
        raise ValueError("Phase 1 local baseline requires one worker numerical thread.")
    if int(training["hidden_dimension"]) != 64:
        raise ValueError("Phase 1 MAPPO GRU hidden dimension must be 64.")
    if provisional and training["environment_steps"] is None:
        if training.get("optimization_schedule_steps") is not None:
            raise ValueError("Provisional schedule cannot precede budget approval.")
        if "astar_schedule_steps" in training:
            raise ValueError("A*KD schedule horizon requires an approved budget.")
        if training.get("astar_stop_steps") is not None:
            raise ValueError("A*KD stop steps require an approved budget.")
        if training.get("milestone_checkpoint_steps", []) != []:
            raise ValueError("Provisional milestones cannot precede approval.")
        return
    if int(training["environment_steps"]) < 1:
        raise ValueError("MAPPO baseline environment-step budget is invalid.")
    budget = int(training["environment_steps"])
    schedule_steps = int(training.get("optimization_schedule_steps", budget))
    if schedule_steps < 1 or schedule_steps > budget:
        raise ValueError("MAPPO optimization schedule horizon is invalid.")
    if "astar_schedule_steps" in training:
        astar_steps = training["astar_schedule_steps"]
        if (
            not isinstance(astar_steps, int)
            or isinstance(astar_steps, bool)
            or astar_steps < 1
            or astar_steps > budget
        ):
            raise ValueError("A*KD schedule horizon must be an integer within budget.")
    stop_steps = training.get("astar_stop_steps")
    if stop_steps is not None and (
        not isinstance(stop_steps, int) or isinstance(stop_steps, bool)
        or stop_steps < 1 or stop_steps > budget
    ):
        raise ValueError("A*KD stop steps must be an integer within budget.")
    post_schedule_lr = float(training.get("post_schedule_learning_rate", 0.0))
    maximum_lr = min(
        float(training["actor_learning_rate"]),
        float(training["critic_learning_rate"]),
    )
    if post_schedule_lr < 0.0 or post_schedule_lr > maximum_lr:
        raise ValueError("MAPPO post-schedule learning rate is invalid.")
    if schedule_steps < budget and post_schedule_lr == 0.0:
        raise ValueError("MAPPO extended training needs a positive tail rate.")
    milestones = training.get("milestone_checkpoint_steps", [])
    _validate_milestone_checkpoints(milestones, budget)
    _validate_post_schedule_astar_lambda(training)


def _validate_post_schedule_astar_lambda(training: Mapping[str, Any]) -> None:
    post_kd_lambda = float(training.get("post_schedule_astar_lambda", 0.01))
    if post_kd_lambda < 0.0 or post_kd_lambda > 0.05:
        raise ValueError("MAPPO post-schedule A*KD lambda is invalid.")


def _validate_milestone_checkpoints(milestones: Any, budget: int) -> None:
    if not isinstance(milestones, list) or not all(
        isinstance(step, int) for step in milestones
    ):
        raise ValueError("MAPPO milestone checkpoints must be a list of steps.")
    if any(step < 1 or step > budget for step in milestones):
        raise ValueError("MAPPO milestone checkpoint step is outside the budget.")


def _validate_evaluation(  # noqa: C901
    evaluation: Mapping[str, Any], *, provisional: bool = False
) -> None:
    if provisional:
        if evaluation.get("replenishment_mode") != "rware_delivery":
            raise ValueError("Three-scene evaluation requires rware_delivery.")
        if tuple(evaluation.get("seeds", ())) != EVALUATION_SEEDS:
            raise ValueError("Three-scene evaluation instance list differs.")
        if tuple(evaluation.get("durability_seeds", ())) != DURABILITY_SEEDS:
            raise ValueError("Three-scene durability seed list differs.")
        if evaluation.get("deterministic_policy") != "argmax":
            raise ValueError("Three-scene evaluation must be deterministic.")
        if evaluation.get("environment_overrides", {}) != {}:
            raise ValueError("Three-scene training and evaluation load must match.")
        if evaluation.get("render_replay") is not False:
            raise ValueError("Three-scene main evaluation disables automatic replay.")
        return
    if list(evaluation.get("seeds", ())) != [9400, 9401]:
        raise ValueError("Phase 1 evaluation seeds must be 9400 and 9401.")
    if evaluation.get("deterministic_policy") != "argmax":
        raise ValueError("Phase 1 evaluation must use deterministic argmax.")
    if int(evaluation.get("replay_frame_duration_ms", 0)) < 500:
        raise ValueError("Phase 1 replay must be slow enough for owner review.")
    if evaluation.get("replenishment_mode", "dynamic_ingress") not in {
        "dynamic_ingress", "rware_delivery"
    }:
        raise ValueError("MAPPO baseline evaluation replenishment mode is invalid.")


class BaselineRun:
    """Train, checkpoint and independently evaluate one frozen baseline seed."""

    UPDATE_FIELDS = (
        "global_environment_steps",
        "updates",
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "actor_grad_norm",
        "critic_grad_norm",
        "grad_norm",
        "actor_learning_rate",
        "critic_learning_rate",
        "learning_rate",
        "environment_steps_per_second",
        "rollout_seconds",
        "inference_seconds",
        "ppo_update_seconds",
        "total_elapsed_seconds",
        "peak_cuda_memory_allocated",
        "peak_cuda_memory_reserved",
        "teacher_query_count",
        "teacher_valid_count",
        "teacher_valid_rate",
        "teacher_supervision_scope_count",
        "teacher_effective_supervision_count",
        "teacher_effective_supervision_rate",
        "teacher_invalid_counts",
        "teacher_cache_build_count",
        "teacher_cache_hit_count",
        "teacher_cache_hit_rate",
        "dynamic_attenuated_count",
        "mean_dynamic_weight",
        "astar_loss",
        "astar_lambda",
        "motion_prior_top1_agreement",
        "policy_teacher_motion_agreement",
        "static_route_regret",
        "static_route_regret_count",
        "static_invalid_top1_rate",
        "ppo_shared_grad_norm",
        "astar_shared_grad_norm",
        "astar_to_ppo_shared_grad_ratio",
        "raw_advantage_count",
        "raw_advantage_mean",
        "raw_advantage_std",
        "raw_advantage_min",
        "raw_advantage_max",
        "raw_advantage_p05",
        "raw_advantage_p50",
        "raw_advantage_p95",
        "raw_advantage_max_abs",
        "raw_return_count",
        "raw_return_mean",
        "raw_return_std",
        "raw_return_min",
        "raw_return_max",
        "raw_return_p05",
        "raw_return_p50",
        "raw_return_p95",
        "raw_return_max_abs",
    )
    EPISODE_FIELDS = (
        "global_environment_steps",
        "worker_index",
        "episode_index",
        "episode_seed",
        "completed_tasks",
        "created_tasks",
        "task_completion_target",
        "task_completion_rate",
        "reward",
        "collisions",
        "deadlocked",
        "agent_deaths",
        "picked_tasks",
        "progress_positive_events",
        "progress_negative_events",
        "blocked_forwards",
        "low_battery_triggers",
        "charging_target_steps",
        "charging_exposure_rate",
        "charger_arrivals",
        "charged_events",
        "charging_wait_steps",
        "task_recoveries",
        "energy_deaths",
        "minimum_battery",
        "steps",
        "steps_per_task",
        "reward_completion",
        "reward_pickup",
        "reward_progress",
        "reward_blocked_forward",
        "reward_collision",
        "reward_charging",
        "reward_deadlock",
        "reward_energy",
        "reward_step_cost",
        "success",
    ) + EPISODE_DIAGNOSTIC_FIELDS

    def __init__(  # noqa: C901
        self,
        experiment: BaselineExperiment,
        *,
        device: str,
        output_root: str | Path | None = None,
        run_directory_name: str | None = None,
        resume: str | Path | None = None,
        evaluation_only: bool = False,
        diagnostic_authorized: bool = False,
        diagnostic_reward_definition: Mapping[str, Any] | None = None,
        diagnostic_reward_definition_sha256: str | None = None,
    ) -> None:
        reward_diagnostic = experiment.schema == REWARD_DIAGNOSTIC_SCHEMA
        diagnostic_schema = reward_diagnostic or (
            experiment.schema == ASTAR_TAIL_DIAGNOSTIC_SCHEMA
        )
        if diagnostic_schema and not diagnostic_authorized:
            raise ValueError("Diagnostic runs require diagnostic authorization.")
        provisional_diagnostic = (
            experiment.schema == SCHEMA and diagnostic_authorized
            and experiment.method == "MAPPO-FP-DG"
            and experiment.training["seed"] == 7
            and experiment.training["environment_steps"] is not None
            and 1 <= int(experiment.training["environment_steps"]) <= 50_000
            and tuple(experiment.evaluation["seeds"]) == DIAGNOSTIC_SEEDS
            and experiment.artifact_root.startswith(
                "artifacts/formal_environment_training/diagnostics/"
            )
        )
        if (
            not diagnostic_schema and diagnostic_authorized
            and not provisional_diagnostic
        ):
            raise ValueError("Diagnostic authorization requires a diagnostic schema.")
        if reward_diagnostic and (
            not isinstance(diagnostic_reward_definition, Mapping)
            or not isinstance(diagnostic_reward_definition_sha256, str)
            or len(diagnostic_reward_definition_sha256) != 64
        ):
            raise ValueError("Reward diagnostic identity is incomplete.")
        if not evaluation_only and experiment.approval_status == "provisional" and not (
            (diagnostic_schema and diagnostic_authorized) or provisional_diagnostic
        ):
            raise ValueError("provisional configuration cannot start real training.")
        if evaluation_only and resume is not None:
            raise ValueError("Evaluation-only runs cannot resume training state.")
        if resume is not None and run_directory_name is not None:
            raise ValueError("Resumed runs cannot select a new output directory.")
        self.experiment = experiment
        self.evaluation_only = evaluation_only
        self._astar_kd_enabled = experiment.method in {
            "MAPPO-FP-ASTARKD",
            "MAPPO-FP-ASTARKD-NO-VALIDITY",
            "MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT",
        }
        self._astar_kd_variant = {
            "MAPPO-FP-ASTARKD": "full",
            "MAPPO-FP-ASTARKD-NO-VALIDITY": "no_validity",
            "MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT": "no_validity",
        }.get(experiment.method)
        if self._astar_kd_enabled != (self._astar_kd_variant is not None):
            raise ValueError("Phase 2 A*KD schema and method must agree.")
        self._checkpoint_schema = checkpoint_schema_for_experiment(experiment)
        self.device = torch.device(device)
        torch.set_num_threads(int(experiment.training["learner_torch_threads"]))
        self.schema = FPStateSchema(
            int(experiment.environment["n_agents"]),
            n_shelves=SCENES[experiment.environment["scene_id"]]["n_shelves"]
            if experiment.schema in CONTINUOUS_DELIVERY_SCHEMAS else 144,
            n_chargers=len(experiment.environment["charging_stations"]),
            n_picking_stations=len(experiment.environment["picking_stations"]),
        )
        self.actor_schema = StructuredObservationSchema(
            observation_schema=str(experiment.environment["observation_schema"])
        )
        actor_schema_version = (
            "v2"
            if self.actor_schema.observation_schema == DIRECT_GOAL_OBSERVATION_V2
            else "v1"
        )
        self.goal_interface = str(experiment.environment["goal_interface"])
        self._evaluation_planner = None
        if self.goal_interface == "waypoint":
            from llm_mappo.planner import AStarPlanner

            self._evaluation_planner = AStarPlanner()
        self.identity = {
            "schema": experiment.schema,
            "method": experiment.method,
            "code_commit": _commit(),
            "config_sha256": experiment.config_sha256,
            "seed": int(experiment.training["seed"]),
            "observation_schema": experiment.environment["observation_schema"],
            "actor_observation_schema": (
                f"structured-{self.goal_interface}-{actor_schema_version}"
            ),
            "actor_observation_dimension": self.actor_schema.dimension,
            "fp_state_dimension": self.schema.dimension,
            "global_step_unit": "cumulative_joint_environment_transition",
            "reward_version": experiment.environment["reward_version"],
            "profile_id": experiment.environment["profile_id"],
        }
        if experiment.schema in CONTINUOUS_DELIVERY_SCHEMAS:
            self.identity.update({
                "scene_id": experiment.environment["scene_id"],
                "environment_id": experiment.environment["environment_id"],
                "layout_sha256": experiment.environment.get("layout_sha256"),
                "n_agents": int(experiment.environment["n_agents"]),
                "charging_stations": experiment.environment["charging_stations"],
                "picking_stations": experiment.environment["picking_stations"],
            })
        if reward_diagnostic:
            self.identity["reward_definition_sha256"] = (
                diagnostic_reward_definition_sha256
            )
        if experiment.schema == "mappo-collection-209d-v1":
            environment_contract = dict(experiment.environment)
            encoded_environment = json.dumps(
                environment_contract, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            self.identity.update({
                "environment_id": environment_contract["environment_id"],
                "n_agents": int(environment_contract["n_agents"]),
                "goal_interface": environment_contract["goal_interface"],
                "environment_sha256": sha256(encoded_environment).hexdigest(),
            })
        if resume is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            root = Path(output_root or experiment.artifact_root)
            self.directory = root / _run_directory_name(
                run_directory_name,
                seed=int(self.identity["seed"]),
                timestamp=timestamp,
            )
            if self.directory.exists():
                raise FileExistsError(
                    f"MAPPO artifact directory already exists: {self.directory}"
                )
            self.directory.mkdir(parents=True, exist_ok=False)
            _write_json(self.directory / "run_manifest.json", {
                "identity": self.identity,
                "environment": dict(experiment.environment),
                "training": dict(experiment.training),
                "evaluation": dict(experiment.evaluation),
                "teacher_modules_enabled": self._astar_kd_enabled,
                "astar_kd_variant": self._astar_kd_variant,
            })
            if reward_diagnostic:
                manifest_path = self.directory / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["reward_diagnostic"] = {
                    "reward_definition": dict(diagnostic_reward_definition),
                    "reward_definition_sha256": diagnostic_reward_definition_sha256,
                }
                _write_json(manifest_path, manifest)
            _write_json(self.directory / "state.json", {"status": "running"})
            self._create_csv(self.directory / "updates.csv", self.UPDATE_FIELDS)
            self._create_csv(self.directory / "episodes.csv", self.EPISODE_FIELDS)
        else:
            self.directory = Path(resume)
            manifest = json.loads(
                (self.directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("identity") != self.identity:
                raise ValueError("MAPPO baseline resume identity differs.")
            state = json.loads(
                (self.directory / "state.json").read_text(encoding="utf-8")
            )
            if state.get("status") != "running":
                raise ValueError("Only a running MAPPO baseline artifact can resume.")
        budget = int(experiment.training["environment_steps"])
        self.optimization_schedule_steps = int(
            experiment.training.get("optimization_schedule_steps", budget)
        )
        self.astar_schedule_steps = int(
            experiment.training.get(
                "astar_schedule_steps", self.optimization_schedule_steps
            )
        )
        self.post_schedule_learning_rate = float(
            experiment.training.get("post_schedule_learning_rate", 0.0)
        )
        self.milestone_checkpoint_steps = [
            int(step)
            for step in experiment.training.get("milestone_checkpoint_steps", [])
        ]
        self.post_schedule_astar_lambda = float(
            experiment.training.get("post_schedule_astar_lambda", 0.01)
        )
        self.astar_stop_steps = experiment.training.get("astar_stop_steps")
        config = MAPPOConfig(
            observation_dim=self.actor_schema.dimension,
            state_dim=self.schema.dimension,
            action_dim=5,
            hidden_dim=int(experiment.training["hidden_dimension"]),
            actor_lr=float(experiment.training["actor_learning_rate"]),
            critic_lr=float(experiment.training["critic_learning_rate"]),
            clip_param=float(experiment.training["clip_param"]),
            ppo_epochs=int(experiment.training["ppo_epochs"]),
            num_minibatches=int(experiment.training["num_minibatches"]),
            entropy_coef=float(experiment.training["entropy_coefficient"]),
            value_loss_coef=float(experiment.training["value_loss_coefficient"]),
            max_grad_norm=float(experiment.training["max_gradient_norm"]),
            huber_delta=float(experiment.training["huber_delta"]),
        )
        self._seed(int(experiment.training["seed"]))
        if self._astar_kd_enabled:
            from llm_mappo.mappo_astar_kd.trainer import AStarKDTrainer

            self.trainer = AStarKDTrainer(
                config,
                self.device,
                total_environment_steps=self.astar_schedule_steps,
                post_schedule_astar_lambda=self.post_schedule_astar_lambda,
                astar_stop_steps=self.astar_stop_steps,
            )
        else:
            self.trainer = MAPPOTrainer(config, self.device)
        self.representation_initialization_receipt = None
        initialization_values = experiment.training.get("representation_initialization")
        if initialization_values is not None and resume is None:
            from llm_mappo.mappo_llm_kd.initialization import (
                initialize_final_actor,
                load_representation_initialization,
            )

            initialization = load_representation_initialization(initialization_values)
            self.representation_initialization_receipt = initialize_final_actor(
                self.trainer.actor, initialization
            )
            manifest_path = self.directory / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["representation_initialization"] = {
                "schema": "mappo-representation-initialization-v1",
                "artifact_sha256": initialization.artifact_sha256,
                "identity": dict(initialization.identity),
                "receipt": self.representation_initialization_receipt,
            }
            _write_json(manifest_path, manifest)
        self.pool = None
        if not evaluation_only:
            worker_values = {
                **experiment.environment,
                "experiment_schema": experiment.schema,
                "worker_numerical_threads": experiment.training[
                    "worker_numerical_threads"
                ],
            }
            if self._astar_kd_enabled:
                worker_values["astar_kd_variant"] = self._astar_kd_variant
            self.pool = MAPPOVectorEnvironment(
                worker_values,
                int(experiment.training["num_env_workers"]),
                int(experiment.training["seed"]),
                failure_directory=self.directory / "worker_failures",
            )
        self.global_steps = 0
        self._milestone_progress_steps = 0
        self.updates = 0
        hidden_shape = (
            int(experiment.training["num_env_workers"]),
            self.schema.n_agents,
            config.hidden_dim,
        )
        self.actor_hidden_states = np.zeros(hidden_shape, dtype=np.float32)
        self.critic_hidden_states = np.zeros(hidden_shape, dtype=np.float32)
        self.started_at = time.perf_counter()
        self.elapsed_offset = 0.0
        self.inference_seconds = 0.0
        self.rollout_seconds = 0.0
        self.update_seconds = 0.0
        self.planner_query_count = 0
        self._board = self._tensorboard()
        if resume is not None:
            self._restore(self.directory / "checkpoint_latest.pt")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def train(self) -> dict[str, Any]:  # noqa: C901
        if self.pool is None:
            raise RuntimeError("Evaluation-only MAPPO runs cannot train.")
        budget = int(self.experiment.training["environment_steps"])
        workers = int(self.experiment.training["num_env_workers"])
        rollout_length = int(self.experiment.training["rollout_length"])
        checkpoint_interval = int(
            self.experiment.training["checkpoint_interval_steps"]
        )
        last_checkpoint_bucket = self.global_steps // checkpoint_interval
        try:
            while self.global_steps < budget:
                remaining = budget - self.global_steps
                steps_this_rollout = min(
                    rollout_length, (remaining + workers - 1) // workers
                )
                storage = MAPPORolloutStorage(
                    steps_this_rollout,
                    workers,
                    self.schema.n_agents,
                    self.actor_schema.dimension,
                    self.schema.dimension,
                    5,
                    self.trainer.config.hidden_dim,
                    astar_kd_enabled=self._astar_kd_enabled,
                )
                episodes = []
                teacher_counts = Counter()
                teacher_query_count = 0
                teacher_valid_count = 0
                teacher_supervision_scope_count = 0
                teacher_effective_supervision_count = 0
                teacher_cache_build_count = 0
                teacher_cache_hit_count = 0
                dynamic_attenuated_count = 0
                dynamic_weight_sum = 0.0
                rollout_started = time.perf_counter()
                for _ in range(steps_this_rollout):
                    active_workers = min(workers, budget - self.global_steps)
                    observations = self.pool.observations.copy()
                    fp_states = self.pool.fp_states.copy()
                    action_masks = self.pool.action_masks.copy()
                    active_masks = self.pool.active_masks.copy()
                    astar_kd = self.pool.astar_kd
                    if self._astar_kd_enabled and astar_kd is None:
                        raise RuntimeError("A*KD worker did not provide teacher labels.")
                    if self._astar_kd_enabled:
                        teacher_query_count += int(astar_kd["query_count"])
                        teacher_cache_build_count += int(astar_kd["cache_build_count"])
                        teacher_cache_hit_count += int(astar_kd["cache_hit_count"])
                        valid = np.asarray(astar_kd["validity"], dtype=bool)
                        scope = np.asarray(
                            astar_kd["supervision_scope"], dtype=bool
                        )
                        supervised = np.logical_and(valid, scope)
                        weights = np.asarray(astar_kd["attenuation"], dtype=np.float32)
                        teacher_valid_count += int(valid.sum())
                        teacher_supervision_scope_count += int(scope.sum())
                        teacher_effective_supervision_count += int(
                            supervised.sum()
                        )
                        dynamic_attenuated_count += int(
                            np.logical_and(supervised, weights == 0.25).sum()
                        )
                        dynamic_weight_sum += float(weights[supervised].sum())
                        for reasons in astar_kd["failure_reasons"]:
                            teacher_counts.update(
                                reason for reason in reasons if reason != "ok"
                            )
                    inference_started = time.perf_counter()
                    (
                        actions,
                        log_probs,
                        values,
                        value_predictions,
                        next_actor_hidden,
                        next_critic_hidden,
                    ) = self.trainer.act(
                        observations,
                        fp_states,
                        action_masks,
                        self.actor_hidden_states,
                        self.critic_hidden_states,
                        deterministic=False,
                    )
                    self.inference_seconds += time.perf_counter() - inference_started
                    response = self.pool.step(
                        actions,
                        global_step=self.global_steps,
                        active_workers=active_workers,
                    )
                    self.global_steps += active_workers
                    active_masks[active_workers:] = 0.0
                    self.planner_query_count = max(
                        self.planner_query_count, response.planner_query_count
                    )
                    continuation = (
                        (~response.dones).astype(np.float32)[..., None]
                        * response.active_masks[..., None]
                    )
                    episodes.extend(response.episodes)
                    storage.insert(
                        observations=observations,
                        fp_states=fp_states,
                        actions=actions,
                        action_masks=action_masks,
                        log_probs=log_probs,
                        values=values,
                        value_predictions=value_predictions,
                        rewards=response.rewards,
                        dones=response.dones,
                        active_masks=active_masks,
                        actor_hidden_states=self.actor_hidden_states,
                        critic_hidden_states=self.critic_hidden_states,
                        **({
                            "astar_soft_targets": astar_kd["soft_targets"],
                            "astar_root_costs": astar_kd["root_costs"],
                            "astar_validity": astar_kd["validity"],
                            "astar_supervision_scope": astar_kd[
                                "supervision_scope"
                            ],
                            "astar_attenuation": astar_kd["attenuation"],
                        } if self._astar_kd_enabled else {}),
                    )
                    self.actor_hidden_states = next_actor_hidden * continuation
                    self.critic_hidden_states = next_critic_hidden * continuation
                self.rollout_seconds += time.perf_counter() - rollout_started
                next_values, _ = self.trainer.values(
                    self.pool.fp_states, self.critic_hidden_states
                )
                batch = storage.compute_returns(
                    next_values,
                    gamma=float(self.experiment.training["gamma"]),
                    gae_lambda=float(self.experiment.training["gae_lambda"]),
                )
                self._apply_learning_rate_schedule()
                if self._astar_kd_enabled:
                    self.trainer.set_global_environment_steps(self.global_steps)
                update_started = time.perf_counter()
                metrics = self.trainer.update(batch)
                if self._astar_kd_enabled:
                    metrics.update({
                        "teacher_query_count": teacher_query_count,
                        "teacher_valid_count": teacher_valid_count,
                        "teacher_valid_rate": teacher_valid_count / max(
                            teacher_query_count, 1
                        ),
                        "teacher_supervision_scope_count": (
                            teacher_supervision_scope_count
                        ),
                        "teacher_effective_supervision_count": (
                            teacher_effective_supervision_count
                        ),
                        "teacher_effective_supervision_rate": (
                            teacher_effective_supervision_count
                            / max(teacher_query_count, 1)
                        ),
                        "teacher_invalid_counts": json.dumps(
                            dict(sorted(teacher_counts.items())), sort_keys=True
                        ),
                        "teacher_cache_build_count": teacher_cache_build_count,
                        "teacher_cache_hit_count": teacher_cache_hit_count,
                        "teacher_cache_hit_rate": teacher_cache_hit_count / max(
                            teacher_cache_build_count + teacher_cache_hit_count, 1
                        ),
                        "dynamic_attenuated_count": dynamic_attenuated_count,
                        "mean_dynamic_weight": dynamic_weight_sum / max(
                            teacher_effective_supervision_count, 1
                        ),
                        "astar_lambda": self._current_astar_lambda(),
                    })
                else:
                    metrics.update({
                        "teacher_query_count": 0,
                        "teacher_valid_count": 0,
                        "teacher_valid_rate": 0.0,
                        "teacher_supervision_scope_count": 0,
                        "teacher_effective_supervision_count": 0,
                        "teacher_effective_supervision_rate": 0.0,
                        "teacher_invalid_counts": "{}",
                        "teacher_cache_build_count": 0,
                        "teacher_cache_hit_count": 0,
                        "teacher_cache_hit_rate": 0.0,
                        "dynamic_attenuated_count": 0,
                        "mean_dynamic_weight": 0.0,
                        "astar_lambda": 0.0,
                        "motion_prior_top1_agreement": 0.0,
                        "policy_teacher_motion_agreement": 0.0,
                        "static_route_regret": 0.0,
                        "static_route_regret_count": 0,
                        "static_invalid_top1_rate": 0.0,
                        "ppo_shared_grad_norm": 0.0,
                        "astar_shared_grad_norm": 0.0,
                        "astar_to_ppo_shared_grad_ratio": 0.0,
                    })
                self.update_seconds += time.perf_counter() - update_started
                self.updates += 1
                row = self._update_row(metrics)
                self._append_csv(self.directory / "updates.csv", row, self.UPDATE_FIELDS)
                for episode in episodes:
                    self._append_csv(
                        self.directory / "episodes.csv", episode, self.EPISODE_FIELDS
                    )
                    self._write_episode_board(episode)
                self._write_update_board(row)
                current_bucket = self.global_steps // checkpoint_interval
                if current_bucket > last_checkpoint_bucket:
                    self._checkpoint(self.directory / "checkpoint_latest.pt")
                    last_checkpoint_bucket = current_bucket
                self._write_due_milestone_checkpoints()
            if self.goal_interface == "direct_goal" and self.planner_query_count != 0:
                raise RuntimeError("DirectGoal baseline observed a planner query.")
            self._checkpoint(self.directory / "checkpoint_final.pt")
            evaluation = self.evaluate()
            summary = self._summary(evaluation)
            _write_json(self.directory / "summary.json", summary)
            _write_json(self.directory / "review_receipt.json", {
                "schema": "phase1-mappo-review-receipt-v1",
                "identity": self.identity,
                "checkpoint": "checkpoint_final.pt",
                "evaluation": evaluation,
                "tensorboard": "tensorboard",
                "owner_architect_review": "pending",
            })
            _write_json(self.directory / "state.json", {"status": "complete"})
            return summary
        except BaseException as error:
            _write_json(self.directory / "state.json", {
                "status": "failed",
                "reason": type(error).__name__,
                "message": str(error),
                "worker_failure_directory": "worker_failures",
            })
            raise
        finally:
            self.close()

    def _apply_learning_rate_schedule(self) -> None:
        self.trainer.linear_lr_decay(
            self.global_steps,
            self.optimization_schedule_steps,
            self.post_schedule_learning_rate,
        )

    def _current_astar_lambda(self) -> float:
        from llm_mappo.mappo_astar_kd.trainer import astar_kd_lambda

        return astar_kd_lambda(
            self.global_steps,
            self.astar_schedule_steps,
            post_schedule_lambda=self.post_schedule_astar_lambda,
            stop_steps=self.astar_stop_steps,
        )

    def _write_due_milestone_checkpoints(self) -> None:
        """Persist named milestone checkpoints when this session crosses them."""
        for milestone in self.milestone_checkpoint_steps:
            if not (
                self._milestone_progress_steps
                < milestone
                <= self.global_steps
            ):
                continue
            self._checkpoint(self.directory / f"checkpoint_{milestone}.pt")
        self._milestone_progress_steps = self.global_steps

    def evaluate(
        self, *, checkpoint_reference: str = "checkpoint_final.pt"
    ) -> dict[str, Any]:
        results = []
        self.trainer.actor.eval()
        for seed in self.experiment.evaluation["seeds"]:
            results.append(self._evaluation_episode(int(seed)))
        if self.experiment.schema in CONTINUOUS_DELIVERY_SCHEMAS:
            legacy_fields = {
                "success", "task_completion_rate", "task_completion_target"
            }
            filtered = [
                {key: value for key, value in row.items() if key not in legacy_fields}
                for row in results
            ]
            total_steps = sum(int(row["steps"]) for row in filtered)
            total_completed = sum(int(row["completed_tasks"]) for row in filtered)
            result = {
                "schema": "mappo-continuous-delivery-evaluation-v1",
                "policy": "deterministic_argmax",
                "checkpoint": checkpoint_reference,
                "episodes": filtered,
                "total_completed_tasks": total_completed,
                "total_episode_steps": total_steps,
                "completed_tasks_per_1000_actual_steps": (
                    1000.0 * total_completed / total_steps if total_steps else 0.0
                ),
                "mean_completed_tasks": float(np.mean([
                    row["completed_tasks"] for row in filtered
                ])),
                "episode_diagnostics": _continuous_episode_window(results),
                "fixed_replay": None,
                "metrics_plot": None,
            }
            output = self.directory / "evaluation"
            output.mkdir(exist_ok=True)
            _write_json(output / "evaluation.json", result)
            return result
        mean_completion = float(np.mean([
            row["task_completion_rate"] for row in results
        ]))
        replay_root = self.directory / "evaluation" / "replays"
        replay_root.mkdir(parents=True, exist_ok=True)
        fixed_seed = int(self.experiment.evaluation["seeds"][0])
        fixed_path = replay_root / f"fixed_seed_{fixed_seed}.gif"
        if self.experiment.evaluation.get("render_replay", True):
            self._evaluation_episode(fixed_seed, replay_path=fixed_path)
        result = {
            "schema": "mappo-fp-dg-evaluation-v1",
            "policy": "deterministic_argmax",
            "checkpoint": checkpoint_reference,
            "episodes": results,
            "mean_task_completion_rate": mean_completion,
            "mean_completed_tasks": float(np.mean([
                row["completed_tasks"] for row in results
            ])),
            "episode_diagnostics": _episode_window(results),
            "fixed_replay": (
                str(fixed_path.relative_to(self.directory))
                if self.experiment.evaluation.get("render_replay", True) else None
            ),
            "metrics_plot": (
                None if self.experiment.schema in CONTINUOUS_DELIVERY_SCHEMAS
                else "evaluation/evaluation_metrics.png"
            ),
        }
        output = self.directory / "evaluation"
        output.mkdir(exist_ok=True)
        if self.experiment.schema not in CONTINUOUS_DELIVERY_SCHEMAS:
            _write_evaluation_plot(output / "evaluation_metrics.png", results)
        _write_json(output / "evaluation.json", result)
        return result

    def evaluate_checkpoint(self, checkpoint: str | Path) -> dict[str, Any]:
        """Evaluate one compatible completed checkpoint without resuming training."""

        source = Path(checkpoint).resolve()
        try:
            payload = torch.load(source, map_location="cpu", weights_only=False)
            source_identity = self._validate_evaluation_checkpoint(payload)
            self.trainer.load_state_dict(payload["trainer"])
            evaluation = self.evaluate(checkpoint_reference=str(source))
            manifest = {
                "schema": "mappo-fp-checkpoint-evaluation-v1",
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": sha256(source.read_bytes()).hexdigest(),
                "source_identity": source_identity,
                "evaluation_identity": self.identity,
                "evaluation_environment": self._evaluation_environment_values(),
                "evaluation": dict(self.experiment.evaluation),
                "checkpoint_evaluation_only": True,
            }
            _write_json(self.directory / "evaluation_manifest.json", manifest)
            result = {
                "schema": "mappo-fp-checkpoint-evaluation-summary-v1",
                "status": "complete",
                "source_checkpoint": str(source),
                "evaluation": evaluation,
            }
            _write_json(self.directory / "summary.json", result)
            _write_json(self.directory / "review_receipt.json", {
                "schema": "phase1-mappo-checkpoint-evaluation-receipt-v1",
                "source_checkpoint": str(source),
                "evaluation": evaluation,
                "owner_architect_review": "pending",
            })
            _write_json(self.directory / "state.json", {"status": "complete"})
            return result
        except BaseException as error:
            _write_json(self.directory / "state.json", {
                "status": "failed",
                "reason": type(error).__name__,
                "message": str(error),
            })
            raise
        finally:
            self.close()

    def _validate_evaluation_checkpoint(self, payload: Mapping[str, Any]) -> dict:
        if payload.get("schema") != self._checkpoint_schema:
            raise ValueError("MAPPO baseline checkpoint schema differs.")
        source_identity = payload.get("identity")
        if not isinstance(source_identity, dict):
            raise ValueError("MAPPO baseline checkpoint has no identity.")
        compatibility_fields = (
            "schema",
            "method",
            "observation_schema",
            "actor_observation_schema",
            "actor_observation_dimension",
            "fp_state_dimension",
            "reward_version",
            "reward_definition_sha256",
        )
        if self.experiment.schema in CONTINUOUS_DELIVERY_SCHEMAS:
            compatibility_fields += (
                "scene_id", "environment_id", "layout_sha256", "n_agents",
                "charging_stations", "picking_stations", "config_sha256",
                "seed", "profile_id",
            )
        mismatched = [
            field for field in compatibility_fields
            if source_identity.get(field) != self.identity.get(field)
        ]
        if mismatched:
            raise ValueError(
                "MAPPO checkpoint is incompatible with the evaluation profile: "
                + ", ".join(mismatched)
            )
        if "trainer" not in payload:
            raise ValueError("MAPPO baseline checkpoint has no trainer state.")
        return source_identity

    def _evaluation_episode(
        self, seed: int, replay_path: Path | None = None
    ) -> dict[str, Any]:
        values = self._evaluation_environment_values()
        environment = new_baseline_environment(
            values,
            replenishment_mode=str(
                self.experiment.evaluation.get(
                    "replenishment_mode", "dynamic_ingress"
                )
            ),
        )
        frames = []
        try:
            environment.reset(seed=seed)
            durability = (
                self.experiment.schema in CONTINUOUS_DELIVERY_SCHEMAS
                and self.experiment.environment["scene_id"] == "formal-8ag"
                and values["max_steps"] == 5000
                and self.experiment.evaluation.get("durability_max_steps") == 5000
            )
            durability_snapshots = [environment.metrics.as_dict()] if durability else []
            actor_hidden = np.zeros(
                (1, environment.n_agents, self.trainer.config.hidden_dim),
                dtype=np.float32,
            )
            transition = None
            stride = int(self.experiment.evaluation["replay_frame_stride"])
            for step in range(int(values["max_steps"])):
                if replay_path is not None and step % stride == 0:
                    from llm_mappo.visualization import render_warehouse_frame

                    frames.append(render_warehouse_frame(environment.env, cell_size=16))
                observations = build_structured_observations(
                    environment,
                    self.actor_schema,
                    goal_interface=self.goal_interface,
                    planner=self._evaluation_planner,
                )
                masks = environment.action_masks()
                actions, next_actor = self.trainer.act_policy(
                    observations[None],
                    masks[None],
                    actor_hidden,
                    deterministic=True,
                )
                transition = environment.step(actions[0])
                finished = (
                    transition.terminated or transition.truncated
                    or transition.metrics.deadlocked
                )
                if durability and (
                    transition.metrics.steps % 1000 == 0 or finished
                    or step + 1 == int(values["max_steps"])
                ):
                    durability_snapshots.append(transition.metrics.as_dict())
                continuation = 0.0 if (
                    transition.terminated
                    or transition.truncated
                    or transition.metrics.deadlocked
                ) else 1.0
                actor_hidden = next_actor * continuation
                if (
                    transition.terminated
                    or transition.truncated
                    or transition.metrics.deadlocked
                ):
                    break
            if transition is None:
                raise RuntimeError(
                    "MAPPO evaluation did not execute an environment step."
                )
            if replay_path is not None:
                from llm_mappo.visualization import render_warehouse_frame
                from PIL import Image

                frames.append(render_warehouse_frame(environment.env, cell_size=16))
                images = [Image.fromarray(frame) for frame in frames]
                images[0].save(
                    replay_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(
                        self.experiment.evaluation["replay_frame_duration_ms"]
                    ),
                    loop=0,
                    optimize=False,
                )
            result = {"seed": seed, **transition.metrics.as_dict()}
            if durability:
                result.update({
                    "durability_windows": build_durability_windows(durability_snapshots),
                    "ran_full_horizon": transition.metrics.steps == 5000,
                    "termination": {
                        "reason": (
                            "deadlock" if transition.metrics.deadlocked
                            else "terminated" if transition.terminated
                            else "horizon" if transition.metrics.steps == 5000
                            else "truncated" if transition.truncated else "stopped"
                        ),
                        "terminated": bool(transition.terminated),
                        "truncated": bool(transition.truncated),
                        "deadlocked": bool(transition.metrics.deadlocked),
                    },
                    "teacher_query_count": environment.planner_query_counter.count,
                    "llm_teacher_request_count": 0,
                })
            return result
        finally:
            environment.close()

    def _evaluation_environment_values(self) -> dict[str, Any]:
        """Apply explicit eval-only workload values without changing training."""

        overrides = self.experiment.evaluation.get("environment_overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("Evaluation environment_overrides must be a mapping.")
        values = {**self.experiment.environment, **overrides}
        _validate_environment(
            values,
            provisional=self.experiment.schema in CONTINUOUS_DELIVERY_SCHEMAS,
        )
        return values

    def _checkpoint(self, path: Path) -> None:
        payload = {
            "schema": self._checkpoint_schema,
            "identity": self.identity,
            "trainer": self.trainer.state_dict(),
            "global_environment_steps": self.global_steps,
            "updates": self.updates,
            "planner_query_count": self.planner_query_count,
            "worker_states": self.pool.snapshot(),
            "recurrent_states": {
                "actor": self.actor_hidden_states,
                "critic": self.critic_hidden_states,
            },
            "timing": {
                "rollout_seconds": self.rollout_seconds,
                "inference_seconds": self.inference_seconds,
                "ppo_update_seconds": self.update_seconds,
                "total_elapsed_seconds": self._elapsed(),
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        if self.experiment.schema == "mappo-collection-209d-v1":
            self._export_collection_actor(
                path.with_name(
                    path.stem.replace("checkpoint_", "collection_actor_")
                    + path.suffix
                )
            )

    def _export_collection_actor(self, path: Path) -> None:
        """Export a weights-only-safe Actor artifact for offline collection."""

        config = self.trainer.config
        payload = {
            "schema": "mappo-collection-actor-209d-v1",
            "identity": dict(self.identity),
            "actor_config": {
                "observation_dim": int(config.observation_dim),
                "action_dim": int(config.action_dim),
                "hidden_dim": int(config.hidden_dim),
            },
            "actor_state_dict": self.trainer.actor.state_dict(),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _restore(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != self._checkpoint_schema:
            raise ValueError("MAPPO baseline checkpoint schema differs.")
        if payload.get("identity") != self.identity:
            raise ValueError("MAPPO baseline checkpoint identity differs.")
        self.trainer.load_state_dict(payload["trainer"])
        self.pool.restore(payload["worker_states"])
        recurrent = payload["recurrent_states"]
        self.actor_hidden_states = np.asarray(recurrent["actor"], dtype=np.float32)
        self.critic_hidden_states = np.asarray(recurrent["critic"], dtype=np.float32)
        self.global_steps = int(payload["global_environment_steps"])
        self._milestone_progress_steps = self.global_steps
        self.updates = int(payload["updates"])
        self.planner_query_count = int(payload["planner_query_count"])
        timing = payload["timing"]
        self.rollout_seconds = float(timing["rollout_seconds"])
        self.inference_seconds = float(timing["inference_seconds"])
        self.update_seconds = float(timing["ppo_update_seconds"])
        self.elapsed_offset = float(timing["total_elapsed_seconds"])
        self.started_at = time.perf_counter()
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        if torch.cuda.is_available() and payload["rng"]["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
        self._truncate_evidence()

    def _update_row(self, metrics) -> dict[str, Any]:
        elapsed = self._elapsed()
        peak_allocated = 0
        peak_reserved = 0
        if self.device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        return {
            "global_environment_steps": self.global_steps,
            "updates": self.updates,
            **{name: metrics[name] for name in (
                "policy_loss", "value_loss", "entropy", "total_loss", "approx_kl",
                "clip_fraction", "explained_variance", "actor_grad_norm",
                "critic_grad_norm", "actor_learning_rate", "critic_learning_rate",
                "teacher_query_count", "teacher_valid_count", "teacher_valid_rate",
                "teacher_supervision_scope_count",
                "teacher_effective_supervision_count",
                "teacher_effective_supervision_rate",
                "teacher_invalid_counts", "teacher_cache_build_count",
                "teacher_cache_hit_count", "teacher_cache_hit_rate",
                "dynamic_attenuated_count", "mean_dynamic_weight", "astar_loss",
                "astar_lambda", "motion_prior_top1_agreement",
                "policy_teacher_motion_agreement", "static_route_regret",
                "static_route_regret_count", "static_invalid_top1_rate",
                "ppo_shared_grad_norm", "astar_shared_grad_norm",
                "astar_to_ppo_shared_grad_ratio",
                "raw_advantage_count", "raw_advantage_mean", "raw_advantage_std",
                "raw_advantage_min", "raw_advantage_max", "raw_advantage_p05",
                "raw_advantage_p50", "raw_advantage_p95", "raw_advantage_max_abs",
                "raw_return_count", "raw_return_mean", "raw_return_std",
                "raw_return_min", "raw_return_max", "raw_return_p05",
                "raw_return_p50", "raw_return_p95", "raw_return_max_abs",
            )},
            "grad_norm": max(
                float(metrics["actor_grad_norm"]),
                float(metrics["critic_grad_norm"]),
            ),
            "learning_rate": float(metrics["actor_learning_rate"]),
            "environment_steps_per_second": self.global_steps / max(elapsed, 1e-9),
            "rollout_seconds": self.rollout_seconds,
            "inference_seconds": self.inference_seconds,
            "ppo_update_seconds": self.update_seconds,
            "total_elapsed_seconds": elapsed,
            "peak_cuda_memory_allocated": peak_allocated,
            "peak_cuda_memory_reserved": peak_reserved,
        }

    def _summary(self, evaluation) -> dict[str, Any]:
        episodes = self._read_csv(self.directory / "episodes.csv")
        window = episodes[-100:]
        continuous_throughput = not self.experiment.environment.get(
            "terminate_on_task_target", True
        )
        return {
            "identity": self.identity,
            "global_environment_steps": self.global_steps,
            "updates": self.updates,
            "completed_training_episodes": len(episodes),
            "planner_query_count": self.planner_query_count,
            "training_episode_window": _episode_window(
                window,
                primary_metric="steps_per_task" if continuous_throughput else None,
            ) if self.experiment.schema not in CONTINUOUS_DELIVERY_SCHEMAS else (
                _continuous_episode_window(window)
            ),
            "evaluation": evaluation,
            "total_elapsed_seconds": self._elapsed(),
        }

    def _tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(self.directory / "tensorboard"))

    def _write_update_board(self, row) -> None:
        step = int(row["global_environment_steps"])
        for name in (
            "policy_loss", "value_loss", "entropy", "total_loss", "approx_kl",
            "clip_fraction", "explained_variance", "actor_grad_norm",
            "critic_grad_norm", "grad_norm", "actor_learning_rate",
            "critic_learning_rate", "learning_rate",
        ):
            self._board.add_scalar(f"train/{name}", float(row[name]), step)
        for name in (
            "environment_steps_per_second", "rollout_seconds", "inference_seconds",
            "ppo_update_seconds",
        ):
            self._board.add_scalar(f"performance/{name}", float(row[name]), step)
        self._board.add_scalar(
            "performance/gpu_memory_mb",
            float(row["peak_cuda_memory_allocated"]) / (1024.0 * 1024.0),
            step,
        )
        for name in (
            "teacher_supervision_scope_count",
            "teacher_effective_supervision_count",
            "teacher_effective_supervision_rate",
            "motion_prior_top1_agreement",
            "policy_teacher_motion_agreement",
            "static_route_regret",
            "static_invalid_top1_rate",
            "ppo_shared_grad_norm",
            "astar_shared_grad_norm",
            "astar_to_ppo_shared_grad_ratio",
        ):
            self._board.add_scalar(f"teacher/{name}", float(row[name]), step)
        self._board.flush()

    def _write_episode_board(self, row) -> None:
        step = int(row["global_environment_steps"])
        mapping = {
            "team_reward": "reward",
            "completed_tasks": "completed_tasks",
            "episode_length": "steps",
            "collisions": "collisions",
            "blocked_forward": "blocked_forwards",
            "deadlocked": "deadlocked",
            "energy_deaths": "energy_deaths",
            "charging_exposure_rate": "charging_exposure_rate",
        }
        if not self.experiment.environment.get("terminate_on_task_target", True):
            mapping["steps_per_task"] = "steps_per_task"
        else:
            mapping["task_completion_rate"] = "task_completion_rate"
        for tag, name in mapping.items():
            if row[name] not in {None, ""}:
                self._board.add_scalar(f"episode/{tag}", float(row[name]), step)
        if row["steps_per_task"] in {None, ""}:
            self._board.add_scalar("episode/zero_completed_tasks", 1.0, step)
        for name in (
            "reward_completion", "reward_pickup", "reward_progress",
            "reward_blocked_forward", "reward_collision", "reward_energy",
            "reward_charging", "reward_deadlock",
            "reward_step_cost",
        ):
            self._board.add_scalar(
                f"reward/{name.removeprefix('reward_')}", float(row[name]), step
            )
        self._board.flush()

    def close(self) -> None:
        if getattr(self, "_board", None) is not None:
            self._board.close()
            self._board = None
        if getattr(self, "pool", None) is not None:
            self.pool.close()
            self.pool = None

    def _elapsed(self) -> float:
        return self.elapsed_offset + time.perf_counter() - self.started_at

    def _truncate_evidence(self) -> None:
        """Drop rows written after the latest durable checkpoint boundary."""

        for name, fields in (
            ("updates.csv", self.UPDATE_FIELDS),
            ("episodes.csv", self.EPISODE_FIELDS),
        ):
            path = self.directory / name
            rows = self._read_csv(path)
            kept = [
                row for row in rows
                if int(row["global_environment_steps"]) <= self.global_steps
            ]
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)
            temporary.replace(path)

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _create_csv(path: Path, fields) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    @staticmethod
    def _append_csv(path: Path, row, fields) -> None:
        if set(row) != set(fields):
            raise ValueError(f"Evidence row for {path.name} has incompatible fields.")
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
            handle.flush()

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def _episode_window(rows, *, primary_metric: str | None = None) -> dict[str, Any]:
    if not rows:
        return {"window_size": 0}
    total_steps = sum(float(row["steps"]) for row in rows)
    total_completed = sum(float(row["completed_tasks"]) for row in rows)
    summary = {
        "window_size": len(rows),
        "mean_task_completion_rate": float(np.mean([
            float(row["task_completion_rate"]) for row in rows
        ])),
        "mean_completed_tasks": float(np.mean([
            float(row["completed_tasks"]) for row in rows
        ])),
        "total_episode_steps": total_steps,
        "total_completed_tasks": total_completed,
        "steps_per_task": total_steps / total_completed if total_completed else None,
        "mean_team_reward": float(np.mean([float(row["reward"]) for row in rows])),
    }
    if primary_metric is not None:
        summary["primary_metric"] = primary_metric
    diagnostic_counts = [
        f"{phase}_{action}_count"
        for phase in EPISODE_TARGET_PHASES
        for action in EPISODE_ACTION_NAMES
    ]
    if all(
        row.get(field) not in (None, "")
        for row in rows
        for field in (*diagnostic_counts, "direction_reversals")
    ):
        summary["direction_reversals"] = sum(
            float(row["direction_reversals"]) for row in rows
        )
        summary["mean_direction_reversals"] = float(np.mean([
            float(row["direction_reversals"]) for row in rows
        ]))
        for phase in EPISODE_TARGET_PHASES:
            phase_total = sum(
                float(row[f"{phase}_{action}_count"])
                for row in rows
                for action in EPISODE_ACTION_NAMES
            )
            for action in EPISODE_ACTION_NAMES:
                count_field = f"{phase}_{action}_count"
                count = sum(float(row[count_field]) for row in rows)
                summary[count_field] = count
                summary[f"{phase}_{action}_rate"] = (
                    count / phase_total if phase_total else 0.0
                )
    if all(
        row.get(field) not in (None, "")
        for row in rows
        for field in ("motion_steps", "active_motion_steps")
    ):
        total_motion = sum(float(row["motion_steps"]) for row in rows)
        total_active_motion = sum(
            float(row["active_motion_steps"]) for row in rows
        )
        summary["motion_steps"] = total_motion
        summary["active_motion_steps"] = total_active_motion
        summary["motion_steps_per_task"] = (
            total_motion / total_completed if total_completed else None
        )
        summary["active_motion_steps_per_task"] = (
            total_active_motion / total_completed if total_completed else None
        )
    return summary


def _continuous_episode_window(rows) -> dict[str, Any]:
    """Summarize fixed-duration replenishment without target-rate semantics."""

    if not rows:
        return {"window_size": 0}
    total_steps = sum(float(row["steps"]) for row in rows)
    total_completed = sum(float(row["completed_tasks"]) for row in rows)
    fields = (
        "energy_deaths", "agent_deaths", "collisions", "blocked_forwards",
        "deadlocked", "charged_events", "charger_arrivals", "picked_tasks",
    )
    summary = {
        "window_size": len(rows),
        "total_episode_steps": total_steps,
        "total_completed_tasks": total_completed,
        "mean_completed_tasks": total_completed / len(rows),
        "completed_tasks_per_1000_actual_steps": (
            1000.0 * total_completed / total_steps if total_steps else 0.0
        ),
        "primary_metric": "completed_tasks_per_1000_actual_steps",
    }
    for field in fields:
        if all(row.get(field) not in (None, "") for row in rows):
            values = (
                {"True": 1.0, "False": 0.0}.get(row[field], row[field])
                if field == "deadlocked" else row[field]
                for row in rows
            )
            summary[f"total_{field}"] = sum(float(value) for value in values)
    return summary


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _write_evaluation_plot(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Create a compact fixed-seed completion chart without GUI dependencies."""

    from PIL import Image, ImageDraw

    width, height = 920, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 16), "MAPPO-FP-DG deterministic evaluation", fill="black")
    left, top, right, bottom = 60, 60, 890, 310
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    target_y = bottom - int(0.90 * (bottom - top))
    draw.line((left, target_y, right, target_y), fill="#d62728", width=2)
    draw.text((right - 95, target_y - 18), "90% Gate", fill="#d62728")
    bar_width = max((right - left) // max(len(rows), 1) - 18, 12)
    for index, row in enumerate(rows):
        value = float(row["task_completion_rate"])
        x0 = left + 12 + index * ((right - left) // len(rows))
        x1 = x0 + bar_width
        y0 = bottom - int(min(max(value, 0.0), 1.0) * (bottom - top))
        color = "#2ca02c" if value >= 0.90 else "#1f77b4"
        draw.rectangle((x0, y0, x1, bottom), fill=color)
        draw.text((x0, bottom + 8), str(row["seed"]), fill="black")
        draw.text((x0, max(y0 - 18, top)), f"{value:.0%}", fill="black")
    image.save(path)
