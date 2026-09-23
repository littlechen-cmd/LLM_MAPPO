"""Strict configuration contracts for owner-approved reward diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from llm_mappo.mappo_baseline.runner import BaselineExperiment


REWARD_DIAGNOSTIC_SCHEMA = "reward-diagnostic-mappo-209d-delivery-v1"
REWARD_DIAGNOSTIC_CHECKPOINT_SCHEMA = (
    "mappo-fp-209d-delivery-reward-diagnostic-checkpoint-v1"
)
ALLOWED_TRAINING_SEEDS = (7, 17, 27)
CALIBRATION_EVALUATION_SEEDS = tuple(range(9400, 9410))
DIAGNOSTIC_ARTIFACT_ROOT = Path("artifacts/reward_diagnostics/group2")

_EXPECTED_ENVIRONMENT = {
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
    "charge_event_reward": 0.00,
    "reward_deadlock_penalty": 5.00,
    "oracle_interaction_mask": True,
}

_EXPECTED_TRAINING = {
    "environment_steps": 300000,
    "optimization_schedule_steps": 300000,
    "milestone_checkpoint_steps": [300000],
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

_EXPECTED_EVALUATION = {
    "replenishment_mode": "rware_delivery",
    "seeds": list(CALIBRATION_EVALUATION_SEEDS),
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

_REWARD_V2_REFERENCE = {
    "version": "reward-v2-reference",
    "completion_reward": 5.0,
    "pickup_reward": 0.5,
    "progress_reward": 0.1,
    "blocked_forward_penalty": 0.1,
    "collision_penalty": 0.1,
    "step_cost": 0.005,
    "charge_event_reward": 0.0,
    "deadlock_penalty": 5.0,
    "priority_weights": {"A": 2.0, "B": 1.5, "C": 1.0},
}

_REWARD_V3_C1_ENERGY_DEATH_LOCAL_MEAN = {
    **_REWARD_V2_REFERENCE,
    "version": "reward-v3-c1-energy-death-local-mean",
    "energy_death_penalty": 20.0,
    "energy_death_aggregation": "local-mean",
}

_REWARD_V3_C2_PROGRESS_0P2 = {
    **_REWARD_V2_REFERENCE,
    "version": "reward-v3-c2-progress-0p2",
    "progress_reward": 0.2,
    "energy_death_penalty": 20.0,
    "energy_death_aggregation": "team-broadcast",
}

_REWARD_V3_C3_COLLISION_1P0 = {
    **_REWARD_V2_REFERENCE,
    "version": "reward-v3-c3-collision-1p0",
    "collision_penalty": 1.0,
    "energy_death_penalty": 20.0,
    "energy_death_aggregation": "team-broadcast",
}

_REWARD_V3_C4_BLOCKED_FORWARD_0P5 = {
    **_REWARD_V2_REFERENCE,
    "version": "reward-v3-c4-blocked-forward-0p5",
    "blocked_forward_penalty": 0.5,
    "energy_death_penalty": 20.0,
    "energy_death_aggregation": "team-broadcast",
}

_APPROVED_REWARD_CONTRACTS = {
    "reward-v2-reference": (
        _REWARD_V2_REFERENCE,
        {},
    ),
    "reward-v3-c1-energy-death-local-mean": (
        _REWARD_V3_C1_ENERGY_DEATH_LOCAL_MEAN,
        {
            "reward_progress_coefficient": 0.1,
            "reward_energy_death_aggregation": "local-mean",
        },
    ),
    "reward-v3-c2-progress-0p2": (
        _REWARD_V3_C2_PROGRESS_0P2,
        {
            "reward_progress_coefficient": 0.2,
            "reward_energy_death_aggregation": "team-broadcast",
        },
    ),
    "reward-v3-c3-collision-1p0": (
        _REWARD_V3_C3_COLLISION_1P0,
        {
            "reward_progress_coefficient": 0.1,
            "reward_energy_death_aggregation": "team-broadcast",
            "reward_collision_penalty": 1.0,
        },
    ),
    "reward-v3-c4-blocked-forward-0p5": (
        _REWARD_V3_C4_BLOCKED_FORWARD_0P5,
        {
            "reward_progress_coefficient": 0.1,
            "reward_energy_death_aggregation": "team-broadcast",
            "reward_blocked_forward_penalty": 0.5,
        },
    ),
}


@dataclass(frozen=True)
class RewardDiagnosticExperiment:
    """One fully frozen provisional reward-diagnostic experiment."""

    baseline: BaselineExperiment
    reward_definition: Mapping[str, Any]
    reward_definition_sha256: str

    @property
    def schema(self) -> str:
        return self.baseline.schema

    @property
    def method(self) -> str:
        return self.baseline.method

    @property
    def approval_status(self) -> str:
        return self.baseline.approval_status

    @property
    def artifact_root(self) -> str:
        return self.baseline.artifact_root

    @property
    def environment(self) -> Mapping[str, Any]:
        return self.baseline.environment

    @property
    def training(self) -> Mapping[str, Any]:
        return self.baseline.training

    @property
    def evaluation(self) -> Mapping[str, Any]:
        return self.baseline.evaluation


def load_reward_diagnostic_experiment(path: str | Path) -> RewardDiagnosticExperiment:
    """Load a pre-registered provisional reference or single-factor candidate."""

    source = Path(path)
    raw = source.read_bytes()
    values = yaml.safe_load(raw)
    if not isinstance(values, dict):
        raise ValueError("Reward diagnostic configuration must be a mapping.")
    expected_keys = {
        "schema",
        "method",
        "approval_status",
        "artifact_root",
        "environment",
        "reward_definition",
        "training",
        "evaluation",
    }
    if set(values) != expected_keys:
        raise ValueError(
            "Reward diagnostic configuration has unexpected top-level keys."
        )
    if values["schema"] != REWARD_DIAGNOSTIC_SCHEMA:
        raise ValueError("Reward diagnostic schema is invalid.")
    if values["method"] != "MAPPO-FP-DG":
        raise ValueError("Reward diagnostics require the original MAPPO baseline.")
    if values["approval_status"] != "provisional":
        raise ValueError("Reward diagnostics must remain provisional.")
    _require_diagnostic_artifact_root(values["artifact_root"])
    reward_definition = _validate_approved_reward_contract(values)
    _validate_training(values["training"])
    _require_exact_mapping(
        values["evaluation"], _EXPECTED_EVALUATION, "evaluation"
    )
    reward_sha256 = _canonical_sha256(reward_definition)
    baseline = BaselineExperiment(
        schema=REWARD_DIAGNOSTIC_SCHEMA,
        method=str(values["method"]),
        artifact_root=str(values["artifact_root"]),
        environment=dict(values["environment"]),
        training=dict(values["training"]),
        evaluation=dict(values["evaluation"]),
        config_sha256=sha256(raw).hexdigest(),
        approval_status=str(values["approval_status"]),
    )
    return RewardDiagnosticExperiment(
        baseline=baseline,
        reward_definition=reward_definition,
        reward_definition_sha256=reward_sha256,
    )


def _validate_approved_reward_contract(values: Mapping[str, Any]) -> dict[str, Any]:
    reward_definition = values["reward_definition"]
    if not isinstance(reward_definition, dict):
        raise ValueError("Reward diagnostic reward definition must be a mapping.")
    version = reward_definition.get("version")
    contract = _APPROVED_REWARD_CONTRACTS.get(version)
    if contract is None:
        raise ValueError("Reward diagnostic candidate is not owner-approved.")
    expected_reward, environment_overrides = contract
    try:
        _require_exact_mapping(
            reward_definition, expected_reward, "reward definition"
        )
        _require_exact_mapping(
            values["environment"],
            {**_EXPECTED_ENVIRONMENT, **environment_overrides},
            "environment",
        )
    except ValueError as error:
        if version != "reward-v2-reference":
            raise ValueError(
                "Reward diagnostic candidate violates its approved "
                "single-factor contract."
            ) from error
        raise
    return dict(reward_definition)


def validate_diagnostic_invocation(
    experiment: RewardDiagnosticExperiment,
    *,
    seed: int,
    output_root: str | Path,
    acknowledged: bool,
) -> None:
    """Reject accidental or out-of-contract execution before artifact creation."""

    if not acknowledged:
        raise ValueError("Reward diagnostic acknowledgement is required.")
    if int(seed) not in ALLOWED_TRAINING_SEEDS:
        raise ValueError("Reward diagnostic seed is not pre-registered.")
    if experiment.approval_status != "provisional":
        raise ValueError("Reward diagnostic approval status is invalid.")
    root = Path(output_root).resolve()
    required_root = DIAGNOSTIC_ARTIFACT_ROOT.resolve()
    try:
        root.relative_to(required_root)
    except ValueError as error:
        raise ValueError(
            "Reward diagnostic output root is outside group2 artifacts."
        ) from error


def resolve_diagnostic_baseline_experiment(
    experiment: RewardDiagnosticExperiment, *, seed: int
) -> BaselineExperiment:
    """Return the exact same diagnostic contract with one pre-registered seed."""

    if int(seed) not in ALLOWED_TRAINING_SEEDS:
        raise ValueError("Reward diagnostic seed is not pre-registered.")
    training = {**experiment.training, "seed": int(seed)}
    return replace(experiment.baseline, training=training)


def make_local_smoke_experiment(
    experiment: RewardDiagnosticExperiment,
) -> BaselineExperiment:
    """Build an explicitly non-server, non-comparable lifecycle probe."""

    training = {
        **experiment.training,
        "environment_steps": 64,
        "optimization_schedule_steps": 64,
        "milestone_checkpoint_steps": [64],
        "checkpoint_interval_steps": 64,
        "num_env_workers": 1,
    }
    evaluation = {
        **experiment.evaluation,
        "seeds": [9400],
        "environment_overrides": {
            **experiment.evaluation["environment_overrides"],
            "max_steps": 32,
            "task_target": 1,
        },
    }
    smoke_identity = {
        "source_config_sha256": experiment.baseline.config_sha256,
        "mode": "local-smoke-v1",
        "training": training,
        "evaluation": evaluation,
    }
    return replace(
        experiment.baseline,
        artifact_root="artifacts/reward_diagnostics/group2/local_smoke",
        training=training,
        evaluation=evaluation,
        config_sha256=_canonical_sha256(smoke_identity),
    )


def _validate_training(values: Any) -> None:
    if not isinstance(values, dict):
        raise ValueError("Reward diagnostic training must be a mapping.")
    expected = {"seed": 7, **_EXPECTED_TRAINING}
    _require_exact_mapping(values, expected, "training")


def _require_diagnostic_artifact_root(value: Any) -> None:
    root = Path(str(value)).resolve()
    required_root = DIAGNOSTIC_ARTIFACT_ROOT.resolve()
    try:
        root.relative_to(required_root)
    except ValueError as error:
        raise ValueError(
            "Reward diagnostic artifact root is outside group2 artifacts."
        ) from error


def _require_exact_mapping(values: Any, expected: Mapping[str, Any], name: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"Reward diagnostic {name} must be a mapping.")
    if values == expected:
        return
    missing = sorted(set(expected) - set(values))
    unexpected = sorted(set(values) - set(expected))
    if missing or unexpected:
        detail = ", ".join([*missing, *unexpected])
        raise ValueError(f"Reward diagnostic {name} keys differ: {detail}.")
    for key in expected:
        if values[key] == expected[key]:
            continue
        label = {
            "num_env_workers": "workers",
            "seeds": "evaluation seeds",
            "delivery_contract": "delivery",
        }.get(key, key)
        raise ValueError(
            f"Reward diagnostic {name} {label} differs from the frozen contract."
        )
    raise RuntimeError(
        "Reward diagnostic mapping comparison did not resolve a difference."
    )


def _canonical_sha256(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def classify_candidate_effect(
    reference_by_seed: Mapping[int, Mapping[str, float | int | None]],
    candidate_by_seed: Mapping[int, Mapping[str, float | int | None]],
    *,
    training_trend_consistent: bool,
    numerical_health: bool,
) -> dict[str, Any]:
    """Apply the pre-registered paired effect rule without selecting a reward."""

    seeds = sorted(reference_by_seed)
    if seeds != sorted(candidate_by_seed) or len(seeds) != 3:
        raise ValueError(
            "Reward candidate classification requires the paired three seeds."
        )
    wins = 0
    losses = 0
    ties = 0
    collision_regression_seeds = []
    blocked_regression_seeds = []
    additional_deadlocks = 0.0
    additional_energy_deaths = 0.0
    for seed in seeds:
        reference = reference_by_seed[seed]
        candidate = candidate_by_seed[seed]
        outcome = _paired_task_outcome(reference, candidate)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            ties += 1
        if float(candidate["collisions"]) > float(reference["collisions"]):
            collision_regression_seeds.append(seed)
        if float(candidate["blocked_forwards"]) > float(reference["blocked_forwards"]):
            blocked_regression_seeds.append(seed)
        additional_deadlocks += float(candidate["deadlocked"]) - float(
            reference["deadlocked"]
        )
        additional_energy_deaths += float(candidate["energy_deaths"]) - float(
            reference["energy_deaths"]
        )
    safe = (
        additional_deadlocks <= 0.0
        and additional_energy_deaths <= 0.0
        and len(collision_regression_seeds) < 2
        and len(blocked_regression_seeds) < 2
    )
    if not numerical_health or (losses >= 2 and wins == 0):
        label = "negative"
    elif wins >= 2 and losses == 0 and safe and training_trend_consistent:
        label = "positive"
    else:
        label = "no-clear-effect"
    return {
        "label": label,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "collision_regression_seeds": collision_regression_seeds,
        "blocked_regression_seeds": blocked_regression_seeds,
        "additional_deadlocks": additional_deadlocks,
        "additional_energy_deaths": additional_energy_deaths,
        "training_trend_consistent": training_trend_consistent,
        "numerical_health": numerical_health,
    }


def _paired_task_outcome(
    reference: Mapping[str, float | int | None],
    candidate: Mapping[str, float | int | None],
) -> str:
    reference_completed = float(reference["completed_tasks"])
    candidate_completed = float(candidate["completed_tasks"])
    if candidate_completed > reference_completed:
        return "win"
    if candidate_completed < reference_completed:
        return "loss"
    reference_steps = reference["steps_per_task"]
    candidate_steps = candidate["steps_per_task"]
    if reference_steps is None or candidate_steps is None:
        return "tie"
    if float(candidate_steps) < float(reference_steps):
        return "win"
    if float(candidate_steps) > float(reference_steps):
        return "loss"
    return "tie"
