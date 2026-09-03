"""Frozen R1-C diagnostic profile and evidence helpers.

This module deliberately models the small 4-AGV diagnostic separately from
the E1 confirmation matrix.  It contains no curriculum or checkpoint-transfer
logic; every arm starts from the same seed-derived initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from llm_mappo.e1_protocol import E1FormalRun


_SCHEMA = "r1c-4agv-lowload-v1"
_ARM_NAMES = (
    "legacy-r128", "reward-v2-r128", "legacy-r32", "reward-v2-r32",
)


@dataclass(frozen=True)
class R1CDiagnostic:
    """One frozen 4-AGV reward/update-cadence diagnostic arm."""

    arm: str
    environment: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation_seeds: tuple[int, ...]
    artifact_root: str

    @property
    def run(self) -> E1FormalRun:
        return E1FormalRun(
            group="MAPPO-DG", seed=int(self.training["seed"]), algorithm="mappo",
            astar_kd="disabled", semantic_teacher="disabled",
            semantic_control="none", observation_schema=str(
                self.environment["observation_schema"]
            ), real_environment_steps=int(self.training["real_environment_steps"]),
            checkpoint_rule="checkpoint_final.pt", artifact_path=self.artifact_root,
        )


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash a configuration in a stable, human-auditable representation."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def load_r1c_diagnostic(path: str | Path, arm: str) -> R1CDiagnostic:
    """Load exactly one preregistered R1-C arm, rejecting contract drift."""
    source = Path(path)
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping) or values.get("schema") != _SCHEMA:
        raise ValueError("R1-C diagnostic configuration schema is incompatible.")
    arms = values.get("arms")
    if not isinstance(arms, Mapping) or tuple(arms) != _ARM_NAMES:
        raise ValueError("R1-C must contain exactly the frozen four-arm diagnostic.")
    if arm not in arms:
        raise ValueError("R1-C requested arm is not preregistered.")
    profile = _mapping(values, "profile")
    training = _mapping(values, "training")
    evaluation = _mapping(values, "evaluation")
    _validate_profile(profile, training, evaluation)
    arm_values = _mapping(arms, arm)
    reward = arm_values.get("reward_version")
    rollout = arm_values.get("rollout_length")
    if reward not in {"legacy-v1", "reward-v2"} or rollout not in {32, 128}:
        raise ValueError("R1-C arm reward or rollout contract is incompatible.")
    environment = {**profile, "reward_version": reward}
    run_training = {**training, "rollout_steps": int(rollout),
                    "rollout_length": int(rollout)}
    return R1CDiagnostic(
        arm=arm, environment=environment, training=run_training,
        evaluation_seeds=tuple(int(seed) for seed in evaluation["seeds"]),
        artifact_root=str(values["artifact_root"]),
    )


def r1c_identity(*, code_commit: str, diagnostic: R1CDiagnostic,
                 raw_records_sha256: str, layout_hash: str,
                 initial_parameter_sha256: str) -> dict[str, Any]:
    """Return the full identity required by artifacts and resumable checkpoints."""
    if not raw_records_sha256 or not layout_hash or not initial_parameter_sha256:
        raise ValueError("R1-C identity is missing evidence, layout, or initialization.")
    profile = dict(diagnostic.environment)
    training = dict(diagnostic.training)
    return {
        "code_commit": str(code_commit), "phase": "R1-C",
        "arm": diagnostic.arm, "group": "MAPPO-DG", "seed": diagnostic.run.seed,
        "raw_records_sha256": raw_records_sha256,
        "environment": profile, "environment_sha256": canonical_sha256(profile),
        "layout_hash": layout_hash, "training": training,
        "training_sha256": canonical_sha256(training),
        "evaluation_seeds": list(diagnostic.evaluation_seeds),
        "initial_parameter_sha256": initial_parameter_sha256,
        "semantic_teacher": "disabled", "astar_kd": "disabled",
        "checkpoint_transfer": "prohibited",
    }


def r1c_trend(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the preregistered complete-episode trend, without selection."""
    rates = [float(row["task_completion_rate"]) for row in episodes]
    if len(rates) < 40:
        return {"complete_episode_count": len(rates), "trend_available": False,
                "trend_pass": False}
    first = sum(rates[:20]) / 20.0
    last = sum(rates[-20:]) / 20.0
    return {"complete_episode_count": len(rates), "trend_available": True,
            "first_20_mean_completion_rate": first,
            "last_20_mean_completion_rate": last, "trend_pass": last > first}


def _mapping(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = values.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"R1-C configuration field {name} must be a mapping.")
    return value


def _validate_profile(profile: Mapping[str, Any], training: Mapping[str, Any],
                      evaluation: Mapping[str, Any]) -> None:
    expected_profile = {
        "environment_id": "llm-mappo-medium-3ag-v1", "n_agents": 4,
        "dynamic_ingress_interval": 40, "batch_size_range": [2, 4],
        "queue_size": 4, "task_target": 20, "max_steps": 1000,
        "deadlock_steps": 180, "initial_priority_label": "A",
        "battery_cost_scale": 1.10, "charge_threshold": 0.30,
        "charge_release_threshold": 0.80,
        "observation_schema": "direct-goal-observation-v1",
    }
    if dict(profile) != expected_profile:
        raise ValueError("R1-C 4-AGV LowLoad profile is incompatible.")
    expected_training = {
        "seed": 9107, "real_environment_steps": 50000,
        "num_env_workers": 16, "update_epochs": 4, "minibatch_steps": 64,
    }
    if dict(training) != expected_training:
        raise ValueError("R1-C training configuration is incompatible.")
    if tuple(evaluation.get("seeds", ())) != (9300, 9301, 9302, 9303, 9304):
        raise ValueError("R1-C evaluation seeds are incompatible.")
    if evaluation.get("deterministic_policy") != "argmax":
        raise ValueError("R1-C evaluation policy must be deterministic argmax.")
