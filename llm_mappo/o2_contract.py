"""Frozen O2 calibration matrix and O1-Go authorization boundary."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from llm_mappo.run_evidence import RunIdentity, verify_o1_gate_receipt


_TOP_LEVEL = {
    "schema",
    "groups",
    "seeds",
    "real_env_steps",
    "llm_kd",
    "fixed_astar_kd_long_runs",
    "environment",
    "training",
}
_FROZEN_ENVIRONMENT = {
    "environment_id": "llm-mappo-medium-3ag-v1",
    "n_agents": 5,
    "dynamic_ingress_interval": 40,
    "batch_size_range": [4, 8],
    "queue_size": 8,
    "task_target": 50,
    "max_steps": 1000,
    "deadlock_steps": 180,
    "battery_cost_scale": 1.10,
    "charge_threshold": 0.30,
    "charge_release_threshold": 0.80,
    "observation_schema": "direct-goal-observation-v1",
}
_FROZEN_TRAINING = {
    "parallel_environments": 1,
    "rollout_steps": 512,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.20,
    "value_coefficient": 0.50,
    "entropy_coefficient": 0.01,
    "learning_rate": 0.0003,
    "max_grad_norm": 0.50,
    "update_epochs": 4,
    "minibatch_steps": 64,
    "checkpoint_interval": 10000,
}


@dataclass(frozen=True)
class O2RunSpec:
    """One pre-registered O2 training run."""

    group: str
    seed: int
    real_env_steps: int


@dataclass(frozen=True)
class O2ExperimentConfig:
    """The exact O2 matrix and all owner-approved execution controls."""

    schema: str
    groups: tuple[str, ...]
    seeds: tuple[int, ...]
    real_env_steps: int
    llm_kd: bool
    fixed_astar_kd_long_runs: int
    environment: Mapping[str, Any]
    training: Mapping[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "O2ExperimentConfig":
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_mapping(yaml.safe_load(stream))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "O2ExperimentConfig":
        if not isinstance(values, Mapping):
            raise ValueError("O2 configuration must be a mapping.")
        actual_keys = set(values)
        unknown = sorted(actual_keys - _TOP_LEVEL)
        missing = sorted(_TOP_LEVEL - actual_keys)
        if unknown:
            raise ValueError(f"Unknown O2 configuration field: {unknown[0]}.")
        if missing:
            raise ValueError(f"Missing O2 configuration field: {missing[0]}.")
        config = cls(
            schema=values["schema"],
            groups=tuple(values["groups"]),
            seeds=tuple(values["seeds"]),
            real_env_steps=int(values["real_env_steps"]),
            llm_kd=values["llm_kd"],
            fixed_astar_kd_long_runs=int(values["fixed_astar_kd_long_runs"]),
            environment=dict(values["environment"]),
            training=dict(values["training"]),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        expected = {
            "schema": "o2-calibration-v1",
            "groups": ("MAPPO-DG", "RC-AStarKD"),
            "seeds": (107, 117, 127),
            "real_env_steps": 150000,
            "llm_kd": False,
            "fixed_astar_kd_long_runs": 0,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Frozen O2 field {name} is incompatible.")
        if not isinstance(self.llm_kd, bool):
            raise ValueError("Frozen O2 field llm_kd must be a boolean.")
        _validate_frozen_mapping("environment", self.environment, _FROZEN_ENVIRONMENT)
        _validate_frozen_mapping("training", self.training, _FROZEN_TRAINING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "groups": list(self.groups),
            "seeds": list(self.seeds),
            "real_env_steps": self.real_env_steps,
            "llm_kd": self.llm_kd,
            "fixed_astar_kd_long_runs": self.fixed_astar_kd_long_runs,
            "environment": dict(self.environment),
            "training": dict(self.training),
        }

    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()


def _validate_frozen_mapping(
    name: str, actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if set(actual) != set(expected):
        mismatch = sorted(set(actual).symmetric_difference(expected))
        field = mismatch[0] if mismatch else name
        raise ValueError(f"Frozen O2 field {name}.{field} is incompatible.")
    for field, value in expected.items():
        if actual[field] != value:
            raise ValueError(f"Frozen O2 field {name}.{field} is incompatible.")


def expand_o2_matrix(config: O2ExperimentConfig) -> tuple[O2RunSpec, ...]:
    """Return the only pre-registered six training run specifications."""
    return tuple(
        O2RunSpec(group, seed, config.real_env_steps)
        for group in config.groups
        for seed in config.seeds
    )


def verify_o1_authorization(run_directory: str | Path) -> dict[str, Any]:
    """Verify a completed O1 Go and bind its receipt to its summary."""
    directory = Path(run_directory)
    try:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
        summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ValueError("O1 authorization artifacts are unreadable.") from error
    if state.get("status") != "complete" or state.get("gate_pass") is not True:
        raise ValueError("O1 state does not record a complete Gate Go.")
    if not all(
        summary.get(name) is True
        for name in ("gate_pass", "runtime_gate_pass", "memory_gate_pass")
    ):
        raise ValueError("O1 summary does not record a complete Gate Go.")
    try:
        identity = RunIdentity(
            code_commit=summary["code_commit"],
            config_sha256=summary["config_hash"],
            immutable_machine_sha256=summary["immutable_machine_sha256"],
            environment_sha256=summary["environment_freeze_hash"],
        )
    except KeyError as error:
        raise ValueError("O1 summary identity is incomplete.") from error
    receipt = verify_o1_gate_receipt(directory / "o1_gate_receipt.json", identity)
    canonical = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    if receipt.get("summary_sha256") != sha256(canonical.encode("utf-8")).hexdigest():
        raise ValueError("O1 receipt summary hash does not match.")
    return {
        "code_commit": identity.code_commit,
        "config_sha256": identity.config_sha256,
        "immutable_machine_sha256": identity.immutable_machine_sha256,
        "environment_sha256": identity.environment_sha256,
        "summary_sha256": receipt["summary_sha256"],
    }
