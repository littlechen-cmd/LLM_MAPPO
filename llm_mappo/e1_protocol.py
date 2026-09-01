"""Frozen E1 governance manifest parsing and formal run-matrix expansion."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


E1_MANIFEST_SCHEMA_VERSION = 9
E1_FORMAL_ENVIRONMENT_STEPS = 150000
E1_FORMAL_SEEDS = (7, 17, 27, 37, 47, 57, 67, 77)
E1_DIAGNOSTIC_SEEDS = (7, 17, 27)
E1_ARTIFACT_ROOT = "artifacts/optimization/e2_formal"
E1_CHECKPOINT_RULE = "checkpoint_final.pt"
_E1_STATUS = "d1_optimization_selected_e1_implementation_in_progress"
_E1_BLOCKERS = ("E1 selected-route protocol freeze",)


@dataclass(frozen=True)
class E1FormalRun:
    """One preregistered learning run, excluding the non-learning heuristic."""

    group: str
    seed: int
    algorithm: str
    astar_kd: str
    semantic_teacher: str
    semantic_control: str
    observation_schema: str
    real_environment_steps: int
    checkpoint_rule: str
    artifact_path: str

    @property
    def identity(self) -> str:
        return f"{self.group}:seed{self.seed:03d}"


def load_e1_governance_manifest(path: str | Path) -> Mapping[str, Any]:
    """Load the repository's E1 governance manifest as a mapping."""

    with Path(path).open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, Mapping):
        raise ValueError("E1 governance manifest must be a mapping.")
    return manifest


def validate_e1_governance_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject any manifest that drifts from the E1 preregistered contract."""

    if manifest.get("schema_version") != E1_MANIFEST_SCHEMA_VERSION:
        raise ValueError("E1 governance manifest schema version is incompatible.")
    if manifest.get("status") != _E1_STATUS:
        raise ValueError("E1 governance manifest status is incompatible.")
    if tuple(manifest.get("freeze_blockers", ())) != _E1_BLOCKERS:
        raise ValueError("E1 governance manifest blockers are incompatible.")
    training = _mapping(manifest, "training")
    if training.get("formal_environment_steps") != E1_FORMAL_ENVIRONMENT_STEPS:
        raise ValueError("E1 formal environment-step budget is incompatible.")
    if training.get("checkpoint_rule") != E1_CHECKPOINT_RULE:
        raise ValueError("E1 checkpoint rule is incompatible.")
    route = _mapping(_mapping(manifest, "route_profiles"), "optimization")
    if tuple(route.get("formal_training_seeds", ())) != E1_FORMAL_SEEDS:
        raise ValueError("E1 formal seeds are incompatible.")
    if tuple(route.get("diagnostic_training_seeds", ())) != E1_DIAGNOSTIC_SEEDS:
        raise ValueError("E1 diagnostic seeds are incompatible.")
    matrix = _mapping(manifest, "e1_formal_matrix")
    if matrix.get("artifact_root") != E1_ARTIFACT_ROOT:
        raise ValueError("E1 artifact root is incompatible.")
    if matrix.get("schema") != "e1-formal-matrix-v1":
        raise ValueError("E1 formal matrix schema is incompatible.")
    o3 = _mapping(_mapping(manifest, "evaluation"), "o3_exploratory_matrix")
    if o3.get("default_state") != "execute":
        raise ValueError("E1 must freeze O3 exploratory execution.")
    if o3.get("total_episodes") != 6400:
        raise ValueError("E1 O3 exploratory episode count is incompatible.")
    expected_episodes = (
        len(o3.get("groups", ()))
        * len(o3.get("training_seeds", ()))
        * len(o3.get("topologies", ()))
        * len(o3.get("held_out_seeds", ()))
        * int(o3.get("episodes_per_seed", 0))
    )
    if expected_episodes != 6400:
        raise ValueError("E1 O3 exploratory matrix does not expand to 6400 episodes.")


def expand_e1_formal_matrix(manifest: Mapping[str, Any]) -> tuple[E1FormalRun, ...]:
    """Expand the declarative E1 matrix into exactly 65 learning run identities."""

    validate_e1_governance_manifest(manifest)
    matrix = _mapping(manifest, "e1_formal_matrix")
    route = _mapping(_mapping(manifest, "route_profiles"), "optimization")
    training = _mapping(manifest, "training")
    runs = []
    for group, profile in _mapping(matrix, "groups").items():
        if not isinstance(profile, Mapping):
            raise ValueError(f"E1 group {group} must be a mapping.")
        seed_kind = profile.get("seed_set")
        if seed_kind == "formal":
            seeds = route["formal_training_seeds"]
        elif seed_kind == "diagnostic":
            seeds = route["diagnostic_training_seeds"]
        else:
            raise ValueError(f"E1 group {group} has an incompatible seed set.")
        for seed in seeds:
            runs.append(E1FormalRun(
                group=group,
                seed=int(seed),
                algorithm=_string(profile, "algorithm", group),
                astar_kd=_string(profile, "astar_kd", group),
                semantic_teacher=_string(profile, "semantic_teacher", group),
                semantic_control=_string(profile, "semantic_control", group),
                observation_schema=_string(profile, "observation_schema", group),
                real_environment_steps=int(training["formal_environment_steps"]),
                checkpoint_rule=str(training["checkpoint_rule"]),
                artifact_path=(
                    f"{matrix['artifact_root']}/{_slug(group)}/seed_{int(seed):03d}"
                ),
            ))
    _validate_expanded_runs(runs)
    return tuple(runs)


def _validate_expanded_runs(runs: list[E1FormalRun]) -> None:
    expected_counts = {
        "MAPPO-DG": 8,
        "RC-AStarKD": 8,
        "LLMKD": 8,
        "RC-AStarKD+LLMKD": 8,
        "Fixed-AStarKD+LLMKD": 8,
        "QMIX-DG": 8,
        "RuleKD-v3": 8,
        "ShuffleKD-v3": 3,
        "NoOOD-v1": 3,
        "NoGoalHint-v1": 3,
    }
    counts = {group: 0 for group in expected_counts}
    for run in runs:
        if run.group not in counts:
            raise ValueError(f"E1 formal matrix has an unexpected group {run.group}.")
        counts[run.group] += 1
        if run.real_environment_steps != E1_FORMAL_ENVIRONMENT_STEPS:
            raise ValueError("E1 formal run budget is incompatible.")
        if run.checkpoint_rule != E1_CHECKPOINT_RULE:
            raise ValueError("E1 formal run checkpoint rule is incompatible.")
    if counts != expected_counts or len(runs) != 65:
        raise ValueError("E1 formal matrix must contain exactly 65 preregistered runs.")
    identities = [run.identity for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("E1 formal matrix contains duplicate run identities.")


def _mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"E1 governance manifest field {key} must be a mapping.")
    return value


def _string(profile: Mapping[str, Any], key: str, group: str) -> str:
    value = profile.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"E1 group {group} field {key} must be a non-empty string.")
    return value


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace("+", "-plus-")
        .replace("*", "star")
        .replace(" ", "-")
    )
