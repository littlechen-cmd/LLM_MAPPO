"""Contracts for the isolated Group 2 reward-diagnostic entry point."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import yaml
import numpy as np

from llm_mappo.mappo_baseline.runner import load_baseline_experiment


DIAGNOSTIC_SCHEMA = "reward-diagnostic-mappo-209d-delivery-v1"


def _diagnostic_config():
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "method": "MAPPO-FP-DG",
        "approval_status": "provisional",
        "artifact_root": "artifacts/reward_diagnostics/group2/reward-v2",
        "environment": {
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
        },
        "reward_definition": {
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
        },
        "training": {
            "seed": 7,
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
        },
        "evaluation": {
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
        },
    }


def _candidate_config(candidate):
    values = deepcopy(_diagnostic_config())
    if candidate == "c1":
        values["artifact_root"] = (
            "artifacts/reward_diagnostics/group2/"
            "reward-v3-c1-energy-death-local-mean"
        )
        values["environment"]["reward_progress_coefficient"] = 0.1
        values["environment"]["reward_energy_death_aggregation"] = "local-mean"
        values["reward_definition"].update(
            {
                "version": "reward-v3-c1-energy-death-local-mean",
                "energy_death_penalty": 20.0,
                "energy_death_aggregation": "local-mean",
            }
        )
    elif candidate == "c2":
        values["artifact_root"] = (
            "artifacts/reward_diagnostics/group2/"
            "reward-v3-c2-progress-0p2"
        )
        values["environment"]["reward_progress_coefficient"] = 0.2
        values["environment"]["reward_energy_death_aggregation"] = "team-broadcast"
        values["reward_definition"].update(
            {
                "version": "reward-v3-c2-progress-0p2",
                "progress_reward": 0.2,
                "energy_death_penalty": 20.0,
                "energy_death_aggregation": "team-broadcast",
            }
        )
    elif candidate == "c3":
        values["artifact_root"] = (
            "artifacts/reward_diagnostics/group2/"
            "reward-v3-c3-collision-1p0"
        )
        values["environment"]["reward_progress_coefficient"] = 0.1
        values["environment"]["reward_energy_death_aggregation"] = "team-broadcast"
        values["environment"]["reward_collision_penalty"] = 1.0
        values["reward_definition"].update(
            {
                "version": "reward-v3-c3-collision-1p0",
                "collision_penalty": 1.0,
                "energy_death_penalty": 20.0,
                "energy_death_aggregation": "team-broadcast",
            }
        )
    elif candidate == "c4":
        values["artifact_root"] = (
            "artifacts/reward_diagnostics/group2/"
            "reward-v3-c4-blocked-forward-0p5"
        )
        values["environment"]["reward_progress_coefficient"] = 0.1
        values["environment"]["reward_energy_death_aggregation"] = "team-broadcast"
        values["environment"]["reward_blocked_forward_penalty"] = 0.5
        values["reward_definition"].update(
            {
                "version": "reward-v3-c4-blocked-forward-0p5",
                "blocked_forward_penalty": 0.5,
                "energy_death_penalty": 20.0,
                "energy_death_aggregation": "team-broadcast",
            }
        )
    else:
        raise ValueError(candidate)
    return values


def _write_config(tmp_path, values):
    path = tmp_path / "reward-diagnostic.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def test_production_loader_rejects_reward_diagnostic_schema(tmp_path):
    """Catch production loading a provisional diagnostic as a normal experiment."""
    path = _write_config(tmp_path, _diagnostic_config())

    with pytest.raises(ValueError, match="diagnostic"):
        load_baseline_experiment(path)


def test_diagnostic_loader_accepts_only_the_frozen_reward_v2_contract(tmp_path):
    """Catch a diagnostic run drifting from its pre-registered environment contract."""
    from llm_mappo.reward_diagnostics import load_reward_diagnostic_experiment

    path = _write_config(tmp_path, _diagnostic_config())

    experiment = load_reward_diagnostic_experiment(path)

    assert experiment.schema == DIAGNOSTIC_SCHEMA
    assert experiment.training["num_env_workers"] == 16
    assert experiment.training["environment_steps"] == 300000
    assert experiment.environment["max_steps"] == 800
    assert experiment.evaluation["seeds"] == list(range(9400, 9410))
    assert len(experiment.reward_definition_sha256) == 64


@pytest.mark.parametrize("candidate", ("c1", "c2", "c3", "c4"))
def test_diagnostic_loader_accepts_only_owner_approved_single_factor_candidates(
    tmp_path, candidate
):
    from llm_mappo.reward_diagnostics import load_reward_diagnostic_experiment

    reference = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _diagnostic_config())
    )
    approved = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _candidate_config(candidate))
    )

    assert approved.reward_definition["version"].startswith("reward-v3-")
    assert approved.reward_definition_sha256 != reference.reward_definition_sha256
    assert approved.artifact_root != reference.artifact_root


def test_diagnostic_loader_rejects_mixed_candidate_factors(tmp_path):
    from llm_mappo.reward_diagnostics import load_reward_diagnostic_experiment

    values = _candidate_config("c1")
    values["environment"]["reward_progress_coefficient"] = 0.2
    values["reward_definition"]["progress_reward"] = 0.2

    with pytest.raises(ValueError, match="single-factor"):
        load_reward_diagnostic_experiment(_write_config(tmp_path, values))


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("environment", "max_steps"), "max_steps"),
        (("training", "num_env_workers"), "workers"),
        (("evaluation", "seeds"), "evaluation seeds"),
        (("environment", "delivery_contract"), "delivery"),
    ],
)
def test_diagnostic_loader_rejects_contract_drift(tmp_path, path, message):
    """Catch a reward comparison whose environment or evaluation has changed."""
    from llm_mappo.reward_diagnostics import load_reward_diagnostic_experiment

    values = deepcopy(_diagnostic_config())
    target = values
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = {
        "max_steps": 500,
        "num_env_workers": 6,
        "seeds": [9400, 9401],
        "delivery_contract": "legacy-v1",
    }[path[-1]]

    with pytest.raises(ValueError, match=message):
        load_reward_diagnostic_experiment(_write_config(tmp_path, values))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("method",), "MAPPO-FP-ASTARKD-NO-VALIDITY", "original MAPPO"),
        (("training", "actor_learning_rate"), 0.0001, "actor_learning_rate"),
        (("environment", "goal_interface"), "waypoint", "goal_interface"),
    ],
)
def test_diagnostic_loader_rejects_teacher_or_hyperparameter_drift(
    tmp_path, path, value, message
):
    """Catch a reward diagnostic silently becoming teacher-assisted or retuned."""
    from llm_mappo.reward_diagnostics import load_reward_diagnostic_experiment

    values = deepcopy(_diagnostic_config())
    target = values
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        load_reward_diagnostic_experiment(_write_config(tmp_path, values))


def test_diagnostic_invocation_requires_explicit_acknowledgement(tmp_path):
    """Catch accidental provisional diagnostic training without an owner opt-in."""
    from llm_mappo.reward_diagnostics import (
        load_reward_diagnostic_experiment,
        validate_diagnostic_invocation,
    )

    experiment = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _diagnostic_config())
    )

    with pytest.raises(ValueError, match="acknowledgement"):
        validate_diagnostic_invocation(
            experiment,
            seed=7,
            output_root=tmp_path / "artifacts" / "reward_diagnostics" / "group2",
            acknowledged=False,
        )


def test_diagnostic_resolution_uses_a_distinct_checkpoint_namespace(tmp_path):
    """Catch a diagnostic seed sharing an ordinary core checkpoint meaning."""
    from llm_mappo.mappo_baseline.runner import checkpoint_schema_for_experiment
    from llm_mappo.reward_diagnostics import (
        REWARD_DIAGNOSTIC_CHECKPOINT_SCHEMA,
        load_reward_diagnostic_experiment,
        resolve_diagnostic_baseline_experiment,
    )

    diagnostic = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _diagnostic_config())
    )
    baseline = resolve_diagnostic_baseline_experiment(diagnostic, seed=17)

    assert baseline.training["seed"] == 17
    assert checkpoint_schema_for_experiment(baseline) == REWARD_DIAGNOSTIC_CHECKPOINT_SCHEMA


def test_diagnostic_invocation_rejects_a_non_diagnostic_artifact_root(tmp_path):
    """Catch a provisional diagnostic polluting formal core artifacts."""
    from llm_mappo.reward_diagnostics import (
        load_reward_diagnostic_experiment,
        validate_diagnostic_invocation,
    )

    experiment = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _diagnostic_config())
    )

    with pytest.raises(ValueError, match="outside group2 artifacts"):
        validate_diagnostic_invocation(
            experiment,
            seed=7,
            output_root=tmp_path / "formal-artifacts",
            acknowledged=True,
        )


def test_diagnostic_loader_rejects_a_non_diagnostic_config_artifact_root(tmp_path):
    """Catch a diagnostic config that points its manifest at formal artifacts."""
    from llm_mappo.reward_diagnostics import load_reward_diagnostic_experiment

    values = _diagnostic_config()
    values["artifact_root"] = "artifacts/core_integration/baseline"

    with pytest.raises(ValueError, match="artifact root"):
        load_reward_diagnostic_experiment(_write_config(tmp_path, values))


def test_diagnostic_script_refuses_training_without_acknowledgement(tmp_path):
    """Catch the CLI starting a provisional 300k run without owner opt-in."""
    config = _write_config(tmp_path, _diagnostic_config())
    script = Path(__file__).parents[1] / "scripts" / "run_reward_diagnostic.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--seed",
            "7",
            "--device",
            "cpu",
            "--output-root",
            str(tmp_path / "formal-artifacts"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "acknowledgement" in result.stderr.lower()
    assert not (tmp_path / "formal-artifacts").exists()


def test_update_evidence_contract_persists_raw_return_and_advantage_scales():
    """Catch diagnostic distribution statistics being lost before CSV evidence writing."""
    from llm_mappo.mappo_baseline.runner import BaselineRun

    for prefix in ("raw_advantage", "raw_return"):
        for suffix in (
            "count",
            "mean",
            "std",
            "min",
            "max",
            "p05",
            "p50",
            "p95",
            "max_abs",
        ):
            assert f"{prefix}_{suffix}" in BaselineRun.UPDATE_FIELDS


def test_local_smoke_experiment_is_isolated_from_server_diagnostic_budget(tmp_path):
    """Catch a local lifecycle probe inheriting the owner-only 300k contract."""
    from llm_mappo.reward_diagnostics import (
        load_reward_diagnostic_experiment,
        make_local_smoke_experiment,
    )

    diagnostic = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _diagnostic_config())
    )
    smoke = make_local_smoke_experiment(diagnostic)

    assert smoke.training["environment_steps"] == 64
    assert smoke.training["optimization_schedule_steps"] == 64
    assert smoke.training["num_env_workers"] == 1
    assert smoke.evaluation["seeds"] == [9400]
    assert smoke.evaluation["environment_overrides"]["max_steps"] == 32
    assert smoke.config_sha256 != diagnostic.baseline.config_sha256


def test_diagnostic_run_uses_one_stable_nonreusable_directory_per_seed(tmp_path):
    """Catch timestamped seed aliases or silent replacement of diagnostic evidence."""
    from llm_mappo.mappo_baseline.runner import BaselineRun
    from llm_mappo.reward_diagnostics import (
        load_reward_diagnostic_experiment,
        make_local_smoke_experiment,
    )

    diagnostic = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _diagnostic_config())
    )
    experiment = make_local_smoke_experiment(diagnostic)
    root = tmp_path / "artifacts" / "reward_diagnostics" / "group2" / "reward-v2"
    run = BaselineRun(
        experiment,
        device="cpu",
        output_root=root,
        run_directory_name="seed_0007",
        diagnostic_authorized=True,
        diagnostic_reward_definition=diagnostic.reward_definition,
        diagnostic_reward_definition_sha256=diagnostic.reward_definition_sha256,
    )
    try:
        assert run.directory == root / "seed_0007"
    finally:
        run.close()

    with pytest.raises(FileExistsError, match="already exists"):
        BaselineRun(
            experiment,
            device="cpu",
            output_root=root,
            run_directory_name="seed_0007",
            diagnostic_authorized=True,
            diagnostic_reward_definition=diagnostic.reward_definition,
            diagnostic_reward_definition_sha256=diagnostic.reward_definition_sha256,
        )


@pytest.mark.parametrize("candidate", (None, "c1", "c2", "c3", "c4"))
def test_diagnostic_checkpoint_resume_requires_the_reward_hash_identity(
    tmp_path, candidate
):
    """Catch a resumed diagnostic losing its reward-definition checkpoint identity."""
    from llm_mappo.mappo_baseline.runner import BaselineRun
    from llm_mappo.reward_diagnostics import (
        load_reward_diagnostic_experiment,
        make_local_smoke_experiment,
    )

    diagnostic = load_reward_diagnostic_experiment(
        _write_config(
            tmp_path,
            _diagnostic_config() if candidate is None else _candidate_config(candidate),
        )
    )
    experiment = make_local_smoke_experiment(diagnostic)
    run = BaselineRun(
        experiment,
        device="cpu",
        output_root=tmp_path,
        diagnostic_authorized=True,
        diagnostic_reward_definition=diagnostic.reward_definition,
        diagnostic_reward_definition_sha256=diagnostic.reward_definition_sha256,
    )
    try:
        run.actor_hidden_states.fill(0.25)
        run.critic_hidden_states.fill(-0.5)
        run._checkpoint(run.directory / "checkpoint_latest.pt")
        directory = run.directory
        actions = np.zeros((1, 5), dtype=np.int64)
        expected = run.pool.step(actions, global_step=0)
        expected_observations = run.pool.observations.copy()
        expected_fp_states = run.pool.fp_states.copy()
        expected_masks = run.pool.action_masks.copy()
    finally:
        run.close()

    resumed = BaselineRun(
        experiment,
        device="cpu",
        resume=directory,
        diagnostic_authorized=True,
        diagnostic_reward_definition=diagnostic.reward_definition,
        diagnostic_reward_definition_sha256=diagnostic.reward_definition_sha256,
    )
    try:
        np.testing.assert_array_equal(
            resumed.actor_hidden_states, np.full((1, 5, 64), 0.25)
        )
        np.testing.assert_array_equal(
            resumed.critic_hidden_states, np.full((1, 5, 64), -0.5)
        )
        actual = resumed.pool.step(actions, global_step=0)
        np.testing.assert_array_equal(actual.rewards, expected.rewards)
        np.testing.assert_array_equal(actual.dones, expected.dones)
        np.testing.assert_array_equal(actual.active_masks, expected.active_masks)
        np.testing.assert_array_equal(resumed.pool.observations, expected_observations)
        np.testing.assert_array_equal(resumed.pool.fp_states, expected_fp_states)
        np.testing.assert_array_equal(resumed.pool.action_masks, expected_masks)
    finally:
        resumed.close()


def test_diagnostic_checkpoint_rejects_cross_candidate_resume(tmp_path):
    from llm_mappo.mappo_baseline.runner import BaselineRun
    from llm_mappo.reward_diagnostics import (
        load_reward_diagnostic_experiment,
        make_local_smoke_experiment,
    )

    reference = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _diagnostic_config())
    )
    reference_smoke = make_local_smoke_experiment(reference)
    run = BaselineRun(
        reference_smoke,
        device="cpu",
        output_root=tmp_path,
        diagnostic_authorized=True,
        diagnostic_reward_definition=reference.reward_definition,
        diagnostic_reward_definition_sha256=reference.reward_definition_sha256,
    )
    try:
        run._checkpoint(run.directory / "checkpoint_latest.pt")
        directory = run.directory
    finally:
        run.close()

    candidate = load_reward_diagnostic_experiment(
        _write_config(tmp_path, _candidate_config("c1"))
    )
    candidate_smoke = make_local_smoke_experiment(candidate)
    with pytest.raises(ValueError, match="identity"):
        mismatched = BaselineRun(
            candidate_smoke,
            device="cpu",
            resume=directory,
            diagnostic_authorized=True,
            diagnostic_reward_definition=candidate.reward_definition,
            diagnostic_reward_definition_sha256=(
                candidate.reward_definition_sha256
            ),
        )
        mismatched.close()
