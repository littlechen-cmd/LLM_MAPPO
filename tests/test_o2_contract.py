"""O2 must expand only its frozen calibration matrix and consume an O1 Go."""

import json
from dataclasses import asdict

import pytest

from llm_mappo.run_evidence import RunIdentity, write_o1_gate_receipt


def _config_mapping():
    return {
        "schema": "o2-calibration-v1",
        "groups": ["MAPPO-DG", "RC-AStarKD"],
        "seeds": [107, 117, 127],
        "real_env_steps": 150000,
        "llm_kd": False,
        "fixed_astar_kd_long_runs": 0,
        "environment": {
            "environment_id": "llm-mappo-medium-3ag-v1",
            "n_agents": 5,
            "dynamic_ingress_interval": 40,
            "batch_size_range": [4, 8],
            "initial_priority_label": "A",
            "queue_size": 8,
            "task_target": 50,
            "max_steps": 1000,
            "deadlock_steps": 180,
            "battery_cost_scale": 1.10,
            "charge_threshold": 0.30,
            "charge_release_threshold": 0.80,
            "observation_schema": "direct-goal-observation-v1",
        },
        "training": {
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
        },
    }


def test_o2_contract_expands_only_the_six_frozen_runs():
    """A changed group, seed, budget, or training contract must reject O2."""
    from llm_mappo.o2_contract import O2ExperimentConfig, expand_o2_matrix

    config = O2ExperimentConfig.from_mapping(_config_mapping())

    assert [asdict(spec) for spec in expand_o2_matrix(config)] == [
        {"group": "MAPPO-DG", "seed": 107, "real_env_steps": 150000},
        {"group": "MAPPO-DG", "seed": 117, "real_env_steps": 150000},
        {"group": "MAPPO-DG", "seed": 127, "real_env_steps": 150000},
        {"group": "RC-AStarKD", "seed": 107, "real_env_steps": 150000},
        {"group": "RC-AStarKD", "seed": 117, "real_env_steps": 150000},
        {"group": "RC-AStarKD", "seed": 127, "real_env_steps": 150000},
    ]
    assert config.llm_kd is False
    assert config.training["parallel_environments"] == 1

    changed = _config_mapping()
    changed["real_env_steps"] = 150001
    with pytest.raises(ValueError, match="real_env_steps"):
        O2ExperimentConfig.from_mapping(changed)


def test_o2_contract_rejects_any_unknown_or_changed_environment_field():
    """A permissive parser could silently change the formal O2 environment."""
    from llm_mappo.o2_contract import O2ExperimentConfig

    unknown = _config_mapping()
    unknown["extra"] = True
    with pytest.raises(ValueError, match="Unknown"):
        O2ExperimentConfig.from_mapping(unknown)

    changed = _config_mapping()
    changed["environment"]["charge_threshold"] = 0.25
    with pytest.raises(ValueError, match="charge_threshold"):
        O2ExperimentConfig.from_mapping(changed)

    changed = _config_mapping()
    changed["environment"]["initial_priority_label"] = "B"
    with pytest.raises(ValueError, match="initial_priority_label"):
        O2ExperimentConfig.from_mapping(changed)

    invalid_type = _config_mapping()
    invalid_type["llm_kd"] = 0
    with pytest.raises(ValueError, match="llm_kd"):
        O2ExperimentConfig.from_mapping(invalid_type)


def test_o2_authorization_requires_a_complete_receipted_o1_go(tmp_path):
    """A forged, stale, or failed O1 artifact must not unlock an O2 run."""
    from llm_mappo.o2_contract import verify_o1_authorization

    identity = RunIdentity("7c305ea", "config", "machine", "environment")
    summary = {
        "code_commit": identity.code_commit,
        "config_hash": identity.config_sha256,
        "immutable_machine_sha256": identity.immutable_machine_sha256,
        "environment_freeze_hash": identity.environment_sha256,
        "gate_pass": True,
        "runtime_gate_pass": True,
        "memory_gate_pass": True,
    }
    (tmp_path / "state.json").write_text(
        json.dumps({"status": "complete", "gate_pass": True}), encoding="utf-8"
    )
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_o1_gate_receipt(summary, identity, tmp_path / "o1_gate_receipt.json")

    verified = verify_o1_authorization(tmp_path)
    assert verified["code_commit"] == "7c305ea"

    summary["gate_pass"] = False
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="complete Gate Go"):
        verify_o1_authorization(tmp_path)

    summary["gate_pass"] = True
    summary["device"] = "forged-after-receipt"
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="summary hash"):
        verify_o1_authorization(tmp_path)
