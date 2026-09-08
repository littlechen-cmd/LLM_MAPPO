"""Group 3 regression tests for executable Phase 1 MAPPO diagnostics."""

from pathlib import Path

import pytest
import yaml

from llm_mappo.environment import DynamicWarehouse
from llm_mappo.mappo_baseline.runner import load_baseline_experiment
from rware.warehouse import RewardType


def _write_config(path: Path, environment: dict, training: dict) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "phase1-mappo-baseline-v1",
                "method": "MAPPO-FP-DG",
                "artifact_root": "artifacts/phase1/test",
                "environment": environment,
                "training": training,
                "evaluation": {
                    "seeds": [9400, 9401],
                    "deterministic_policy": "argmax",
                    "replay_frame_stride": 1,
                    "replay_frame_duration_ms": 600,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_phase1_config_accepts_the_six_worker_low_workload_profile(tmp_path):
    config = _write_config(
        tmp_path / "medium3.yaml",
        {
            "profile_id": "medium-3ag-low-workload",
            "environment_id": "llm-mappo-medium-3ag-v1",
            "n_agents": 3,
            "max_steps": 500,
            "dynamic_ingress_interval": 120,
            "batch_size_range": [1, 1],
            "queue_size": 3,
            "task_target": 6,
            "initial_priority_label": "A",
            "priority_schedule": ["A", "B", "C", "D"],
            "charging_stations": [[0, 0], [19, 0], [0, 23]],
            "picking_stations": [[9, 21]],
            "battery_cost_scale": 1.10,
            "charge_threshold": 0.30,
            "charge_release_threshold": 0.80,
            "deadlock_steps": 180,
            "observation_schema": "direct-goal-observation-v1",
            "reward_version": "reward-v2",
            "oracle_interaction_mask": True,
        },
        {
            "seed": 7,
            "environment_steps": 300000,
            "num_env_workers": 6,
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
    )

    experiment = load_baseline_experiment(config)

    assert experiment.training["num_env_workers"] == 6
    assert experiment.environment["picking_stations"] == [[9, 21]]


def test_dynamic_warehouse_uses_only_configured_picking_stations():
    warehouse = DynamicWarehouse(
        column_height=8,
        shelf_rows=2,
        shelf_columns=5,
        n_agents=3,
        msg_bits=0,
        sensor_range=4,
        outer_aisle_width=2,
        request_queue_size=3,
        max_inactivity_steps=None,
        max_steps=10,
        reward_type=RewardType.INDIVIDUAL,
        picking_stations=((9, 21),),
    )
    try:
        assert warehouse.picking_stations == ((9, 21),)
    finally:
        warehouse.close()


def test_phase1_config_rejects_a_missing_worker_thread_contract(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs" / "phase1" / (
        "medium-3ag-low-workload.yaml"
    )
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    del values["training"]["worker_numerical_threads"]
    config = tmp_path / "incomplete.yaml"
    config.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="worker_numerical_threads"):
        load_baseline_experiment(config)
