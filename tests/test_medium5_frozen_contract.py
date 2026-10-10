"""Regression tests for the owner-approved medium-5AG C1 training contract."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from llm_mappo.mappo_baseline.runner import (
    checkpoint_schema_for_experiment,
    load_baseline_experiment,
)
from llm_mappo.warehouse_environment import Phase2Warehouse
from rware.warehouse import Action, Direction


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "core_integration"
CONFIG_FILES = (
    "medium-5ag-baseline.yaml",
    "medium-5ag-llm-kd.yaml",
    "medium-5ag-astar-kd-no-validity.yaml",
    "medium-5ag-astar-llm-kd-no-validity.yaml",
)
C1_SHA256 = "3b570b7d25f2fc264c113027def4291872db95825fce1c6d23f1eec389c688a6"


def test_four_medium5_configs_share_the_frozen_c1_contract():
    """Catch a method silently retaining the obsolete 450k/reward contract."""

    experiments = [load_baseline_experiment(CONFIG_ROOT / name) for name in CONFIG_FILES]
    baseline = experiments[0]
    for experiment in experiments:
        assert experiment.schema == "core-mappo-209d-delivery-c1-v3"
        expected_status = (
            "provisional" if experiment.method.endswith("LLM-INIT") else "approved"
        )
        assert experiment.approval_status == expected_status
        assert experiment.environment == baseline.environment
        assert experiment.evaluation == baseline.evaluation
        assert experiment.reward_definition_sha256 == C1_SHA256
        assert experiment.artifact_root.startswith(
            "artifacts/core_integration/medium5_c1_v3/"
        )
        assert experiment.environment["max_steps"] == 800
        assert experiment.environment["completion_metric_contract"] == "task-target-v1"
        assert experiment.training["environment_steps"] == 600000
        assert experiment.training["optimization_schedule_steps"] == 600000
        assert experiment.training["num_env_workers"] == 16
        assert experiment.training["rollout_length"] == 64
        assert experiment.evaluation["seeds"] == list(range(9400, 9410))
        assert checkpoint_schema_for_experiment(experiment).endswith(
            "209d-delivery-c1-checkpoint-v3"
        )


def test_c1_loader_rejects_reward_drift_before_training(tmp_path):
    """Catch a C1 label paired with changed reward magnitudes."""

    values = yaml.safe_load((CONFIG_ROOT / CONFIG_FILES[0]).read_text(encoding="utf-8"))
    values["reward_definition"]["collision_penalty"] = 1.0
    source = tmp_path / "changed-reward.yaml"
    source.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="reward definition"):
        load_baseline_experiment(source)


def test_c1_loader_rejects_approved_llm_placeholder(tmp_path):
    """Catch provisional representation placeholders being approved for training."""

    values = yaml.safe_load((CONFIG_ROOT / CONFIG_FILES[1]).read_text(encoding="utf-8"))
    values["approval_status"] = "approved"
    source = tmp_path / "approved-placeholder.yaml"
    source.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="placeholder"):
        load_baseline_experiment(source)


def test_c1_checkpoint_namespace_does_not_accept_delivery_v2():
    """Catch the changed reward reusing the old checkpoint namespace."""

    experiment = load_baseline_experiment(CONFIG_ROOT / CONFIG_FILES[0])
    previous = replace(experiment, schema="core-mappo-209d-delivery-v2")

    assert checkpoint_schema_for_experiment(previous) != (
        checkpoint_schema_for_experiment(experiment)
    )


def test_delivery_v2_cannot_opt_into_new_target_metrics(tmp_path):
    """Catch old checkpoints silently acquiring the new evaluation protocol."""

    values = yaml.safe_load((CONFIG_ROOT / CONFIG_FILES[0]).read_text(encoding="utf-8"))
    values["schema"] = "core-mappo-209d-delivery-v2"
    del values["reward_definition"]
    values["training"]["num_env_workers"] = 6
    values["evaluation"]["seeds"] = [9400, 9401]
    source = tmp_path / "delivery-v2-with-new-metrics.yaml"
    source.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="metric contract"):
        load_baseline_experiment(source)


def test_target_metrics_survive_replenishment_and_stop_at_target():
    """Catch delivery replenishment resetting the completion target to zero."""

    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=16,
        deadlock_steps=15,
        reward_version="reward-v2",
        batch_interval=40,
        batch_size_range=(1, 1),
        request_queue_size=1,
        task_completion_target=1,
        terminate_on_task_target=True,
        replenishment_mode="rware_delivery",
        completion_metric_contract="task-target-v1",
    )
    try:
        environment.reset(seed=3)
        task = environment.env.task_queue.active_tasks[0]
        shelf = environment.env.shelfs[task.shelf_id - 1]
        agent = environment.env.agents[0]
        goal_x, goal_y = environment.env.goals[0]
        agent.x, agent.y, agent.dir = goal_x, goal_y - 1, Direction.DOWN
        agent.carrying_shelf = shelf
        shelf.x, shelf.y = agent.x, agent.y
        environment.env._recalc_grid()

        transition = environment.step([Action.FORWARD])

        assert transition.metrics.completed_tasks == 1
        assert transition.metrics.task_completion_target == 1
        assert transition.metrics.completion_rate == 1.0
        assert transition.metrics.success
        assert transition.terminated
    finally:
        environment.close()
