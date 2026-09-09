"""Regression contract for the Phase 2 geometric A*KD teacher."""

import numpy as np
import torch
from pathlib import Path

from llm_mappo.mappo_astar_kd.teacher import (
    AStarKDQuery,
    AStarKDTeacher,
    StaticMap,
)
from llm_mappo.mappo_astar_kd.labels import AStarKDLabeler
from llm_mappo.mappo_astar_kd.networks import AStarKDActor
from llm_mappo.mappo_astar_kd.trainer import astar_kd_loss, astar_kd_lambda
from llm_mappo.mappo_astar_kd.trainer import AStarKDTrainer
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage
from llm_mappo.mappo_baseline.trainer import MAPPOConfig
from llm_mappo.mappo_astar_kd.runner import load_astar_kd_experiment
from llm_mappo.mappo_baseline.runner import (
    _episode_window,
    load_baseline_experiment,
)
from llm_mappo.mappo_baseline.vector_env import (
    MAPPOVectorEnvironment,
    new_baseline_environment,
)
from llm_mappo.phase2 import EpisodeMetrics, Phase2Warehouse
from rware.warehouse import Action


def _map() -> StaticMap:
    """A 3x2 map whose left cell is storage and the rest highway."""

    return StaticMap(
        width=3,
        height=2,
        highway=frozenset({(1, 0), (2, 0), (0, 1), (1, 1), (2, 1)}),
        workstations=frozenset(),
        chargers=frozenset(),
        static_obstacles=frozenset(),
    )


def test_cost_table_is_reused_across_poses_without_peer_occupancy():
    teacher = AStarKDTeacher(_map())
    first = teacher.query(
        AStarKDQuery((0, 0), "RIGHT", (2, 0), "unloaded")
    )
    second = teacher.query(
        AStarKDQuery((1, 0), "RIGHT", (2, 0), "unloaded")
    )

    assert first.valid and second.valid
    assert first.diagnostics["cache_hit"] is False
    assert second.diagnostics["cache_hit"] is True
    assert teacher.cache_build_count == 1
    assert first.soft_target.shape == (3,)
    assert np.isclose(first.soft_target.sum(), 1.0)


def test_loaded_static_geometry_permits_egress_but_not_storage_entry():
    teacher = AStarKDTeacher(_map())

    egress = teacher.query(
        AStarKDQuery((0, 0), "RIGHT", (2, 0), "loaded")
    )
    entry = teacher.query(
        AStarKDQuery((1, 0), "LEFT", (0, 0), "loaded")
    )

    assert egress.valid
    assert egress.root_costs[0] == 2.0
    assert not entry.valid
    assert entry.failure_reason == "no_finite_root_cost"


def test_no_validity_removes_only_semantic_gates():
    teacher = AStarKDTeacher(_map())
    semantic = AStarKDQuery(
        (1, 0),
        "RIGHT",
        (1, 0),
        "unloaded",
        mandatory_toggle_load=True,
    )

    full = teacher.query(semantic, use_semantic_validity=True)
    ablation = teacher.query(semantic, use_semantic_validity=False)

    assert not full.valid
    assert full.failure_reason == "mandatory_toggle_load"
    assert ablation.valid
    assert np.isclose(ablation.soft_target.sum(), 1.0)


def test_labeler_uses_direct_rule_goal_and_keeps_peer_occupancy_out_of_target():
    values = {
        "environment_id": "llm-mappo-medium-3ag-v1",
        "n_agents": 3,
        "max_steps": 20,
        "dynamic_ingress_interval": 30,
        "batch_size_range": [1, 1],
        "queue_size": 3,
        "task_target": 1,
        "training_replenishment_mode": "rware_delivery",
        "initial_priority_label": "A",
        "priority_schedule": ["A", "B", "C", "D"],
        "charging_stations": [[0, 0], [19, 0], [0, 23]],
        "picking_stations": [[9, 21]],
        "battery_cost_scale": 1.1,
        "charge_threshold": 0.3,
        "charge_release_threshold": 0.8,
        "deadlock_steps": 20,
        "observation_schema": "direct-goal-observation-v1",
        "reward_version": "reward-v2",
        "oracle_interaction_mask": True,
    }
    environment = new_baseline_environment(values)
    try:
        environment.reset(seed=7)
        labels = AStarKDLabeler(environment).labels(use_semantic_validity=True)
        assert labels.soft_targets.shape == (3, 3)
        assert labels.validity.shape == (3,)
        assert labels.attenuation.shape == (3,)
        assert labels.query_count == 3
        assert np.all((labels.attenuation == 1.0) | (labels.attenuation == 0.25))
        assert all(reason for reason in labels.failure_reasons)
    finally:
        environment.close()


def test_continuous_training_timeout_is_a_truncation_not_a_task_termination():
    """Catch target-driven or terminal timeout endings in continuous training."""

    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=1,
        request_queue_size=1,
        task_completion_target=18,
        replenishment_mode="rware_delivery",
        terminate_on_task_target=False,
    )
    try:
        environment.reset(seed=7)
        transition = environment.step([Action.NOOP])

        assert not transition.terminated
        assert transition.truncated
        assert transition.metrics.steps == 1
    finally:
        environment.close()


def test_steps_per_task_keeps_zero_completion_undefined_and_aggregates_totals():
    """Catch finite zero-completion throughput and mean-of-ratios summaries."""

    assert EpisodeMetrics(steps=800, completed_tasks=0).steps_per_task is None
    assert EpisodeMetrics(steps=800, completed_tasks=4).steps_per_task == 200.0

    window = _episode_window(
        [
            {"steps": "800", "completed_tasks": "4", "reward": "1.0",
             "task_completion_rate": "0.5"},
            {"steps": "800", "completed_tasks": "0", "reward": "2.0",
             "task_completion_rate": "0.0"},
        ],
        primary_metric="steps_per_task",
    )

    assert window["steps_per_task"] == 400.0
    assert window["primary_metric"] == "steps_per_task"


def test_phase2_training_profiles_disable_task_target_termination_only_in_training():
    """Catch a training profile that restores target termination or alters eval."""

    root = Path("configs/phase2")
    names = (
        "medium-3ag-astar-kd.yaml",
        "medium-5ag-baseline.yaml",
        "medium-5ag-astar-kd-no-validity.yaml",
        "medium-5ag-astar-kd.yaml",
    )

    for name in names:
        profile = load_baseline_experiment(root / name)
        assert profile.environment["terminate_on_task_target"] is False
        assert profile.evaluation["environment_overrides"][
            "terminate_on_task_target"
        ] is True


def test_vector_step_refreshes_current_astar_kd_payload():
    """A rollout must consume the labels returned for its current state."""

    values = {
        "environment_id": "llm-mappo-medium-3ag-v1",
        "n_agents": 3,
        "max_steps": 20,
        "dynamic_ingress_interval": 30,
        "batch_size_range": [1, 1],
        "queue_size": 3,
        "task_target": 1,
        "training_replenishment_mode": "rware_delivery",
        "initial_priority_label": "A",
        "priority_schedule": ["A", "B", "C", "D"],
        "charging_stations": [[0, 0], [19, 0], [0, 23]],
        "picking_stations": [[9, 21]],
        "battery_cost_scale": 1.1,
        "charge_threshold": 0.3,
        "charge_release_threshold": 0.8,
        "deadlock_steps": 20,
        "observation_schema": "direct-goal-observation-v1",
        "reward_version": "reward-v2",
        "oracle_interaction_mask": True,
        "astar_kd_variant": "full",
        "worker_numerical_threads": 1,
    }
    pool = MAPPOVectorEnvironment(values, 1, 7)
    try:
        pool.step(np.zeros((1, 3), dtype=np.int64), global_step=0)
        current = pool.astar_kd

        assert current["query_count"] == 3
        assert current["cache_build_count"] == 0
        assert current["cache_hit_count"] == 3
    finally:
        pool.close()


def test_partial_vector_step_counts_only_active_teacher_queries():
    """A final partial vector batch must not re-count inactive worker labels."""

    values = {
        "environment_id": "llm-mappo-medium-3ag-v1",
        "n_agents": 3,
        "max_steps": 20,
        "dynamic_ingress_interval": 30,
        "batch_size_range": [1, 1],
        "queue_size": 3,
        "task_target": 1,
        "training_replenishment_mode": "rware_delivery",
        "initial_priority_label": "A",
        "priority_schedule": ["A", "B", "C", "D"],
        "charging_stations": [[0, 0], [19, 0], [0, 23]],
        "picking_stations": [[9, 21]],
        "battery_cost_scale": 1.1,
        "charge_threshold": 0.3,
        "charge_release_threshold": 0.8,
        "deadlock_steps": 20,
        "observation_schema": "direct-goal-observation-v1",
        "reward_version": "reward-v2",
        "oracle_interaction_mask": True,
        "astar_kd_variant": "full",
        "worker_numerical_threads": 1,
    }
    pool = MAPPOVectorEnvironment(values, 2, 7)
    try:
        pool.step(
            np.zeros((2, 3), dtype=np.int64),
            global_step=0,
            active_workers=1,
        )
        current = pool.astar_kd

        assert current["soft_targets"].shape == (2, 3, 3)
        assert current["query_count"] == 3
        assert current["cache_build_count"] == 0
        assert current["cache_hit_count"] == 3
    finally:
        pool.close()


def test_motion_prior_head_uses_the_actor_gru_without_changing_policy_shape():
    actor = AStarKDActor(observation_dim=7, action_dim=5, hidden_dim=8)
    observations = __import__("torch").zeros((2, 3, 7))

    policy_logits, motion_logits, hidden = actor.forward_with_motion(observations)

    assert policy_logits.shape == (2, 3, 5)
    assert motion_logits.shape == (2, 3, 3)
    assert hidden.shape == (2, 3, 8)


def test_kd_loss_uses_valid_count_not_attenuation_sum_as_denominator():
    logits = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    validity = torch.tensor([True, True])
    attenuation = torch.tensor([1.0, 0.25])

    loss = astar_kd_loss(logits, targets, validity, attenuation, lambda_value=1.0)

    assert torch.isclose(loss, torch.tensor(0.625 * np.log(3.0), dtype=loss.dtype))


def test_lambda_schedule_has_a_nonzero_tail():
    assert astar_kd_lambda(0, 300_000) == 0.05
    assert astar_kd_lambda(210_000, 300_000) == 0.01
    assert astar_kd_lambda(300_000, 300_000) == 0.01


def test_astar_kd_trainer_updates_motion_head_from_recurrent_rollout():
    trainer = AStarKDTrainer(
        MAPPOConfig(observation_dim=4, state_dim=3, action_dim=5, ppo_epochs=1),
        "cpu",
        total_environment_steps=20,
    )
    storage = MAPPORolloutStorage(2, 1, 1, 4, 3, 5, 64, astar_kd_enabled=True)
    actor_hidden = np.zeros((1, 1, 64), dtype=np.float32)
    critic_hidden = np.zeros((1, 1, 64), dtype=np.float32)
    for _ in range(2):
        observations = np.zeros((1, 1, 4), dtype=np.float32)
        states = np.zeros((1, 1, 3), dtype=np.float32)
        masks = np.ones((1, 1, 5), dtype=bool)
        output = trainer.act(
            observations, states, masks, actor_hidden, critic_hidden, deterministic=False
        )
        actions, log_probs, values, predictions, next_actor, next_critic = output
        storage.insert(
            observations=observations, fp_states=states, actions=actions,
            action_masks=masks, log_probs=log_probs, values=values,
            value_predictions=predictions, rewards=np.ones((1, 1), dtype=np.float32),
            dones=np.zeros((1, 1), dtype=bool),
            active_masks=np.ones((1, 1), dtype=np.float32),
            actor_hidden_states=actor_hidden, critic_hidden_states=critic_hidden,
            astar_soft_targets=np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32),
            astar_validity=np.ones((1, 1), dtype=bool),
            astar_attenuation=np.ones((1, 1), dtype=np.float32),
        )
        actor_hidden, critic_hidden = next_actor, next_critic
    trainer.set_global_environment_steps(4)
    batch = storage.compute_returns(np.zeros((1, 1), dtype=np.float32), gamma=.99, gae_lambda=.95)
    metrics = trainer.update(batch)
    assert metrics["astar_loss"] > 0.0
    assert np.isfinite(metrics["total_loss"])


def test_phase2_training_profile_uses_target_18_only_for_training():
    profile = load_astar_kd_experiment(
        Path("configs/phase2/medium-5ag-astar-kd.yaml")
    )

    assert profile.environment["task_target"] == 18
    assert profile.evaluation["environment_overrides"]["task_target"] == 8
    assert profile.training["seed"] == 7
    assert profile.evaluation["seeds"] == [9400, 9401]


def test_large_pressure_profiles_are_dynamic_high_workload_evaluations():
    root = Path("configs/phase2")
    names = (
        "large-5ag-high-baseline-eval.yaml",
        "large-5ag-high-astar-kd-no-validity-eval.yaml",
        "large-5ag-high-astar-kd-eval.yaml",
    )

    for name in names:
        profile = load_baseline_experiment(root / name)
        assert profile.environment["environment_id"] == "llm-mappo-large-5ag-v1"
        assert profile.environment["max_steps"] == 800
        assert profile.environment["dynamic_ingress_interval"] == 60
        assert profile.environment["batch_size_range"] == [2, 2]
        assert profile.environment["task_target"] == 12
        assert profile.evaluation["seeds"] == [9400, 9401]
