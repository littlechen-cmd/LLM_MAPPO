"""Regression contract for the Phase 2 geometric A*KD teacher."""

from dataclasses import replace
import numpy as np
import pytest
import torch
from pathlib import Path

from llm_mappo.mappo_astar_kd.teacher import (
    AStarKDQuery,
    AStarKDTeacher,
    StaticMap,
)
from llm_mappo.mappo_astar_kd.labels import AStarKDLabeler
from llm_mappo.mappo_astar_kd.networks import AStarKDActor
from llm_mappo.mappo_astar_kd.trainer import (
    astar_kd_diagnostics,
    astar_kd_loss,
    astar_kd_lambda,
)
from llm_mappo.mappo_astar_kd.trainer import AStarKDTrainer
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer
from llm_mappo.mappo_astar_kd.runner import load_astar_kd_experiment
from llm_mappo.mappo_baseline.runner import (
    BaselineRun,
    _episode_window,
    _validate_training,
    load_baseline_experiment,
)
from llm_mappo.mappo_baseline.vector_env import (
    MAPPOVectorEnvironment,
    new_baseline_environment,
)
from llm_mappo.warehouse_environment import EpisodeMetrics, Phase2Warehouse
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
        assert labels.root_costs.shape == (3, 3)
        assert labels.validity.shape == (3,)
        assert labels.supervision_scope.shape == (3,)
        assert labels.supervision_scope.tolist() == [True, True, True]
        assert labels.attenuation.shape == (3,)
        assert labels.query_count == 3
        assert np.all((labels.attenuation == 1.0) | (labels.attenuation == 0.25))
        assert all(reason for reason in labels.failure_reasons)
    finally:
        environment.close()


def test_full_validity_labels_forced_pickup_as_loaded_delivery_transition():
    """Keep the post-pickup route supervised across the forced-toggle boundary."""

    values = {
        "environment_id": "llm-mappo-medium-3ag-v1",
        "n_agents": 1,
        "max_steps": 20,
        "dynamic_ingress_interval": 30,
        "batch_size_range": [1, 1],
        "queue_size": 1,
        "task_target": 1,
        "training_replenishment_mode": "rware_delivery",
        "initial_priority_label": "A",
        "priority_schedule": ["A", "B", "C", "D"],
        "charging_stations": [[0, 0]],
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
        agent = environment.env.agents[0]
        task = environment.env.task_queue.task_for_agent(agent.id)
        shelf = environment.env.shelfs[task.shelf_id - 1]
        agent.x, agent.y = shelf.x, shelf.y
        labeler = AStarKDLabeler(environment)
        queries = []
        query = labeler.teacher.query

        def record(candidate, *, use_semantic_validity=True):
            queries.append(candidate)
            return query(
                candidate, use_semantic_validity=use_semantic_validity
            )

        labeler.teacher.query = record
        labels = labeler.labels(use_semantic_validity=True)

        assert labels.validity.tolist() == [True]
        assert labels.supervision_scope.tolist() == [True]
        assert labels.failure_reasons == ("ok",)
        assert queries[0].pose == (shelf.x, shelf.y)
        assert queries[0].goal == (9, 21)
        assert queries[0].footprint == "loaded"
        assert not queries[0].mandatory_toggle_load
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


def test_continuous_training_does_not_end_when_inner_target_is_reached():
    """Catch an inner warehouse target termination leaking past the wrapper."""

    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=2,
        request_queue_size=1,
        task_completion_target=1,
        replenishment_mode="rware_delivery",
        terminate_on_task_target=False,
    )
    try:
        environment.reset(seed=7)
        task = environment.env.task_queue.active_tasks[0]
        environment.env.task_queue.complete(task.task_id, completed_step=0)

        transition = environment.step([Action.NOOP])

        assert not transition.terminated
        assert not transition.truncated
        assert "task_completion_target_reached" not in {
            event["type"] for event in transition.info["events"]
        }
    finally:
        environment.close()


def test_core_evaluation_profiles_use_two_initial_batches_then_fixed_ingress():
    """Keep evaluation supplied without randomizing AGV-sized arrivals."""

    root = Path("configs/core_integration")
    names = (
        "medium-5ag-baseline.yaml",
        "medium-5ag-astar-kd-no-validity.yaml",
    )

    for name in names:
        profile = load_baseline_experiment(root / name)
        values = {
            **profile.environment,
            **profile.evaluation.get("environment_overrides", {}),
        }
        agents = int(values["n_agents"])
        assert values["dynamic_ingress_interval"] == 100
        assert values["batch_size_range"] == [agents, agents]
        assert values["initial_ingress_batches"] == 2
        assert values["task_target"] == 5 * agents

        environment = new_baseline_environment(
            values, replenishment_mode="dynamic_ingress"
        )
        try:
            environment.reset(seed=9400)
            assert len(environment.env.task_queue.active_tasks) == 2 * agents
        finally:
            environment.close()


def test_non_phase2_dynamic_ingress_keeps_two_initial_batches_by_default():
    """Catch Phase 2's one-batch evaluation policy leaking into other phases."""

    environment = Phase2Warehouse(
        n_agents=1,
        max_steps=20,
        batch_interval=10,
        batch_size_range=(1, 1),
        request_queue_size=1,
        task_completion_target=6,
        replenishment_mode="dynamic_ingress",
    )
    try:
        environment.reset(seed=7)
        assert len(environment.env.task_queue.active_tasks) == 2
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


def test_core_training_profiles_disable_task_target_termination_only_in_training():
    """Catch a training profile that restores target termination or alters eval."""

    root = Path("configs/core_integration")
    names = (
        "medium-5ag-baseline.yaml",
        "medium-5ag-astar-kd-no-validity.yaml",
    )

    for name in names:
        profile = load_baseline_experiment(root / name)
        assert profile.environment["terminate_on_task_target"] is False
        assert profile.evaluation["environment_overrides"][
            "terminate_on_task_target"
        ] is True


def test_core_training_horizon_is_shorter_than_its_evaluation_horizon():
    """Catch a 500-step training cutoff leaking into independent evaluation."""

    root = Path("configs/core_integration")
    names = (
        "medium-5ag-baseline.yaml",
        "medium-5ag-astar-kd-no-validity.yaml",
    )

    for name in names:
        profile = load_baseline_experiment(root / name)
        evaluation_values = {
            **profile.environment,
            **profile.evaluation["environment_overrides"],
        }
        assert profile.environment["max_steps"] == 500
        assert evaluation_values["max_steps"] == 800


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

        assert current["root_costs"].shape == (1, 3, 3)
        assert current["supervision_scope"].shape == (1, 3)
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
        assert current["root_costs"].shape == (2, 3, 3)
        assert current["supervision_scope"].shape == (2, 3)
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

    loss = astar_kd_loss(
        logits,
        targets,
        validity,
        attenuation,
        torch.ones(2, dtype=torch.bool),
        lambda_value=1.0,
    )

    assert torch.isclose(loss, torch.tensor(0.625 * np.log(3.0), dtype=loss.dtype))


def test_kd_loss_excludes_valid_samples_outside_supervision_scope():
    """Keep geometrically valid but non-motion states out of A*KD gradients."""
    logits = torch.zeros((2, 3), requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    loss = astar_kd_loss(
        logits,
        targets,
        torch.tensor([True, True]),
        torch.ones(2),
        torch.tensor([True, False]),
        lambda_value=1.0,
    )

    loss.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_update_schema_exposes_supervision_scope_density():
    """Keep scope filtering auditable rather than hidden inside the loss."""
    assert {
        "teacher_supervision_scope_count",
        "teacher_effective_supervision_count",
        "teacher_effective_supervision_rate",
    } <= set(BaselineRun.UPDATE_FIELDS)


def test_astar_diagnostics_report_head_agreement_and_finite_route_regret():
    """Catch diagnostics that conflate policy actions or hide invalid roots."""

    policy_logits = torch.tensor([
        [0.0, 1.0, 0.0, 3.0, -1.0],
        [0.0, 0.0, 4.0, 1.0, -1.0],
        [0.0, 0.0, 1.0, 5.0, -1.0],
    ])
    motion_logits = torch.tensor([
        [3.0, 0.0, 1.0],
        [0.0, 3.0, 1.0],
        [2.0, 0.0, 1.0],
    ])
    soft_targets = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
    ])
    root_costs = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 2.0, float("nan")],
        [1.0, 2.0, float("nan")],
    ])

    metrics = astar_kd_diagnostics(
        policy_logits,
        motion_logits,
        soft_targets,
        root_costs,
        torch.tensor([True, True, True]),
        torch.tensor([True, True, True]),
        torch.tensor([1.0, 1.0, 1.0]),
    )

    assert metrics["motion_prior_top1_agreement"] == 1.0
    assert metrics["policy_teacher_motion_agreement"] == pytest.approx(1 / 3)
    assert metrics["static_route_regret"] == 1.0
    assert metrics["static_route_regret_count"] == 2
    assert metrics["static_invalid_top1_rate"] == pytest.approx(1 / 3)


def test_motion_prior_agreement_uses_valid_teacher_samples_even_when_inactive():
    """Keep policy-only active masking out of the auxiliary-head diagnostic."""

    metrics = astar_kd_diagnostics(
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]]),
        torch.tensor([[3.0, 0.0, 1.0]]),
        torch.tensor([[0.8, 0.1, 0.1]]),
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([True]),
        torch.tensor([True]),
        torch.tensor([0.0]),
    )

    assert metrics["motion_prior_top1_agreement"] == 1.0
    assert metrics["policy_teacher_motion_agreement"] == 0.0
    assert metrics["static_route_regret_count"] == 0


def test_astar_root_costs_survive_rollout_storage():
    """Catch root costs being dropped before the learner diagnostics."""

    storage = MAPPORolloutStorage(
        1, 1, 1, 2, 3, 5, 4, astar_kd_enabled=True
    )
    storage.insert(
        observations=np.zeros((1, 1, 2), dtype=np.float32),
        fp_states=np.zeros((1, 1, 3), dtype=np.float32),
        actions=np.zeros((1, 1), dtype=np.int64),
        action_masks=np.ones((1, 1, 5), dtype=bool),
        log_probs=np.zeros((1, 1), dtype=np.float32),
        values=np.zeros((1, 1), dtype=np.float32),
        rewards=np.zeros((1, 1), dtype=np.float32),
        dones=np.zeros((1, 1), dtype=bool),
        active_masks=np.ones((1, 1), dtype=np.float32),
        actor_hidden_states=np.zeros((1, 1, 4), dtype=np.float32),
        critic_hidden_states=np.zeros((1, 1, 4), dtype=np.float32),
        astar_soft_targets=np.asarray([[[0.6, 0.3, 0.1]]], dtype=np.float32),
        astar_root_costs=np.asarray([[[2.0, 3.0, np.nan]]], dtype=np.float32),
        astar_validity=np.ones((1, 1), dtype=bool),
        astar_supervision_scope=np.ones((1, 1), dtype=bool),
        astar_attenuation=np.ones((1, 1), dtype=np.float32),
    )

    batch = storage.compute_returns(
        np.zeros((1, 1), dtype=np.float32), gamma=0.99, gae_lambda=0.95
    )

    assert batch.astar_root_costs.shape == (1, 1, 1, 3)
    assert batch.astar_root_costs[0, 0, 0, :2].tolist() == [2.0, 3.0]
    assert np.isnan(batch.astar_root_costs[0, 0, 0, 2])
    assert batch.astar_supervision_scope.tolist() == [[[True]]]


def test_lambda_schedule_has_a_nonzero_tail():
    assert astar_kd_lambda(0, 300_000) == 0.05
    assert astar_kd_lambda(210_000, 300_000) == 0.01
    assert astar_kd_lambda(300_000, 300_000) == 0.01


def test_lambda_tail_can_be_disabled_after_the_schedule_horizon():
    """Catch the decoupled tail forcing a nonzero geometric prior forever."""
    assert astar_kd_lambda(
        300_000, 300_000, post_schedule_lambda=0.0
    ) == 0.0
    assert astar_kd_lambda(
        450_000, 300_000, post_schedule_lambda=0.0
    ) == 0.0
    assert astar_kd_lambda(
        299_999, 300_000, post_schedule_lambda=0.0
    ) == 0.01
    assert astar_kd_lambda(0, 300_000, post_schedule_lambda=0.0) == 0.05
    assert astar_kd_lambda(150_000, 300_000, post_schedule_lambda=0.0) == (
        astar_kd_lambda(150_000, 300_000)
    )


def test_post_schedule_astar_lambda_is_validated():
    """Catch a tail lambda that is negative or above the starting weight."""
    profile = load_astar_kd_experiment(
        Path("configs/core_integration/medium-5ag-astar-kd-no-validity.yaml")
    )
    for bad in (-0.1, 0.06):
        training = {
            **profile.training,
            "post_schedule_astar_lambda": bad,
        }
        with pytest.raises(ValueError, match="lambda"):
            _validate_training(training)
    training = {**profile.training, "post_schedule_astar_lambda": 0.0}
    _validate_training(training)


def test_astar_kd_training_profile_disables_the_kd_tail():
    """Formal A*KD profiles must drop the geometric prior after the schedule."""
    path = Path(
        "configs/core_integration/medium-5ag-astar-kd-no-validity.yaml"
    )
    profile = load_astar_kd_experiment(path)
    assert profile.training["post_schedule_astar_lambda"] == 0.0


def test_runner_replays_300k_schedule_before_low_rate_450k_tail(tmp_path):
    """Catch the formal budget stretching the proven optimization schedule."""

    profile = load_astar_kd_experiment(
        Path("configs/core_integration/medium-5ag-astar-kd-no-validity.yaml")
    )
    profile = replace(
        profile,
        approval_status="approved",
        training={
            **profile.training,
            "num_env_workers": 1,
            "optimization_schedule_steps": 300_000,
            "post_schedule_learning_rate": 1e-5,
        },
    )
    run = BaselineRun(
        profile,
        device="cpu",
        output_root=tmp_path,
        evaluation_only=True,
    )
    try:
        assert getattr(run, "optimization_schedule_steps", None) == 300_000
        assert run.trainer.total_environment_steps == 300_000
        assert run.post_schedule_learning_rate == 1e-5

        run.global_steps = 150_000
        run._apply_learning_rate_schedule()
        assert run.trainer.actor_optimizer.param_groups[0]["lr"] == pytest.approx(
            1.5e-4
        )

        run.global_steps = 300_000
        run._apply_learning_rate_schedule()
        assert run.trainer.actor_optimizer.param_groups[0]["lr"] == 0.0

        run.global_steps = 300_384
        run._apply_learning_rate_schedule()
        assert run.trainer.actor_optimizer.param_groups[0]["lr"] == 1e-5
        assert run.trainer.critic_optimizer.param_groups[0]["lr"] == 1e-5
    finally:
        run.close()


def test_logged_astar_lambda_uses_schedule_not_total_budget(tmp_path):
    """Catch updates.csv astar_lambda drifting from the trained schedule."""
    profile = load_astar_kd_experiment(
        Path("configs/core_integration/medium-5ag-astar-kd-no-validity.yaml")
    )
    profile = replace(
        profile,
        approval_status="approved",
        training={
            **profile.training,
            "num_env_workers": 1,
            "optimization_schedule_steps": 300_000,
            "post_schedule_learning_rate": 1e-5,
        },
    )
    run = BaselineRun(
        profile,
        device="cpu",
        output_root=tmp_path,
        evaluation_only=True,
    )
    try:
        run.global_steps = 150_000
        scheduled_lambda = astar_kd_lambda(150_000, 300_000)
        budget_lambda = astar_kd_lambda(150_000, 450_000)
        assert scheduled_lambda != budget_lambda
        assert run._current_astar_lambda() == pytest.approx(scheduled_lambda)
    finally:
        run.close()


def _milestone_profile():
    profile = load_astar_kd_experiment(
        Path("configs/core_integration/medium-5ag-astar-kd-no-validity.yaml")
    )
    return replace(
        profile,
        training={
            **profile.training,
            "num_env_workers": 1,
            "milestone_checkpoint_steps": [300_000],
        },
    )


def test_milestone_checkpoint_steps_are_validated_against_the_budget():
    """Catch milestone checkpoints that can never fire inside the budget."""
    profile = load_astar_kd_experiment(
        Path("configs/core_integration/medium-5ag-astar-kd-no-validity.yaml")
    )
    training = {**profile.training, "milestone_checkpoint_steps": [0]}
    with pytest.raises(ValueError, match="milestone"):
        _validate_training(training)
    training = {**profile.training, "milestone_checkpoint_steps": [450_001]}
    with pytest.raises(ValueError, match="milestone"):
        _validate_training(training)
    training = {**profile.training, "milestone_checkpoint_steps": [300_000]}
    _validate_training(training)


def test_milestone_checkpoint_is_written_once_when_crossed(tmp_path, monkeypatch):
    """Catch the 300k milestone being overwritten by later latest-checkpoints."""
    run = BaselineRun(
        _milestone_profile(),
        device="cpu",
        output_root=tmp_path,
        evaluation_only=True,
    )
    try:
        written = []
        monkeypatch.setattr(
            run, "_checkpoint", lambda path: written.append(path.name)
        )
        run.global_steps = 299_994
        run._write_due_milestone_checkpoints()
        assert written == []

        run.global_steps = 300_000
        run._write_due_milestone_checkpoints()
        assert written == ["checkpoint_300000.pt"]

        run._write_due_milestone_checkpoints()
        assert written == ["checkpoint_300000.pt"]
    finally:
        run.close()


def test_milestone_checkpoint_skips_steps_passed_before_resume(
    tmp_path, monkeypatch
):
    """Catch a resumed run rewriting milestones with post-milestone state."""
    run = BaselineRun(
        _milestone_profile(),
        device="cpu",
        output_root=tmp_path,
        evaluation_only=True,
    )
    try:
        written = []
        monkeypatch.setattr(
            run, "_checkpoint", lambda path: written.append(path.name)
        )
        run.global_steps = 320_000
        run._milestone_progress_steps = 320_000
        run._write_due_milestone_checkpoints()
        assert written == []
    finally:
        run.close()


def test_train_smoke_persists_milestone_checkpoint_at_its_step(
    tmp_path, monkeypatch
):
    """End-to-end wiring smoke: train() must keep the milestone checkpoint.

    The final evaluation pass is stubbed because this smoke only verifies
    checkpoint wiring, not policy quality.
    """
    profile = load_astar_kd_experiment(
        Path("configs/core_integration/medium-5ag-astar-kd-no-validity.yaml")
    )
    profile = replace(
        profile,
        approval_status="approved",
        training={
            **profile.training,
            "num_env_workers": 1,
            "environment_steps": 16,
            "rollout_length": 4,
            "checkpoint_interval_steps": 4,
            "milestone_checkpoint_steps": [8],
        },
    )
    run = BaselineRun(
        profile,
        device="cpu",
        output_root=tmp_path,
        evaluation_only=False,
    )
    try:
        monkeypatch.setattr(run, "evaluate", lambda **kwargs: {"episodes": []})
        monkeypatch.setattr(run, "_summary", lambda evaluation: {"schema": "smoke"})
        run.train()

        milestone_path = run.directory / "checkpoint_8.pt"
        assert milestone_path.exists()
        assert (run.directory / "checkpoint_latest.pt").exists()
        assert (run.directory / "checkpoint_final.pt").exists()

        milestone = torch.load(
            milestone_path, map_location="cpu", weights_only=False
        )
        latest = torch.load(
            run.directory / "checkpoint_latest.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert milestone["global_environment_steps"] == 8
        assert milestone["identity"] == run.identity
        assert latest["global_environment_steps"] == 16
    finally:
        run.close()


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
            astar_root_costs=np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32),
            astar_validity=np.ones((1, 1), dtype=bool),
            astar_supervision_scope=np.ones((1, 1), dtype=bool),
            astar_attenuation=np.ones((1, 1), dtype=np.float32),
        )
        actor_hidden, critic_hidden = next_actor, next_critic
    trainer.set_global_environment_steps(4)
    batch = storage.compute_returns(
        np.zeros((1, 1), dtype=np.float32), gamma=.99, gae_lambda=.95
    )
    reference = AStarKDTrainer(
        MAPPOConfig(observation_dim=4, state_dim=3, action_dim=5, ppo_epochs=1),
        "cpu",
        total_environment_steps=20,
    )
    reference.actor.load_state_dict(trainer.actor.state_dict())
    reference.critic.load_state_dict(trainer.critic.state_dict())
    reference.set_global_environment_steps(4)
    reference._actor_diagnostics = lambda *args: {}

    torch.manual_seed(101)
    metrics = trainer.update(batch)
    torch.manual_seed(101)
    reference.update(batch)

    assert metrics["astar_loss"] > 0.0
    assert np.isfinite(metrics["total_loss"])
    assert 0.0 <= metrics["motion_prior_top1_agreement"] <= 1.0
    assert 0.0 <= metrics["policy_teacher_motion_agreement"] <= 1.0
    assert metrics["static_route_regret"] >= 0.0
    assert metrics["static_route_regret_count"] > 0
    assert metrics["ppo_shared_grad_norm"] > 0.0
    assert metrics["astar_shared_grad_norm"] > 0.0
    assert np.isfinite(metrics["astar_to_ppo_shared_grad_ratio"])
    for name, parameter in trainer.actor.state_dict().items():
        assert torch.equal(parameter, reference.actor.state_dict()[name])


def test_astar_training_profile_uses_target_18_only_for_training():
    profile = load_astar_kd_experiment(
        Path("configs/core_integration/medium-5ag-astar-kd-no-validity.yaml")
    )

    assert profile.environment["task_target"] == 18
    assert profile.evaluation["environment_overrides"]["task_target"] == 25
    assert profile.training["seed"] == 7
    assert profile.evaluation["seeds"] == [9400, 9401]


def test_all_core_profiles_describe_the_formal_450k_training_budget():
    """Keep the formal budget separate from the proven optimization schedule."""
    root = Path("configs/core_integration")

    for path in root.glob("*.yaml"):
        profile = load_baseline_experiment(path)
        assert profile.training["environment_steps"] == 450_000, path.name
        assert profile.training.get("optimization_schedule_steps") == 300_000, (
            path.name
        )
        assert profile.training.get("post_schedule_learning_rate") == 1e-5, (
            path.name
        )
