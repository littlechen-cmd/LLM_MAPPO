import csv

import numpy as np
import pytest
import torch

from eval.evaluate_phase3 import _write_json
from llm_mappo.mappo import (
    DualHeadMAPPOPolicy,
    PPOHyperparameters,
    RolloutBuffer,
)
from llm_mappo.phase2 import ACTION_COUNT, Phase2Warehouse
from llm_mappo.phase3_training import (
    Phase3TrainingConfig,
    _append_csv,
    _checkpoint_semantic_dim,
    _engagement_targets,
    _resolve_device,
    _training_episode_seed,
    _validate_training_seed_groups,
    _write_csv,
    evaluate_phase3,
    load_phase3_policy,
    train_phase3,
)


def test_phase3_priority_schedule_and_observation_features():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        priority_schedule=("A", "B", "C"),
    )
    try:
        observations = env.reset(seed=3)
        assert observations.shape == (3, env.actor_observation_dim)
        assert env.actor_observation_dim == 615
        assert [task.label for task in env.env.task_queue.tasks] == ["A1", "B1", "C1"]
    finally:
        env.close()


def test_phase3_dual_head_outputs_engagement_and_motion_distribution():
    policy = DualHeadMAPPOPolicy(observation_dim=615, action_dim=ACTION_COUNT)
    observations = np.zeros((3, 615), dtype=np.float32)
    actions, log_probs, value, engagement = policy.act(observations)
    assert actions.shape == (3,)
    assert log_probs.shape == (3,)
    assert isinstance(value, float)
    assert engagement.shape == (3,)
    assert np.all((engagement >= 0.0) & (engagement <= 1.0))


def test_phase3_r2_motion_loss_does_not_update_engagement_branch():
    policy = DualHeadMAPPOPolicy(observation_dim=615, action_dim=ACTION_COUNT)
    observations = torch.randn(3, 615)
    policy.actor(observations).sum().backward()
    engagement_gradients = [
        parameter.grad
        for parameter in policy.actor.engagement_encoder.parameters()
    ]
    engagement_gradients.extend(
        parameter.grad for parameter in policy.actor.engagement_head.parameters()
    )
    assert all(gradient is None for gradient in engagement_gradients)
    assert any(
        parameter.grad is not None
        for parameter in policy.actor.motion_encoder.parameters()
    )


def test_phase3_r2_engagement_loss_does_not_update_motion_branch():
    policy = DualHeadMAPPOPolicy(observation_dim=615, action_dim=ACTION_COUNT)
    observations = torch.randn(3, 615)
    policy.engagement(observations).sum().backward()
    assert all(
        parameter.grad is None
        for parameter in policy.actor.motion_encoder.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in policy.actor.engagement_encoder.parameters()
    )


def test_phase3_r2_uses_lower_label_for_idle_and_locked_agvs():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        priority_schedule=("A", "B", "C"),
    )
    try:
        env.reset(seed=3)
        env.env.task_queue.release_agent(1)
        env.env.agents[0].task_id = None
        env.env.agents[1].picking_lock_steps = 1
        assert np.allclose(_engagement_targets(env), [0.1, 0.1, 0.3])
    finally:
        env.close()


def test_phase3_dynamic_ingress_rotates_batches_and_rank_labels():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        priority_schedule=None,
        batch_interval=2,
        batch_size_range=(1, 1),
        initial_priority_label="A",
        request_queue_size=1,
        task_completion_target=4,
        include_priority_features=True,
    )
    try:
        observations = env.reset(seed=3)
        assert observations.shape == (3, 615)
        assert [task.label for task in env.env.task_queue.active_tasks] == [
            "A1",
            "B1",
        ]
        for _ in range(4):
            env.step([0, 0, 0])
        assert [task.label for task in env.env.task_queue.active_tasks] == [
            "A1",
            "B1",
            "C1",
            "D1",
        ]
        assert np.allclose(
            _engagement_targets(env),
            [0.1 + 0.7 * 3 / 4, 0.1 + 0.7 * 2 / 4, 0.1 + 0.7 / 4],
        )
        assert env.metrics.created_tasks == 4
        assert env.metrics.task_completion_target == 4
    finally:
        env.close()


def test_phase3_dynamic_ingress_does_not_arrive_after_the_time_limit():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=2,
        priority_schedule=None,
        batch_interval=2,
        batch_size_range=(1, 1),
        initial_priority_label="A",
        request_queue_size=1,
        task_completion_target=4,
        include_priority_features=True,
    )
    try:
        env.reset(seed=3)
        env.step([0, 0, 0])
        transition = env.step([0, 0, 0])
        assert transition.metrics.steps == 2
        assert [task.label for task in env.env.task_queue.tasks] == ["A1", "B1"]
    finally:
        env.close()


def test_phase3_dynamic_ingress_terminates_once_the_completion_target_is_met():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=8,
        priority_schedule=None,
        batch_interval=8,
        batch_size_range=(1, 1),
        initial_priority_label="A",
        request_queue_size=1,
        task_completion_target=2,
        include_priority_features=True,
    )
    try:
        env.reset(seed=3)
        for task in env.env.task_queue.active_tasks:
            env.env.task_queue.complete(task.task_id, completed_step=0)
        env.env._refresh_request_queue()
        transition = env.step([0, 0, 0])
        assert transition.terminated
        assert not transition.truncated
        assert transition.metrics.completed_tasks == 2
        assert transition.metrics.task_completion_target == 2
        assert transition.metrics.completion_rate == 1.0
        assert transition.metrics.success
        assert transition.info["task_target_reached"]
    finally:
        env.close()


def test_phase3_config_disables_astar_kl_and_enables_rule_labels():
    config = Phase3TrainingConfig()
    assert config.ppo.reservation_kl_coefficient == 0.0
    assert config.ppo.engagement_coefficient > 0.0


def test_phase3_uses_twelve_persistent_environment_processes(tmp_path):
    config = Phase3TrainingConfig(
        phase="3b",
        seed=3,
        n_agents=3,
        max_steps=1,
        episodes=12,
        parallel_envs=12,
        rollout_steps=12,
        checkpoint_interval=20,
        metrics_write_interval=12,
        output_dir=str(tmp_path / "run"),
        ppo=PPOHyperparameters(
            update_epochs=1,
            minibatch_steps=12,
            engagement_coefficient=0.1,
            reservation_kl_coefficient=0.05,
        ),
    )
    summary = train_phase3(config)
    assert summary["episodes"] == 12
    assert summary["parallel_envs"] == 12
    assert summary["steps"] == 12
    assert summary["reservation_teacher"]["cache_misses"] == 12


def test_phase3_round_robins_isolated_training_seed_groups():
    config = Phase3TrainingConfig(seed=7, training_seed_groups=(100, 101, 102))
    assert _training_episode_seed(config, 0) == (1_000_000, 100, 0)
    assert _training_episode_seed(config, 1) == (1_010_000, 101, 0)
    assert _training_episode_seed(config, 3) == (1_000_001, 100, 1)


def test_phase3_rejects_duplicate_training_seed_groups():
    with pytest.raises(ValueError, match="duplicates"):
        _validate_training_seed_groups((100, 100))


def test_phase3b_config_enables_only_astar_kl_ablation():
    config = Phase3TrainingConfig.from_yaml("configs/phase3b_dual_head_astar_kl.yaml")
    assert config.phase == "3b"
    assert config.n_agents == 3
    assert config.priority_schedule == ("A", "B", "C")
    assert config.ppo.reservation_kl_coefficient == 0.05
    assert config.ppo.engagement_coefficient == 0.1


def test_phase3_r2_configs_differ_only_in_phase_and_path_kl():
    phase3a = Phase3TrainingConfig.from_yaml("configs/phase3a_r2_semantic.yaml")
    phase3b = Phase3TrainingConfig.from_yaml(
        "configs/phase3b_r2_semantic_astar_kl.yaml"
    )
    assert phase3a.phase == "3a"
    assert phase3b.phase == "3b"
    assert phase3a.output_dir != phase3b.output_dir
    assert phase3a.ppo.reservation_kl_coefficient == 0.0
    assert phase3b.ppo.reservation_kl_coefficient == 0.05
    assert phase3a.ppo.engagement_coefficient == phase3b.ppo.engagement_coefficient
    assert phase3a.priority_schedule == phase3b.priority_schedule


def test_phase3_dynamic_configs_keep_ingress_fixed_for_the_kl_ablation():
    phase3a = Phase3TrainingConfig.from_yaml(
        "configs/phase3a_r2_dynamic_ingress.yaml"
    )
    phase3b = Phase3TrainingConfig.from_yaml(
        "configs/phase3b_r2_dynamic_ingress_astar_kl.yaml"
    )
    assert phase3a.priority_schedule is None
    assert phase3b.priority_schedule is None
    assert phase3a.batch_interval == phase3b.batch_interval == 40
    assert phase3a.batch_size_range == phase3b.batch_size_range == (1, 3)
    assert phase3a.initial_priority_label == phase3b.initial_priority_label == "A"
    assert phase3a.request_queue_size == phase3b.request_queue_size == 4
    assert phase3a.task_completion_target == phase3b.task_completion_target == 9
    assert phase3a.ppo.reservation_kl_coefficient == 0.0
    assert phase3b.ppo.reservation_kl_coefficient == 0.05


def test_phase3_csv_writer_preserves_late_priority_columns(tmp_path):
    path = tmp_path / "episodes.csv"
    _write_csv(
        path,
        [
            {"episode": 1, "task_completion_rate": 0.0},
            {
                "episode": 2,
                "task_completion_rate": 1.0,
                "priority_A_mean_completion_steps": 10.0,
            },
        ],
    )
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[1]["priority_A_mean_completion_steps"] == "10.0"
    assert rows[0]["priority_A_mean_completion_steps"] == ""


def test_phase3_csv_writer_retries_transient_windows_replace_error(
    tmp_path, monkeypatch
):
    path = tmp_path / "episodes.csv"
    original_replace = type(path).replace
    attempts = 0

    def transient_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporarily held by a reader")
        return original_replace(source, target)

    monkeypatch.setattr(type(path), "replace", transient_replace)

    _write_csv(path, [{"episode": 1, "reward": 1.0}])

    assert attempts == 2
    assert path.exists()


def test_phase3_update_csv_appends_without_rewriting_prior_rows(tmp_path):
    path = tmp_path / "updates.csv"
    _append_csv(path, {"update": 1, "steps": 8, "loss": 0.5})
    first_size = path.stat().st_size
    _append_csv(path, {"update": 2, "steps": 16, "loss": 0.25})

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["update"] for row in rows] == ["1", "2"]
    assert path.stat().st_size > first_size


def test_rollout_buffer_computes_gae_per_parallel_environment_stream():
    buffer = RolloutBuffer(n_agents=1)
    observation = np.zeros((1, 2), dtype=np.float32)
    action = np.zeros(1, dtype=np.int64)
    log_prob = np.zeros(1, dtype=np.float32)
    for stream_id, reward, done in (
        (0, 1.0, False),
        (1, 10.0, False),
        (0, 2.0, True),
        (1, 20.0, True),
    ):
        buffer.add(
            observation,
            action,
            log_prob,
            reward,
            done,
            value=0.0,
            stream_id=stream_id,
        )
    data = buffer.tensors(
        {0: 0.0, 1: 0.0},
        PPOHyperparameters(gamma=1.0, gae_lambda=1.0),
        "cpu",
    )
    assert np.allclose(data["advantages"].numpy(), [3.0, 30.0, 2.0, 20.0])


def test_dual_head_policy_batches_independent_environments():
    policy = DualHeadMAPPOPolicy(6, ACTION_COUNT, semantic_dim=2)
    observations = np.zeros((3, 5, 6), dtype=np.float32)
    masks = np.ones((3, 5, ACTION_COUNT), dtype=bool)
    actions, log_probs, values, semantics = policy.act(observations, masks)
    assert actions.shape == log_probs.shape == (3, 5)
    assert values.shape == (3,)
    assert semantics.shape == (3, 5, 2)


def test_dual_head_policy_uses_fixed_zero_semantics_when_disabled():
    policy = DualHeadMAPPOPolicy(
        6,
        ACTION_COUNT,
        semantic_dim=2,
        semantic_features_enabled=False,
    )
    observations = torch.randn(5, 6)

    before = policy.actor(observations).detach().clone()
    with torch.no_grad():
        for parameter in policy.actor.engagement_encoder.parameters():
            parameter.add_(torch.randn_like(parameter))
        for parameter in policy.actor.engagement_head.parameters():
            parameter.add_(torch.randn_like(parameter))
    after = policy.actor(observations).detach()

    assert torch.equal(before, after)
    assert torch.count_nonzero(policy.actor.motion_semantics(observations)) == 0


@pytest.mark.parametrize(("phase", "semantic_dim"), (("3b", 1), ("4", 2)))
def test_phase3_policy_loader_restores_single_and_dual_semantics(
    tmp_path, phase, semantic_dim
):
    source = DualHeadMAPPOPolicy(6, ACTION_COUNT, semantic_dim=semantic_dim)
    checkpoint = tmp_path / f"phase_{phase}.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "config": {"phase": phase},
            "actor_observation_dim": 6,
            "episodes": 1,
            "steps": 1,
            "phase": phase,
        },
        checkpoint,
    )

    loaded, config, payload = load_phase3_policy(checkpoint)

    assert loaded.actor.semantic_dim == semantic_dim
    assert loaded.actor.motion_head.in_features == 64 + semantic_dim
    assert loaded.training is False
    assert config == {"phase": phase}
    assert payload["phase"] == phase
    for key, expected in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected)


def test_phase3_policy_loader_rejects_inconsistent_semantic_metadata(tmp_path):
    source = DualHeadMAPPOPolicy(6, ACTION_COUNT, semantic_dim=2)
    checkpoint = {
        "model_state": source.state_dict(),
        "config": {"phase": "4"},
        "actor_observation_dim": 6,
        "semantic_dim": 1,
        "phase": "4",
    }
    path = tmp_path / "inconsistent.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="semantic dimensions disagree"):
        load_phase3_policy(path)


def test_phase3_policy_loader_restores_disabled_semantic_features(tmp_path):
    source = DualHeadMAPPOPolicy(
        6,
        ACTION_COUNT,
        semantic_dim=2,
        semantic_features_enabled=False,
    )
    checkpoint = tmp_path / "no_llm.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "config": {"phase": "4", "use_offline_llm_teacher": False},
            "actor_observation_dim": 6,
            "semantic_dim": 2,
            "semantic_features_enabled": False,
            "phase": "4",
        },
        checkpoint,
    )

    loaded, _, _ = load_phase3_policy(checkpoint)

    assert loaded.actor.semantic_features_enabled is False


def test_checkpoint_semantic_dim_rejects_unsupported_width():
    source = DualHeadMAPPOPolicy(6, ACTION_COUNT, semantic_dim=2)
    checkpoint = {"model_state": source.state_dict(), "semantic_dim": 3}
    with pytest.raises(ValueError, match="invalid semantic dimension"):
        _checkpoint_semantic_dim(checkpoint)


def test_training_device_resolution_supports_auto_and_clear_cuda_errors(
    monkeypatch,
):
    assert _resolve_device("cpu").type == "cpu"
    automatic = _resolve_device("auto")
    assert automatic.type in {"cpu", "cuda"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        _resolve_device("cuda:0")


def test_phase3_evaluation_rejects_nonpositive_engagement_sample_rate():
    with pytest.raises(ValueError, match="engagement_sample_rate"):
        evaluate_phase3(None, {}, (), engagement_sample_rate=0)


def test_phase3_evaluation_output_creates_parent_directories(tmp_path):
    output = tmp_path / "nested" / "evaluation.json"
    _write_json('{"success": true}', str(output))
    assert output.read_text(encoding="utf-8") == '{"success": true}'
