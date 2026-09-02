"""E1-C regressions for the formal three-dimensional MAPPO path."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _raw_record(index: int, *, valid: bool = True) -> dict:
    record = {
        "scenario_id": f"formal-{index:03d}",
        "content_hash": f"content-{index:03d}",
        "semantic_view_version": "semantic-view-v3",
        "vector": [float((index * (dimension + 3)) % 797) / 797.0
                   for dimension in range(61)],
        "validity": int(valid),
    }
    if valid:
        record.update({
            "scores": [0.2, 0.4, 0.6],
            "reasons": ["r1", "r2", "r3"],
            "backend_tuple": [
                "deepseek-v4-pro",
                "deepseek-v4-pro",
                "fingerprint-v5",
            ],
        })
    else:
        record["failure_reason"] = "schema_or_content_invalid"
    return record


def _write_raw_labels(directory: Path) -> Path:
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({
        "schema": "semantic-label-session-v1",
        "mode": "formal",
        "request_model": "deepseek-v4-pro",
        "status": "running",
        "prompt_version": "semantic-prompt-v5-state-contract",
        "frozen_backend_tuple": [
            "deepseek-v4-pro", "deepseek-v4-pro", "fingerprint-v5"
        ],
    }), encoding="utf-8")
    records = [_raw_record(index, valid=index != 790) for index in range(800)]
    (directory / "records.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return directory / "records.jsonl"


def test_e1_raw_labels_preserve_the_799_of_800_exploratory_contract(tmp_path):
    from llm_mappo.e1_training import load_e1_raw_semantic_evidence

    records_path = _write_raw_labels(tmp_path / "raw")
    evidence = load_e1_raw_semantic_evidence(records_path)

    assert evidence.total_records == 800
    assert evidence.valid_records == 799
    assert evidence.invalid_records == 1
    assert evidence.dataset.scores.shape == (799, 3)
    assert evidence.backend_fingerprint == "fingerprint-v5"


def test_e1_semantic_loss_keeps_ood_reliability_per_robot():
    from llm_mappo.e1_training import E1Rollout

    rollout = E1Rollout(n_agents=2)
    rollout.add(
        physical_observations=np.zeros((2, 613), dtype=np.float32),
        semantic_observations=np.zeros((2, 61), dtype=np.float32),
        actions=np.asarray([1, 2], dtype=np.int64),
        log_probs=np.zeros(2, dtype=np.float32),
        action_masks=np.ones((2, 5), dtype=bool),
        astar_preferences=np.zeros((2, 3), dtype=np.float32),
        astar_valid=np.zeros(2, dtype=bool),
        calibration_selected=False,
        reward_confidence=0.0,
        semantic_targets=np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        semantic_validity=np.asarray([1.0, 1.0]),
        semantic_ood_reliability=np.asarray([1.0, 0.0]),
        reward=0.0,
        done=True,
        value=0.0,
    )
    data = rollout.tensors(last_value=0.0, device="cpu")
    scores = torch.zeros((1, 2, 3), requires_grad=True)

    loss, denominator = data.semantic_mse_loss(scores, lambda_l=0.10)

    assert denominator == 2
    assert loss.item() == pytest.approx(0.05)
    loss.backward()
    assert torch.all(scores.grad[0, 0] != 0)
    assert torch.all(scores.grad[0, 1] == 0)


def test_e1_llm_updater_produces_nonzero_three_dimensional_loss():
    from llm_mappo.e1_training import E1PPOUpdater, E1Rollout
    from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor

    rollout = E1Rollout(n_agents=2)
    for _ in range(2):
        rollout.add(
            physical_observations=np.zeros((2, 613), dtype=np.float32),
            semantic_observations=np.zeros((2, 61), dtype=np.float32),
            actions=np.asarray([1, 2], dtype=np.int64),
            log_probs=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_masks=np.ones((2, 5), dtype=bool),
            astar_preferences=np.zeros((2, 3), dtype=np.float32),
            astar_valid=np.zeros(2, dtype=bool), calibration_selected=False,
            reward_confidence=0.0,
            semantic_targets=np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            semantic_validity=np.ones(2), semantic_ood_reliability=np.ones(2),
            reward=1.0, done=True, value=0.0,
        )
    updater = E1PPOUpdater(
        actor=O0StudentActor(), critic=O0CentralizedCritic(), method="LLMKD",
        device="cpu", update_epochs=1, minibatch_steps=2,
    )
    metrics = updater.update(rollout, last_value=0.0, lambda_a=0.05, lambda_l=0.10)

    assert metrics["semantic_loss"] > 0.0
    assert metrics["semantic_valid_denominator"] == 4


def test_e1_checkpoint_rejects_missing_rng_or_raw_evidence_identity(tmp_path):
    from llm_mappo.e1_evidence import load_e1_checkpoint, save_e1_checkpoint
    from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor

    actor, critic = O0StudentActor(), O0CentralizedCritic()
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
    identity = {"group": "LLMKD", "seed": 7, "raw_records_sha256": "abc"}
    checkpoint = tmp_path / "checkpoint.pt"
    save_e1_checkpoint(checkpoint, identity=identity, actor=actor, critic=critic,
                       optimizer=optimizer, schedule_state={"schedule_version": "linear-env-step-v1"},
                       calibration_state=None, trainer_state={"schema": "e1-runtime-v1"})
    restored = load_e1_checkpoint(checkpoint, expected_identity=identity, actor=actor,
                                  critic=critic, optimizer=optimizer)
    assert restored["trainer_state"]["schema"] == "e1-runtime-v1"

    payload = torch.load(checkpoint, weights_only=False)
    del payload["rng"]["torch_cpu"]
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="RNG"):
        load_e1_checkpoint(checkpoint, expected_identity=identity, actor=actor,
                           critic=critic, optimizer=optimizer)


def test_e1_no_goal_hint_keeps_613d_shape_without_planner_queries():
    from llm_mappo.optimization_observation import ObservationSchema
    from llm_mappo.phase2 import Phase2Warehouse

    environment = Phase2Warehouse(n_agents=2, max_steps=8,
        observation_schema=ObservationSchema.NO_GEOMETRIC_GOAL_HINT_V1)
    try:
        observations = environment.reset(seed=7)
        assert observations.shape == (2, 613)
        assert environment.planner_query_counter.count == 0
    finally:
        environment.close()


def test_e1_calibration_import_initializes_the_cross_process_mirror(tmp_path):
    from llm_mappo.e1_protocol import E1FormalRun
    from llm_mappo.e1_training import E1Trainer, load_e1_raw_semantic_evidence
    from llm_mappo.shadow_state import ShadowStateAdapter

    labels = load_e1_raw_semantic_evidence(_write_raw_labels(tmp_path / "raw"))
    run = E1FormalRun(
        group="RC-AStarKD", seed=7, algorithm="mappo",
        astar_kd="reward_calibrated", semantic_teacher="disabled",
        semantic_control="none", observation_schema="direct-goal-observation-v1",
        real_environment_steps=150000, checkpoint_rule="checkpoint_final.pt",
        artifact_path="unused",
    )
    environment = {
        "environment_id": "llm-mappo-medium-3ag-v1", "n_agents": 5,
        "dynamic_ingress_interval": 40, "batch_size_range": [4, 8],
        "queue_size": 8, "task_target": 50, "max_steps": 1000,
        "deadlock_steps": 180, "charge_threshold": .30,
        "charge_release_threshold": .8, "battery_cost_scale": 1.1,
    }
    trainer = E1Trainer(run=run, environment=environment,
        training={"rollout_steps": 128, "rollout_length": 128,
                  "num_env_workers": 16, "update_epochs": 1,
                  "minibatch_steps": 64}, labels=labels, device="cpu")
    source = trainer._new_environment()
    try:
        source.reset(seed=1_000_010)
        address = {
            "run_seed": 7, "episode_index": 0, "episode_seed": 1_000_010,
            "environment_index": 1, "episode_step": 0,
        }
        address["real_global_step"] = next(
            step for step in range(1, 128)
            if trainer.calibrator.select(**{**address, "real_global_step": step})
        )
        snapshot = ShadowStateAdapter(source, code_commit="e1-vector-v1").capture(
            **address,
        )

        restored = trainer._restore_worker_snapshot(snapshot.to_bytes())

        assert trainer.real_adapter.state_hash() == restored.state_hash
    finally:
        source.close()
        trainer.close()


def test_e1_worker_snapshot_runs_a_paired_rc_shadow_without_cardinality_error(tmp_path):
    from llm_mappo.e1_protocol import E1FormalRun
    from llm_mappo.e1_training import E1Trainer, load_e1_raw_semantic_evidence
    from llm_mappo.shadow_state import ShadowStateAdapter

    labels = load_e1_raw_semantic_evidence(_write_raw_labels(tmp_path / "raw"))
    run = E1FormalRun(
        group="RC-AStarKD", seed=7, algorithm="mappo",
        astar_kd="reward_calibrated", semantic_teacher="disabled",
        semantic_control="none", observation_schema="direct-goal-observation-v1",
        real_environment_steps=150000, checkpoint_rule="checkpoint_final.pt",
        artifact_path="unused",
    )
    environment = {
        "environment_id": "llm-mappo-medium-3ag-v1", "n_agents": 5,
        "dynamic_ingress_interval": 40, "batch_size_range": [4, 8],
        "queue_size": 8, "task_target": 50, "max_steps": 1000,
        "deadlock_steps": 180, "charge_threshold": .30,
        "charge_release_threshold": .8, "battery_cost_scale": 1.1,
    }
    trainer = E1Trainer(run=run, environment=environment,
        training={"rollout_steps": 128, "rollout_length": 128,
                  "num_env_workers": 16, "update_epochs": 1,
                  "minibatch_steps": 64}, labels=labels, device="cpu")
    source = trainer._new_environment()
    try:
        source.reset(seed=1_000_010)
        address = {
            "run_seed": 7, "episode_index": 0, "episode_seed": 1_000_010,
            "environment_index": 1, "episode_step": 0,
        }
        address["real_global_step"] = next(
            step for step in range(1, 128)
            if trainer.calibrator.select(**{**address, "real_global_step": step})
        )
        snapshot = ShadowStateAdapter(source, code_commit="e1-vector-v1").capture(
            **address,
        )
        restored = trainer._restore_worker_snapshot(snapshot.to_bytes())
        _, valid = trainer._teacher_batch(trainer.environment)

        result = trainer.calibrator.run_paired_shadows(
            snapshot=restored,
            real_adapter=trainer.real_adapter,
            student_adapter=trainer.student_adapter,
            teacher_adapter=trainer.teacher_adapter,
            student_logits=lambda env, obs: trainer._shadow_logits(env, obs),
            teacher_preferences=lambda env: trainer._teacher_batch(env),
            initial_valid_mask=valid,
            critic_value=lambda obs: trainer._value(obs),
            gamma=0.99,
            address=restored.payload["address"],
        )

        assert np.isfinite(result.delta_g)
        assert 0.0 <= result.confidence <= 1.0
    finally:
        source.close()
        trainer.close()
