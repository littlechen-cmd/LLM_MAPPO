"""Contract tests for deterministic reward calibration primitives."""

import hashlib
import json

import numpy as np
import pytest

from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.reward_calibration import (
    CalibrationResult,
    CalibrationSamplerV1,
    DeltaGEMA,
    RewardCalibrator,
    RewardCalibrationNoGo,
    deterministic_masked_argmax,
)
from llm_mappo.shadow_state import ShadowStateAdapter


def _environment() -> Phase2Warehouse:
    environment = Phase2Warehouse(
        n_agents=3,
        max_steps=80,
        batch_interval=8,
        batch_size_range=(1, 2),
        request_queue_size=3,
        task_completion_target=9,
        observation_schema=ObservationSchema.DIRECT_GOAL_V1,
    )
    environment.reset(seed=19)
    return environment


def test_sampler_uses_the_frozen_json_key_and_sha256_modulo():
    sampler = CalibrationSamplerV1()
    values = ("calibration-sampler-v1", 7, 2, 19, 0, 3, 1)
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 16 == 0
    assert sampler.select(
        run_seed=7,
        episode_index=2,
        episode_seed=19,
        environment_index=0,
        real_global_step=3,
        episode_step=1,
    ) is expected


def test_masked_argmax_is_deterministic_and_rejects_empty_masks():
    logits = np.asarray([[1.0, 3.0, 3.0, -1.0, 2.0]], dtype=np.float32)
    masks = np.asarray([[False, True, True, True, False]])
    assert deterministic_masked_argmax(logits, masks).tolist() == [1]
    with pytest.raises(ValueError, match="legal action"):
        deterministic_masked_argmax(logits, np.zeros_like(masks))


def test_delta_g_ema_uses_welford_then_pre_update_ema_scale():
    ema = DeltaGEMA()
    for _ in range(64):
        assert ema.observe(2.0) == 0.0
    assert ema.initialized
    assert ema.count == 64
    assert ema.observe(3.0) == 1.0
    assert ema.count == 65
    with pytest.raises(RewardCalibrationNoGo, match="non-finite"):
        ema.observe(float("nan"))


def test_paired_shadows_keep_real_rollout_unchanged_and_bootstrap_only_at_horizon():
    sampler = CalibrationSamplerV1()
    address = next(
        {
            "run_seed": 7,
            "episode_index": 2,
            "episode_seed": 19,
            "environment_index": 0,
            "real_global_step": step,
            "episode_step": 1,
        }
        for step in range(256)
        if sampler.select(
            run_seed=7,
            episode_index=2,
            episode_seed=19,
            environment_index=0,
            real_global_step=step,
            episode_step=1,
        )
    )
    real = _environment()
    real_adapter = ShadowStateAdapter(real, code_commit="test-commit")
    snapshot = real_adapter.capture(**address)
    student_adapter = ShadowStateAdapter(_environment(), code_commit="test-commit")
    teacher_adapter = ShadowStateAdapter(_environment(), code_commit="test-commit")

    def teacher_provider(environment):
        masks = environment.action_masks()[:, 1:4]
        preferences = masks.astype(np.float32)
        preferences /= preferences.sum(axis=1, keepdims=True)
        return preferences, masks.any(axis=1)

    result = RewardCalibrator(sampler=sampler).run_paired_shadows(
        snapshot=snapshot,
        real_adapter=real_adapter,
        student_adapter=student_adapter,
        teacher_adapter=teacher_adapter,
        student_logits=lambda observations: np.zeros((3, 5), dtype=np.float32),
        teacher_preferences=teacher_provider,
        initial_valid_mask=teacher_provider(real)[1],
        critic_value=lambda observations: 0.0,
        gamma=0.99,
        address=address,
    )
    assert result.success
    assert result.student_length == 12
    assert result.teacher_length == 12
    assert real_adapter.state_hash() == snapshot.state_hash


def test_fixed_and_rc_weights_differ_only_by_reward_confidence():
    result = CalibrationResult(True, True, True, 0.25, 2.0)
    valid = [True, False, True]
    fixed = RewardCalibrator.astar_weights(
        mode="fixed-kd", lambda_a=0.05, valid_mask=valid, result=result
    )
    calibrated = RewardCalibrator.astar_weights(
        mode="reward-calibrated-kd", lambda_a=0.05, valid_mask=valid, result=result
    )
    assert fixed.tolist() == pytest.approx([0.05, 0.0, 0.05])
    assert calibrated.tolist() == pytest.approx([0.0125, 0.0, 0.0125])
    assert not RewardCalibrator.astar_weights(
        mode="fixed-kd",
        lambda_a=0.05,
        valid_mask=valid,
        result=CalibrationResult(False, False, False, 0.0, None),
    ).any()
