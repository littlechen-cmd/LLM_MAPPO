"""Deterministic O0 paired-shadow reward calibration primitives."""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Callable, Iterable

import numpy as np


SAMPLER_VERSION = "calibration-sampler-v1"
EMA_SCHEMA_VERSION = "reward-calibration-ema-v1"


class RewardCalibrationNoGo(RuntimeError):
    """A frozen calibration contract failed and O1 must not silently continue."""


class CalibrationSamplerV1:
    """The stateless shared 1/16 sampler for Fixed-KD and RC-KD."""

    version = SAMPLER_VERSION
    divisor = 16

    def key(
        self,
        *,
        run_seed: int,
        episode_index: int,
        episode_seed: int,
        environment_index: int,
        real_global_step: int,
        episode_step: int,
    ) -> bytes:
        values = (
            self.version,
            int(run_seed),
            int(episode_index),
            int(episode_seed),
            int(environment_index),
            int(real_global_step),
            int(episode_step),
        )
        return json.dumps(
            values, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

    def select(self, **address: int) -> bool:
        digest = sha256(self.key(**address)).digest()
        return int.from_bytes(digest[:8], "big") % self.divisor == 0


def deterministic_masked_argmax(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Choose the lowest-index legal maximizer without sampling or gradients."""
    values = np.asarray(logits, dtype=np.float64)
    legal = np.asarray(masks, dtype=bool)
    if values.shape != legal.shape or values.ndim != 2:
        raise ValueError("Masked argmax requires matching [agents, actions] arrays.")
    if not np.isfinite(values).all():
        raise RewardCalibrationNoGo("Student logits are non-finite.")
    if not legal.any(axis=1).all():
        raise ValueError("Every robot must retain one legal action.")
    masked = np.where(legal, values, -np.inf)
    return np.argmax(masked, axis=1).astype(np.int64)


@dataclass
class DeltaGEMA:
    """The frozen Welford-then-EMA state machine for `c_A_reward`."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    variance: float = 0.0
    initialized: bool = False
    decay: float = 0.99
    minimum_scale: float = 1e-3
    initialization_sample_count: int = 64

    def observe(self, delta_g: float) -> float:
        if not np.isfinite(delta_g):
            raise RewardCalibrationNoGo("Delta G is non-finite.")
        value = float(delta_g)
        if not self.initialized:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)
            if self.count == self.initialization_sample_count:
                self.variance = max(self.m2 / self.count, 0.0)
                self.initialized = True
            return 0.0
        scale = max(np.sqrt(max(self.variance, 0.0)), self.minimum_scale)
        confidence = float(np.clip(max(value, 0.0) / scale, 0.0, 1.0))
        delta = value - self.mean
        self.mean = self.decay * self.mean + (1.0 - self.decay) * value
        self.variance = self.decay * (
            self.variance + (1.0 - self.decay) * delta * delta
        )
        self.count += 1
        return confidence

    def state_dict(self) -> dict:
        return {
            "schema_version": EMA_SCHEMA_VERSION,
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "variance": self.variance,
            "initialized": self.initialized,
            "decay": self.decay,
            "minimum_scale": self.minimum_scale,
            "initialization_sample_count": self.initialization_sample_count,
            "weight_clip": [0.0, 1.0],
        }


@dataclass(frozen=True)
class CalibrationResult:
    selected: bool
    attempted: bool
    success: bool
    confidence: float
    delta_g: float | None
    failure_reason: str | None = None
    student_return: float | None = None
    teacher_return: float | None = None
    student_length: int = 0
    teacher_length: int = 0


class RewardCalibrator:
    """Coordinate shared sampler/EMA behavior; branch stepping is injected later."""

    def __init__(
        self,
        sampler: CalibrationSamplerV1 | None = None,
        ema: DeltaGEMA | None = None,
    ):
        self.sampler = sampler or CalibrationSamplerV1()
        self.ema = ema or DeltaGEMA()

    def select(self, **address: int) -> bool:
        return self.sampler.select(**address)

    def record_delta_g(
        self, *, selected: bool, any_valid: bool, delta_g: float
    ) -> CalibrationResult:
        if not selected:
            return CalibrationResult(False, False, False, 0.0, None)
        if not any_valid:
            return CalibrationResult(True, False, False, 0.0, None, "selected_no_valid")
        confidence = self.ema.observe(delta_g)
        return CalibrationResult(True, True, True, confidence, float(delta_g))

    def run_paired_shadows(
        self,
        *,
        snapshot,
        real_adapter,
        student_adapter,
        teacher_adapter,
        student_logits: Callable[[object, np.ndarray], np.ndarray],
        teacher_preferences: Callable[[object], tuple[np.ndarray, np.ndarray]],
        initial_valid_mask: np.ndarray,
        critic_value: Callable[[np.ndarray], object],
        gamma: float,
        address: dict[str, int],
        horizon: int = 12,
    ) -> CalibrationResult:
        """Evaluate frozen Student/A* branches without mutating real rollout state."""
        if horizon != 12:
            raise ValueError("H=12 is the only formal reward-calibration horizon.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("Calibration gamma must be within [0, 1].")
        selected = self.select(**address)
        if not selected:
            return CalibrationResult(False, False, False, 0.0, None)
        canonical_address = {name: int(address[name]) for name in sorted(address)}
        if snapshot.payload["address"] != canonical_address:
            raise ValueError("Calibration address must match the captured snapshot.")
        valid = np.asarray(initial_valid_mask, dtype=bool)
        if not valid.any():
            return CalibrationResult(True, False, False, 0.0, None, "selected_no_valid")
        real_hash = real_adapter.state_hash()
        try:
            student = self._shadow_return(
                adapter=student_adapter,
                snapshot=snapshot,
                student_logits=student_logits,
                teacher_preferences=None,
                critic_value=critic_value,
                gamma=gamma,
                address=address,
                horizon=horizon,
            )
            teacher = self._shadow_return(
                adapter=teacher_adapter,
                snapshot=snapshot,
                student_logits=student_logits,
                teacher_preferences=teacher_preferences,
                critic_value=critic_value,
                gamma=gamma,
                address=address,
                horizon=horizon,
            )
            delta_g = teacher[0] - student[0]
            confidence = self.ema.observe(delta_g)
        except RewardCalibrationNoGo:
            raise
        except Exception as error:
            raise RewardCalibrationNoGo("Shadow branch execution failed.") from error
        if real_adapter.state_hash() != real_hash:
            raise RewardCalibrationNoGo("Shadow calibration mutated real rollout state.")
        real_adapter.assert_global_rng_guard(snapshot)
        return CalibrationResult(
            selected=True,
            attempted=True,
            success=True,
            confidence=confidence,
            delta_g=delta_g,
            student_return=student[0],
            teacher_return=teacher[0],
            student_length=student[1],
            teacher_length=teacher[1],
        )

    def _shadow_return(
        self,
        *,
        adapter,
        snapshot,
        student_logits: Callable[[object, np.ndarray], np.ndarray],
        teacher_preferences: Callable[[object], tuple[np.ndarray, np.ndarray]] | None,
        critic_value: Callable[[np.ndarray], object],
        gamma: float,
        address: dict[str, int],
        horizon: int,
    ) -> tuple[float, int]:
        adapter.restore(snapshot)
        environment = adapter.environment
        team_return = 0.0
        observations = environment._observations()
        completed_horizon = True
        try:
            for shadow_offset in range(horizon):
                masks = environment.action_masks()
                actions = deterministic_masked_argmax(
                    student_logits(environment, observations), masks
                )
                if teacher_preferences is not None:
                    preferences, valid = teacher_preferences(environment)
                    actions = self._teacher_actions(
                        actions, masks, preferences, np.asarray(valid, dtype=bool)
                    )
                environment.env.set_shadow_randomness(
                    self._event_randomness,
                    episode_seed=address["episode_seed"],
                    real_global_step=address["real_global_step"],
                    shadow_offset=shadow_offset,
                )
                transition = environment.step(actions)
                team_return += gamma ** shadow_offset * transition.team_reward
                observations = transition.observations
                if (
                    transition.terminated
                    or transition.truncated
                    or transition.metrics.deadlocked
                ):
                    completed_horizon = False
                    return team_return, shadow_offset + 1
        finally:
            environment.env.clear_shadow_randomness()
        if completed_horizon:
            bootstrap = self._detached_scalar(critic_value(observations))
            team_return += gamma ** horizon * bootstrap
        return team_return, horizon

    @staticmethod
    def _teacher_actions(
        student_actions: np.ndarray,
        masks: np.ndarray,
        preferences: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(preferences, dtype=np.float64)
        if (
            values.shape != (student_actions.shape[0], 3)
            or valid.shape != student_actions.shape
        ):
            raise RewardCalibrationNoGo("Pure Motion Teacher output shape is invalid.")
        if not np.isfinite(values).all():
            raise RewardCalibrationNoGo("Pure Motion Teacher output is non-finite.")
        actions = np.asarray(student_actions, dtype=np.int64).copy()
        motion_actions = np.asarray([1, 2, 3], dtype=np.int64)
        for index in np.flatnonzero(valid):
            root_action = int(motion_actions[np.argmax(values[index])])
            if not masks[index, root_action]:
                raise RewardCalibrationNoGo("teacher_mask_mismatch")
            actions[index] = root_action
        return actions

    @staticmethod
    def _detached_scalar(value: object) -> float:
        candidate = value
        if hasattr(candidate, "detach"):
            candidate = candidate.detach()
        if hasattr(candidate, "cpu"):
            candidate = candidate.cpu()
        if hasattr(candidate, "item"):
            candidate = candidate.item()
        scalar = float(candidate)
        if not np.isfinite(scalar):
            raise RewardCalibrationNoGo("Critic bootstrap is non-finite.")
        return scalar

    @property
    def _event_randomness(self):
        from llm_mappo.shadow_state import EventAddressedRandomness

        return EventAddressedRandomness()

    @staticmethod
    def astar_weights(
        *,
        mode: str,
        lambda_a: float,
        valid_mask: Iterable[bool],
        result: CalibrationResult,
    ) -> np.ndarray:
        if mode not in {"fixed-kd", "reward-calibrated-kd"}:
            raise ValueError("Unsupported optimization KD mode.")
        valid = np.asarray(list(valid_mask), dtype=bool)
        if not result.selected:
            return np.zeros(valid.shape, dtype=np.float32)
        confidence = 1.0 if mode == "fixed-kd" else result.confidence
        return float(lambda_a) * valid.astype(np.float32) * confidence
