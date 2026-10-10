"""Versioned raw Actor-observation trajectory records."""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)


ACTOR_OBSERVATION_TRAJECTORY_V1 = "actor-observation-trajectory-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_V2_DIMENSION = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
).dimension


@dataclass(frozen=True)
class ActorObservationTrajectory:
    """One focal AGV's complete episode, without teacher or outcome fields."""

    episode_id: str
    agent_id: int
    timesteps: np.ndarray
    observations: np.ndarray
    reset_masks: np.ndarray
    behavior_policy_id: str
    environment_sha256: str
    config_sha256: str
    observation_schema: str = DIRECT_GOAL_OBSERVATION_V2
    schema_version: str = ACTOR_OBSERVATION_TRAJECTORY_V1

    def __post_init__(self) -> None:
        self._validate_identity()
        observations, timesteps, reset_masks = self._validated_arrays()
        observations.flags.writeable = False
        timesteps.flags.writeable = False
        reset_masks.flags.writeable = False
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "timesteps", timesteps)
        object.__setattr__(self, "reset_masks", reset_masks)

    def _validate_identity(self) -> None:
        if self.schema_version != ACTOR_OBSERVATION_TRAJECTORY_V1:
            raise ValueError("Unknown raw trajectory schema version.")
        if self.observation_schema != DIRECT_GOAL_OBSERVATION_V2:
            raise ValueError("Raw LLM KD trajectories require the 209D v2 schema.")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string.")
        if (
            isinstance(self.agent_id, (bool, np.bool_))
            or not isinstance(self.agent_id, (int, np.integer))
            or self.agent_id < 1
        ):
            raise ValueError("agent_id must be a positive integer.")
        if not isinstance(self.behavior_policy_id, str) or not self.behavior_policy_id:
            raise ValueError("behavior_policy_id must be a non-empty string.")
        self._validate_hash("environment_sha256", self.environment_sha256)
        self._validate_hash("config_sha256", self.config_sha256)

    def _validated_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        supplied_timesteps = np.asarray(self.timesteps)
        supplied_reset_masks = np.asarray(self.reset_masks)
        if not np.issubdtype(supplied_timesteps.dtype, np.integer):
            raise ValueError("timesteps must use an integer dtype.")
        if supplied_reset_masks.dtype != np.bool_:
            raise ValueError("reset masks must use a boolean dtype.")
        observations = np.array(self.observations, dtype=np.float32, copy=True)
        timesteps = np.array(supplied_timesteps, dtype=np.int64, copy=True)
        reset_masks = np.array(supplied_reset_masks, dtype=np.bool_, copy=True)
        if observations.ndim != 2 or observations.shape[1] != _V2_DIMENSION:
            raise ValueError("observations must have shape [time, 209].")
        length = observations.shape[0]
        if length == 0:
            raise ValueError("A raw trajectory must contain at least one frame.")
        if timesteps.shape != (length,):
            raise ValueError("timesteps must contain one value per observation.")
        if reset_masks.shape != (length,):
            raise ValueError("reset masks must contain one value per observation.")
        if not np.isfinite(observations).all():
            raise ValueError("observations must contain only finite values.")
        if np.any(np.diff(timesteps) <= 0):
            raise ValueError("timesteps must be strictly increasing.")
        if not reset_masks[0] or np.count_nonzero(reset_masks) != 1:
            raise ValueError(
                "reset masks must mark exactly the first frame of one episode."
            )
        return observations, timesteps, reset_masks

    @staticmethod
    def _validate_hash(name: str, value: str) -> None:
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA-256 digest.")

    def to_record(self) -> dict[str, object]:
        """Return the canonical raw JSON-compatible record."""

        return {
            "schema_version": self.schema_version,
            "observation_schema": self.observation_schema,
            "episode_id": self.episode_id,
            "agent_id": int(self.agent_id),
            "timesteps": self.timesteps.tolist(),
            "behavior_policy_id": self.behavior_policy_id,
            "environment_sha256": self.environment_sha256,
            "config_sha256": self.config_sha256,
            "reset_masks": self.reset_masks.tolist(),
            "observations": self.observations.tolist(),
        }
