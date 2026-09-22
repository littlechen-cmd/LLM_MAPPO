"""Teacher-free MAPPO baseline with a feature-pruned centralized state."""

from llm_mappo.mappo_baseline.fp_state import FPStateSchema, build_fp_state
from llm_mappo.mappo_baseline.networks import MAPPOActor, MAPPOFPCritic
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V1,
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
    build_structured_observations,
)
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer

__all__ = [
    "FPStateSchema",
    "MAPPOActor",
    "MAPPOConfig",
    "MAPPOFPCritic",
    "MAPPOTrainer",
    "DIRECT_GOAL_OBSERVATION_V1",
    "DIRECT_GOAL_OBSERVATION_V2",
    "StructuredObservationSchema",
    "build_fp_state",
    "build_structured_observations",
]
