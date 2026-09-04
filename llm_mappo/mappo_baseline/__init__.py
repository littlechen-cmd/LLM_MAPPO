"""Teacher-free MAPPO baseline with a feature-pruned centralized state."""

from llm_mappo.mappo_baseline.fp_state import FPStateSchema, build_fp_state
from llm_mappo.mappo_baseline.networks import MAPPOActor, MAPPOFPCritic
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer

__all__ = [
    "FPStateSchema",
    "MAPPOActor",
    "MAPPOConfig",
    "MAPPOFPCritic",
    "MAPPOTrainer",
    "build_fp_state",
]
