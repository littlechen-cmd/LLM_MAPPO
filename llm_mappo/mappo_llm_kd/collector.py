"""Deterministic, teacher-free behavior collection for Phase 3 LLM KD."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from llm_mappo.mappo_baseline.networks import MAPPOActor

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
    build_structured_observations,
)
from llm_mappo.mappo_llm_kd.dataset import (
    MAIN_BASELINE_DISTRIBUTION,
    DistributionTrajectory,
)
from llm_mappo.mappo_llm_kd.trajectory import ActorObservationTrajectory


_V2 = StructuredObservationSchema(observation_schema=DIRECT_GOAL_OBSERVATION_V2)
_FROZEN_ENVIRONMENT_SHA256_BY_PROFILE = {
    "medium-5ag-phase2-train": (
        "6031952c1a872bcab3a25243ecd29280b4b350b43d6edea1d55a38386da7da04"
    ),
    "medium-5ag-low-workload-collection": (
        "8a0d6abbea0a63503af3060d66441d5f9d963f053d397f874450a7f0f2cdb3a1"
    ),
}


class BehaviorPolicy(Protocol):
    """Narrow inference boundary for a frozen recurrent MAPPO Actor."""

    hidden_dim: int
    behavior_policy_id: str

    def act(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
        hidden: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class FrozenMAPPOPolicy:
    """Inference-only native 209D Actor loaded from the collection baseline."""

    usage_scope = "offline_behavior_collection_only"

    def __init__(self, actor: MAPPOActor, checkpoint_sha256: str, device: str):
        self.actor = actor.eval()
        self.device = torch.device(device)
        self.hidden_dim = int(actor.hidden_dim)
        self.behavior_policy_id = f"MAPPO-FP-DG:{checkpoint_sha256}"
        self._generator = torch.Generator(device=self.device.type)
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)

    def start_episode(self, seed: int) -> None:
        """Reset only this policy's sampling stream for deterministic replay."""

        self._generator.manual_seed(int(seed))

    @torch.no_grad()
    def act(self, observations, action_masks, hidden):
        observations_tensor = torch.as_tensor(
            observations[None], dtype=torch.float32, device=self.device
        )
        hidden_tensor = torch.as_tensor(
            hidden, dtype=torch.float32, device=self.device
        )
        masks = torch.as_tensor(
            action_masks[None], dtype=torch.bool, device=self.device
        )
        logits, next_hidden = self.actor(observations_tensor, hidden_tensor)
        logits = logits.masked_fill(~masks, -1e10)
        probabilities = torch.softmax(logits, dim=-1)
        actions = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            1,
            generator=self._generator,
        ).reshape(probabilities.shape[:-1])
        return (
            actions[0].cpu().numpy(),
            next_hidden.cpu().numpy(),
        )


def load_phase3_collection_policy(
    checkpoint: str | Path, *, device: str, expected_artifact_sha256: str
) -> FrozenMAPPOPolicy:
    """Strictly load an approved, weights-only-safe Phase 3 Actor artifact."""

    path = Path(checkpoint)
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != expected_artifact_sha256:
        raise ValueError("Collection Actor artifact SHA-256 is not approved.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    identity = payload.get("identity", {})
    expected_identity = {
        "schema": "mappo-collection-209d-v1",
        "method": "MAPPO-FP-DG",
        "observation_schema": DIRECT_GOAL_OBSERVATION_V2,
        "actor_observation_schema": "structured-direct_goal-v2",
        "actor_observation_dimension": 209,
        "reward_version": "reward-v2",
        "environment_id": "llm-mappo-medium-3ag-v1",
        "n_agents": 5,
        "goal_interface": "direct_goal",
    }
    if payload.get("schema") != "mappo-collection-actor-209d-v1" or any(
            identity.get(name) != value for name, value in expected_identity.items()
    ):
        raise ValueError("Artifact is not the native Phase 3 collection contract.")
    expected_environment_sha256 = _FROZEN_ENVIRONMENT_SHA256_BY_PROFILE.get(
        identity.get("profile_id")
    )
    if (
        expected_environment_sha256 is None
        or identity.get("environment_sha256") != expected_environment_sha256
    ):
        raise ValueError("Phase 3 collection environment contract differs.")
    config = payload.get("actor_config")
    if not isinstance(config, dict):
        raise ValueError("Phase 3 collection artifact has no Actor configuration.")
    expected_config = {"observation_dim": 209, "action_dim": 5, "hidden_dim": 64}
    if any(config.get(name) != value for name, value in expected_config.items()):
        raise ValueError("Phase 3 collection checkpoint Actor configuration differs.")
    actor_state = payload.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        raise ValueError("Phase 3 collection artifact has no Actor state.")
    actor = MAPPOActor(209, 5, 64).to(device)
    actor.load_state_dict(actor_state, strict=True)
    return FrozenMAPPOPolicy(actor, digest, device)


def collect_complete_episode(
    environment,
    policy: BehaviorPolicy,
    *,
    seed: int,
    episode_id: str,
    environment_sha256: str,
    config_sha256: str,
    max_steps: int,
) -> list[DistributionTrajectory]:
    """Run one complete five-AGV episode and retain only v2 Actor observations."""

    if int(environment.n_agents) != 5:
        raise ValueError("Phase 3 diagnostic collection requires five AGVs.")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer.")
    environment.reset(seed=int(seed))
    start_episode = getattr(policy, "start_episode", None)
    if start_episode is not None:
        start_episode(int(seed))
    hidden = np.zeros((1, 5, int(policy.hidden_dim)), dtype=np.float32)
    observations_by_agent = [[] for _ in range(5)]
    completed = False
    for _step in range(max_steps):
        observations = build_structured_observations(
            environment, _V2, goal_interface="direct_goal", planner=None
        )
        for agent_index, row in enumerate(observations):
            observations_by_agent[agent_index].append(row.copy())
        actions, hidden = policy.act(
            observations,
            np.asarray(environment.action_masks(), dtype=np.bool_),
            hidden,
        )
        transition = environment.step(np.asarray(actions, dtype=np.int64))
        completed = bool(
            transition.terminated
            or transition.truncated
            or transition.metrics.deadlocked
        )
        if completed:
            break
    if not completed:
        raise RuntimeError("Collector stopped before a complete episode boundary.")
    length = len(observations_by_agent[0])
    return [
        DistributionTrajectory(
            MAIN_BASELINE_DISTRIBUTION,
            ActorObservationTrajectory(
                episode_id=episode_id,
                agent_id=agent_index + 1,
                timesteps=np.arange(length, dtype=np.int64),
                observations=np.stack(rows),
                reset_masks=np.asarray(
                    [True] + [False] * (length - 1), dtype=np.bool_
                ),
                behavior_policy_id=policy.behavior_policy_id,
                environment_sha256=environment_sha256,
                config_sha256=config_sha256,
            ),
        )
        for agent_index, rows in enumerate(observations_by_agent)
    ]
