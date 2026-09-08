"""Axis-preserving recurrent MAPPO rollout storage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MAPPOBatch:
    observations: np.ndarray
    fp_states: np.ndarray
    actions: np.ndarray
    action_masks: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    active_masks: np.ndarray
    dones: np.ndarray
    actor_hidden_states: np.ndarray
    critic_hidden_states: np.ndarray


class MAPPORolloutStorage:
    """Store `[time, worker, agent, ...]` without collapsing team axes."""

    def __init__(
        self,
        rollout_length: int,
        num_workers: int,
        num_agents: int,
        observation_dim: int,
        state_dim: int,
        action_dim: int,
        recurrent_hidden_dim: int = 64,
    ) -> None:
        self.rollout_length = int(rollout_length)
        self.num_workers = int(num_workers)
        self.num_agents = int(num_agents)
        self.observation_dim = int(observation_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.recurrent_hidden_dim = int(recurrent_hidden_dim)
        if min(
            self.rollout_length,
            self.num_workers,
            self.num_agents,
            self.recurrent_hidden_dim,
        ) < 1:
            raise ValueError("MAPPO rollout dimensions must be positive.")
        self._records: list[dict[str, np.ndarray]] = []

    def __len__(self) -> int:
        return len(self._records)

    def insert(self, **values) -> None:
        expected = {
            "observations": (self.num_workers, self.num_agents, self.observation_dim),
            "fp_states": (self.num_workers, self.num_agents, self.state_dim),
            "actions": (self.num_workers, self.num_agents),
            "action_masks": (self.num_workers, self.num_agents, self.action_dim),
            "log_probs": (self.num_workers, self.num_agents),
            "values": (self.num_workers, self.num_agents),
            "rewards": (self.num_workers, self.num_agents),
            "dones": (self.num_workers, self.num_agents),
            "active_masks": (self.num_workers, self.num_agents),
            "actor_hidden_states": (
                self.num_workers,
                self.num_agents,
                self.recurrent_hidden_dim,
            ),
            "critic_hidden_states": (
                self.num_workers,
                self.num_agents,
                self.recurrent_hidden_dim,
            ),
        }
        if len(self._records) >= self.rollout_length:
            raise ValueError("MAPPO rollout is already full.")
        record = {}
        for name, shape in expected.items():
            array = np.asarray(values[name])
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
            record[name] = array.copy()
        bad_masks = np.asarray(
            values.get("bad_masks", np.ones(expected["dones"], dtype=np.float32)),
            dtype=np.float32,
        )
        if bad_masks.shape != expected["dones"]:
            raise ValueError("bad_masks must match the worker and agent axes.")
        record["bad_masks"] = bad_masks.copy()
        value_predictions = np.asarray(
            values.get("value_predictions", record["values"]), dtype=np.float32
        )
        if value_predictions.shape != expected["values"]:
            raise ValueError("value_predictions must match worker and agent axes.")
        record["value_predictions"] = value_predictions.copy()
        self._records.append(record)

    def compute_returns(
        self,
        next_values: np.ndarray,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> MAPPOBatch:
        if not self._records:
            raise ValueError("Cannot compute returns for an empty MAPPO rollout.")
        next_values = np.asarray(next_values, dtype=np.float32)
        shape = (self.num_workers, self.num_agents)
        if next_values.shape != shape:
            raise ValueError(f"next_values must have shape {shape}.")
        rewards = self._stack("rewards", np.float32)
        dones = self._stack("dones", bool)
        values = self._stack("values", np.float32)
        bad_masks = self._stack("bad_masks", np.float32)
        advantages = np.zeros_like(values)
        gae = np.zeros(shape, dtype=np.float32)
        following_values = next_values.copy()
        for step in reversed(range(len(self._records))):
            nonterminal = 1.0 - dones[step].astype(np.float32)
            delta = rewards[step] + gamma * following_values * nonterminal - values[step]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            gae *= bad_masks[step]
            advantages[step] = gae
            following_values = values[step]
        return MAPPOBatch(
            observations=self._stack("observations", np.float32),
            fp_states=self._stack("fp_states", np.float32),
            actions=self._stack("actions", np.int64),
            action_masks=self._stack("action_masks", bool),
            old_log_probs=self._stack("log_probs", np.float32),
            old_values=self._stack("value_predictions", np.float32),
            advantages=advantages,
            returns=advantages + values,
            active_masks=self._stack("active_masks", np.float32),
            dones=dones,
            actor_hidden_states=self._stack("actor_hidden_states", np.float32),
            critic_hidden_states=self._stack("critic_hidden_states", np.float32),
        )

    def _stack(self, name: str, dtype) -> np.ndarray:
        return np.stack([record[name] for record in self._records]).astype(
            dtype, copy=False
        )
