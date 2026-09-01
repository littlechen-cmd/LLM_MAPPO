"""E1 QMIX-DG guardrails, isolated from the legacy waypoint baseline."""

from typing import Mapping
import random

import numpy as np
import torch

from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.qmix import QMIXHyperparameters, QMIXLearner
from llm_mappo.shadow_state import ShadowStateAdapter


def validate_qmix_dg_contract(run, environment: Mapping) -> None:
    """Reject any QMIX setup that could silently fall back to waypoint inputs."""
    if run.group != "QMIX-DG" or run.algorithm != "qmix":
        raise ValueError("E1 QMIX-DG requires the exact QMIX-DG matrix identity.")
    if run.observation_schema != ObservationSchema.DIRECT_GOAL_V1.value:
        raise ValueError("E1 QMIX-DG requires DirectGoal observations.")
    if environment.get("observation_schema") != ObservationSchema.DIRECT_GOAL_V1.value:
        raise ValueError("E1 QMIX-DG environment may not use waypoint fallback.")
    if run.astar_kd != "disabled" or run.semantic_teacher != "disabled":
        raise ValueError("E1 QMIX-DG cannot acquire an A* or semantic teacher.")


class E1QMIXDGTrainer:
    """DirectGoal-only QMIX implementation for the E1 external-MARL baseline."""

    def __init__(self, *, run, environment: Mapping, device: str | torch.device) -> None:
        validate_qmix_dg_contract(run, environment)
        self.run, self.values, self.device = run, dict(environment), torch.device(device)
        self.environment = Phase2Warehouse(
            n_agents=int(self.values["n_agents"]), max_steps=int(self.values["max_steps"]),
            env_id=str(self.values["environment_id"]),
            charge_threshold=float(self.values["charge_threshold"]),
            charge_release_threshold=float(self.values["charge_release_threshold"]),
            battery_cost_scale=float(self.values["battery_cost_scale"]),
            deadlock_steps=int(self.values["deadlock_steps"]),
            batch_interval=int(self.values["dynamic_ingress_interval"]),
            batch_size_range=tuple(self.values["batch_size_range"]), initial_priority_label="A",
            request_queue_size=int(self.values["queue_size"]), task_completion_target=int(self.values["task_target"]),
            observation_schema=ObservationSchema.DIRECT_GOAL_V1,
        )
        self.rng = np.random.default_rng(int(run.seed))
        self.learner = QMIXLearner(613, 5, int(self.values["n_agents"]),
            QMIXHyperparameters(), self.device)
        self.replay: list[dict] = []
        self.adapter = ShadowStateAdapter(self.environment, code_commit="e1-qmix-dg-v1")
        self._runtime = None

    def close(self) -> None: self.environment.close()

    def run_prefix(self, steps: int) -> dict:
        if not 1 <= steps <= self.run.real_environment_steps: raise ValueError("QMIX-DG step budget is incompatible.")
        if self._runtime is None:
            observations = self.environment.reset(seed=int(self.run.seed)); episodes = 0; start = 0
        else:
            random.setstate(self._runtime["python_rng"]); np.random.set_state(self._runtime["numpy_rng"]); torch.set_rng_state(self._runtime["torch_rng"])
            self.environment.reset(seed=int(self.run.seed)); self.adapter.restore_bytes(self._runtime["snapshot"])
            observations = self.environment._observations(); episodes = int(self._runtime["episodes"]); start = int(self._runtime["steps"])
            self.replay = list(self._runtime["replay"]); self.rng.bit_generator.state = self._runtime["rng"]
            self.learner.load_state_dict(self._runtime["learner"])
        losses = []
        for step in range(start, steps):
            epsilon = max(.05, 1.0 - step / 100000.0); masks = self.environment.action_masks()
            actions = self.learner.select_actions(observations, masks, epsilon, self.rng)
            transition = self.environment.step(actions); next_masks = self.environment.action_masks()
            done = bool(transition.terminated or transition.truncated or transition.metrics.deadlocked)
            self.replay.append({"observations": observations.astype(np.float32), "actions": actions,
                "rewards": np.float32(transition.team_reward), "next_observations": transition.observations.astype(np.float32),
                "dones": np.float32(done), "masks": masks, "next_masks": next_masks})
            if len(self.replay) > 10000: self.replay.pop(0)
            if len(self.replay) >= 64:
                indices = self.rng.choice(len(self.replay), 64, replace=False)
                batch = {key: np.stack([self.replay[index][key] for index in indices]) for key in self.replay[0]}
                losses.append(self.learner.update(batch))
            if (step + 1) % 1000 == 0: self.learner.update_targets()
            observations = transition.observations
            if done: episodes += 1; observations = self.environment.reset(seed=int(self.run.seed) + episodes)
        self._runtime = self._capture_runtime(steps=steps, episodes=episodes)
        return {"algorithm": "QMIX-DG", "real_env_steps": steps, "episodes": episodes,
                "mean_qmix_loss": float(np.mean(losses)) if losses else 0.0,
                "planner_query_count": self.environment.planner_query_counter.count}

    def runtime_state(self) -> dict:
        if self._runtime is None: raise RuntimeError("QMIX runtime state is unavailable.")
        return dict(self._runtime)

    def restore_runtime_state(self, state: Mapping) -> None:
        required = {"schema", "snapshot", "steps", "episodes", "replay", "rng", "learner", "python_rng", "numpy_rng", "torch_rng"}
        if set(state) != required or state.get("schema") != "e1-qmix-runtime-v1":
            raise ValueError("E1 QMIX runtime state is incompatible.")
        self._runtime = dict(state)

    def _capture_runtime(self, *, steps: int, episodes: int) -> dict:
        address = {"run_seed": int(self.run.seed), "episode_index": int(episodes),
                   "episode_seed": int(self.run.seed) + int(episodes), "environment_index": 0,
                   "real_global_step": int(steps), "episode_step": int(self.environment._metrics.steps)}
        return {"schema": "e1-qmix-runtime-v1", "snapshot": self.adapter.capture(**address).to_bytes(),
                "steps": int(steps), "episodes": int(episodes), "replay": list(self.replay),
                "rng": self.rng.bit_generator.state, "learner": self.learner.state_dict(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}
