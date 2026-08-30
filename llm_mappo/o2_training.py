"""O2-only MAPPO rollout and PPO update adapter.

This module intentionally composes frozen O0/O1 networks and auxiliary-loss
objects without modifying their implementations.  It is not a replacement for
the legacy MAPPO trainer.
"""

from dataclasses import dataclass
import random
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as functional

from llm_mappo.optimization_buffer import OptimizationBatch, OptimizationRolloutBuffer
from llm_mappo.optimization_buffer import LinearEnvStepSchedule
from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor
from llm_mappo.o2_contract import O2ExperimentConfig, O2RunSpec
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.pure_motion_teacher import PureMotionQuery, PureMotionTeacher
from llm_mappo.reward_calibration import RewardCalibrator
from llm_mappo.shadow_state import ShadowStateAdapter


@dataclass(frozen=True)
class O2PPOHyperparameters:
    """The pre-registered Phase 4 PPO values used by every O2 run."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.50
    update_epochs: int = 4
    minibatch_steps: int = 64


@dataclass(frozen=True)
class O2RolloutTensors:
    """PPO transition tensors plus the frozen O0 auxiliary-loss tensors."""

    physical_observations: Tensor
    semantic_observations: Tensor
    actions: Tensor
    old_log_probs: Tensor
    action_masks: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor
    auxiliary: OptimizationBatch


class O2Rollout:
    """Whole-team O2 transitions with centralized GAE and O0 KD fields."""

    def __init__(self, n_agents: int) -> None:
        self.n_agents = int(n_agents)
        self._auxiliary = OptimizationRolloutBuffer(self.n_agents)
        self._rewards: list[float] = []
        self._dones: list[bool] = []
        self._values: list[float] = []

    def __len__(self) -> int:
        return len(self._rewards)

    def add(
        self,
        *,
        physical_observations: np.ndarray,
        semantic_observations: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        action_masks: np.ndarray,
        astar_preferences: np.ndarray,
        astar_valid: np.ndarray,
        calibration_selected: bool,
        reward_confidence: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        if not np.isfinite(reward) or not np.isfinite(value):
            raise ValueError("O2 rollout reward and value must be finite.")
        semantic_targets = np.zeros((self.n_agents, 3), dtype=np.float32)
        semantic_validity = np.zeros(self.n_agents, dtype=np.float32)
        self._auxiliary.add(
            physical_observations=physical_observations,
            semantic_observations=semantic_observations,
            actions=actions,
            log_probs=log_probs,
            action_masks=action_masks,
            astar_preferences=astar_preferences,
            astar_valid=astar_valid,
            calibration_selected=calibration_selected,
            reward_confidence=reward_confidence,
            semantic_targets=semantic_targets,
            semantic_validity=semantic_validity,
            ood_reliability=0.0,
        )
        self._rewards.append(float(reward))
        self._dones.append(bool(done))
        self._values.append(float(value))

    def tensors(
        self,
        *,
        last_value: float,
        hyperparameters: O2PPOHyperparameters,
        device: str | torch.device,
    ) -> O2RolloutTensors:
        if not self._rewards:
            raise ValueError("Cannot update an empty O2 rollout.")
        advantages = np.zeros(len(self), dtype=np.float32)
        next_value = float(last_value)
        gae = 0.0
        for index in reversed(range(len(self))):
            keep_bootstrap = 1.0 - float(self._dones[index])
            delta = self._rewards[index] + (
                hyperparameters.gamma * next_value * keep_bootstrap
            ) - self._values[index]
            gae = delta + (
                hyperparameters.gamma
                * hyperparameters.gae_lambda
                * keep_bootstrap
                * gae
            )
            advantages[index] = gae
            next_value = self._values[index]
        auxiliary = self._auxiliary.tensors(device)
        return O2RolloutTensors(
            physical_observations=auxiliary.physical_observations,
            semantic_observations=auxiliary.semantic_observations,
            actions=auxiliary.actions,
            old_log_probs=auxiliary.log_probs,
            action_masks=auxiliary.action_masks,
            rewards=torch.as_tensor(self._rewards, dtype=torch.float32, device=device),
            dones=torch.as_tensor(self._dones, dtype=torch.bool, device=device),
            values=torch.as_tensor(self._values, dtype=torch.float32, device=device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
            returns=torch.as_tensor(
                advantages + np.asarray(self._values, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            auxiliary=auxiliary,
        )


class O2PPOUpdater:
    """Clipped PPO plus the frozen group-specific A*KD objective."""

    def __init__(
        self,
        *,
        actor: O0StudentActor,
        critic: O0CentralizedCritic,
        hyperparameters: O2PPOHyperparameters,
        method: str,
        device: str | torch.device,
    ) -> None:
        if method not in {"MAPPO-DG", "RC-AStarKD"}:
            raise ValueError("O2 only supports MAPPO-DG or RC-AStarKD.")
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.hyperparameters = hyperparameters
        self.method = method
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=hyperparameters.learning_rate,
        )

    def update(
        self,
        rollout: O2Rollout,
        *,
        last_value: float,
        lambda_a: float,
    ) -> dict[str, float | bool]:
        data = rollout.tensors(
            last_value=last_value,
            hyperparameters=self.hyperparameters,
            device=self.device,
        )
        advantages = self._normalized_advantages(data.advantages)
        total_steps = data.physical_observations.shape[0]
        sums = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "astar_loss": 0.0,
            "total_loss": 0.0,
        }
        updates = 0
        for _ in range(self.hyperparameters.update_epochs):
            order = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, self.hyperparameters.minibatch_steps):
                indices = order[start:start + self.hyperparameters.minibatch_steps]
                metrics = self._minibatch_update(data, advantages, indices, lambda_a)
                updates += 1
                for name, value in metrics.items():
                    sums[name] += value
        if updates < 1:
            raise RuntimeError("O2 PPO did not construct any minibatch updates.")
        return {
            **{name: value / updates for name, value in sums.items()},
            "semantic_loss": 0.0,
            "finite": True,
        }

    def _minibatch_update(
        self,
        data: O2RolloutTensors,
        advantages: Tensor,
        indices: Tensor,
        lambda_a: float,
    ) -> dict[str, float]:
        physical = data.physical_observations[indices]
        semantic = data.semantic_observations[indices]
        actions = data.actions[indices]
        masks = data.action_masks[indices]
        output = self.actor(physical, semantic)
        masked_logits = output.action_logits.masked_fill(~masks, -1e9)
        distribution = Categorical(logits=masked_logits)
        log_probs = distribution.log_prob(actions)
        ratios = torch.exp(log_probs - data.old_log_probs[indices])
        agent_advantages = advantages[indices].unsqueeze(-1)
        clipped = torch.clamp(
            ratios,
            1.0 - self.hyperparameters.clip_ratio,
            1.0 + self.hyperparameters.clip_ratio,
        )
        policy_loss = -torch.minimum(
            ratios * agent_advantages, clipped * agent_advantages
        ).mean()
        value_loss = functional.mse_loss(self.critic(physical), data.returns[indices])
        entropy = distribution.entropy().mean()
        astar_loss = self._astar_loss(
            data.auxiliary, output.motion_logits, indices, lambda_a
        )
        total = (
            policy_loss
            + self.hyperparameters.value_coefficient * value_loss
            - self.hyperparameters.entropy_coefficient * entropy
            + astar_loss
        )
        if not torch.isfinite(total):
            raise RuntimeError("O2 PPO produced a non-finite minibatch loss.")
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.hyperparameters.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("O2 PPO produced a non-finite gradient norm.")
        self.optimizer.step()
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "astar_loss": float(astar_loss.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }

    def _astar_loss(
        self,
        batch: OptimizationBatch,
        motion_logits: Tensor,
        indices: Tensor,
        lambda_a: float,
    ) -> Tensor:
        if self.method == "MAPPO-DG":
            return motion_logits.sum() * 0.0
        valid = batch.astar_valid[indices]
        selected = batch.calibration_selected[indices].unsqueeze(-1)
        active = valid & selected
        denominator = active.sum()
        if not bool(denominator):
            return motion_logits.sum() * 0.0
        preferences = batch.astar_preferences[indices]
        divergence = functional.kl_div(
            torch.log_softmax(motion_logits, dim=-1), preferences, reduction="none"
        ).sum(dim=-1)
        weights = active.to(motion_logits.dtype) * batch.reward_confidence[
            indices
        ].unsqueeze(-1)
        return float(lambda_a) * (weights * divergence).sum() / denominator

    @staticmethod
    def _normalized_advantages(advantages: Tensor) -> Tensor:
        if not torch.isfinite(advantages).all():
            raise RuntimeError("O2 PPO advantages are non-finite.")
        return (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )


class O2Trainer:
    """One O2 run using stochastic actions and the approved frozen components."""

    def __init__(
        self,
        *,
        experiment: O2ExperimentConfig,
        run: O2RunSpec,
        device: str | torch.device,
        calibration_weight_mode: str = "reward-calibrated",
    ) -> None:
        if run not in set(
            O2RunSpec(group, seed, experiment.real_env_steps)
            for group in experiment.groups
            for seed in experiment.seeds
        ):
            raise ValueError("O2 run is outside the frozen experiment matrix.")
        self.experiment = experiment
        self.run_spec = run
        if calibration_weight_mode not in {"fixed", "reward-calibrated"}:
            raise ValueError("Unsupported O2 calibration weight mode.")
        if run.group == "MAPPO-DG" and calibration_weight_mode != "reward-calibrated":
            raise ValueError("MAPPO-DG cannot enter the Fixed/RC parity path.")
        self.calibration_weight_mode = calibration_weight_mode
        self.device = torch.device(device)
        self.hyperparameters = O2PPOHyperparameters(
            gamma=float(experiment.training["gamma"]),
            gae_lambda=float(experiment.training["gae_lambda"]),
            clip_ratio=float(experiment.training["clip_ratio"]),
            value_coefficient=float(experiment.training["value_coefficient"]),
            entropy_coefficient=float(experiment.training["entropy_coefficient"]),
            learning_rate=float(experiment.training["learning_rate"]),
            max_grad_norm=float(experiment.training["max_grad_norm"]),
            update_epochs=int(experiment.training["update_epochs"]),
            minibatch_steps=int(experiment.training["minibatch_steps"]),
        )
        self._seed_everything(run.seed)
        self.actor = O0StudentActor().to(self.device)
        self.critic = O0CentralizedCritic().to(self.device)
        self.updater = O2PPOUpdater(
            actor=self.actor,
            critic=self.critic,
            hyperparameters=self.hyperparameters,
            method=run.group,
            device=self.device,
        )
        self.schedule = LinearEnvStepSchedule(run.real_env_steps)
        self.teacher = PureMotionTeacher() if run.group == "RC-AStarKD" else None
        self.calibrator = RewardCalibrator() if self.teacher is not None else None
        self.environment = self._new_environment()
        self.student_shadow = self._new_environment()
        self.teacher_shadow = self._new_environment()
        self.real_adapter = ShadowStateAdapter(self.environment, code_commit="o2-v1")
        self.student_adapter = ShadowStateAdapter(
            self.student_shadow, code_commit="o2-v1"
        )
        self.teacher_adapter = ShadowStateAdapter(
            self.teacher_shadow, code_commit="o2-v1"
        )

    def run(
        self,
        max_steps: int | None = None,
        *,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        on_update: Callable[[dict[str, Any]], None] | None = None,
        on_episode: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run a bounded prefix; caller marks any prefix diagnostic-only in evidence."""
        target_steps = (
            self.run_spec.real_env_steps if max_steps is None else int(max_steps)
        )
        if target_steps < 1 or target_steps > self.run_spec.real_env_steps:
            raise ValueError("O2 run length must be within its frozen step budget.")
        observations = self.environment.reset(seed=self.run_spec.seed)
        self.student_shadow.reset(seed=self.run_spec.seed)
        self.teacher_shadow.reset(seed=self.run_spec.seed)
        rollout = O2Rollout(self.experiment.environment["n_agents"])
        episode_index = 0
        episode_step = 0
        updates = 0
        episodes = 0
        cumulative_completed_tasks = 0
        cumulative_episode_steps = 0
        counts = {
            "teacher_queries": 0,
            "calibration_selected": 0,
            "calibration_selected_agent_slots": 0,
            "valid_teacher_selected_slots": 0,
            "shadow_calls": 0,
            "ema_updates": 0,
        }
        latest_metrics: dict[str, float | bool | int] = {}
        for step in range(target_steps):
            masks = self.environment.action_masks()
            actions, log_probs, value = self._stochastic_actions(observations, masks)
            preferences, valid = self._teacher_batch(self.environment)
            result = self._calibration_result(
                observations=observations,
                valid=valid,
                episode_index=episode_index,
                episode_step=episode_step,
                real_global_step=step,
                counts=counts,
            )
            if result.selected:
                counts["calibration_selected_agent_slots"] += valid.size
                counts["valid_teacher_selected_slots"] += int(valid.sum())
            transition = self.environment.step(actions)
            done = bool(
                transition.terminated
                or transition.truncated
                or transition.metrics.deadlocked
            )
            rollout.add(
                physical_observations=observations,
                semantic_observations=np.zeros(
                    (self.experiment.environment["n_agents"], 61), dtype=np.float32
                ),
                actions=actions,
                log_probs=log_probs,
                action_masks=masks,
                astar_preferences=preferences,
                astar_valid=valid,
                calibration_selected=result.selected,
                reward_confidence=self._reward_confidence(result),
                reward=transition.team_reward,
                done=done,
                value=value,
            )
            self.schedule.advance_real_env_steps(1)
            observations = transition.observations
            latest_metrics = transition.metrics.as_dict()
            if on_step is not None:
                on_step(
                    {
                        "real_env_steps": self.schedule.global_env_steps,
                        **counts,
                    }
                )
            if len(rollout) == self.experiment.training["rollout_steps"] or (
                step + 1 == target_steps
            ):
                last_value = 0.0 if done else self._critic_value(observations)
                update_metrics = self.updater.update(
                    rollout,
                    last_value=last_value,
                    lambda_a=self.schedule.weights()[0],
                )
                if not update_metrics["finite"]:
                    raise RuntimeError("O2 PPO update reported non-finite metrics.")
                updates += 1
                if on_update is not None:
                    on_update(
                        {
                            "real_env_steps": self.schedule.global_env_steps,
                            **update_metrics,
                        }
                    )
                rollout = O2Rollout(self.experiment.environment["n_agents"])
            if done:
                episodes += 1
                cumulative_completed_tasks += int(transition.metrics.completed_tasks)
                cumulative_episode_steps += int(transition.metrics.steps)
                if on_episode is not None:
                    on_episode(
                        {
                            "real_env_steps": self.schedule.global_env_steps,
                            "cumulative_completed_tasks": cumulative_completed_tasks,
                            "cumulative_episode_steps": cumulative_episode_steps,
                        }
                    )
                episode_index += 1
                episode_step = 0
                observations = self.environment.reset(
                    seed=self.run_spec.seed + episode_index
                )
            else:
                episode_step += 1
        return {
            "group": self.run_spec.group,
            "seed": self.run_spec.seed,
            "real_env_steps": self.schedule.global_env_steps,
            "updates": updates,
            "episodes": episodes,
            "semantic_loss": 0.0,
            "planner_query_count": self.environment.planner_query_counter.count,
            "latest_episode_metrics": latest_metrics,
            **counts,
        }

    def _new_environment(self) -> Phase2Warehouse:
        values = self.experiment.environment
        return Phase2Warehouse(
            n_agents=int(values["n_agents"]),
            max_steps=int(values["max_steps"]),
            env_id=str(values["environment_id"]),
            charge_threshold=float(values["charge_threshold"]),
            charge_release_threshold=float(values["charge_release_threshold"]),
            battery_cost_scale=float(values["battery_cost_scale"]),
            deadlock_steps=int(values["deadlock_steps"]),
            batch_interval=int(values["dynamic_ingress_interval"]),
            batch_size_range=tuple(values["batch_size_range"]),
            request_queue_size=int(values["queue_size"]),
            task_completion_target=int(values["task_target"]),
            observation_schema=ObservationSchema.DIRECT_GOAL_V1,
        )

    def _stochastic_actions(
        self, observations: np.ndarray, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        physical = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        semantic = torch.zeros((physical.shape[0], 61), dtype=torch.float32,
                               device=self.device)
        action_masks = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            output = self.actor(physical, semantic)
            logits = output.action_logits.masked_fill(~action_masks, -1e9)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            value = self.critic(physical.unsqueeze(0)).item()
        return (
            actions.cpu().numpy().astype(np.int64),
            log_probs.cpu().numpy().astype(np.float32),
            float(value),
        )

    def _teacher_batch(
        self, environment: Phase2Warehouse
    ) -> tuple[np.ndarray, np.ndarray]:
        n_agents = int(self.experiment.environment["n_agents"])
        if self.teacher is None:
            return np.zeros((n_agents, 3), dtype=np.float32), np.zeros(
                n_agents, dtype=bool
            )
        warehouse = environment.env
        shelf_coordinates = tuple(
            (x, y)
            for y in range(warehouse.grid_size[0])
            for x in range(warehouse.grid_size[1])
            if not warehouse._is_highway(x, y)
        )
        results = []
        for agent in warehouse.agents:
            goal, _ = environment._target_for_agent(agent.id)
            blocked = shelf_coordinates if agent.carrying_shelf is not None else ()
            results.append(
                self.teacher.query(
                    PureMotionQuery(
                        layout_hash=warehouse.shadow_layout_hash(),
                        width=int(warehouse.grid_size[1]),
                        height=int(warehouse.grid_size[0]),
                        blocked_coordinates=tuple((int(x), int(y)) for x, y in blocked),
                        own_pose=(int(agent.x), int(agent.y)),
                        orientation=agent.dir.name,
                        goal=(int(goal[0]), int(goal[1])),
                        occupied_coordinates=tuple(
                            (int(other.x), int(other.y))
                            for other in warehouse.agents
                            if other.id != agent.id
                        ),
                        pure_motion_mask=(False, True, True, True, False),
                        dead=bool(agent.dead),
                        picking_lock=bool(agent.picking_lock_steps),
                        mandatory_toggle_load=environment._requires_pickup(agent.id),
                        footprint_class=(
                            "loaded" if agent.carrying_shelf is not None else "unloaded"
                        ),
                    )
                )
            )
        return (
            np.stack([result.motion_preferences[1:4] for result in results]),
            np.asarray([result.valid for result in results], dtype=bool),
        )

    def _calibration_result(
        self,
        *,
        observations: np.ndarray,
        valid: np.ndarray,
        episode_index: int,
        episode_step: int,
        real_global_step: int,
        counts: dict[str, int],
    ):
        if self.calibrator is None:
            return self._no_calibration_result()
        address = {
            "run_seed": self.run_spec.seed,
            "episode_index": episode_index,
            "episode_seed": self.run_spec.seed + episode_index,
            "environment_index": 0,
            "real_global_step": real_global_step,
            "episode_step": episode_step,
        }
        selected = self.calibrator.select(**address)
        counts["teacher_queries"] += int(valid.size)
        counts["calibration_selected"] += int(selected)
        if not selected or not valid.any():
            return self.calibrator.record_delta_g(
                selected=selected, any_valid=bool(valid.any()), delta_g=0.0
            )
        snapshot = self.real_adapter.capture(**address)
        result = self.calibrator.run_paired_shadows(
            snapshot=snapshot,
            real_adapter=self.real_adapter,
            student_adapter=self.student_adapter,
            teacher_adapter=self.teacher_adapter,
            student_logits=self._shadow_student_logits,
            teacher_preferences=self._shadow_teacher_batch,
            initial_valid_mask=valid,
            critic_value=self._critic_value,
            gamma=self.hyperparameters.gamma,
            address=address,
        )
        counts["shadow_calls"] += 1
        counts["ema_updates"] += 1
        return result

    def _shadow_teacher_batch(
        self, environment: Phase2Warehouse
    ) -> tuple[np.ndarray, np.ndarray]:
        original = self.teacher
        self.teacher = PureMotionTeacher()
        try:
            return self._teacher_batch(environment)
        finally:
            self.teacher = original

    def _shadow_student_logits(
        self, environment: Phase2Warehouse, observations: np.ndarray
    ) -> np.ndarray:
        del environment
        physical = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        semantic = torch.zeros((physical.shape[0], 61), dtype=torch.float32,
                               device=self.device)
        with torch.no_grad():
            return self.actor(physical, semantic).action_logits.cpu().numpy()

    def _critic_value(self, observations: np.ndarray) -> torch.Tensor:
        physical = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.critic(physical.unsqueeze(0))

    @staticmethod
    def _no_calibration_result():
        from llm_mappo.reward_calibration import CalibrationResult

        return CalibrationResult(False, False, False, 0.0, None)

    def _reward_confidence(self, result) -> float:
        if self.run_spec.group == "MAPPO-DG":
            return 0.0
        if self.calibration_weight_mode == "fixed":
            return 1.0 if result.selected else 0.0
        return float(result.confidence)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
