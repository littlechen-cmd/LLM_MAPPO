"""Frozen O1 functional-smoke composition for the optimization route."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as functional
import yaml

from llm_mappo.optimization_buffer import (
    LinearEnvStepSchedule,
    OptimizationRolloutBuffer,
)
from llm_mappo.optimization_logging import O0RunLogger
from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.pure_motion_teacher import PureMotionQuery, PureMotionTeacher
from llm_mappo.reward_calibration import RewardCalibrator, deterministic_masked_argmax
from llm_mappo.semantic_v3 import SemanticDatasetV3, SemanticViewV3
from llm_mappo.shadow_state import ShadowStateAdapter


@dataclass(frozen=True)
class OptimizationTrainingConfig:
    environment_id: str
    n_agents: int
    dynamic_ingress_interval: int
    batch_size_range: tuple[int, int]
    queue_size: int
    task_target: int
    max_steps: int
    deadlock_steps: int
    battery_cost_scale: float
    charge_threshold: float
    charge_release_threshold: float
    observation_schema: str
    method: str
    seed: int
    real_env_steps: int
    rollout_length: int
    optimizer_epochs: int
    minibatch_size: int
    device: str
    fixture_path: str
    fixture_only: bool
    k_motion: int
    h_reward: int
    expansion_budget: int
    sampler_divisor: int
    lambda_a_start: float
    lambda_l_start: float
    ema_decay: float
    ema_minimum_scale: float
    ema_initialization_samples: int
    gamma: float
    online_llm: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OptimizationTrainingConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OptimizationTrainingConfig":
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(values) - expected
        missing = expected - set(values)
        if unknown or missing:
            name = sorted(unknown or missing)[0]
            raise ValueError(f"Unknown or missing optimization config field: {name}.")
        normalized = dict(values)
        normalized["batch_size_range"] = tuple(normalized["batch_size_range"])
        config = cls(**normalized)
        config._validate()
        return config

    def _validate(self) -> None:
        frozen = {
            "environment_id": "llm-mappo-medium-3ag-v1",
            "n_agents": 5,
            "dynamic_ingress_interval": 40,
            "batch_size_range": (4, 8),
            "queue_size": 8,
            "task_target": 50,
            "max_steps": 1000,
            "deadlock_steps": 180,
            "battery_cost_scale": 1.10,
            "charge_threshold": 0.30,
            "charge_release_threshold": 0.80,
            "observation_schema": "direct-goal-observation-v1",
            "k_motion": 12,
            "h_reward": 12,
            "expansion_budget": 512,
            "sampler_divisor": 16,
            "lambda_a_start": 0.05,
            "lambda_l_start": 0.10,
            "ema_decay": 0.99,
            "ema_minimum_scale": 1e-3,
            "ema_initialization_samples": 64,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                display = "H_reward" if name == "h_reward" else name
                raise ValueError(f"Frozen optimization field {display} is incompatible.")
        if self.online_llm:
            raise ValueError("Optimization training forbids online LLM access.")
        if self.method not in {"fixed-kd", "reward-calibrated-kd", "mappo-dg"}:
            raise ValueError("Unsupported optimization method.")
        if self.real_env_steps < 1 or self.rollout_length < 1:
            raise ValueError("Optimization step counts must be positive.")
        if self.optimizer_epochs != 1 or self.minibatch_size != 32:
            raise ValueError("Functional smoke optimizer settings are frozen.")
        if not self.fixture_only:
            raise ValueError("O1 functional smoke accepts fixture-only semantics only.")

    def as_dict(self) -> dict:
        return asdict(self)


class OptimizationTrainer:
    """A deliberately small, non-claim-bearing 128-step integration smoke."""

    def __init__(self, config: OptimizationTrainingConfig, output_directory: str | Path):
        self.config = config
        self.output_directory = Path(output_directory)
        self.device = torch.device(config.device)
        self.actor = O0StudentActor().to(self.device)
        self.critic = O0CentralizedCritic().to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4
        )
        self.schedule = LinearEnvStepSchedule(config.real_env_steps)
        self.teacher = PureMotionTeacher()
        self.calibrator = RewardCalibrator()
        self.dataset = self._load_fixture(config.fixture_path)
        self.logger = O0RunLogger(output_directory)
        self.environment = self._new_environment()
        self.student_shadow = self._new_environment()
        self.teacher_shadow = self._new_environment()
        self.real_adapter = ShadowStateAdapter(self.environment, code_commit="o1-smoke")
        self.student_adapter = ShadowStateAdapter(
            self.student_shadow, code_commit="o1-smoke"
        )
        self.teacher_adapter = ShadowStateAdapter(
            self.teacher_shadow, code_commit="o1-smoke"
        )

    def _new_environment(self) -> Phase2Warehouse:
        return Phase2Warehouse(
            n_agents=self.config.n_agents,
            max_steps=self.config.max_steps,
            env_id=self.config.environment_id,
            charge_threshold=self.config.charge_threshold,
            charge_release_threshold=self.config.charge_release_threshold,
            battery_cost_scale=self.config.battery_cost_scale,
            deadlock_steps=self.config.deadlock_steps,
            batch_interval=self.config.dynamic_ingress_interval,
            batch_size_range=self.config.batch_size_range,
            request_queue_size=self.config.queue_size,
            task_completion_target=self.config.task_target,
            observation_schema=ObservationSchema.DIRECT_GOAL_V1,
        )

    def _load_fixture(self, fixture_path: str) -> SemanticDatasetV3:
        manifest_path = Path(fixture_path).with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fixture_only") is not True:
            raise ValueError("Functional smoke requires a fixture_only manifest.")
        records = [
            json.loads(line)
            for line in Path(fixture_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return SemanticDatasetV3.from_records(records)

    def train(self) -> dict:
        observations = self.environment.reset(seed=self.config.seed)
        self.student_shadow.reset(seed=self.config.seed)
        self.teacher_shadow.reset(seed=self.config.seed)
        buffer = OptimizationRolloutBuffer(self.config.n_agents)
        rewards: list[float] = []
        episode_index = 0
        update_count = 0
        calibration_calls = 0
        for step in range(self.config.real_env_steps):
            semantic, targets, semantic_valid, reliability = self._semantic_batch()
            masks = self.environment.action_masks()
            physical = torch.as_tensor(
                observations, dtype=torch.float32, device=self.device
            )
            semantic_tensor = torch.as_tensor(
                semantic, dtype=torch.float32, device=self.device
            )
            with torch.inference_mode():
                output = self.actor(physical, semantic_tensor)
            actions = deterministic_masked_argmax(
                output.action_logits.cpu().numpy(), masks
            )
            log_probs = self._selected_log_probs(output.action_logits, actions, masks)
            preferences, valid = self._teacher_batch(self.environment)
            address = {
                "run_seed": self.config.seed,
                "episode_index": episode_index,
                "episode_seed": self.config.seed + episode_index,
                "environment_index": 0,
                "real_global_step": step,
                "episode_step": step,
            }
            result = self.calibrator.record_delta_g(
                selected=False, any_valid=bool(valid.any()), delta_g=0.0
            )
            if self.calibrator.select(**address) and valid.any():
                snapshot = self.real_adapter.capture(**address)
                result = self.calibrator.run_paired_shadows(
                    snapshot=snapshot,
                    real_adapter=self.real_adapter,
                    student_adapter=self.student_adapter,
                    teacher_adapter=self.teacher_adapter,
                    student_logits=self._student_logits,
                    teacher_preferences=self._teacher_batch,
                    initial_valid_mask=valid,
                    critic_value=self._critic_value,
                    gamma=self.config.gamma,
                    address=address,
                )
                calibration_calls += 1
            transition = self.environment.step(actions)
            buffer.add(
                physical_observations=observations,
                semantic_observations=semantic,
                actions=actions,
                log_probs=log_probs,
                action_masks=masks,
                astar_preferences=preferences,
                astar_valid=valid,
                calibration_selected=result.selected,
                reward_confidence=(
                    1.0 if self.config.method == "fixed-kd" else result.confidence
                ),
                semantic_targets=targets,
                semantic_validity=semantic_valid,
                ood_reliability=reliability,
            )
            rewards.append(transition.team_reward)
            self.schedule.advance_real_env_steps(1)
            observations = transition.observations
            if (
                len(rewards) == self.config.rollout_length
                or step + 1 == self.config.real_env_steps
            ):
                loss = self._update(buffer, rewards)
                update_count += 1
                self.logger.write(
                    {
                        "event": "update",
                        "real_env_steps": self.schedule.global_env_steps,
                        "loss_total": loss,
                        "planner_query_count": (
                            self.environment.planner_query_counter.count
                        ),
                        "teacher_valid_count": int(valid.sum()),
                        "shadow_attempted_count": calibration_calls,
                        "pollution_counters": {"astar": 0, "llm": 0, "planner": 0},
                    }
                )
                buffer = OptimizationRolloutBuffer(self.config.n_agents)
                rewards = []
            if transition.terminated or transition.truncated:
                episode_index += 1
                observations = self.environment.reset(
                    seed=self.config.seed + episode_index
                )
        summary = {
            "real_env_steps": self.schedule.global_env_steps,
            "updates": update_count,
            "calibration_calls": calibration_calls,
            "planner_query_count": self.environment.planner_query_counter.count,
            "fixture_only": True,
        }
        (self.output_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def _update(self, buffer: OptimizationRolloutBuffer, rewards: list[float]) -> float:
        batch = buffer.tensors(self.device)
        output = self.actor(batch.physical_observations, batch.semantic_observations)
        logits = output.action_logits.masked_fill(~batch.action_masks, -1e9)
        log_probs = torch.log_softmax(logits, dim=-1).gather(
            -1, batch.actions.unsqueeze(-1)
        ).squeeze(-1)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        policy_loss = -(log_probs * reward_tensor.unsqueeze(-1)).mean()
        lambda_a, lambda_l = self.schedule.weights()
        astar_loss = batch.astar_kl_loss(output.motion_logits, lambda_a)
        semantic_loss = batch.semantic_mse_loss(output.semantic_scores, lambda_l)
        critic_values = self.critic(batch.physical_observations)
        value_loss = functional.mse_loss(critic_values, reward_tensor)
        total = policy_loss + astar_loss + semantic_loss + 0.1 * value_loss
        if not torch.isfinite(total):
            raise RuntimeError("O1 functional smoke produced a non-finite loss.")
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        return float(total.detach().cpu())

    def _semantic_batch(self):
        views = self._semantic_views(self.environment)
        targets, validities, reliabilities = zip(
            *(self.dataset.retrieve(view.vector) for view in views)
        )
        return (
            np.stack([view.vector for view in views]).astype(np.float32),
            np.stack(targets).astype(np.float32),
            np.asarray(validities, dtype=np.float32),
            float(reliabilities[0]),
        )

    def _semantic_views(self, environment: Phase2Warehouse) -> list[SemanticViewV3]:
        warehouse = environment.env
        width, height = warehouse.grid_size[1], warehouse.grid_size[0]
        layout_hash = warehouse.shadow_layout_hash()
        views = []
        for agent in warehouse.agents:
            _, target_kind = environment._target_for_agent(agent.id)
            focal = self._semantic_agent_state(environment, agent, target_kind)
            peers = [
                self._semantic_agent_state(
                    environment, other, environment._target_for_agent(other.id)[1]
                )
                for other in warehouse.agents
                if other.id != agent.id
            ]
            views.append(
                SemanticViewV3.from_state(layout_hash, width, height, focal, peers)
            )
        return views

    def _semantic_agent_state(self, environment, agent, target_kind: str) -> dict:
        warehouse = environment.env
        direction = agent.dir.name.lower()
        deltas = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        forward = deltas[direction]
        right_direction = {
            "up": "right", "right": "down", "down": "left", "left": "up"
        }[direction]
        right = deltas[right_direction]
        back = (-forward[0], -forward[1])
        left = (-right[0], -right[1])

        def highway(delta):
            x, y = agent.x + delta[0], agent.y + delta[1]
            return (
                0 <= x < warehouse.grid_size[1]
                and 0 <= y < warehouse.grid_size[0]
                and warehouse._is_highway(x, y)
            )

        return {
            "position": (agent.x, agent.y),
            "orientation": direction,
            "battery_ratio": float(agent.battery),
            "loaded": agent.carrying_shelf is not None,
            "dead": bool(agent.dead),
            "priority_present": (
                environment.env.task_queue.task_for_agent(agent.id) is not None
            ),
            "priority_rank": 0.0, "target_kind": target_kind,
            "on_highway": warehouse._is_highway(agent.x, agent.y),
            "at_charging_station": (agent.x, agent.y) in warehouse.charging_stations,
            "at_picking_station": (agent.x, agent.y) in warehouse.picking_stations,
            "adjacent_highway": {
                "forward": highway(forward),
                "right": highway(right),
                "backward": highway(back),
                "left": highway(left),
            },
        }

    def _teacher_batch(
        self, environment: Phase2Warehouse
    ) -> tuple[np.ndarray, np.ndarray]:
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
            mask = [False, True, True, True, False]
            blocked = shelf_coordinates if agent.carrying_shelf is not None else ()
            query = PureMotionQuery(
                layout_hash=warehouse.shadow_layout_hash(),
                width=int(warehouse.grid_size[1]),
                height=int(warehouse.grid_size[0]),
                blocked_coordinates=tuple(
                    (int(x), int(y)) for x, y in blocked
                ),
                own_pose=(int(agent.x), int(agent.y)),
                orientation=agent.dir.name,
                goal=(int(goal[0]), int(goal[1])),
                occupied_coordinates=tuple(
                    (int(other.x), int(other.y))
                    for other in warehouse.agents
                    if other.id != agent.id
                ),
                pure_motion_mask=tuple(mask), dead=bool(agent.dead),
                picking_lock=bool(agent.picking_lock_steps),
                mandatory_toggle_load=environment._requires_pickup(agent.id),
                footprint_class=(
                    "loaded" if agent.carrying_shelf is not None else "unloaded"
                ),
            )
            results.append(self.teacher.query(query))
        return (
            np.stack([result.motion_preferences[1:4] for result in results]),
            np.asarray([result.valid for result in results], dtype=bool),
        )

    def _student_logits(
        self, environment: Phase2Warehouse, observations: np.ndarray
    ) -> np.ndarray:
        views = self._semantic_views(environment)
        semantic = np.stack([view.vector for view in views]).astype(np.float32)
        with torch.inference_mode():
            result = self.actor(
                torch.as_tensor(observations, dtype=torch.float32, device=self.device),
                torch.as_tensor(semantic, dtype=torch.float32, device=self.device),
            )
        return result.action_logits.cpu().numpy()

    def _critic_value(self, observations: np.ndarray) -> torch.Tensor:
        with torch.inference_mode():
            physical = torch.as_tensor(
                observations, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return self.critic(physical)

    @staticmethod
    def _selected_log_probs(logits, actions, masks) -> np.ndarray:
        values = logits.masked_fill(
            ~torch.as_tensor(masks, dtype=torch.bool, device=logits.device), -1e9
        )
        log_probs = torch.log_softmax(values, dim=-1)
        indices = torch.as_tensor(actions, dtype=torch.long, device=logits.device)
        return log_probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1).cpu().numpy()


def train_optimization(
    config: OptimizationTrainingConfig, output_directory: str | Path
) -> dict:
    return OptimizationTrainer(config, output_directory).train()
