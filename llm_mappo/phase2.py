"""Phase 2 oracle-waypoint environment adapter and evaluation metrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

import rware  # noqa: F401 - importing registers the dynamic environment.
from llm_mappo.planner import AStarPlanner
from rware.warehouse import Action


ACTION_COUNT = len(Action)


@dataclass
class EpisodeMetrics:
    """Measurements required by the Phase 2 Go/No-Go gate."""

    completed_tasks: int = 0
    created_tasks: int = 0
    task_completion_target: int = 0
    collisions: int = 0
    deadlocked: bool = False
    agent_deaths: int = 0
    picked_tasks: int = 0
    blocked_forwards: int = 0
    low_battery_triggers: int = 0
    charging_target_steps: int = 0
    charger_arrivals: int = 0
    charged_events: int = 0
    charging_wait_steps: int = 0
    task_recoveries: int = 0
    energy_deaths: int = 0
    minimum_battery: float = 1.0
    agent_steps: int = 0
    steps: int = 0
    reward: float = 0.0

    @property
    def completion_rate(self) -> float:
        denominator = self.task_completion_target or self.created_tasks
        if not denominator:
            return 0.0
        return min(self.completed_tasks, denominator) / denominator

    @property
    def success(self) -> bool:
        if self.task_completion_target:
            return (
                not self.deadlocked
                and self.completed_tasks >= self.task_completion_target
            )
        return not self.deadlocked and self.completion_rate >= 0.95

    @property
    def charging_exposure_rate(self) -> float:
        if not self.agent_steps:
            return 0.0
        return self.charging_target_steps / self.agent_steps

    def as_dict(self) -> Dict[str, float | bool | int]:
        return {
            "completed_tasks": self.completed_tasks,
            "created_tasks": self.created_tasks,
            "task_completion_target": self.task_completion_target,
            "task_completion_rate": self.completion_rate,
            "collisions": self.collisions,
            "deadlocked": self.deadlocked,
            "agent_deaths": self.agent_deaths,
            "picked_tasks": self.picked_tasks,
            "blocked_forwards": self.blocked_forwards,
            "low_battery_triggers": self.low_battery_triggers,
            "charging_target_steps": self.charging_target_steps,
            "charging_exposure_rate": self.charging_exposure_rate,
            "charger_arrivals": self.charger_arrivals,
            "charged_events": self.charged_events,
            "charging_wait_steps": self.charging_wait_steps,
            "task_recoveries": self.task_recoveries,
            "energy_deaths": self.energy_deaths,
            "minimum_battery": self.minimum_battery,
            "steps": self.steps,
            "reward": self.reward,
            "success": self.success,
        }


@dataclass
class Phase2Step:
    """A decentralized observation transition with one centralized reward."""

    observations: np.ndarray
    team_reward: float
    terminated: bool
    truncated: bool
    info: dict
    metrics: EpisodeMetrics


@dataclass
class Phase2Warehouse:
    """Expose oracle waypoints while retaining decentralized actor inputs.

    The adapter deliberately reads warehouse state only to derive task/charging
    waypoints and compact feature vectors. The MAPPO actor receives its own
    observation; only the critic receives all agents' observations together.
    """

    n_agents: int = 3
    max_steps: int = 400
    env_id: str = "llm-mappo-medium-3ag-v1"
    charge_threshold: float = 0.2
    charge_release_threshold: float = 0.8
    battery_cost_scale: float = 1.0
    deadlock_steps: int = 120
    waypoint_reward: float = 1.0
    oracle_interaction_mask: bool = True
    priority_schedule: Optional[Sequence[str]] = None
    batch_interval: Optional[int] = None
    batch_size_range: Optional[Tuple[int, int]] = None
    initial_priority_label: str = "B"
    request_queue_size: Optional[int] = None
    task_completion_target: Optional[int] = None
    include_priority_features: bool = False
    render_mode: Optional[str] = None
    _env: gym.Env = field(init=False, repr=False)
    _planner: AStarPlanner = field(init=False, repr=False)
    _raw_observations: Sequence[np.ndarray] = field(init=False, repr=False)
    _metrics: EpisodeMetrics = field(init=False, repr=False)
    _last_progress_step: int = field(init=False, default=0, repr=False)
    _last_completed: int = field(init=False, default=0, repr=False)
    _last_picked: int = field(init=False, default=0, repr=False)
    _low_battery_active: set[int] = field(
        init=False, default_factory=set, repr=False
    )
    _charged_pending_recovery: set[int] = field(
        init=False, default_factory=set, repr=False
    )
    _charging_active: set[int] = field(init=False, default_factory=set, repr=False)

    def __post_init__(self) -> None:  # noqa: C901
        if self.n_agents < 1:
            raise ValueError("Phase 2 requires at least one AGV.")
        if self.waypoint_reward < 0.0:
            raise ValueError("waypoint_reward must not be negative.")
        if self.battery_cost_scale <= 0.0:
            raise ValueError("battery_cost_scale must be positive.")
        if not self.charge_threshold < self.charge_release_threshold <= 1.0:
            raise ValueError(
                "charge_release_threshold must be above charge_threshold and at most 1."
            )
        if self.batch_interval is not None and self.batch_interval < 2:
            raise ValueError("batch_interval must be at least two when provided.")
        if self.batch_size_range is not None:
            if len(self.batch_size_range) != 2:
                raise ValueError("batch_size_range must contain exactly two values.")
            lower, upper = self.batch_size_range
            if lower < 1 or lower > upper:
                raise ValueError("batch_size_range must be a positive ordered pair.")
        if self.request_queue_size is not None and self.request_queue_size < 1:
            raise ValueError("request_queue_size must be positive when provided.")
        if self.task_completion_target is not None and self.task_completion_target < 1:
            raise ValueError("task_completion_target must be positive when provided.")
        make_options = {
            "disable_env_checker": True,
            "n_agents": self.n_agents,
            "request_queue_size": self.request_queue_size or self.n_agents,
            "max_steps": self.max_steps,
            "batch_interval": self.batch_interval or self.max_steps + 1,
            "batch_size_range": self.batch_size_range or (1, 1),
            "initial_priority_label": self.initial_priority_label,
            "task_completion_target": self.task_completion_target,
            "battery_cost_scale": self.battery_cost_scale,
        }
        if self.priority_schedule is not None:
            make_options["priority_schedule"] = tuple(self.priority_schedule)
        if self.render_mode is not None:
            make_options["render_mode"] = self.render_mode
        self._env = gym.make(self.env_id, **make_options)
        self._planner = AStarPlanner()

    @property
    def env(self):
        """Return the unwrapped DynamicWarehouse for diagnostics and rendering."""
        return self._env.unwrapped

    @property
    def actor_observation_dim(self) -> int:
        return int(self._observations().shape[-1])

    @property
    def metrics(self) -> EpisodeMetrics:
        return self._metrics

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._raw_observations, info = self._env.reset(seed=seed)
        target = (
            int(info["task_completion_target"])
            if self._dynamic_ingress_enabled()
            else 0
        )
        self._metrics = EpisodeMetrics(
            created_tasks=len(info["tasks"]), task_completion_target=target
        )
        self._last_progress_step = 0
        self._last_completed = 0
        self._last_picked = 0
        self._low_battery_active = {
            agent.id
            for agent in self.env.agents
            if agent.battery < self.charge_threshold
        }
        self._charged_pending_recovery = set()
        self._charging_active = set(self._low_battery_active)
        return self._observations()

    def step(self, actions: Sequence[int]) -> Phase2Step:
        if len(actions) != self.n_agents:
            raise ValueError("Expected one discrete action for every AGV.")

        charging_targets = self._charging_targets()
        before = self._waypoint_distances(charging_targets)
        carrying_before = {
            agent.id: agent.carrying_shelf is not None for agent in self.env.agents
        }
        positions_before = {
            agent.id: (agent.x, agent.y) for agent in self.env.agents
        }
        batteries_before = {
            agent.id: float(agent.battery) for agent in self.env.agents
        }
        raw_obs, rewards, terminated, truncated, info = self._env.step(actions)
        self._raw_observations = raw_obs
        shaped_rewards = np.asarray(rewards, dtype=np.float32)
        movement_rewards = self._movement_rewards(before, charging_targets)
        shaped_rewards += movement_rewards
        shaped_rewards -= 0.01
        shaped_rewards += self._low_battery_penalties(actions)
        picked_tasks = sum(
            not carrying_before[agent.id] and agent.carrying_shelf is not None
            for agent in self.env.agents
        )

        self._update_metrics(
            info,
            float(np.mean(shaped_rewards)),
            bool(np.any(movement_rewards)),
            picked_tasks,
            charging_targets,
            positions_before,
            batteries_before,
            actions,
        )
        observations = self._observations()
        return Phase2Step(
            observations=observations,
            team_reward=float(np.mean(shaped_rewards)),
            terminated=bool(
                terminated
                or (
                    self._metrics.task_completion_target > 0
                    and self._metrics.completed_tasks
                    >= self._metrics.task_completion_target
                )
                or (
                    self._metrics.completed_tasks == self._metrics.created_tasks
                    and not self._dynamic_ingress_enabled()
                )
            ),
            truncated=bool(truncated),
            info=info,
            metrics=self._metrics,
        )

    def close(self) -> None:
        self._env.close()

    def render(self):
        """Render the current warehouse state in the selected Gymnasium mode."""
        return self._env.render()

    def action_masks(self) -> np.ndarray:
        """Return decentralized masks for valid Phase 2 motion interactions."""
        masks = np.ones((self.n_agents, ACTION_COUNT), dtype=bool)
        for index, agent in enumerate(self.env.agents):
            if agent.dead or agent.picking_lock_steps:
                masks[index] = False
                masks[index, Action.NOOP.value] = True
            elif self.oracle_interaction_mask:
                masks[index, Action.TOGGLE_LOAD.value] = self._requires_pickup(
                    agent.id
                )
                if self._requires_pickup(agent.id):
                    masks[index] = False
                    masks[index, Action.TOGGLE_LOAD.value] = True
        return masks

    def _dynamic_ingress_enabled(self) -> bool:
        """Keep episodes open until truncation while future batches can arrive."""
        return (
            self.batch_interval is not None
            and self.batch_interval <= self.max_steps
        )

    def _observations(self) -> np.ndarray:
        warehouse = self.env
        height, width = warehouse.grid_size
        rows = []
        for index, agent in enumerate(warehouse.agents):
            target, target_kind = self._target_for_agent(agent.id)
            waypoint = self._planner.plan(warehouse, agent.id, target).waypoints
            if len(waypoint) > 1:
                next_point = waypoint[1]
                desired_direction = self._planner._direction_between(
                    waypoint[0], next_point
                )
            else:
                next_point = target
                desired_direction = None
            dx = (next_point[0] - agent.x) / max(width - 1, 1)
            dy = (next_point[1] - agent.y) / max(height - 1, 1)
            direction_features = np.zeros(4, dtype=np.float32)
            direction_features[agent.dir.value] = 1.0
            desired_direction_features = np.zeros(4, dtype=np.float32)
            waypoint_relation_features = np.zeros(3, dtype=np.float32)
            if desired_direction is not None:
                desired_direction_features[desired_direction.value] = 1.0
                if desired_direction == agent.dir:
                    waypoint_relation_features[0] = 1.0
                elif self._planner._turn(agent.dir, right=False) == desired_direction:
                    waypoint_relation_features[1] = 1.0
                else:
                    waypoint_relation_features[2] = 1.0
            own = np.asarray(
                [
                    agent.x / max(width - 1, 1),
                    agent.y / max(height - 1, 1),
                    agent.battery,
                    float(agent.carrying_shelf is not None),
                    float(target_kind == "charging"),
                    float(target_kind == "delivery"),
                    float(self._requires_pickup(agent.id)),
                ],
                dtype=np.float32,
            )
            if self.priority_schedule is not None or self.include_priority_features:
                priority_weight, priority_rank = self._priority_features(agent.id)
                own = np.concatenate(
                    (
                        own,
                        np.asarray(
                            [priority_weight / 2.0, priority_rank], dtype=np.float32
                        ),
                    )
                )
            nearby = self._nearby_features(agent.id, width, height)
            global_features = self._global_features(agent.id)
            raw = np.asarray(self._raw_observations[index], dtype=np.float32)
            rows.append(
                np.concatenate(
                    (
                        raw,
                        own,
                        direction_features,
                        [dx, dy],
                        desired_direction_features,
                        waypoint_relation_features,
                        nearby,
                        global_features,
                    )
                )
            )
        return np.stack(rows).astype(np.float32, copy=False)

    def _target_for_agent(
        self,
        agent_id: int,
        charging_targets: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> Tuple[Tuple[int, int], str]:
        warehouse = self.env
        agent = warehouse.agents[agent_id - 1]
        if agent.dead:
            return (agent.x, agent.y), "idle"
        charging_targets = charging_targets or self._charging_targets()
        station = charging_targets.get(agent_id)
        if station is not None:
            return station, "charging"
        if agent.carrying_shelf is not None:
            station = self._nearest(
                (agent.x, agent.y), warehouse.picking_stations
            )
            return station, "delivery"
        task = warehouse.task_queue.task_for_agent(agent_id)
        if task is not None:
            shelf = warehouse.shelfs[task.shelf_id - 1]
            return (shelf.x, shelf.y), "task"
        return (agent.x, agent.y), "idle"

    def _charging_targets(self) -> Dict[int, Tuple[int, int]]:
        """Assign at-risk AGVs distinct stations by urgency and occupancy."""
        warehouse = self.env
        available_agents = {agent.id for agent in warehouse.agents if not agent.dead}
        self._charging_active.intersection_update(available_agents)
        self._charging_active.update(
            agent.id
            for agent in warehouse.agents
            if not agent.dead and agent.battery < self.charge_threshold
        )
        self._charging_active.difference_update(
            agent.id
            for agent in warehouse.agents
            if agent.battery >= self.charge_release_threshold
        )
        charging_agents = [
            agent
            for agent in warehouse.agents
            if agent.id in self._charging_active
        ]
        charging_agents.sort(
            key=lambda agent: (
                agent.battery,
                agent.id,
            )
        )
        occupied = {
            (agent.x, agent.y): agent.id
            for agent in warehouse.agents
            if (agent.x, agent.y) in warehouse.charging_stations
        }
        targets = {}
        reserved = set()
        for agent in charging_agents:
            position = agent.x, agent.y
            if position in warehouse.charging_stations:
                targets[agent.id] = position
                reserved.add(position)
        for agent in charging_agents:
            if agent.id in targets:
                continue
            available = [
                station
                for station in warehouse.charging_stations
                if station not in reserved and station not in occupied
            ]
            if not available:
                raise RuntimeError("No unoccupied charging station is available.")
            station = min(
                available,
                key=lambda point: (
                    abs(point[0] - agent.x) + abs(point[1] - agent.y),
                    point[0],
                    point[1],
                ),
            )
            targets[agent.id] = station
            reserved.add(station)
        warehouse.charging_reservations = {
            station: agent_id for agent_id, station in targets.items()
        }
        return targets

    def _requires_pickup(self, agent_id: int) -> bool:
        warehouse = self.env
        agent = warehouse.agents[agent_id - 1]
        if agent.carrying_shelf is not None or agent.dead:
            return False
        task = warehouse.task_queue.task_for_agent(agent_id)
        if task is None:
            return False
        shelf = warehouse.shelfs[task.shelf_id - 1]
        return (agent.x, agent.y) == (shelf.x, shelf.y)

    @staticmethod
    def _nearest(point, choices):
        return min(
            choices,
            key=lambda candidate: abs(candidate[0] - point[0])
            + abs(candidate[1] - point[1]),
        )

    def _nearby_features(self, agent_id: int, width: int, height: int) -> np.ndarray:
        warehouse = self.env
        agent = warehouse.agents[agent_id - 1]
        peers = [peer for peer in warehouse.agents if peer.id != agent_id]
        peers.sort(key=lambda peer: abs(peer.x - agent.x) + abs(peer.y - agent.y))
        features: List[float] = []
        for peer in peers[:3]:
            features.extend(
                [
                    (peer.x - agent.x) / max(width - 1, 1),
                    (peer.y - agent.y) / max(height - 1, 1),
                    float(peer.carrying_shelf is not None),
                    peer.battery,
                    float(peer.dead),
                ]
            )
        features.extend([0.0] * (15 - len(features)))
        return np.asarray(features, dtype=np.float32)

    def _global_features(self, agent_id: int) -> np.ndarray:
        warehouse = self.env
        agent = warehouse.agents[agent_id - 1]
        active = warehouse.task_queue.active_tasks
        highest_priority = (
            max(
                warehouse.task_queue.priority_weight(task.label) for task in active
            )
            / 2.0
            if active
            else 0.0
        )
        occupied_stations = sum(
            (agent.x, agent.y) in warehouse.charging_stations
            for agent in warehouse.agents
        )
        same_region = sum(
            abs(peer.x - agent.x)
            + abs(peer.y - agent.y)
            <= 4
            for peer in warehouse.agents
        )
        return np.asarray(
            [
                highest_priority,
                same_region / self.n_agents,
                occupied_stations / self.n_agents,
            ],
            dtype=np.float32,
        )

    def _priority_features(self, agent_id: int) -> Tuple[float, float]:
        """Encode the assigned task's dynamic priority for the Phase 3 actor."""
        task = self.env.task_queue.task_for_agent(agent_id)
        if task is None:
            return 0.0, 0.0
        weight = self.env.task_queue.priority_weight(task.label)
        rank = (ord(task.label[0]) - ord("A")) / 25.0
        return weight, rank

    def _waypoint_distances(
        self, charging_targets: Optional[Dict[int, Tuple[int, int]]] = None
    ) -> List[int]:
        distances = []
        for agent in self.env.agents:
            target, _ = self._target_for_agent(agent.id, charging_targets)
            distances.append(abs(target[0] - agent.x) + abs(target[1] - agent.y))
        return distances

    def _movement_rewards(
        self,
        before: Sequence[int],
        charging_targets: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> np.ndarray:
        after = (
            self._waypoint_distances()
            if charging_targets is None
            else self._waypoint_distances(charging_targets)
        )
        return np.asarray(
            [
                self.waypoint_reward if next_distance < prior else 0.0
                for prior, next_distance in zip(before, after)
            ],
            dtype=np.float32,
        )

    def _low_battery_penalties(self, actions: Sequence[int]) -> np.ndarray:
        penalties = np.zeros(self.n_agents, dtype=np.float32)
        for index, (agent, action) in enumerate(zip(self.env.agents, actions)):
            charging = (agent.x, agent.y) in self.env.charging_stations
            if agent.battery < self.charge_threshold and not (
                charging and Action(action) == Action.NOOP
            ):
                penalties[index] = -0.5
        return penalties

    def _update_metrics(
        self,
        info: dict,
        reward: float,
        waypoint_progress: bool,
        picked_tasks: int,
        charging_targets: Dict[int, Tuple[int, int]],
        positions_before: Dict[int, Tuple[int, int]],
        batteries_before: Dict[int, float],
        actions: Sequence[int],
    ) -> None:
        completed = sum(task["status"] == "completed" for task in info["tasks"])
        events = info["events"]
        progressed = (
            waypoint_progress
            or picked_tasks > 0
            or completed > self._last_completed
            or any(
                event["type"] in {"task_completed", "charged"} for event in events
            )
        )
        if progressed:
            self._last_progress_step = info["step"]
        self._last_completed = completed
        self._metrics.completed_tasks = completed
        self._metrics.created_tasks = len(info["tasks"])
        self._metrics.task_completion_target = (
            int(info["task_completion_target"])
            if self._dynamic_ingress_enabled()
            else 0
        )
        self._metrics.collisions = int(info["collisions"])
        self._metrics.agent_deaths = sum(agent["dead"] for agent in info["agents"])
        self._metrics.picked_tasks += picked_tasks
        self._metrics.blocked_forwards = int(info["blocked_forwards"])
        self._update_charging_metrics(
            info, charging_targets, positions_before, batteries_before, actions
        )
        self._metrics.steps = int(info["step"])
        self._metrics.reward += reward
        if info["step"] - self._last_progress_step >= self.deadlock_steps:
            self._metrics.deadlocked = True

    def _update_charging_metrics(
        self,
        info: dict,
        charging_targets: Dict[int, Tuple[int, int]],
        positions_before: Dict[int, Tuple[int, int]],
        batteries_before: Dict[int, float],
        actions: Sequence[int],
    ) -> None:
        low_before = {
            agent_id
            for agent_id, battery in batteries_before.items()
            if battery < self.charge_threshold
        }
        current_low = {
            agent.id
            for agent in self.env.agents
            if agent.battery < self.charge_threshold
        }
        self._metrics.low_battery_triggers += len(
            (low_before | current_low) - self._low_battery_active
        )
        self._low_battery_active = current_low
        self._metrics.agent_steps += self.n_agents
        self._metrics.charging_target_steps += len(charging_targets)
        self._metrics.minimum_battery = min(
            self._metrics.minimum_battery,
            *batteries_before.values(),
            *(float(agent.battery) for agent in self.env.agents),
        )

        for agent_id, target in charging_targets.items():
            agent = self.env.agents[agent_id - 1]
            if positions_before[agent_id] != target and (agent.x, agent.y) == target:
                self._metrics.charger_arrivals += 1
            if (
                positions_before[agent_id] == target
                and Action(actions[agent_id - 1]) == Action.NOOP
            ):
                self._metrics.charging_wait_steps += 1

        charged_ids = {
            int(event["agent_id"])
            for event in info["events"]
            if event["type"] == "charged"
        }
        self._metrics.charged_events += len(charged_ids)
        self._charged_pending_recovery.update(charged_ids)
        self._metrics.energy_deaths += sum(
            event["type"] == "agent_dead" for event in info["events"]
        )
        for agent_id in tuple(self._charged_pending_recovery):
            agent = self.env.agents[agent_id - 1]
            has_task = (
                agent.carrying_shelf is not None
                or self.env.task_queue.task_for_agent(agent_id) is not None
            )
            if agent.battery >= self.charge_release_threshold and has_task:
                self._metrics.task_recoveries += 1
                self._charged_pending_recovery.remove(agent_id)
