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
    collisions: int = 0
    deadlocked: bool = False
    agent_deaths: int = 0
    picked_tasks: int = 0
    blocked_forwards: int = 0
    steps: int = 0
    reward: float = 0.0

    @property
    def completion_rate(self) -> float:
        if not self.created_tasks:
            return 0.0
        return self.completed_tasks / self.created_tasks

    @property
    def success(self) -> bool:
        return not self.deadlocked and self.completion_rate >= 0.95

    def as_dict(self) -> Dict[str, float | bool | int]:
        return {
            "completed_tasks": self.completed_tasks,
            "created_tasks": self.created_tasks,
            "task_completion_rate": self.completion_rate,
            "collisions": self.collisions,
            "deadlocked": self.deadlocked,
            "agent_deaths": self.agent_deaths,
            "picked_tasks": self.picked_tasks,
            "blocked_forwards": self.blocked_forwards,
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
    deadlock_steps: int = 120
    waypoint_reward: float = 1.0
    oracle_interaction_mask: bool = True
    render_mode: Optional[str] = None
    _env: gym.Env = field(init=False, repr=False)
    _planner: AStarPlanner = field(init=False, repr=False)
    _raw_observations: Sequence[np.ndarray] = field(init=False, repr=False)
    _metrics: EpisodeMetrics = field(init=False, repr=False)
    _last_progress_step: int = field(init=False, default=0, repr=False)
    _last_completed: int = field(init=False, default=0, repr=False)
    _last_picked: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.n_agents < 1:
            raise ValueError("Phase 2 requires at least one AGV.")
        if self.waypoint_reward < 0.0:
            raise ValueError("waypoint_reward must not be negative.")
        make_options = {
            "disable_env_checker": True,
            "n_agents": self.n_agents,
            "request_queue_size": self.n_agents,
            "max_steps": self.max_steps,
            "batch_interval": self.max_steps + 1,
            "batch_size_range": (1, 1),
            "initial_priority_label": "B",
        }
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
        self._metrics = EpisodeMetrics(created_tasks=len(info["tasks"]))
        self._last_progress_step = 0
        self._last_completed = 0
        self._last_picked = 0
        return self._observations()

    def step(self, actions: Sequence[int]) -> Phase2Step:
        if len(actions) != self.n_agents:
            raise ValueError("Expected one discrete action for every AGV.")

        before = self._waypoint_distances()
        carrying_before = {
            agent.id: agent.carrying_shelf is not None for agent in self.env.agents
        }
        raw_obs, rewards, terminated, truncated, info = self._env.step(actions)
        self._raw_observations = raw_obs
        shaped_rewards = np.asarray(rewards, dtype=np.float32)
        movement_rewards = self._movement_rewards(before)
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
        )
        observations = self._observations()
        return Phase2Step(
            observations=observations,
            team_reward=float(np.mean(shaped_rewards)),
            terminated=bool(
                terminated
                or self._metrics.completed_tasks == self._metrics.created_tasks
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

    def _target_for_agent(self, agent_id: int) -> Tuple[Tuple[int, int], str]:
        warehouse = self.env
        agent = warehouse.agents[agent_id - 1]
        if agent.dead:
            return (agent.x, agent.y), "idle"
        if agent.battery < self.charge_threshold:
            station = self._nearest(
                (agent.x, agent.y), warehouse.charging_stations
            )
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
        return self._nearest((agent.x, agent.y), warehouse.charging_stations), "idle"

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
        highest_priority = 0.5 if active else 0.0
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

    def _waypoint_distances(self) -> List[int]:
        distances = []
        for agent in self.env.agents:
            target, _ = self._target_for_agent(agent.id)
            distances.append(abs(target[0] - agent.x) + abs(target[1] - agent.y))
        return distances

    def _movement_rewards(self, before: Sequence[int]) -> np.ndarray:
        after = self._waypoint_distances()
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
        self, info: dict, reward: float, waypoint_progress: bool, picked_tasks: int
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
        self._metrics.collisions = int(info["collisions"])
        self._metrics.agent_deaths = sum(agent["dead"] for agent in info["agents"])
        self._metrics.picked_tasks += picked_tasks
        self._metrics.blocked_forwards = int(info["blocked_forwards"])
        self._metrics.steps = int(info["step"])
        self._metrics.reward += reward
        if info["step"] - self._last_progress_step >= self.deadlock_steps:
            self._metrics.deadlocked = True
