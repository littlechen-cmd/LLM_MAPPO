"""Phase 2 oracle-waypoint environment adapter and evaluation metrics."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

import rware  # noqa: F401 - importing registers the dynamic environment.
from llm_mappo.warehouse_observation import (
    ObservationSchema,
    PlannerQueryCounter,
    build_direct_goal_observation,
    build_no_goal_hint_observation,
)
from llm_mappo.reward_v2 import (
    BLOCKED_FORWARD_PENALTY,
    COLLISION_PENALTY,
    LEGACY_REWARD_V1,
    PROGRESS_REWARD,
    REWARD_V2,
    SUPPORTED_REWARD_VERSIONS,
    RewardGoalSnapshot,
    reward_v2_progress_deltas,
    reward_v2_team_reward_breakdown,
    static_grid_distance,
)
from rware.warehouse import Action


if TYPE_CHECKING:
    from llm_mappo.planner import AStarPlanner


ACTION_COUNT = len(Action)
EPISODE_TARGET_PHASES = ("idle", "task", "delivery", "charging")
EPISODE_ACTION_NAMES = tuple(action.name.lower() for action in Action)
EPISODE_ACTION_COUNT_FIELDS = tuple(
    f"{phase}_{action}_count"
    for phase in EPISODE_TARGET_PHASES
    for action in EPISODE_ACTION_NAMES
)
EPISODE_ACTION_RATE_FIELDS = tuple(
    f"{phase}_{action}_rate"
    for phase in EPISODE_TARGET_PHASES
    for action in EPISODE_ACTION_NAMES
)
EPISODE_DIAGNOSTIC_FIELDS = (
    *EPISODE_ACTION_COUNT_FIELDS,
    *EPISODE_ACTION_RATE_FIELDS,
    "direction_reversals",
    "motion_steps",
    "active_motion_steps",
    "motion_steps_per_task",
    "active_motion_steps_per_task",
)


def _new_phase_action_counts() -> Dict[str, int]:
    """Build the fixed action-count schema for one episode."""
    return {
        f"{phase}_{action}_count": 0
        for phase in EPISODE_TARGET_PHASES
        for action in EPISODE_ACTION_NAMES
    }


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
    progress_positive_events: int = 0
    progress_negative_events: int = 0
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
    reward_completion: float = 0.0
    reward_pickup: float = 0.0
    reward_progress: float = 0.0
    reward_blocked_forward: float = 0.0
    reward_collision: float = 0.0
    reward_charging: float = 0.0
    reward_deadlock: float = 0.0
    reward_energy: float = 0.0
    reward_step_cost: float = 0.0
    phase_action_counts: Dict[str, int] = field(
        default_factory=_new_phase_action_counts,
        repr=False,
    )
    direction_reversals: int = 0
    motion_steps: int = 0
    active_motion_steps: int = 0

    @property
    def completion_rate(self) -> float:
        denominator = self.task_completion_target or self.created_tasks
        if not denominator:
            return 0.0
        return min(self.completed_tasks, denominator) / denominator

    @property
    def steps_per_task(self) -> float | None:
        """Return the continuous-workload cost per completion, if defined."""

        if self.completed_tasks < 1:
            return None
        return float(self.steps) / float(self.completed_tasks)

    @property
    def motion_steps_per_task(self) -> float | None:
        """Return requested turning/forward actions per completed task."""
        if self.completed_tasks < 1:
            return None
        return float(self.motion_steps) / float(self.completed_tasks)

    @property
    def active_motion_steps_per_task(self) -> float | None:
        """Return non-idle requested turning/forward actions per completion."""
        if self.completed_tasks < 1:
            return None
        return float(self.active_motion_steps) / float(self.completed_tasks)

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

    def as_dict(self) -> Dict[str, float | bool | int | None]:
        values = {
            "completed_tasks": self.completed_tasks,
            "created_tasks": self.created_tasks,
            "task_completion_target": self.task_completion_target,
            "task_completion_rate": self.completion_rate,
            "steps_per_task": self.steps_per_task,
            "collisions": self.collisions,
            "deadlocked": self.deadlocked,
            "agent_deaths": self.agent_deaths,
            "picked_tasks": self.picked_tasks,
            "progress_positive_events": self.progress_positive_events,
            "progress_negative_events": self.progress_negative_events,
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
            "reward_completion": self.reward_completion,
            "reward_pickup": self.reward_pickup,
            "reward_progress": self.reward_progress,
            "reward_blocked_forward": self.reward_blocked_forward,
            "reward_collision": self.reward_collision,
            "reward_charging": self.reward_charging,
            "reward_deadlock": self.reward_deadlock,
            "reward_energy": self.reward_energy,
            "reward_step_cost": self.reward_step_cost,
            "direction_reversals": self.direction_reversals,
            "motion_steps": self.motion_steps,
            "active_motion_steps": self.active_motion_steps,
            "motion_steps_per_task": self.motion_steps_per_task,
            "active_motion_steps_per_task": self.active_motion_steps_per_task,
            "success": self.success,
        }
        for phase in EPISODE_TARGET_PHASES:
            phase_total = sum(
                self.phase_action_counts[f"{phase}_{action}_count"]
                for action in EPISODE_ACTION_NAMES
            )
            for action in EPISODE_ACTION_NAMES:
                count_key = f"{phase}_{action}_count"
                values[count_key] = self.phase_action_counts[count_key]
                values[f"{phase}_{action}_rate"] = (
                    self.phase_action_counts[count_key] / phase_total
                    if phase_total else 0.0
                )
        return values


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
    reward_version: str = LEGACY_REWARD_V1
    reward_blocked_forward_penalty: float = BLOCKED_FORWARD_PENALTY
    reward_collision_penalty: float = COLLISION_PENALTY
    charge_event_reward: float = 0.0
    reward_deadlock_penalty: float = 0.0
    reward_progress_coefficient: float = PROGRESS_REWARD
    reward_energy_death_aggregation: str = "team-broadcast"
    include_waypoint_features: bool = True
    oracle_interaction_mask: bool = True
    priority_schedule: Optional[Sequence[str]] = None
    batch_interval: Optional[int] = None
    batch_size_range: Optional[Tuple[int, int]] = None
    initial_ingress_batches: int = 2
    charging_stations: Optional[Sequence[Tuple[int, int]]] = None
    picking_stations: Optional[Sequence[Tuple[int, int]]] = None
    initial_priority_label: str = "B"
    request_queue_size: Optional[int] = None
    task_completion_target: Optional[int] = None
    terminate_on_task_target: bool = True
    replenishment_mode: str = "dynamic_ingress"
    delivery_contract: str = "legacy-v1"
    include_priority_features: bool = False
    render_mode: Optional[str] = None
    observation_schema: ObservationSchema = field(
        default=ObservationSchema.LEGACY_WAYPOINT_V1,
        kw_only=True,
    )
    _env: gym.Env = field(init=False, repr=False)
    _planner: Optional["AStarPlanner"] = field(
        init=False, default=None, repr=False
    )
    _planner_query_counter: PlannerQueryCounter = field(
        init=False,
        default_factory=PlannerQueryCounter,
        repr=False,
    )
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
    _previous_actions: np.ndarray = field(init=False, repr=False)
    _previous_blocked: np.ndarray = field(init=False, repr=False)
    _last_successful_moves: List[
        Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    ] = field(init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: C901
        try:
            self.observation_schema = ObservationSchema(self.observation_schema)
        except ValueError as error:
            raise ValueError("Unsupported observation_schema.") from error
        if self.n_agents < 1:
            raise ValueError("Phase 2 requires at least one AGV.")
        if self.waypoint_reward < 0.0:
            raise ValueError("waypoint_reward must not be negative.")
        if self.reward_version not in SUPPORTED_REWARD_VERSIONS:
            raise ValueError(
                "reward_version must be 'legacy-v1' or 'reward-v2'."
            )
        if (
            self.reward_blocked_forward_penalty < 0.0
            or self.reward_collision_penalty < 0.0
            or self.charge_event_reward < 0.0
            or self.reward_deadlock_penalty < 0.0
            or self.reward_progress_coefficient < 0.0
        ):
            raise ValueError("Reward-v2 event magnitudes must not be negative.")
        if self.reward_energy_death_aggregation not in {
            "team-broadcast",
            "local-mean",
        }:
            raise ValueError("Reward-v2 energy-death aggregation is invalid.")
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
        if self.initial_ingress_batches not in {1, 2}:
            raise ValueError("initial_ingress_batches must be one or two.")
        if self.request_queue_size is not None and self.request_queue_size < 1:
            raise ValueError("request_queue_size must be positive when provided.")
        if self.task_completion_target is not None and self.task_completion_target < 1:
            raise ValueError("task_completion_target must be positive when provided.")
        if not isinstance(self.terminate_on_task_target, bool):
            raise ValueError("terminate_on_task_target must be a boolean.")
        if self.replenishment_mode not in {"dynamic_ingress", "rware_delivery"}:
            raise ValueError("replenishment_mode is not supported.")
        if (
            self.observation_schema != ObservationSchema.LEGACY_WAYPOINT_V1
            and self.include_priority_features
        ):
            raise ValueError(
                "Optimization observation schemas require the frozen 613D layout."
            )
        make_options = {
            "disable_env_checker": True,
            "n_agents": self.n_agents,
            "request_queue_size": self.request_queue_size or self.n_agents,
            "max_steps": self.max_steps,
            "batch_interval": self.batch_interval or self.max_steps + 1,
            "batch_size_range": self.batch_size_range or (1, 1),
            "initial_ingress_batches": self.initial_ingress_batches,
            "initial_priority_label": self.initial_priority_label,
            "task_completion_target": self.task_completion_target,
            "terminate_on_task_target": self.terminate_on_task_target,
            "battery_cost_scale": self.battery_cost_scale,
            "replenishment_mode": self.replenishment_mode,
            "delivery_contract": self.delivery_contract,
            "assignment_battery_threshold": self.charge_threshold,
        }
        if self.priority_schedule is not None:
            make_options["priority_schedule"] = tuple(self.priority_schedule)
        if self.charging_stations is not None:
            make_options["charging_stations"] = tuple(self.charging_stations)
        if self.picking_stations is not None:
            make_options["picking_stations"] = tuple(self.picking_stations)
        if self.render_mode is not None:
            make_options["render_mode"] = self.render_mode
        self._env = gym.make(self.env_id, **make_options)
        if self.observation_schema == ObservationSchema.LEGACY_WAYPOINT_V1:
            from llm_mappo.planner import AStarPlanner

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

    @property
    def planner_query_counter(self) -> PlannerQueryCounter:
        """Return per-episode planner instrumentation for execution audits."""
        return self._planner_query_counter

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._planner_query_counter.reset()
        self._raw_observations, info = self._env.reset(seed=seed)
        target = (
            int(info["task_completion_target"])
            if self._uses_task_completion_target()
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
        self._previous_actions = np.zeros(self.n_agents, dtype=np.int64)
        self._previous_blocked = np.zeros(self.n_agents, dtype=bool)
        self._last_successful_moves = [None] * self.n_agents
        return self._observations()

    def step(self, actions: Sequence[int]) -> Phase2Step:
        if len(actions) != self.n_agents:
            raise ValueError("Expected one discrete action for every AGV.")

        charging_targets = self._charging_targets()
        before = self._waypoint_distances(charging_targets)
        reward_goals_before = self._reward_goal_snapshots(charging_targets)
        task_weights = {
            task.task_id: self.env.task_queue.priority_weight(task.label)
            for task in self.env.task_queue.active_tasks
        }
        assigned_before = {
            agent.id: self._assigned_task_reward_state(agent.id)
            for agent in self.env.agents
        }
        carrying_before = {
            agent.id: (
                agent.carrying_shelf.id if agent.carrying_shelf is not None else None
            )
            for agent in self.env.agents
        }
        positions_before = {
            agent.id: (agent.x, agent.y) for agent in self.env.agents
        }
        target_phases_before = tuple(
            self._target_for_agent(agent.id, charging_targets)[1]
            for agent in self.env.agents
        )
        batteries_before = {
            agent.id: float(agent.battery) for agent in self.env.agents
        }
        raw_obs, rewards, terminated, truncated, info = self._env.step(actions)
        self._raw_observations = raw_obs
        low_battery_penalties = self._low_battery_entry_penalties(
            batteries_before
        )
        charge_rewards = self._successful_charge_rewards(
            charging_targets, positions_before, batteries_before, actions
        )
        picked_tasks = sum(
            carrying_before[agent.id] is None and agent.carrying_shelf is not None
            for agent in self.env.agents
        )
        progress_deltas = np.zeros(self.n_agents, dtype=np.float64)
        if self.reward_version == REWARD_V2:
            charging_targets_after = self._charging_targets()
            reward_goals_after = self._reward_goal_snapshots(
                charging_targets_after
            )
            progress_deltas = reward_v2_progress_deltas(
                reward_goals_before, reward_goals_after
            )
            pickup_weights = self._successful_pickup_weights(
                carrying_before, assigned_before
            )
            waypoint_progress = bool(np.any(progress_deltas > 0.0))
            terminal_deadlock = self._will_trigger_deadlock(
                info, waypoint_progress, picked_tasks
            )
            reward_breakdown = reward_v2_team_reward_breakdown(
                raw_rewards=rewards,
                events=info["events"],
                task_weights=task_weights,
                pickup_weights=pickup_weights,
                progress_deltas=progress_deltas,
                low_battery_penalties=low_battery_penalties,
                charge_rewards=charge_rewards,
                terminal_deadlock=terminal_deadlock,
                legacy_blocked_forward_penalty=self.env.blocked_forward_penalty,
                blocked_forward_penalty=self.reward_blocked_forward_penalty,
                collision_penalty=self.reward_collision_penalty,
                deadlock_penalty=self.reward_deadlock_penalty,
                progress_reward=self.reward_progress_coefficient,
                energy_death_aggregation=self.reward_energy_death_aggregation,
            )
            team_reward = reward_breakdown.team_reward
            reward_components = reward_breakdown.as_dict()
        else:
            shaped_rewards = np.asarray(rewards, dtype=np.float32)
            movement_rewards = self._movement_rewards(before, charging_targets)
            shaped_rewards += movement_rewards
            shaped_rewards -= 0.01
            shaped_rewards += low_battery_penalties
            team_reward = float(np.mean(shaped_rewards))
            reward_components = {}
            waypoint_progress = bool(np.any(movement_rewards))
            terminal_deadlock = self._will_trigger_deadlock(
                info, waypoint_progress, picked_tasks
            )

        self._update_metrics(
            info,
            team_reward,
            reward_components,
            waypoint_progress,
            picked_tasks,
            progress_deltas,
            charging_targets,
            positions_before,
            batteries_before,
            actions,
            terminal_deadlock,
            target_phases_before,
        )
        self._previous_actions = np.asarray(
            [
                action.value if isinstance(action, Action) else int(action)
                for action in actions
            ],
            dtype=np.int64,
        )
        self._previous_blocked = np.asarray(
            [
                any(
                    event.get("type") == "blocked_forward"
                    and int(event.get("agent_id", -1)) == agent.id
                    for event in info["events"]
                )
                for agent in self.env.agents
            ],
            dtype=bool,
        )
        observations = self._observations()
        reached_task_terminal = self.terminate_on_task_target and (
            (
                self._metrics.task_completion_target > 0
                and self._metrics.completed_tasks
                >= self._metrics.task_completion_target
            )
            or (
                self._metrics.completed_tasks == self._metrics.created_tasks
                and not self._dynamic_ingress_enabled()
            )
        )
        timeout = int(info["step"]) >= self.max_steps
        return Phase2Step(
            observations=observations,
            team_reward=team_reward,
            terminated=bool((terminated and not timeout) or reached_task_terminal),
            truncated=bool(truncated or timeout),
            info=info,
            metrics=self._metrics,
        )

    def close(self) -> None:
        self._env.close()

    def render(self):
        """Render the current warehouse state in the selected Gymnasium mode."""
        return self._env.render()

    def export_shadow_state(self) -> dict:
        """Export only mutable adapter state for `o0-shadow-state-v1`."""
        return {
            "raw_observations": [
                np.asarray(item).copy() for item in self._raw_observations
            ],
            "metrics": self._metrics.as_dict(),
            "last_progress_step": self._last_progress_step,
            "last_completed": self._last_completed,
            "last_picked": self._last_picked,
            "low_battery_active": sorted(self._low_battery_active),
            "charged_pending_recovery": sorted(self._charged_pending_recovery),
            "charging_active": sorted(self._charging_active),
            "previous_actions": self._previous_actions.tolist(),
            "previous_blocked": self._previous_blocked.tolist(),
            "phase_action_counts": dict(self._metrics.phase_action_counts),
            "last_successful_moves": self._last_successful_moves,
            "planner_query_count": self._planner_query_counter.count,
        }

    def import_shadow_state(self, state: dict) -> None:
        """Restore adapter state without querying a planner or resetting the env."""
        self._raw_observations = tuple(
            np.asarray(item, dtype=np.float32).copy()
            for item in state["raw_observations"]
        )
        metrics = EpisodeMetrics()
        for name, value in state["metrics"].items():
            if name in EpisodeMetrics.__dataclass_fields__:
                setattr(metrics, name, value)
        metrics.phase_action_counts = {
            **_new_phase_action_counts(),
            **state.get("phase_action_counts", {}),
        }
        metrics.agent_steps = int(metrics.steps) * self.n_agents
        self._metrics = metrics
        self._last_progress_step = int(state["last_progress_step"])
        self._last_completed = int(state["last_completed"])
        self._last_picked = int(state["last_picked"])
        self._low_battery_active = set(state["low_battery_active"])
        self._charged_pending_recovery = set(state["charged_pending_recovery"])
        self._charging_active = set(state["charging_active"])
        self._previous_actions = np.asarray(
            state["previous_actions"], dtype=np.int64
        )
        self._previous_blocked = np.asarray(
            state["previous_blocked"], dtype=bool
        )
        self._last_successful_moves = [
            None
            if move is None
            else (
                (int(move[0][0]), int(move[0][1])),
                (int(move[1][0]), int(move[1][1])),
            )
            for move in state.get("last_successful_moves", [None] * self.n_agents)
        ]
        self._planner_query_counter.count = int(state["planner_query_count"])

    def previous_action(self, agent_id: int) -> int:
        """Return the current episode's immediately preceding action."""

        return int(self._previous_actions[int(agent_id) - 1])

    def previous_blocked(self, agent_id: int) -> bool:
        """Return whether the AGV's immediately preceding forward was blocked."""

        return bool(self._previous_blocked[int(agent_id) - 1])

    def action_masks(self) -> np.ndarray:
        """Return decentralized masks for valid Phase 2 motion interactions."""
        masks = np.ones((self.n_agents, ACTION_COUNT), dtype=bool)
        for index, agent in enumerate(self.env.agents):
            if agent.dead or agent.picking_lock_steps:
                masks[index] = False
                masks[index, Action.NOOP.value] = True
            elif self.oracle_interaction_mask:
                requires_toggle = self._requires_pickup(
                    agent.id
                ) or self._requires_dropoff(agent.id)
                masks[index, Action.TOGGLE_LOAD.value] = requires_toggle
                if requires_toggle:
                    masks[index] = False
                    masks[index, Action.TOGGLE_LOAD.value] = True
        return masks

    def _dynamic_ingress_enabled(self) -> bool:
        """Keep episodes open until truncation while future batches can arrive."""
        return (
            self.replenishment_mode == "dynamic_ingress"
            and self.batch_interval is not None
            and self.batch_interval <= self.max_steps
        )

    def _uses_task_completion_target(self) -> bool:
        return (
            self._dynamic_ingress_enabled()
            or self.replenishment_mode == "rware_delivery"
        )

    def _observations(self) -> np.ndarray:
        warehouse = self.env
        height, width = warehouse.grid_size
        rows = []
        for index, agent in enumerate(warehouse.agents):
            target, target_kind = self._target_for_agent(agent.id)
            direction_features = np.zeros(4, dtype=np.float32)
            direction_features[agent.dir.value] = 1.0
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
                self._build_observation_row(
                    warehouse,
                    agent.id,
                    target,
                    raw,
                    own,
                    direction_features,
                    nearby,
                    global_features,
                    width,
                    height,
                )
            )
        return np.stack(rows).astype(np.float32, copy=False)

    def _build_observation_row(
        self,
        warehouse,
        agent_id: int,
        target: Tuple[int, int],
        raw: np.ndarray,
        own: np.ndarray,
        direction_features: np.ndarray,
        nearby: np.ndarray,
        global_features: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        agent = warehouse.agents[agent_id - 1]
        if self.observation_schema == ObservationSchema.DIRECT_GOAL_V1:
            return build_direct_goal_observation(
                raw,
                own,
                direction_features,
                (target[0] - agent.x) / max(width - 1, 1),
                (target[1] - agent.y) / max(height - 1, 1),
                nearby,
                global_features,
            )
        if self.observation_schema == ObservationSchema.NO_GEOMETRIC_GOAL_HINT_V1:
            return build_no_goal_hint_observation(
                raw,
                own,
                direction_features,
                nearby,
                global_features,
            )
        return self._build_legacy_observation_row(
            warehouse,
            agent_id,
            target,
            raw,
            own,
            direction_features,
            nearby,
            global_features,
            width,
            height,
        )

    def _build_legacy_observation_row(
        self,
        warehouse,
        agent_id: int,
        target: Tuple[int, int],
        raw: np.ndarray,
        own: np.ndarray,
        direction_features: np.ndarray,
        nearby: np.ndarray,
        global_features: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        if self._planner is None:
            raise RuntimeError("Legacy waypoint schema requires an A* planner.")
        agent = warehouse.agents[agent_id - 1]
        waypoint = ()
        if self.include_waypoint_features:
            self._planner_query_counter.record_query()
            waypoint = self._planner.plan(warehouse, agent_id, target).waypoints
        if len(waypoint) > 1:
            next_point = waypoint[1]
            desired_direction = self._planner._direction_between(
                waypoint[0], next_point
            )
        elif self.include_waypoint_features:
            next_point = target
            desired_direction = None
        else:
            next_point = (agent.x, agent.y)
            desired_direction = None
        dx = (next_point[0] - agent.x) / max(width - 1, 1)
        dy = (next_point[1] - agent.y) / max(height - 1, 1)
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
        return np.concatenate(
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
            task = warehouse.task_queue.task_for_agent(agent_id)
            if task is None and self.delivery_contract == "balanced-random-v1":
                return warehouse._shelf_home[agent.carrying_shelf.id], "delivery"
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

    def _requires_dropoff(self, agent_id: int) -> bool:
        warehouse = self.env
        agent = warehouse.agents[agent_id - 1]
        if (
            self.delivery_contract != "balanced-random-v1"
            or agent.dead
            or agent.carrying_shelf is None
            or warehouse.task_queue.task_for_agent(agent_id) is not None
        ):
            return False
        return (agent.x, agent.y) == warehouse._shelf_home[agent.carrying_shelf.id]

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

    def _reward_goal_snapshots(
        self, charging_targets: Dict[int, Tuple[int, int]]
    ) -> List[RewardGoalSnapshot]:
        snapshots = []
        for agent in self.env.agents:
            target, target_kind = self._target_for_agent(
                agent.id, charging_targets
            )
            task = self.env.task_queue.task_for_agent(agent.id)
            if target_kind == "task" and task is not None:
                entity = ("task_id", task.task_id)
            elif target_kind == "delivery" and agent.carrying_shelf is not None:
                entity = ("shelf_id", agent.carrying_shelf.id)
            elif target_kind == "charging":
                entity = ("station", target[0], target[1])
            else:
                entity = ("agent_id", agent.id)
            snapshots.append(
                RewardGoalSnapshot(
                    identity=(target_kind, *entity, target[0], target[1]),
                    coordinate=(int(target[0]), int(target[1])),
                    distance=self._static_reward_distance(
                        agent.id, (int(target[0]), int(target[1]))
                    ),
                )
            )
        return snapshots

    def _static_reward_distance(
        self, agent_id: int, target: Tuple[int, int]
    ) -> int:
        """Measure route progress on static geometry, never on peer occupancy."""
        warehouse = self.env
        agent = warehouse.agents[agent_id - 1]
        blocked = set()
        if agent.carrying_shelf is not None:
            blocked.update((shelf.x, shelf.y) for shelf in warehouse.shelfs)
        height, width = warehouse.grid_size
        return static_grid_distance(
            (int(agent.x), int(agent.y)),
            target,
            width=int(width),
            height=int(height),
            blocked=blocked,
        )

    def _assigned_task_reward_state(self, agent_id: int) -> Optional[dict]:
        task = self.env.task_queue.task_for_agent(agent_id)
        if task is None:
            return None
        return {
            "task_id": task.task_id,
            "shelf_id": task.shelf_id,
            "weight": self.env.task_queue.priority_weight(task.label),
        }

    def _successful_pickup_weights(
        self,
        carrying_before: Dict[int, Optional[int]],
        assigned_before: Dict[int, Optional[dict]],
    ) -> Dict[int, float]:
        weights = {}
        for agent in self.env.agents:
            assigned = assigned_before[agent.id]
            carried = (
                agent.carrying_shelf.id if agent.carrying_shelf is not None else None
            )
            if (
                carrying_before[agent.id] is None
                and assigned is not None
                and carried == assigned["shelf_id"]
            ):
                weights[agent.id] = float(assigned["weight"])
        return weights

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

    def _low_battery_entry_penalties(
        self, batteries_before: Dict[int, float]
    ) -> np.ndarray:
        """Apply the low-charge cost once per threshold crossing.

        Remaining below threshold is a state to recover from, not a new failure
        on every step.  Battery depletion is still handled separately by the
        environment's terminal energy-death consequence.
        """
        penalties = np.zeros(self.n_agents, dtype=np.float32)
        for index, agent in enumerate(self.env.agents):
            if (
                batteries_before[agent.id] >= self.charge_threshold
                and agent.battery < self.charge_threshold
            ):
                penalties[index] = -0.5
        return penalties

    def _successful_charge_rewards(
        self,
        charging_targets: Dict[int, Tuple[int, int]],
        positions_before: Dict[int, Tuple[int, int]],
        batteries_before: Dict[int, float],
        actions: Sequence[int],
    ) -> np.ndarray:
        """Reward only a requested NOOP that actually increases battery."""
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        if self.charge_event_reward == 0.0:
            return rewards
        for index, agent in enumerate(self.env.agents):
            target = charging_targets.get(agent.id)
            if (
                target is not None
                and positions_before[agent.id] == target
                and (agent.x, agent.y) == target
                and Action(actions[index]) == Action.NOOP
                and agent.battery > batteries_before[agent.id]
            ):
                rewards[index] = self.charge_event_reward
        return rewards

    def _will_trigger_deadlock(
        self,
        info: dict,
        waypoint_progress: bool,
        picked_tasks: int,
    ) -> bool:
        """Return whether this transition first reaches the deadlock boundary."""
        completed = sum(task["status"] == "completed" for task in info["tasks"])
        progressed = (
            waypoint_progress
            or picked_tasks > 0
            or completed > self._last_completed
            or any(
                event["type"] in {"task_completed", "charged"}
                for event in info["events"]
            )
        )
        return bool(
            not self._metrics.deadlocked
            and not progressed
            and info["step"] - self._last_progress_step >= self.deadlock_steps
        )

    def _update_metrics(
        self,
        info: dict,
        reward: float,
        reward_components: Dict[str, float],
        waypoint_progress: bool,
        picked_tasks: int,
        progress_deltas: np.ndarray,
        charging_targets: Dict[int, Tuple[int, int]],
        positions_before: Dict[int, Tuple[int, int]],
        batteries_before: Dict[int, float],
        actions: Sequence[int],
        terminal_deadlock: bool,
        target_phases_before: Sequence[str],
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
        self._metrics.progress_positive_events += int(
            np.count_nonzero(progress_deltas > 0)
        )
        self._metrics.progress_negative_events += int(
            np.count_nonzero(progress_deltas < 0)
        )
        self._metrics.blocked_forwards = int(info["blocked_forwards"])
        self._update_episode_diagnostics(
            positions_before, actions, target_phases_before
        )
        self._update_charging_metrics(
            info, charging_targets, positions_before, batteries_before, actions
        )
        self._metrics.steps = int(info["step"])
        self._metrics.reward += reward
        for name, value in reward_components.items():
            setattr(self._metrics, name, getattr(self._metrics, name) + float(value))
        if terminal_deadlock:
            self._metrics.deadlocked = True

    def _update_episode_diagnostics(
        self,
        positions_before: Dict[int, Tuple[int, int]],
        actions: Sequence[int],
        target_phases_before: Sequence[str],
    ) -> None:
        """Record requested actions against their pre-transition rule target."""
        for index, agent in enumerate(self.env.agents):
            phase = target_phases_before[index]
            action = Action(actions[index])
            action_name = action.name.lower()
            self._metrics.phase_action_counts[
                f"{phase}_{action_name}_count"
            ] += 1
            if action in {Action.FORWARD, Action.LEFT, Action.RIGHT}:
                self._metrics.motion_steps += 1
                if phase != "idle":
                    self._metrics.active_motion_steps += 1

            origin = positions_before[agent.id]
            destination = (agent.x, agent.y)
            if origin == destination:
                continue
            previous = self._last_successful_moves[index]
            if previous == (destination, origin):
                self._metrics.direction_reversals += 1
            self._last_successful_moves[index] = (origin, destination)

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
