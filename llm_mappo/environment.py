"""Dynamic warehouse environment used by the LLM-MAPPO implementation."""

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from llm_mappo.rules import TaskQueue
from llm_mappo.types import PriorityAdjustment, Task
from rware.warehouse import Action, Direction, RewardType, Warehouse


class DynamicWarehouse(Warehouse):
    """RWARE with dynamic tasks, hard scheduling rules, energy, and stations."""

    def __init__(
        self,
        *args,
        batch_interval: int = 20,
        batch_size_range: Tuple[int, int] = (1, 3),
        charging_stations: Optional[Sequence[Tuple[int, int]]] = None,
        charging_rate: float = 0.02,
        picking_lock_steps: int = 3,
        auto_assign: bool = True,
        **kwargs,
    ):
        if batch_interval <= 0:
            raise ValueError("batch_interval must be positive.")
        if batch_size_range[0] <= 0 or batch_size_range[0] > batch_size_range[1]:
            raise ValueError("batch_size_range must be a positive ordered pair.")
        if not 0.0 < charging_rate <= 1.0:
            raise ValueError("charging_rate must be within (0, 1].")
        if picking_lock_steps < 0:
            raise ValueError("picking_lock_steps must not be negative.")

        self.batch_interval = batch_interval
        self.batch_size_range = batch_size_range
        self.charging_rate = charging_rate
        self.picking_lock_steps = picking_lock_steps
        self.auto_assign = auto_assign
        self.task_queue = TaskQueue()
        self._batch_index = 0
        self._shelf_home = {}
        self.total_collisions = 0
        self.last_events: List[dict] = []
        super().__init__(*args, **kwargs)
        stations = charging_stations or self._default_charging_stations()
        if len(stations) != self.n_agents:
            raise ValueError("DynamicWarehouse requires one charging station per AGV.")
        if len(set(stations)) != len(stations):
            raise ValueError("Charging station coordinates must be unique.")
        for station in stations:
            x, y = station
            if not (0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]):
                raise ValueError(f"Charging station outside grid: {station}")
            if not self._is_highway(x, y):
                raise ValueError(f"Charging station must be accessible: {station}")
        self.charging_stations = tuple(stations)
        self.picking_stations = tuple(self.goals)

    def reset(self, seed=None, options=None):
        self.task_queue = TaskQueue()
        self._batch_index = 0
        self.total_collisions = 0
        self.last_events = []
        super().reset(seed=seed, options=options)
        for agent in self.agents:
            agent.battery = 1.0
            agent.dead = False
            agent.picking_lock_steps = 0
            agent.task_id = None
            agent.collision_count = 0

        self._shelf_home = {
            shelf.id: (shelf.x, shelf.y) for shelf in self.shelfs
        }
        self._create_batch([shelf.id for shelf in self.request_queue])
        self._refresh_request_queue()
        self._assign_available_tasks()
        observations = tuple(self._make_obs(agent) for agent in self.agents)
        return observations, self._get_info()

    def step(self, actions):
        if self.msg_bits:
            raise NotImplementedError(
                "DynamicWarehouse does not support message actions."
            )
        if len(actions) != self.n_agents:
            raise ValueError("Expected exactly one action per AGV.")

        requested_actions = [Action(action) for action in actions]
        locked_before = {
            agent.id: agent.picking_lock_steps > 0 for agent in self.agents
        }
        pre_state = {
            agent.id: {
                "position": (agent.x, agent.y),
                "loaded": agent.carrying_shelf is not None,
                "battery": agent.battery,
                "carrying_shelf_id": (
                    agent.carrying_shelf.id if agent.carrying_shelf else None
                ),
            }
            for agent in self.agents
        }
        effective_actions = [
            Action.NOOP if agent.dead or locked_before[agent.id] else action
            for agent, action in zip(self.agents, requested_actions)
        ]
        collision_initiators = self._collision_initiators(effective_actions)
        active_before = tuple(self.task_queue.active_tasks)
        _, rewards, terminated, truncated, _ = super().step(
            [action.value for action in effective_actions]
        )
        rewards = np.asarray(rewards, dtype=np.float64)
        self.last_events = []

        self._complete_delivered_tasks(active_before, rewards)
        self._apply_collision_penalties(collision_initiators, rewards)
        self._update_batteries(pre_state, locked_before, rewards)
        self._advance_picking_locks(locked_before)
        self._spawn_scheduled_batch()
        self._assign_available_tasks()
        self._refresh_request_queue()

        if all(agent.dead for agent in self.agents):
            terminated = True
        observations = tuple(self._make_obs(agent) for agent in self.agents)
        return observations, list(rewards), terminated, truncated, self._get_info()

    def apply_priority_adjustments(
        self, adjustments: Iterable[PriorityAdjustment]
    ) -> Sequence[Task]:
        updated = self.task_queue.apply_adjustments(adjustments)
        self._refresh_request_queue()
        self.last_events.append(
            {"type": "priority_adjusted", "labels": [task.label for task in updated]}
        )
        return updated

    def _get_info(self):
        info = super()._get_info()
        if not self.agents or not hasattr(self.agents[0], "battery"):
            return info
        info.update(
            {
                "step": self._cur_steps,
                "queue": [task.label for task in self.task_queue.active_tasks],
                "tasks": self.task_queue.as_dict(),
                "charging_stations": list(self.charging_stations),
                "picking_stations": list(self.picking_stations),
                "collisions": self.total_collisions,
                "events": list(self.last_events),
                "agents": [
                    {
                        "agent_id": agent.id,
                        "battery": round(float(agent.battery), 6),
                        "dead": agent.dead,
                        "task_id": agent.task_id,
                        "picking_lock_steps": agent.picking_lock_steps,
                        "collision_count": agent.collision_count,
                    }
                    for agent in self.agents
                ],
            }
        )
        return info

    def _create_batch(self, shelf_ids: Sequence[int]) -> Sequence[Task]:
        if not shelf_ids:
            return ()
        letter = chr(ord("A") + min(self._batch_index, 25))
        batch_id = self._batch_index + 1
        self._batch_index += 1
        tasks = self.task_queue.create_batch(
            shelf_ids, batch_id, letter, self._cur_steps
        )
        self.last_events.append(
            {
                "type": "batch_arrived",
                "batch_id": batch_id,
                "labels": [task.label for task in tasks],
            }
        )
        return tasks

    def _spawn_scheduled_batch(self):
        if self._cur_steps == 0 or self._cur_steps % self.batch_interval:
            return
        active_shelf_ids = {task.shelf_id for task in self.task_queue.active_tasks}
        carried_shelf_ids = {
            agent.carrying_shelf.id
            for agent in self.agents
            if agent.carrying_shelf is not None
        }
        candidates = [
            shelf.id
            for shelf in self.shelfs
            if shelf.id not in active_shelf_ids and shelf.id not in carried_shelf_ids
        ]
        count = int(
            self.np_random.integers(
                self.batch_size_range[0], self.batch_size_range[1] + 1
            )
        )
        count = min(count, len(candidates))
        if count:
            chosen = self.np_random.choice(candidates, size=count, replace=False)
            self._create_batch([int(shelf_id) for shelf_id in chosen])

    def _refresh_request_queue(self):
        self.request_queue = [
            self.shelfs[task.shelf_id - 1] for task in self.task_queue.active_tasks
        ]

    def _assign_available_tasks(self):
        if not self.auto_assign:
            return
        for agent in self.agents:
            if agent.dead or agent.picking_lock_steps or agent.carrying_shelf:
                continue
            task = self.task_queue.assign_next(agent.id, agent.battery)
            if task is not None:
                agent.task_id = task.task_id

    def _complete_delivered_tasks(self, active_before, rewards):
        delivered = []
        for task in active_before:
            shelf = self.shelfs[task.shelf_id - 1]
            if (shelf.x, shelf.y) not in self.goals or shelf in self.request_queue:
                continue
            agent_id = self.grid[0, shelf.y, shelf.x]
            if not agent_id:
                continue
            agent = self.agents[agent_id - 1]
            self._remove_base_delivery_reward(agent.id, rewards)
            reward = 5.0 * self.task_queue.priority_weight(task.label)
            rewards[agent.id - 1] += reward
            completed = self.task_queue.complete(task.task_id, self._cur_steps)
            agent.task_id = None
            agent.picking_lock_steps = self.picking_lock_steps
            delivered.append(completed)
            self.last_events.append(
                {
                    "type": "task_completed",
                    "task_id": completed.task_id,
                    "label": completed.label,
                    "agent_id": agent.id,
                }
            )
        return delivered

    def _remove_base_delivery_reward(self, agent_id: int, rewards: np.ndarray):
        if self.reward_type == RewardType.GLOBAL:
            rewards -= 1.0
        elif self.reward_type == RewardType.INDIVIDUAL:
            rewards[agent_id - 1] -= 1.0
        elif self.reward_type == RewardType.TWO_STAGE:
            rewards[agent_id - 1] -= 0.5

    def _collision_initiators(self, actions: Sequence[Action]) -> List[int]:
        positions = {(agent.x, agent.y) for agent in self.agents}
        initiators = []
        for agent, action in zip(self.agents, actions):
            if action == Action.FORWARD and self._forward_target(agent) in positions:
                initiators.append(agent.id)
        return initiators

    def _forward_target(self, agent) -> Tuple[int, int]:
        x, y = agent.x, agent.y
        if agent.dir == Direction.UP:
            return x, max(0, y - 1)
        if agent.dir == Direction.DOWN:
            return x, min(self.grid_size[0] - 1, y + 1)
        if agent.dir == Direction.LEFT:
            return max(0, x - 1), y
        return min(self.grid_size[1] - 1, x + 1), y

    def _apply_collision_penalties(self, initiators: Sequence[int], rewards):
        for agent_id in initiators:
            agent = self.agents[agent_id - 1]
            rewards[agent_id - 1] -= 2.0
            agent.collision_count += 1
            self.total_collisions += 1
            self.last_events.append({"type": "collision", "agent_id": agent_id})

    def _update_batteries(self, pre_state, locked_before, rewards):
        newly_dead = []
        for agent in self.agents:
            if agent.dead:
                continue
            agent.battery = self._next_battery(
                agent, pre_state[agent.id], locked_before[agent.id]
            )
            if agent.battery == 0.0:
                agent.dead = True
                agent.task_id = None
                self.task_queue.release_agent(agent.id)
                newly_dead.append(agent.id)
        if newly_dead:
            rewards -= 20.0 * len(newly_dead)
            for agent_id in newly_dead:
                self.last_events.append({"type": "agent_dead", "agent_id": agent_id})

    def _next_battery(self, agent, state, was_locked):
        if was_locked:
            return max(0.0, state["battery"] - 0.0002)
        action = agent.req_action
        if (agent.x, agent.y) in self.charging_stations and action == Action.NOOP:
            self.last_events.append({"type": "charged", "agent_id": agent.id})
            return min(1.0, state["battery"] + self.charging_rate)
        return max(0.0, state["battery"] - self._battery_cost(agent, state))

    @staticmethod
    def _battery_cost(agent, state):
        action = agent.req_action
        if action == Action.FORWARD and (agent.x, agent.y) != state["position"]:
            return 0.002 if state["loaded"] else 0.001
        if action in (Action.LEFT, Action.RIGHT):
            return 0.002 if state["loaded"] else 0.001
        current_shelf_id = agent.carrying_shelf.id if agent.carrying_shelf else None
        if (
            action == Action.TOGGLE_LOAD
            and current_shelf_id != state["carrying_shelf_id"]
        ):
            return 0.002
        return 0.0002

    def _advance_picking_locks(self, locked_before):
        released_shelf = False
        for agent in self.agents:
            if locked_before[agent.id]:
                agent.picking_lock_steps -= 1
                if agent.picking_lock_steps == 0 and agent.carrying_shelf:
                    shelf = agent.carrying_shelf
                    shelf.x, shelf.y = self._shelf_home[shelf.id]
                    agent.carrying_shelf = None
                    released_shelf = True
                    self.last_events.append(
                        {"type": "picking_complete", "agent_id": agent.id}
                    )
        if released_shelf:
            self._recalc_grid()

    def _default_charging_stations(self) -> Sequence[Tuple[int, int]]:
        width = self.grid_size[1]
        height = self.grid_size[0]
        preferred = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width // 2, 0),
            (width // 2, height - 1),
        ]
        stations = []
        for point in preferred:
            if point not in self.goals and self._is_highway(*point):
                stations.append(point)
            if len(stations) == self.n_agents:
                return tuple(stations)
        for y in range(height):
            for x in range(width):
                point = x, y
                if (
                    point not in stations
                    and point not in self.goals
                    and self._is_highway(x, y)
                ):
                    stations.append(point)
                if len(stations) == self.n_agents:
                    return tuple(stations)
        raise ValueError(
            "Warehouse layout has insufficient accessible charging stations."
        )
