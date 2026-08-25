"""Dynamic warehouse environment used by the LLM-MAPPO implementation."""

import copy
from hashlib import sha256
import json
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from llm_mappo.rules import TaskQueue
from llm_mappo.types import PriorityAdjustment, Task, TaskStatus
from rware.warehouse import Action, Agent, Direction, RewardType, Shelf, Warehouse


class DynamicWarehouse(Warehouse):
    """RWARE with dynamic tasks, hard scheduling rules, energy, and stations."""

    def __init__(
        self,
        *args,
        batch_interval: int = 40,
        batch_size_range: Tuple[int, int] = (1, 3),
        charging_stations: Optional[Sequence[Tuple[int, int]]] = None,
        charging_rate: float = 0.02,
        battery_cost_scale: float = 1.0,
        picking_lock_steps: int = 3,
        auto_assign: bool = True,
        initial_priority_label: str = "A",
        priority_schedule: Optional[Sequence[str]] = None,
        blocked_forward_penalty: float = 0.05,
        task_completion_target: Optional[int] = None,
        **kwargs,
    ):
        self._validate_options(
            batch_interval,
            batch_size_range,
            charging_rate,
            battery_cost_scale,
            picking_lock_steps,
            initial_priority_label,
            priority_schedule,
            blocked_forward_penalty,
            task_completion_target,
        )
        self.batch_interval = batch_interval
        self.batch_size_range = batch_size_range
        self.charging_rate = charging_rate
        self.battery_cost_scale = battery_cost_scale
        self.picking_lock_steps = picking_lock_steps
        self.auto_assign = auto_assign
        self.initial_priority_label = initial_priority_label
        self.priority_schedule = tuple(priority_schedule or ())
        self.blocked_forward_penalty = blocked_forward_penalty
        self.task_queue = TaskQueue()
        self._batch_index = 0
        self._shelf_home = {}
        self.total_collisions = 0
        self.total_blocked_forwards = 0
        self.last_events: List[dict] = []
        self.charging_reservations = {}
        self._shadow_randomness = None
        self._shadow_randomness_address = None
        super().__init__(*args, **kwargs)
        self.task_completion_target = (
            task_completion_target
            if task_completion_target is not None
            else self.n_agents * 3
        )
        if self._dynamic_ingress_enabled() and self.task_completion_target < 2:
            raise ValueError(
                "Dynamic ingress requires task_completion_target to allow "
                "the initial A and B batches."
            )
        stations = charging_stations or self._default_charging_stations()
        if len(stations) < self.n_agents:
            raise ValueError(
                "DynamicWarehouse requires at least one charging station per AGV."
            )
        if len(set(stations)) != len(stations):
            raise ValueError("Charging station coordinates must be unique.")
        for station in stations:
            x, y = station
            if not (0 <= x < self.grid_size[1] and 0 <= y < self.grid_size[0]):
                raise ValueError(f"Charging station outside grid: {station}")
            if not self._is_highway(x, y):
                raise ValueError(f"Charging station must be accessible: {station}")
            if not self._has_highway_neighbor(x, y):
                raise ValueError(
                    f"Charging station must have an accessible neighbor: {station}"
                )
        self.charging_stations = tuple(stations)
        self.picking_stations = tuple(self.goals)

    @staticmethod
    def _validate_options(  # noqa: C901
        batch_interval,
        batch_size_range,
        charging_rate,
        battery_cost_scale,
        picking_lock_steps,
        initial_priority_label,
        priority_schedule,
        blocked_forward_penalty,
        task_completion_target,
    ):
        if batch_interval <= 0:
            raise ValueError("batch_interval must be positive.")
        if batch_size_range[0] <= 0 or batch_size_range[0] > batch_size_range[1]:
            raise ValueError("batch_size_range must be a positive ordered pair.")
        if not 0.0 < charging_rate <= 1.0:
            raise ValueError("charging_rate must be within (0, 1].")
        if battery_cost_scale <= 0.0:
            raise ValueError("battery_cost_scale must be positive.")
        if picking_lock_steps < 0:
            raise ValueError("picking_lock_steps must not be negative.")
        if len(initial_priority_label) != 1 or not initial_priority_label.isupper():
            raise ValueError("initial_priority_label must be one uppercase character.")
        if priority_schedule is not None:
            if not priority_schedule:
                raise ValueError("priority_schedule must not be empty when provided.")
            if any(
                len(label) != 1 or not label.isupper() for label in priority_schedule
            ):
                raise ValueError("priority_schedule labels must be uppercase letters.")
        if blocked_forward_penalty < 0.0:
            raise ValueError("blocked_forward_penalty must not be negative.")
        if task_completion_target is not None and (
            isinstance(task_completion_target, bool) or task_completion_target < 1
        ):
            raise ValueError("task_completion_target must be a positive integer.")

    def reset(self, seed=None, options=None):
        self.task_queue = TaskQueue()
        self._batch_index = 0
        self.total_collisions = 0
        self.total_blocked_forwards = 0
        self.last_events = []
        self.charging_reservations = {}
        super().reset(seed=seed, options=options)
        for agent in self.agents:
            agent.battery = 1.0
            agent.dead = False
            agent.picking_lock_steps = 0
            agent.task_id = None
            agent.collision_count = 0
            agent.blocked_forward_count = 0

        self._shelf_home = {
            shelf.id: (shelf.x, shelf.y) for shelf in self.shelfs
        }
        shelf_ids = [shelf.id for shelf in self.request_queue]
        if self._dynamic_ingress_enabled():
            self._create_initial_batches(shelf_ids)
        elif self.priority_schedule:
            for shelf_id in shelf_ids:
                self._create_batch([shelf_id])
        else:
            self._create_batch(shelf_ids)
        self._refresh_request_queue()
        self._assign_available_tasks()
        observations = tuple(self._make_obs(agent) for agent in self.agents)
        return observations, self._get_info()

    def shadow_config_payload(self) -> dict:
        """Immutable settings covered by the branch import config hash."""
        return {
            "grid_size": self.grid_size,
            "goals": self.goals,
            "highways": self.highways.copy(),
            "n_agents": self.n_agents,
            "max_steps": self.max_steps,
            "reward_type": self.reward_type.name,
            "batch_interval": self.batch_interval,
            "batch_size_range": self.batch_size_range,
            "charging_stations": self.charging_stations,
            "charging_rate": self.charging_rate,
            "battery_cost_scale": self.battery_cost_scale,
            "picking_lock_steps": self.picking_lock_steps,
            "auto_assign": self.auto_assign,
            "task_completion_target": self.task_completion_target,
        }

    def shadow_layout_hash(self) -> str:
        payload = {
            "grid_size": self.grid_size,
            "goals": self.goals,
            "highways": self.highways.astype(np.uint8).tolist(),
            "charging_stations": self.charging_stations,
            "picking_stations": self.picking_stations,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(raw).hexdigest()

    def export_shadow_state(self) -> dict:
        """Serialize explicit mutable state without cross-branch entity references."""
        return {
            "agents": [
                {
                    "id": agent.id,
                    "x": agent.x,
                    "y": agent.y,
                    "prev_x": agent.prev_x,
                    "prev_y": agent.prev_y,
                    "direction": agent.dir.name,
                    "message": agent.message.copy(),
                    "req_action": agent.req_action.name if agent.req_action else None,
                    "carrying_shelf_id": (
                        agent.carrying_shelf.id if agent.carrying_shelf else None
                    ),
                    "canceled_action": agent.canceled_action,
                    "has_delivered": agent.has_delivered,
                    "battery": float(agent.battery),
                    "dead": bool(agent.dead),
                    "picking_lock_steps": int(agent.picking_lock_steps),
                    "task_id": agent.task_id,
                    "collision_count": int(agent.collision_count),
                    "blocked_forward_count": int(agent.blocked_forward_count),
                }
                for agent in self.agents
            ],
            "shelves": [
                {"id": shelf.id, "x": shelf.x, "y": shelf.y,
                 "prev_x": shelf.prev_x, "prev_y": shelf.prev_y}
                for shelf in self.shelfs
            ],
            "request_queue_ids": [shelf.id for shelf in self.request_queue],
            "grid_hash": sha256(self.grid.tobytes()).hexdigest(),
            "task_queue": {
                "tasks": self.task_queue.as_dict(),
                "next_task_index": self.task_queue._next_task_index,
                "next_label_number": dict(self.task_queue._next_label_number),
            },
            "batch_index": self._batch_index,
            "shelf_home": dict(self._shelf_home),
            "charging_reservations": dict(self.charging_reservations),
            "total_collisions": self.total_collisions,
            "total_blocked_forwards": self.total_blocked_forwards,
            "last_events": copy.deepcopy(self.last_events),
            "cur_steps": self._cur_steps,
            "cur_inactive_steps": self._cur_inactive_steps,
            "np_random_seed": self._np_random_seed,
            "np_random_state": copy.deepcopy(self.np_random.bit_generator.state),
            "agent_counter": Agent.counter,
            "shelf_counter": Shelf.counter,
        }

    def import_shadow_state(self, state: dict) -> None:
        """Restore a snapshot into a preconstructed environment without reset."""
        if (
            len(state["agents"]) != len(self.agents)
            or len(state["shelves"]) != len(self.shelfs)
        ):
            raise ValueError("Shadow entity cardinality mismatch.")
        shelves = {}
        for shelf, record in zip(self.shelfs, state["shelves"]):
            for name in ("id", "x", "y", "prev_x", "prev_y"):
                setattr(shelf, name, record[name])
            shelves[shelf.id] = shelf
        for agent, record in zip(self.agents, state["agents"]):
            agent.id = record["id"]
            agent.x = record["x"]
            agent.y = record["y"]
            agent.prev_x = record["prev_x"]
            agent.prev_y = record["prev_y"]
            agent.dir = Direction[record["direction"]]
            agent.message = np.asarray(record["message"], dtype=np.float64).copy()
            agent.req_action = (
                Action[record["req_action"]] if record["req_action"] else None
            )
            agent.carrying_shelf = shelves.get(record["carrying_shelf_id"])
            agent.canceled_action = record["canceled_action"]
            agent.has_delivered = record["has_delivered"]
            agent.battery = float(record["battery"])
            agent.dead = bool(record["dead"])
            agent.picking_lock_steps = int(record["picking_lock_steps"])
            agent.task_id = record["task_id"]
            agent.collision_count = int(record["collision_count"])
            agent.blocked_forward_count = int(record["blocked_forward_count"])
        self.request_queue = [shelves[item] for item in state["request_queue_ids"]]
        self.task_queue._tasks.clear()
        for record in state["task_queue"]["tasks"]:
            self.task_queue._tasks[record["task_id"]] = Task(
                task_id=record["task_id"], shelf_id=record["shelf_id"],
                batch_id=record["batch_id"], label=record["label"],
                arrival_step=record["arrival_step"],
                status=TaskStatus(record["status"]),
                assigned_agent_id=record["assigned_agent_id"],
                completed_step=record["completed_step"],
            )
        self.task_queue._next_task_index = int(state["task_queue"]["next_task_index"])
        self.task_queue._next_label_number = dict(
            state["task_queue"]["next_label_number"]
        )
        self._batch_index = int(state["batch_index"])
        self._shelf_home = {
            int(key): tuple(value) for key, value in state["shelf_home"].items()
        }
        self.charging_reservations = dict(state["charging_reservations"])
        self.total_collisions = int(state["total_collisions"])
        self.total_blocked_forwards = int(state["total_blocked_forwards"])
        self.last_events = copy.deepcopy(state["last_events"])
        self._cur_steps = int(state["cur_steps"])
        self._cur_inactive_steps = int(state["cur_inactive_steps"])
        self._np_random_seed = state["np_random_seed"]
        self.np_random.bit_generator.state = copy.deepcopy(state["np_random_state"])
        Agent.counter = int(state["agent_counter"])
        Shelf.counter = int(state["shelf_counter"])
        self.global_image = None
        self._recalc_grid()
        if sha256(self.grid.tobytes()).hexdigest() != state["grid_hash"]:
            raise ValueError("Shadow grid reconstruction hash mismatch.")

    def set_shadow_randomness(self, randomness, **address: int) -> None:
        self._shadow_randomness = randomness
        self._shadow_randomness_address = {
            name: int(value) for name, value in address.items()
        }

    def clear_shadow_randomness(self) -> None:
        self._shadow_randomness = None
        self._shadow_randomness_address = None

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
        forward_attempts = self._forward_attempts(effective_actions, pre_state)
        active_before = tuple(self.task_queue.active_tasks)
        _, rewards, terminated, truncated, _ = super().step(
            [action.value for action in effective_actions]
        )
        rewards = np.asarray(rewards, dtype=np.float64)
        self.last_events = []
        collision_initiators, blocked_forwards = self._classify_failed_forwards(
            forward_attempts, pre_state
        )

        self._complete_delivered_tasks(active_before, rewards)
        self._apply_collision_penalties(collision_initiators, rewards)
        self._apply_blocked_forward_penalties(blocked_forwards, rewards)
        self._update_batteries(pre_state, locked_before, rewards)
        self._advance_picking_locks(locked_before)
        if self._completion_target_reached():
            terminated = True
            self.last_events.append(
                {
                    "type": "task_completion_target_reached",
                    "target": self.task_completion_target,
                }
            )
        else:
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
                "task_completion_target": self.task_completion_target,
                "completed_task_count": self._completed_task_count(),
                "task_target_reached": self._completion_target_reached(),
                "charging_stations": list(self.charging_stations),
                "charging_station_status": self._charging_station_status(),
                "battery_cost_scale": self.battery_cost_scale,
                "charging_rate": self.charging_rate,
                "picking_stations": list(self.picking_stations),
                "collisions": self.total_collisions,
                "blocked_forwards": self.total_blocked_forwards,
                "events": list(self.last_events),
                "agents": [
                    {
                        "agent_id": agent.id,
                        "battery": round(float(agent.battery), 6),
                        "dead": agent.dead,
                        "task_id": agent.task_id,
                        "picking_lock_steps": agent.picking_lock_steps,
                        "collision_count": agent.collision_count,
                        "blocked_forward_count": agent.blocked_forward_count,
                    }
                    for agent in self.agents
                ],
            }
        )
        return info

    def _charging_station_status(self) -> List[dict]:
        occupants = {
            (agent.x, agent.y): agent.id
            for agent in self.agents
            if (agent.x, agent.y) in self.charging_stations
        }
        return [
            {
                "position": [x, y],
                "occupant_agent_id": occupants.get((x, y)),
                "reserved_agent_id": self.charging_reservations.get((x, y)),
            }
            for x, y in self.charging_stations
        ]

    def _has_highway_neighbor(self, x: int, y: int) -> bool:
        for next_x, next_y in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (
                0 <= next_x < self.grid_size[1]
                and 0 <= next_y < self.grid_size[0]
                and self._is_highway(next_x, next_y)
            ):
                return True
        return False

    def _create_batch(self, shelf_ids: Sequence[int]) -> Sequence[Task]:
        if not shelf_ids:
            return ()
        if self.priority_schedule:
            letter = self.priority_schedule[
                self._batch_index % len(self.priority_schedule)
            ]
        else:
            letter_code = ord(self.initial_priority_label) + self._batch_index
            if letter_code > ord("Z"):
                raise RuntimeError("Dynamic priority labels are exhausted at Z.")
            letter = chr(letter_code)
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

    def _create_initial_batches(self, preferred_shelf_ids: Sequence[int]) -> None:
        """Create the required A and B arrivals before the first action."""
        candidates = list(dict.fromkeys(preferred_shelf_ids))
        candidates.extend(
            shelf.id for shelf in self.shelfs if shelf.id not in candidates
        )
        minimum = self.batch_size_range[0]
        if len(candidates) < minimum * 2:
            raise RuntimeError(
                "Dynamic ingress requires enough shelves for two initial batches."
            )
        first_upper = min(self.batch_size_range[1], len(candidates) - minimum)
        first_count = int(self.np_random.integers(minimum, first_upper + 1))
        first = candidates[:first_count]
        second_count = self._draw_initial_batch_size(len(candidates) - first_count)
        second = candidates[first_count:first_count + second_count]
        self._create_batch(first)
        self._create_batch(second)

    def _draw_initial_batch_size(self, available: int) -> int:
        upper = min(self.batch_size_range[1], available)
        lower = min(self.batch_size_range[0], upper)
        return int(self.np_random.integers(lower, upper + 1))

    def _dynamic_ingress_enabled(self) -> bool:
        return self.max_steps is not None and self.batch_interval <= self.max_steps

    def _completed_task_count(self) -> int:
        return sum(task.status.value == "completed" for task in self.task_queue.tasks)

    def _completion_target_reached(self) -> bool:
        return self._completed_task_count() >= self.task_completion_target

    def _spawn_scheduled_batch(self):
        if (
            self._cur_steps == 0
            or self._cur_steps >= self.max_steps
            or self._cur_steps % self.batch_interval
        ):
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
        count = self._scheduled_batch_count()
        count = min(count, len(candidates))
        if count:
            chosen = self._scheduled_batch_shelves(candidates, count)
            self._create_batch(chosen)

    def _scheduled_batch_count(self) -> int:
        lower, upper = self.batch_size_range
        if self._shadow_randomness is None:
            return int(self.np_random.integers(lower, upper + 1))
        return self._shadow_randomness.integer(
            **self._shadow_randomness_address,
            event_type="dynamic_ingress_batch_size",
            event_slot=0,
            low=lower,
            high=upper + 1,
        )

    def _scheduled_batch_shelves(
        self, candidates: Sequence[int], count: int
    ) -> list[int]:
        if self._shadow_randomness is None:
            return [
                int(shelf_id)
                for shelf_id in self.np_random.choice(
                    candidates, size=count, replace=False
                )
            ]
        return self._shadow_randomness.choose_without_replacement(
            candidates,
            count,
            **self._shadow_randomness_address,
            event_type="dynamic_ingress_shelf_choice",
            event_slot=0,
        )

    def _refresh_request_queue(self):
        self.request_queue = [
            self.shelfs[task.shelf_id - 1] for task in self.task_queue.active_tasks
        ]

    def _renew_delivered_request(self, delivered_shelf) -> None:
        """Let the dynamic TaskQueue, rather than base RWARE, create new work."""
        self.request_queue.remove(delivered_shelf)

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

    def _forward_attempts(self, actions, pre_state):
        positions = {
            state["position"]: agent_id for agent_id, state in pre_state.items()
        }
        attempts = []
        for agent, action in zip(self.agents, actions):
            if action != Action.FORWARD:
                continue
            raw_target = self._unbounded_forward_target(agent)
            target = self._forward_target(agent)
            attempts.append(
                {
                    "agent_id": agent.id,
                    "target": target,
                    "at_boundary": raw_target != target,
                    "target_agent_id": positions.get(target),
                    "static_obstacle": (
                        pre_state[agent.id]["loaded"]
                        and target != pre_state[agent.id]["position"]
                        and any(
                            (shelf.x, shelf.y) == target
                            for shelf in self.shelfs
                        )
                    ),
                }
            )
        return attempts

    def _classify_failed_forwards(self, attempts, pre_state):
        collisions = []
        blocked = []
        for attempt in attempts:
            agent_id = attempt["agent_id"]
            agent = self.agents[agent_id - 1]
            if (agent.x, agent.y) != pre_state[agent_id]["position"]:
                continue
            other_agent_id = attempt["target_agent_id"]
            if other_agent_id is not None and other_agent_id != agent_id:
                collisions.append(agent_id)
                continue
            if attempt["at_boundary"]:
                reason = "boundary"
            elif attempt["static_obstacle"]:
                reason = "static_obstacle"
            else:
                reason = "movement_blocked"
            blocked.append({"agent_id": agent_id, "reason": reason})
        return collisions, blocked

    def _forward_target(self, agent) -> Tuple[int, int]:
        x, y = self._unbounded_forward_target(agent)
        return (
            min(max(0, x), self.grid_size[1] - 1),
            min(max(0, y), self.grid_size[0] - 1),
        )

    @staticmethod
    def _unbounded_forward_target(agent) -> Tuple[int, int]:
        x, y = agent.x, agent.y
        if agent.dir == Direction.UP:
            return x, y - 1
        if agent.dir == Direction.DOWN:
            return x, y + 1
        if agent.dir == Direction.LEFT:
            return x - 1, y
        return x + 1, y

    def _apply_collision_penalties(self, initiators: Sequence[int], rewards):
        for agent_id in initiators:
            agent = self.agents[agent_id - 1]
            rewards[agent_id - 1] -= 2.0
            agent.collision_count += 1
            self.total_collisions += 1
            self.last_events.append({"type": "collision", "agent_id": agent_id})

    def _apply_blocked_forward_penalties(self, blocked_forwards, rewards):
        for blocked in blocked_forwards:
            agent_id = blocked["agent_id"]
            agent = self.agents[agent_id - 1]
            rewards[agent_id - 1] -= self.blocked_forward_penalty
            agent.blocked_forward_count += 1
            self.total_blocked_forwards += 1
            self.last_events.append(
                {
                    "type": "blocked_forward",
                    "agent_id": agent_id,
                    "reason": blocked["reason"],
                }
            )

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
            return max(
                0.0, state["battery"] - 0.0002 * self.battery_cost_scale
            )
        action = agent.req_action
        if (agent.x, agent.y) in self.charging_stations and action == Action.NOOP:
            next_battery = min(1.0, state["battery"] + self.charging_rate)
            if next_battery > state["battery"]:
                self.last_events.append(
                    {
                        "type": "charged",
                        "agent_id": agent.id,
                        "battery_before": state["battery"],
                        "battery_after": next_battery,
                    }
                )
            return next_battery
        cost = self._battery_cost(agent, state) * self.battery_cost_scale
        return max(0.0, state["battery"] - cost)

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
        outer_aisle_width = getattr(self, "outer_aisle_width", 0)
        if outer_aisle_width >= 2:
            return (
                (0, 0),
                (1, 0),
                (width - 2, 0),
                (width - 1, 0),
                (0, height - 1),
                (1, height - 1),
                (width - 2, height - 1),
                (width - 1, height - 1),
            )
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
