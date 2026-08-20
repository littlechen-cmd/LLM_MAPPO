"""A* demonstrations and behavior cloning for the Phase 2 curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from llm_mappo.mappo import MAPPOPolicy
from llm_mappo.phase2 import ACTION_COUNT, Phase2Warehouse
from llm_mappo.planner import ReservationTable
from rware.warehouse import Action


@dataclass
class ExpertDataset:
    """Actor inputs and masked A* soft targets collected from successful runs."""

    observations: List[np.ndarray]
    action_masks: List[np.ndarray]
    preferences: List[np.ndarray]

    def __len__(self) -> int:
        return len(self.observations)

    def append(
        self,
        observations: np.ndarray,
        action_masks: np.ndarray,
        preferences: np.ndarray,
    ) -> None:
        self.observations.extend(np.asarray(observations, dtype=np.float32))
        self.action_masks.extend(np.asarray(action_masks, dtype=bool))
        self.preferences.extend(np.asarray(preferences, dtype=np.float32))


class AStarExpert:
    """Deterministic task controller using the Phase 2 A* waypoint teacher."""

    def __init__(
        self,
        terminal_hold_steps: int = 2,
        legacy_terminal_reservation: bool = False,
    ):
        if terminal_hold_steps < 0:
            raise ValueError("terminal hold steps cannot be negative.")
        self.terminal_hold_steps = terminal_hold_steps
        self.legacy_terminal_reservation = legacy_terminal_reservation
        self._last_state = {}
        self._stalled_steps = {}
        self._last_env_step = None
        self.path_livelocks = 0
        self.state_deadlocks = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.reached_goal_plans = 0
        self.partial_paths = 0
        self.terminal_conflicts = 0
        self.reservation_false_no_paths = 0
        self.explicit_waits = 0
        self.replans = 0
        self.expanded_nodes = 0
        self.planning_times_ms = []
        self._cached_signature = None
        self._cached_preferences = None

    def action_preferences(self, env: Phase2Warehouse) -> np.ndarray:
        self._update_progress(env)
        targets = tuple(
            self._target_for_agent(env, agent.id)[0] for agent in env.env.agents
        )
        signature = self._planning_signature(env, targets)
        if signature == self._cached_signature:
            self.cache_hits += 1
            return self._cached_preferences.copy()
        if self._cached_signature is not None:
            self.replans += 1
        self.cache_misses += 1
        if env.n_agents > 1:
            preferences = self._reserved_action_preferences(env, targets)
        else:
            preferences = np.zeros((env.n_agents, ACTION_COUNT), dtype=np.float32)
            for index, agent in enumerate(env.env.agents):
                if agent.dead or agent.picking_lock_steps:
                    preferences[index, Action.NOOP.value] = 1.0
                    continue
                if env._requires_pickup(agent.id):
                    preferences[index, Action.TOGGLE_LOAD.value] = 1.0
                    continue
                plan = env._planner.plan(env.env, agent.id, targets[index])
                preferences[index] = np.asarray(
                    plan.action_preferences, dtype=np.float32
                )
        self._cached_signature = signature
        self._cached_preferences = preferences.copy()
        return preferences

    def _reserved_action_preferences(
        self, env: Phase2Warehouse, targets
    ) -> np.ndarray:
        horizon = self._reservation_horizon(env)
        reservations = ReservationTable(horizon)
        preferences = np.zeros((env.n_agents, ACTION_COUNT), dtype=np.float32)
        priorities = sorted(
            range(env.n_agents), key=lambda index: self._priority_key(env, index)
        )
        for index in priorities:
            agent = env.env.agents[index]
            if agent.dead:
                preferences[index, Action.NOOP.value] = 1.0
                reservations.reserve([(agent.x, agent.y)], persistent=True)
                continue
            if agent.picking_lock_steps:
                preferences[index, Action.NOOP.value] = 1.0
                reservations.reserve(
                    [(agent.x, agent.y)],
                    terminal_hold_steps=agent.picking_lock_steps,
                )
                continue
            if env._requires_pickup(agent.id):
                preferences[index, Action.TOGGLE_LOAD.value] = 1.0
                reservations.reserve(
                    [(agent.x, agent.y)], terminal_hold_steps=1
                )
                continue
            target = targets[index]
            plan = env._planner.plan_with_reservations(
                env.env, agent.id, target, reservations
            )
            self.reached_goal_plans += int(plan.reached_goal)
            self.partial_paths += int(
                not plan.reached_goal and bool(plan.timed_positions)
            )
            self.reservation_false_no_paths += int(
                plan.reservation_false_no_path
            )
            self.expanded_nodes += plan.expanded_nodes
            self.planning_times_ms.append(plan.planning_time_ms)
            first_action = Action(plan.first_action)
            if (
                first_action == Action.NOOP
                and not plan.reached_goal
                and plan.timed_positions
            ):
                self.explicit_waits += 1
            if not plan.waypoints:
                preferences[index, Action.NOOP.value] = 1.0
                reservations.reserve(
                    [(agent.x, agent.y)], terminal_hold_steps=1
                )
                continue
            preferences[index] = np.asarray(
                plan.action_preferences, dtype=np.float32
            )
            reservations.reserve(
                plan.timed_positions,
                terminal_hold_steps=(
                    self.terminal_hold_steps if plan.reached_goal else 1
                ),
                persistent=(
                    self.legacy_terminal_reservation and plan.reached_goal
                ),
            )
        self.terminal_conflicts += reservations.terminal_conflicts
        return preferences

    @staticmethod
    def _planning_signature(env: Phase2Warehouse, targets) -> tuple:
        return tuple(
            (
                agent.id,
                agent.x,
                agent.y,
                int(agent.dir.value),
                getattr(agent.carrying_shelf, "id", None),
                bool(agent.dead),
                int(agent.picking_lock_steps),
                targets[index],
            )
            for index, agent in enumerate(env.env.agents)
        ) + (int(env.env._cur_steps),)

    def _target_for_agent(self, env: Phase2Warehouse, agent_id: int):
        if agent_id in self._yielding_agents(env):
            agent = env.env.agents[agent_id - 1]
            target = min(
                env.env.charging_stations,
                key=lambda point: abs(point[0] - agent.x) + abs(point[1] - agent.y),
            )
            return target, "parking"
        return env._target_for_agent(agent_id)

    def _yielding_agents(self, env: Phase2Warehouse) -> set[int]:
        stalled_loaded = {
            agent.id
            for agent in env.env.agents
            if agent.carrying_shelf is not None
            and self._stalled_steps.get(agent.id, 0) >= 20
        }
        if not stalled_loaded:
            return set()
        priorities = sorted(
            (agent.id for agent in env.env.agents if agent.id not in stalled_loaded),
            reverse=True,
        )
        return set(priorities[: max(1, len(priorities) - 1)])

    def _reservation_horizon(self, env: Phase2Warehouse) -> int:
        escape = 16
        for agent in env.env.agents:
            if agent.carrying_shelf is None:
                continue
            distance = min(
                abs(agent.x - goal[0]) + abs(agent.y - goal[1])
                for goal in env.env.picking_stations
            )
            escape = max(escape, distance + 8)
        return min(64, escape)

    def _update_progress(self, env: Phase2Warehouse) -> None:
        current_step = int(env.env._cur_steps)
        new_episode = (
            self._last_env_step is not None
            and current_step < self._last_env_step
        )
        if not self._last_state or new_episode:
            self._last_state = {}
            self._stalled_steps = {}
            self._cached_signature = None
            self._cached_preferences = None
        self._last_env_step = current_step
        for agent in env.env.agents:
            target, _ = self._target_for_agent(env, agent.id)
            distance = abs(agent.x - target[0]) + abs(agent.y - target[1])
            state = (
                (agent.x, agent.y),
                agent.dir,
                agent.carrying_shelf is not None,
                distance,
                agent.picking_lock_steps,
            )
            previous = self._last_state.get(agent.id)
            if previous is not None and agent.carrying_shelf is not None:
                moved = state[0] != previous[0]
                improved = distance < previous[3]
                if not moved and not improved:
                    self._stalled_steps[agent.id] = (
                        self._stalled_steps.get(agent.id, 0) + 1
                    )
                    if state[1] != previous[1]:
                        self.path_livelocks += 1
                else:
                    self._stalled_steps[agent.id] = 0
            self._last_state[agent.id] = state
            if previous == state and agent.carrying_shelf is not None:
                self.state_deadlocks += 1

    def statistics(self) -> Dict[str, float | int]:
        timings = np.asarray(self.planning_times_ms, dtype=np.float64)
        return {
            "path_livelocks": self.path_livelocks,
            "state_deadlocks": self.state_deadlocks,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "reached_goal_plans": self.reached_goal_plans,
            "partial_paths": self.partial_paths,
            "terminal_conflicts": self.terminal_conflicts,
            "reservation_false_no_paths": self.reservation_false_no_paths,
            "explicit_waits": self.explicit_waits,
            "replans": self.replans,
            "expanded_nodes": self.expanded_nodes,
            "planning_time_count": int(timings.size),
            "planning_time_ms_total": float(timings.sum()),
            "planning_time_ms_p95": (
                float(np.percentile(timings, 95)) if timings.size else 0.0
            ),
        }

    @staticmethod
    def _priority_key(env: Phase2Warehouse, index: int) -> tuple[int, int]:
        agent = env.env.agents[index]
        task = env.env.task_queue.task_for_agent(agent.id)
        if agent.carrying_shelf is not None:
            return 0, agent.id
        if task is not None:
            return 1, agent.id
        return 2, agent.id

    def act(self, env: Phase2Warehouse, action_masks: np.ndarray) -> tuple:
        preferences = self.action_preferences(env)
        masked = _mask_and_normalize(preferences, action_masks)
        actions = np.argmax(masked, axis=-1).astype(np.int64)
        coordinated = self._coordinate_actions(env, actions)
        for index, action in enumerate(coordinated):
            if action != actions[index]:
                masked[index] = 0.0
                masked[index, action] = 1.0
        return coordinated, masked

    @staticmethod
    def _coordinate_actions(env: Phase2Warehouse, actions: np.ndarray) -> np.ndarray:
        """Break immediate movement conflicts before RWARE resolves actions."""
        safe_actions = actions.copy()
        positions = {
            (agent.x, agent.y): index for index, agent in enumerate(env.env.agents)
        }
        while True:
            forward_indices = _forward_indices(safe_actions)
            targets = {
                index: env.env._forward_target(env.env.agents[index])
                for index in forward_indices
            }
            yielding = _occupied_target_yields(
                env, safe_actions, positions, targets
            )
            yielding.update(_contested_target_yields(forward_indices, targets))
            if not yielding:
                return safe_actions
            for index in yielding:
                safe_actions[index] = Action.NOOP.value


def _forward_indices(actions: np.ndarray) -> list[int]:
    return [
        index
        for index, action in enumerate(actions)
        if Action(action) == Action.FORWARD
    ]


def _occupied_target_yields(env, actions, positions, targets) -> set[int]:
    yielding = set()
    for index, target in targets.items():
        occupant = positions.get(target)
        if occupant is None:
            continue
        occupant_moving = Action(actions[occupant]) == Action.FORWARD
        agent_position = (env.env.agents[index].x, env.env.agents[index].y)
        if occupant == index or not occupant_moving:
            yielding.add(index)
        elif targets.get(occupant) == agent_position:
            yielding.update((index, occupant))
    return yielding


def _contested_target_yields(forward_indices, targets) -> set[int]:
    yielding = set()
    for position in set(targets.values()):
        contenders = [index for index in forward_indices if targets[index] == position]
        if len(contenders) > 1:
            yielding.update(contenders[1:])
    return yielding


def collect_expert_episodes(
    env: Phase2Warehouse,
    episodes: int,
    seed: int,
    max_transitions: int | None = None,
) -> tuple[ExpertDataset, Dict[str, float | int]]:
    """Run A* to validate the curriculum and optionally collect demonstrations."""
    if episodes < 1:
        raise ValueError("expert episodes must be positive.")
    expert = AStarExpert()
    dataset = ExpertDataset([], [], [])
    records = []
    livelock_count = 0
    state_deadlock_count = 0
    for episode in range(episodes):
        expert.path_livelocks = 0
        expert.state_deadlocks = 0
        observations = env.reset(seed=seed + episode)
        episode_observations = []
        episode_masks = []
        episode_preferences = []
        while True:
            masks = env.action_masks()
            actions, preferences = expert.act(env, masks)
            episode_observations.append(observations)
            episode_masks.append(masks)
            episode_preferences.append(preferences)
            transition = env.step(actions)
            observations = transition.observations
            if (
                transition.terminated
                or transition.truncated
                or transition.metrics.deadlocked
            ):
                records.append(transition.metrics.as_dict())
                if transition.metrics.success:
                    dataset.append(
                        np.concatenate(episode_observations),
                        np.concatenate(episode_masks),
                        np.concatenate(episode_preferences),
                    )
                livelock_count += expert.path_livelocks
                state_deadlock_count += expert.state_deadlocks
                break
            if max_transitions is not None and len(dataset) >= max_transitions:
                break
        if max_transitions is not None and len(dataset) >= max_transitions:
            break
    if not records:
        raise RuntimeError("A* collection ended before completing an episode.")
    completions = np.asarray(
        [record["task_completion_rate"] for record in records], dtype=np.float64
    )
    pickups = np.asarray([record["picked_tasks"] for record in records])
    deliveries = np.asarray([record["completed_tasks"] for record in records])
    collisions = np.asarray([record["collisions"] for record in records])
    blocked = np.asarray([record["blocked_forwards"] for record in records])
    return dataset, {
        "episodes": len(records),
        "transitions": len(dataset),
        "task_completion_rate": float(completions.mean()),
        "pickup_delivery_match": bool(np.array_equal(pickups, deliveries)),
        "mean_collisions": float(collisions.mean()),
        "mean_blocked_forwards": float(blocked.mean()),
        "path_livelocks": livelock_count,
        "state_deadlocks": state_deadlock_count,
        **{
            key: value
            for key, value in expert.statistics().items()
            if key not in {"path_livelocks", "state_deadlocks"}
        },
    }


def behavior_clone(
    policy: MAPPOPolicy,
    dataset: ExpertDataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Dict[str, float | int]:
    """Fit the shared Actor to masked A* distributions before PPO fine-tuning."""
    if not dataset:
        raise ValueError("Behavior cloning requires at least one demonstration.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("Behavior-cloning hyperparameters must be positive.")
    observations = torch.as_tensor(
        np.stack(dataset.observations), dtype=torch.float32, device=policy.device
    )
    masks = torch.as_tensor(
        np.stack(dataset.action_masks), dtype=torch.bool, device=policy.device
    )
    targets = torch.as_tensor(
        np.stack(dataset.preferences), dtype=torch.float32, device=policy.device
    )
    preferred_actions = torch.argmax(targets, dim=-1)
    counts = torch.bincount(preferred_actions, minlength=ACTION_COUNT).float()
    class_weights = torch.zeros_like(counts)
    present = counts > 0
    class_weights[present] = len(dataset) / (present.sum() * counts[present])
    sample_weights = class_weights[preferred_actions]
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=learning_rate)
    loss_sum = 0.0
    updates = 0
    policy.train()
    for _ in range(epochs):
        order = torch.randperm(len(dataset), device=policy.device)
        for start in range(0, len(dataset), batch_size):
            indices = order[start:start + batch_size]
            logits = policy._masked_logits(
                policy.actor(observations[indices]), masks[indices]
            )
            cross_entropy = -(
                targets[indices] * torch.log_softmax(logits, dim=-1)
            ).sum(dim=-1)
            loss = (cross_entropy * sample_weights[indices]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), 0.5)
            optimizer.step()
            loss_sum += loss.item()
            updates += 1
    return {
        "demonstrations": len(dataset),
        "epochs": epochs,
        "bc_loss": loss_sum / updates,
        "class_balanced": True,
    }


def _mask_and_normalize(preferences: np.ndarray, masks: np.ndarray) -> np.ndarray:
    masked = np.asarray(preferences, dtype=np.float32) * np.asarray(
        masks, dtype=np.float32
    )
    totals = masked.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("A* preferences must include one valid action per AGV.")
    return masked / totals
