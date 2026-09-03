"""Spawned CPU-only vector environments for the E1 MAPPO learner.

The parent owns every CUDA tensor and policy call.  A worker owns exactly one
Gym environment and returns compact NumPy payloads over a pipe.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Any, Mapping

import numpy as np

from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.pure_motion_teacher import PureMotionQuery, PureMotionTeacher
from llm_mappo.semantic_v3 import SemanticViewV3
from llm_mappo.shadow_state import ShadowStateAdapter


_EPISODE_METRIC_FIELDS = (
    "completed_tasks", "created_tasks", "task_completion_target",
    "task_completion_rate", "reward", "collisions", "deadlocked",
    "agent_deaths", "picked_tasks", "blocked_forwards",
    "low_battery_triggers", "charging_target_steps",
    "charging_exposure_rate", "charger_arrivals", "charged_events",
    "charging_wait_steps", "task_recoveries", "energy_deaths",
    "minimum_battery", "steps", "success",
)


def completed_episode_record(
    metrics: Mapping[str, Any],
    *,
    worker_index: int,
    episode_index: int,
    episode_seed: int,
    terminal_global_step: int,
) -> dict[str, Any]:
    """Bind one terminal metric snapshot to its exact worker episode."""
    missing = set(_EPISODE_METRIC_FIELDS) - set(metrics)
    if missing:
        raise ValueError(f"Completed E1 episode metrics are incomplete: {sorted(missing)}")
    return {
        "real_env_steps": int(terminal_global_step),
        "worker_index": int(worker_index),
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        **{name: metrics[name] for name in _EPISODE_METRIC_FIELDS},
    }


def _new_environment(values: Mapping[str, Any], run) -> Phase2Warehouse:
    return Phase2Warehouse(
        n_agents=int(values["n_agents"]), max_steps=int(values["max_steps"]),
        env_id=str(values["environment_id"]),
        charge_threshold=float(values["charge_threshold"]),
        charge_release_threshold=float(values["charge_release_threshold"]),
        battery_cost_scale=float(values["battery_cost_scale"]),
        deadlock_steps=int(values["deadlock_steps"]),
        batch_interval=int(values["dynamic_ingress_interval"]),
        batch_size_range=tuple(values["batch_size_range"]),
        initial_priority_label=str(values.get("initial_priority_label", "A")),
        request_queue_size=int(values["queue_size"]),
        task_completion_target=int(values["task_target"]),
        reward_version=str(values.get("reward_version", "legacy-v1")),
        observation_schema=ObservationSchema(run.observation_schema),
    )


def _agent_state(environment, agent, target_kind):
    warehouse = environment.env
    direction = agent.dir.name.lower()
    deltas = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    forward = deltas[direction]
    right = deltas[{"up": "right", "right": "down", "down": "left", "left": "up"}[direction]]

    def highway(delta):
        x, y = agent.x + delta[0], agent.y + delta[1]
        return 0 <= x < warehouse.grid_size[1] and 0 <= y < warehouse.grid_size[0] and warehouse._is_highway(x, y)

    task = warehouse.task_queue.task_for_agent(agent.id)
    return {
        "position": (agent.x, agent.y), "orientation": direction,
        "battery_ratio": float(agent.battery), "loaded": agent.carrying_shelf is not None,
        "dead": bool(agent.dead), "priority_present": task is not None,
        "priority_rank": 0.0 if task is None else (ord(task.label[0]) - ord("A")) / 25.0,
        "target_kind": target_kind, "on_highway": warehouse._is_highway(agent.x, agent.y),
        "at_charging_station": (agent.x, agent.y) in warehouse.charging_stations,
        "at_picking_station": (agent.x, agent.y) in warehouse.picking_stations,
        "adjacent_highway": {"forward": highway(forward), "right": highway(right),
                              "backward": highway((-forward[0], -forward[1])), "left": highway((-right[0], -right[1]))},
    }


def _semantic_batch(environment, dataset, semantic_control):
    warehouse = environment.env
    width, height = warehouse.grid_size[1], warehouse.grid_size[0]
    views = [SemanticViewV3.from_state(
        warehouse.shadow_layout_hash(), width, height,
        _agent_state(environment, agent, environment._target_for_agent(agent.id)[1]),
        [_agent_state(environment, peer, environment._target_for_agent(peer.id)[1])
         for peer in warehouse.agents if peer.id != agent.id],
    ) for agent in warehouse.agents]
    targets, validity, ood = zip(*(dataset.retrieve(view.vector) for view in views))
    if semantic_control == "none":
        validity = [0.0] * len(views)
    if semantic_control == "no_ood_v1":
        ood = [1.0 if value else 0.0 for value in validity]
    return (np.stack([view.vector for view in views]).astype(np.float32),
            np.stack(targets).astype(np.float32), np.asarray(validity, dtype=np.float32),
            np.asarray(ood, dtype=np.float32))


def _teacher_batch(environment, teacher):
    n_agents = environment.n_agents
    if teacher is None:
        return np.zeros((n_agents, 3), dtype=np.float32), np.zeros(n_agents, dtype=bool)
    warehouse = environment.env
    shelves = tuple((x, y) for y in range(warehouse.grid_size[0]) for x in range(warehouse.grid_size[1]) if not warehouse._is_highway(x, y))
    results = []
    for agent in warehouse.agents:
        goal, _ = environment._target_for_agent(agent.id)
        results.append(teacher.query(PureMotionQuery(
            layout_hash=warehouse.shadow_layout_hash(), width=int(warehouse.grid_size[1]), height=int(warehouse.grid_size[0]),
            blocked_coordinates=shelves if agent.carrying_shelf is not None else (), own_pose=(int(agent.x), int(agent.y)),
            orientation=agent.dir.name, goal=(int(goal[0]), int(goal[1])),
            occupied_coordinates=tuple((int(peer.x), int(peer.y)) for peer in warehouse.agents if peer.id != agent.id),
            pure_motion_mask=(False, True, True, True, False), dead=bool(agent.dead),
            picking_lock=bool(agent.picking_lock_steps), mandatory_toggle_load=environment._requires_pickup(agent.id),
            footprint_class="loaded" if agent.carrying_shelf is not None else "unloaded")))
    return np.stack([item.motion_preferences[1:4] for item in results]), np.asarray([item.valid for item in results], dtype=bool)


def _payload(environment, dataset, semantic_control, teacher):
    semantic, targets, validity, ood = _semantic_batch(environment, dataset, semantic_control)
    preferences, astar_valid = _teacher_batch(environment, teacher)
    return {"observations": environment._observations(), "semantic": semantic, "semantic_targets": targets,
            "semantic_validity": validity, "semantic_ood_reliability": ood,
            "action_masks": environment.action_masks(), "astar_preferences": preferences,
            "astar_valid": astar_valid}


def _current_payload(environment, dataset, semantic_control, teacher, *, episode_index, episode_step, episode_seed):
    payload = _payload(environment, dataset, semantic_control, teacher)
    payload.update({"episode_index": int(episode_index), "episode_step": int(episode_step), "episode_seed": int(episode_seed)})
    return payload


def _worker(connection, values, run, dataset, worker_index):
    """CPU-only process. CUDA visibility is removed before importing torch."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment = None
    try:
        import torch
        torch.set_num_threads(1)
        environment = _new_environment(values, run)
        teacher = None if run.astar_kd == "disabled" else PureMotionTeacher()
        adapter = ShadowStateAdapter(environment, code_commit="e1-vector-v1")
        episode_index = 0
        episode_step = 0
        base_seed = int(run.seed) + int(worker_index) * 1_000_003
        environment.reset(seed=base_seed)
        connection.send(("ok", {"worker_index": worker_index, "payload": _current_payload(environment, dataset, run.semantic_control, teacher, episode_index=episode_index, episode_step=episode_step, episode_seed=base_seed)}))
        while True:
            command, message = connection.recv()
            if command == "step":
                actions = np.asarray(message["actions"], dtype=np.int64)
                snapshot = None
                if message["capture"]:
                    snapshot = adapter.capture(run_seed=int(run.seed), episode_index=episode_index,
                        episode_seed=base_seed + episode_index, environment_index=worker_index,
                        real_global_step=int(message["global_step"]), episode_step=episode_step).to_bytes()
                transition = environment.step(actions)
                done = bool(transition.terminated or transition.truncated or transition.metrics.deadlocked)
                latest = transition.metrics.as_dict()
                completed_episode = None
                if done:
                    completed_episode = completed_episode_record(
                        latest,
                        worker_index=worker_index,
                        episode_index=episode_index,
                        episode_seed=base_seed + episode_index,
                        terminal_global_step=int(message["global_step"]) + 1,
                    )
                    episode_index += 1
                    episode_step = 0
                    environment.reset(seed=base_seed + episode_index)
                else:
                    episode_step += 1
                connection.send(("ok", {"team_reward": float(transition.team_reward), "done": done,
                    "latest_metrics": latest, "completed_episode": completed_episode,
                    "payload": _current_payload(environment, dataset, run.semantic_control, teacher, episode_index=episode_index, episode_step=episode_step, episode_seed=base_seed + episode_index),
                    "snapshot": snapshot, "episode_index": episode_index, "episode_step": episode_step,
                    "planner_query_count": environment.planner_query_counter.count}))
            elif command == "snapshot":
                connection.send(("ok", {"snapshot": adapter.capture(**message).to_bytes(),
                    "episode_index": episode_index, "episode_step": episode_step}))
            elif command == "restore":
                adapter.restore_bytes(message["snapshot"])
                episode_index, episode_step = int(message["episode_index"]), int(message["episode_step"])
                connection.send(("ok", {"payload": _current_payload(environment, dataset, run.semantic_control, teacher, episode_index=episode_index, episode_step=episode_step, episode_seed=base_seed + episode_index)}))
            elif command == "close":
                connection.send(("ok", {})); break
            else:
                raise ValueError(f"Unknown E1 vector worker command: {command}")
    except EOFError:
        pass
    except BaseException:
        try: connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError): pass
    finally:
        if environment is not None: environment.close()
        connection.close()


class E1VectorEnvironmentPool:
    """Persistent spawned workers; calls sent to all workers before receiving."""
    def __init__(self, values, run, dataset, count: int):
        if count < 1: raise ValueError("E1 worker count must be positive.")
        self.count, self._closed = int(count), False
        context = mp.get_context("spawn")
        self._connections, self._processes = [], []
        for index in range(self.count):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, dict(values), run, dataset, index), name=f"e1-env-{index:02d}")
            process.start(); child.close(); self._connections.append(parent); self._processes.append(process)
        self.payloads = [self._receive(index)["payload"] for index in range(self.count)]

    def _receive(self, index):
        try: status, payload = self._connections[index].recv()
        except EOFError as error:
            raise RuntimeError(f"E1 environment worker {index} exited unexpectedly with code {self._processes[index].exitcode}.") from error
        if status != "ok": raise RuntimeError(f"E1 environment worker {index} failed:\n{payload}")
        return payload

    def step(self, actions, capture, global_steps):
        for index, connection in enumerate(self._connections):
            connection.send(("step", {"actions": actions[index], "capture": bool(capture[index]), "global_step": int(global_steps[index])}))
        responses = [self._receive(index) for index in range(self.count)]
        self.payloads = [item["payload"] for item in responses]
        return responses

    def snapshot(self, addresses):
        for connection, address in zip(self._connections, addresses): connection.send(("snapshot", address))
        return [self._receive(index) for index in range(self.count)]

    def restore(self, states):
        for connection, state in zip(self._connections, states): connection.send(("restore", state))
        self.payloads = [self._receive(index)["payload"] for index in range(self.count)]

    def close(self):
        if self._closed: return
        self._closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try: connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError): pass
        for index, connection in enumerate(self._connections):
            if self._processes[index].is_alive():
                try: self._receive(index)
                except RuntimeError: pass
            connection.close()
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive(): process.terminate(); process.join(timeout=5)
