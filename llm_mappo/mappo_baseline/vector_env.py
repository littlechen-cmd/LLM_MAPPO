"""Spawned CPU environments for the teacher-free MAPPO baseline."""

from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback

import numpy as np

from llm_mappo.mappo_baseline.fp_state import (
    FPStateSchema,
    build_active_masks,
    build_fp_state,
)
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V1,
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
    build_structured_observations,
)
from llm_mappo.warehouse_observation import ObservationSchema
from llm_mappo.warehouse_environment import Phase2Warehouse


def new_baseline_environment(
    values, *, replenishment_mode: str | None = None
) -> Phase2Warehouse:
    """Construct the exact DirectGoal environment from its frozen profile."""

    if values["observation_schema"] not in {
        DIRECT_GOAL_OBSERVATION_V1,
        DIRECT_GOAL_OBSERVATION_V2,
    }:
        raise ValueError("MAPPO-FP-DG only accepts the DirectGoal observation.")
    return Phase2Warehouse(
        n_agents=int(values["n_agents"]),
        max_steps=int(values["max_steps"]),
        env_id=str(values["environment_id"]),
        charge_threshold=float(values["charge_threshold"]),
        charge_release_threshold=float(values["charge_release_threshold"]),
        battery_cost_scale=float(values["battery_cost_scale"]),
        deadlock_steps=int(values["deadlock_steps"]),
        waypoint_reward=float(values.get("goal_progress_reward", 0.0)),
        reward_version=str(values["reward_version"]),
        reward_blocked_forward_penalty=float(
            values.get("reward_blocked_forward_penalty", 0.1)
        ),
        reward_collision_penalty=float(values.get("reward_collision_penalty", 0.1)),
        charge_event_reward=float(values.get("charge_event_reward", 0.0)),
        reward_deadlock_penalty=float(values.get("reward_deadlock_penalty", 0.0)),
        oracle_interaction_mask=bool(values["oracle_interaction_mask"]),
        batch_interval=int(values["dynamic_ingress_interval"]),
        batch_size_range=tuple(values["batch_size_range"]),
        initial_ingress_batches=int(values.get("initial_ingress_batches", 2)),
        initial_priority_label=str(values["initial_priority_label"]),
        priority_schedule=tuple(values["priority_schedule"]),
        request_queue_size=int(values["queue_size"]),
        task_completion_target=int(values["task_target"]),
        terminate_on_task_target=bool(
            values.get("terminate_on_task_target", True)
        ),
        replenishment_mode=str(
            replenishment_mode
            or values.get("training_replenishment_mode", "dynamic_ingress")
        ),
        charging_stations=tuple(tuple(item) for item in values["charging_stations"]),
        picking_stations=tuple(tuple(item) for item in values["picking_stations"]),
        observation_schema=ObservationSchema.DIRECT_GOAL_V1,
    )


def _payload(environment, schema, actor_schema, goal_interface, planner, labeler=None, *,
             use_semantic_validity=True):
    payload = {
        "observations": build_structured_observations(
            environment,
            actor_schema,
            goal_interface=goal_interface,
            planner=planner,
        ),
        "fp_states": build_fp_state(environment, schema),
        "action_masks": environment.action_masks(),
        "active_masks": build_active_masks(environment),
    }
    if labeler is not None:
        labels = labeler.labels(use_semantic_validity=use_semantic_validity)
        payload["astar_kd"] = {
            "soft_targets": labels.soft_targets,
            "root_costs": labels.root_costs,
            "validity": labels.validity,
            "supervision_scope": labels.supervision_scope,
            "attenuation": labels.attenuation,
            "failure_reasons": labels.failure_reasons,
            "query_count": labels.query_count,
            "cache_build_count": labels.cache_build_count,
            "cache_hit_count": labels.cache_hit_count,
        }
    return payload


def _snapshot(environment, episode_index, episode_step):
    return {
        "warehouse": environment.env.export_shadow_state(),
        "adapter": environment.export_shadow_state(),
        "episode_index": int(episode_index),
        "episode_step": int(episode_step),
    }


def _write_failure_evidence(directory, worker_index, payload) -> str | None:
    """Persist a worker failure without relying on a live parent Pipe."""
    if directory is None:
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    source = str(payload.get("source", "unknown"))
    path = root / f"worker_{int(worker_index):02d}_{source}_failure.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)
    return str(path)


def _worker(  # noqa: C901
    connection, values, worker_index, base_seed, failure_directory
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    worker_threads = str(int(values.get("worker_numerical_threads", 1)))
    os.environ.setdefault("OMP_NUM_THREADS", worker_threads)
    os.environ.setdefault("MKL_NUM_THREADS", worker_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", worker_threads)
    environment = None
    try:
        import torch

        torch.set_num_threads(int(values.get("worker_numerical_threads", 1)))
        environment = new_baseline_environment(values)
        goal_interface = str(values.get("goal_interface", "direct_goal"))
        planner = None
        if goal_interface == "waypoint":
            from llm_mappo.planner import AStarPlanner

            planner = AStarPlanner()
        labeler = None
        use_semantic_validity = True
        variant = values.get("astar_kd_variant")
        if variant is not None:
            if variant not in {"full", "no_validity"}:
                raise ValueError("Unknown A*KD variant for MAPPO worker.")
            from llm_mappo.mappo_astar_kd.labels import AStarKDLabeler

            labeler = AStarKDLabeler(environment)
            use_semantic_validity = variant == "full"
        schema = FPStateSchema(
            int(values["n_agents"]),
            n_shelves=144,
            n_chargers=len(environment.env.charging_stations),
            n_picking_stations=len(environment.env.picking_stations),
        )
        actor_schema = StructuredObservationSchema(
            observation_schema=str(values["observation_schema"])
        )
        episode_index = 0
        episode_step = 0
        worker_seed = int(base_seed) + int(worker_index) * 1_000_003
        environment.reset(seed=worker_seed)
        connection.send(("ok", _payload(
            environment, schema, actor_schema, goal_interface, planner, labeler,
            use_semantic_validity=use_semantic_validity,
        )))
        while True:
            command, message = connection.recv()
            if command == "step":
                transition = environment.step(message["actions"])
                done = bool(
                    transition.terminated
                    or transition.truncated
                    or transition.metrics.deadlocked
                )
                episode = None
                if done:
                    episode = {
                        "global_environment_steps": int(message["global_step"]),
                        "worker_index": int(worker_index),
                        "episode_index": int(episode_index),
                        "episode_seed": worker_seed + episode_index,
                        **transition.metrics.as_dict(),
                    }
                    episode_index += 1
                    episode_step = 0
                    environment.reset(seed=worker_seed + episode_index)
                else:
                    episode_step += 1
                connection.send(("ok", {
                    "reward": float(transition.team_reward),
                    "done": done,
                    "episode": episode,
                    "planner_query_count": environment.planner_query_counter.count,
                    "payload": _payload(
                        environment, schema, actor_schema, goal_interface,
                        planner, labeler,
                        use_semantic_validity=use_semantic_validity,
                    ),
                }))
            elif command == "snapshot":
                connection.send(("ok", _snapshot(
                    environment, episode_index, episode_step
                )))
            elif command == "restore":
                environment.env.import_shadow_state(message["warehouse"])
                environment.import_shadow_state(message["adapter"])
                episode_index = int(message["episode_index"])
                episode_step = int(message["episode_step"])
                connection.send(("ok", _payload(
                    environment, schema, actor_schema, goal_interface, planner, labeler,
                    use_semantic_validity=use_semantic_validity,
                )))
            elif command == "close":
                connection.send(("ok", {}))
                break
            else:
                raise ValueError(f"Unknown MAPPO worker command: {command}")
    except EOFError:
        pass
    except BaseException:
        error_traceback = traceback.format_exc()
        evidence_path = _write_failure_evidence(
            failure_directory,
            worker_index,
            {
                "schema": "mappo-vector-worker-failure-v1",
                "source": "worker",
                "worker_index": int(worker_index),
                "pid": os.getpid(),
                "traceback": error_traceback,
            },
        )
        try:
            connection.send(("error", {
                "traceback": error_traceback,
                "evidence_path": evidence_path,
            }))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if environment is not None:
            environment.close()
        connection.close()


@dataclass(frozen=True)
class VectorStep:
    rewards: np.ndarray
    dones: np.ndarray
    active_masks: np.ndarray
    episodes: tuple[dict, ...]
    planner_query_count: int


class MAPPOVectorEnvironment:
    """Persistent processes; all step requests are sent before any receive."""

    def __init__(
        self,
        values,
        count: int,
        base_seed: int,
        *,
        failure_directory: str | Path | None = None,
    ) -> None:
        if count < 1:
            raise ValueError("MAPPO requires at least one environment worker.")
        self.values = dict(values)
        self.count = int(count)
        self.num_agents = int(values["n_agents"])
        self._closed = False
        self._failure_directory = (
            Path(failure_directory) if failure_directory is not None else None
        )
        self._worker_context = [{} for _ in range(self.count)]
        context = mp.get_context("spawn")
        self._connections = []
        self._processes = []
        for index in range(self.count):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(
                    child,
                    self.values,
                    index,
                    int(base_seed),
                    str(self._failure_directory)
                    if self._failure_directory is not None else None,
                ),
                name=f"mappo-fp-env-{index:02d}",
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        self._set_payloads([self._receive(index) for index in range(self.count)])

    def _parent_failure(self, index, reason: str) -> str | None:
        process = self._processes[index]
        return _write_failure_evidence(
            self._failure_directory,
            index,
            {
                "schema": "mappo-vector-worker-failure-v1",
                "source": "parent",
                "worker_index": int(index),
                "pid": process.pid,
                "exit_code": process.exitcode,
                "reason": reason,
                "last_command": self._worker_context[index],
            },
        )

    def _receive(self, index):
        try:
            status, payload = self._connections[index].recv()
        except EOFError as error:
            evidence_path = self._parent_failure(index, "pipe_eof")
            raise RuntimeError(
                f"MAPPO environment worker {index} exited with "
                f"code {self._processes[index].exitcode}; "
                f"evidence={evidence_path}."
            ) from error
        if status != "ok":
            evidence_path = self._parent_failure(index, "worker_error_response")
            details = payload if isinstance(payload, dict) else {"traceback": payload}
            raise RuntimeError(
                f"MAPPO environment worker {index} failed; "
                f"worker_evidence={details.get('evidence_path')}; "
                f"parent_evidence={evidence_path}.\n{details.get('traceback')}"
            )
        return payload

    def _set_payloads(self, payloads) -> None:
        self.observations = np.stack([item["observations"] for item in payloads])
        self.fp_states = np.stack([item["fp_states"] for item in payloads])
        self.action_masks = np.stack([item["action_masks"] for item in payloads])
        self.active_masks = np.stack([item["active_masks"] for item in payloads])
        self.astar_kd = None
        if all("astar_kd" in item for item in payloads):
            self.astar_kd = {
                "soft_targets": np.stack([
                    item["astar_kd"]["soft_targets"] for item in payloads
                ]),
                "root_costs": np.stack([
                    item["astar_kd"]["root_costs"] for item in payloads
                ]),
                "validity": np.stack([
                    item["astar_kd"]["validity"] for item in payloads
                ]),
                "supervision_scope": np.stack([
                    item["astar_kd"]["supervision_scope"] for item in payloads
                ]),
                "attenuation": np.stack([
                    item["astar_kd"]["attenuation"] for item in payloads
                ]),
                "failure_reasons": tuple(
                    item["astar_kd"]["failure_reasons"] for item in payloads
                ),
                "query_count": sum(
                    int(item["astar_kd"]["query_count"]) for item in payloads
                ),
                "cache_build_count": sum(
                    int(item["astar_kd"]["cache_build_count"]) for item in payloads
                ),
                "cache_hit_count": sum(
                    int(item["astar_kd"]["cache_hit_count"]) for item in payloads
                ),
            }

    def _refresh_active_astar_kd(self, payloads) -> None:
        """Replace teacher payloads returned by the workers stepped now."""

        has_labels = ["astar_kd" in item for item in payloads]
        if not any(has_labels):
            return
        if not all(has_labels) or self.astar_kd is None:
            raise RuntimeError("A*KD worker payloads became inconsistent.")
        active_count = len(payloads)
        previous = self.astar_kd
        current = {}
        for name in (
            "soft_targets",
            "root_costs",
            "validity",
            "supervision_scope",
            "attenuation",
        ):
            values = np.asarray(previous[name]).copy()
            values[:active_count] = np.stack([
                item["astar_kd"][name] for item in payloads
            ])
            current[name] = values
        current["failure_reasons"] = (
            tuple(item["astar_kd"]["failure_reasons"] for item in payloads)
            + tuple(previous["failure_reasons"][active_count:])
        )
        for name in ("query_count", "cache_build_count", "cache_hit_count"):
            current[name] = sum(
                int(item["astar_kd"][name]) for item in payloads
            )
        self.astar_kd = current

    def step(
        self, actions, *, global_step: int, active_workers: int | None = None
    ) -> VectorStep:
        actions = np.asarray(actions, dtype=np.int64)
        expected = (self.count, self.num_agents)
        if actions.shape != expected:
            raise ValueError(f"MAPPO actions must have shape {expected}.")
        active_count = self.count if active_workers is None else int(active_workers)
        if not 1 <= active_count <= self.count:
            raise ValueError("active_workers must be within the worker count.")
        terminal_step = int(global_step) + active_count
        for index, (connection, worker_actions) in enumerate(zip(
            self._connections[:active_count], actions[:active_count]
        )):
            self._worker_context[index] = {
                "command": "step", "global_step": terminal_step,
            }
            try:
                connection.send(("step", {
                    "actions": worker_actions,
                    "global_step": terminal_step,
                }))
            except (BrokenPipeError, EOFError, OSError) as error:
                evidence_path = self._parent_failure(index, "pipe_send_error")
                raise RuntimeError(
                    f"MAPPO environment worker {index} rejected step command; "
                    f"evidence={evidence_path}."
                ) from error
        responses = [self._receive(index) for index in range(active_count)]
        for index, item in enumerate(responses):
            self.observations[index] = item["payload"]["observations"]
            self.fp_states[index] = item["payload"]["fp_states"]
            self.action_masks[index] = item["payload"]["action_masks"]
            self.active_masks[index] = item["payload"]["active_masks"]
        self._refresh_active_astar_kd([
            item["payload"] for item in responses
        ])
        rewards = np.zeros((self.count, self.num_agents), dtype=np.float32)
        rewards[:active_count] = np.repeat(
            np.asarray(
                [item["reward"] for item in responses], dtype=np.float32
            )[:, None],
            self.num_agents,
            axis=1,
        )
        dones = np.ones((self.count, self.num_agents), dtype=bool)
        dones[:active_count] = np.repeat(
            np.asarray([item["done"] for item in responses], dtype=bool)[:, None],
            self.num_agents,
            axis=1,
        )
        active_masks = self.active_masks.copy()
        active_masks[active_count:] = 0.0
        return VectorStep(
            rewards=rewards,
            dones=dones,
            active_masks=active_masks,
            episodes=tuple(item["episode"] for item in responses if item["episode"]),
            planner_query_count=sum(
                int(item["planner_query_count"]) for item in responses
            ),
        )

    def snapshot(self) -> list[dict]:
        for index, connection in enumerate(self._connections):
            self._worker_context[index] = {
                "command": "snapshot"
            }
            connection.send(("snapshot", None))
        return [self._receive(index) for index in range(self.count)]

    def restore(self, states) -> None:
        if len(states) != self.count:
            raise ValueError("MAPPO checkpoint has a different worker count.")
        for index, (connection, state) in enumerate(zip(self._connections, states)):
            self._worker_context[index] = {
                "command": "restore"
            }
            connection.send(("restore", state))
        self._set_payloads([self._receive(index) for index in range(self.count)])

    def close(self) -> None:  # noqa: C901
        if self._closed:
            return
        self._closed = True
        for connection, process in zip(self._connections, self._processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for index, process in enumerate(self._processes):
            if process.is_alive():
                try:
                    self._receive(index)
                except RuntimeError:
                    pass
            self._connections[index].close()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
