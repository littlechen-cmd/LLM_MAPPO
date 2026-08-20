"""Local Phase 3a training: dual-head MAPPO with rule priority labels."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
import yaml

from llm_mappo.mappo import (
    DualHeadMAPPOPolicy,
    MAPPOUpdater,
    PPOHyperparameters,
    RolloutBuffer,
)
from llm_mappo.phase2 import ACTION_COUNT, Phase2Warehouse
from llm_mappo.phase2_expert import AStarExpert
from llm_mappo.phase4 import OfflineSemanticTeacher


@dataclass
class Phase3TrainingConfig:
    """Reproducible Phase 3a/3b configuration on the fixed local setting."""

    phase: str = "3a"
    seed: int = 7
    training_seed_groups: tuple[int, ...] = ()
    device: str = "cpu"
    torch_num_threads: int = 1
    cuda_allow_tf32: bool = True
    parallel_envs: int = 1
    n_agents: int = 3
    max_steps: int = 400
    env_id: str = "llm-mappo-medium-3ag-v1"
    priority_schedule: tuple[str, ...] | None = ("A", "B", "C")
    batch_interval: int | None = None
    batch_size_range: tuple[int, int] | None = None
    initial_priority_label: str = "B"
    request_queue_size: int | None = None
    task_completion_target: int | None = None
    charge_threshold: float = 0.2
    charge_release_threshold: float = 0.8
    battery_cost_scale: float = 1.0
    waypoint_reward: float = 0.01
    oracle_interaction_mask: bool = True
    deadlock_steps: int = 180
    episodes: int = 800
    environment_step_budget: int | None = None
    rollout_steps: int = 512
    checkpoint_interval: int = 200
    metrics_write_interval: int = 20
    output_dir: str = "artifacts/phase3a_dual_head"
    offline_semantic_dataset: str | None = None
    offline_semantic_neighbours: int = 3
    use_astar_kl_teacher: bool | None = None
    use_offline_llm_teacher: bool | None = None
    ppo: PPOHyperparameters = field(
        default_factory=lambda: PPOHyperparameters(
            reservation_kl_coefficient=0.0,
            engagement_coefficient=0.1,
        )
    )

    @property
    def astar_kl_enabled(self) -> bool:
        if self.use_astar_kl_teacher is not None:
            return self.use_astar_kl_teacher
        return self.phase in {"3b", "4"}

    @property
    def offline_llm_enabled(self) -> bool:
        if self.use_offline_llm_teacher is not None:
            return self.use_offline_llm_teacher
        return self.phase == "4"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase3TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            source = yaml.safe_load(stream) or {}
        environment = source.get("environment", {})
        training = source.get("training", {})
        ppo_values = dict(source.get("ppo", {}))
        schedule = environment.get("priority_schedule", cls.priority_schedule)
        batch_size_range = environment.get("batch_size_range")
        ppo_values.setdefault("reservation_kl_coefficient", 0.0)
        ppo_values.setdefault("engagement_coefficient", 0.1)
        return cls(
            phase=str(training.get("phase", cls.phase)),
            seed=training.get("seed", cls.seed),
            training_seed_groups=tuple(
                training.get("training_seed_groups", cls.training_seed_groups)
            ),
            device=training.get("device", cls.device),
            torch_num_threads=training.get("torch_num_threads", cls.torch_num_threads),
            cuda_allow_tf32=training.get(
                "cuda_allow_tf32", cls.cuda_allow_tf32
            ),
            parallel_envs=training.get("parallel_envs", cls.parallel_envs),
            n_agents=environment.get("n_agents", cls.n_agents),
            max_steps=environment.get("max_steps", cls.max_steps),
            env_id=environment.get("id", cls.env_id),
            priority_schedule=tuple(schedule) if schedule else None,
            batch_interval=environment.get("batch_interval"),
            batch_size_range=(
                tuple(batch_size_range) if batch_size_range is not None else None
            ),
            initial_priority_label=environment.get(
                "initial_priority_label", cls.initial_priority_label
            ),
            request_queue_size=environment.get("request_queue_size"),
            task_completion_target=environment.get("task_completion_target"),
            charge_threshold=environment.get("charge_threshold", cls.charge_threshold),
            charge_release_threshold=environment.get(
                "charge_release_threshold", cls.charge_release_threshold
            ),
            battery_cost_scale=environment.get(
                "battery_cost_scale", cls.battery_cost_scale
            ),
            waypoint_reward=environment.get("waypoint_reward", cls.waypoint_reward),
            oracle_interaction_mask=environment.get(
                "oracle_interaction_mask", cls.oracle_interaction_mask
            ),
            deadlock_steps=environment.get("deadlock_steps", cls.deadlock_steps),
            episodes=training.get("episodes", cls.episodes),
            environment_step_budget=training.get("environment_step_budget"),
            rollout_steps=training.get("rollout_steps", cls.rollout_steps),
            checkpoint_interval=training.get(
                "checkpoint_interval", cls.checkpoint_interval
            ),
            metrics_write_interval=training.get(
                "metrics_write_interval", cls.metrics_write_interval
            ),
            output_dir=training.get("output_dir", cls.output_dir),
            offline_semantic_dataset=training.get(
                "offline_semantic_dataset", training.get("offline_engagement_dataset")
            ),
            offline_semantic_neighbours=training.get(
                "offline_semantic_neighbours",
                training.get(
                    "offline_engagement_neighbours", cls.offline_semantic_neighbours
                ),
            ),
            use_astar_kl_teacher=training.get("use_astar_kl_teacher"),
            use_offline_llm_teacher=training.get("use_offline_llm_teacher"),
            ppo=PPOHyperparameters(**ppo_values),
        )


def _set_seed(seed: int, threads: int) -> None:
    if threads < 1:
        raise ValueError("torch_num_threads must be positive.")
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic Torch device with actionable errors."""
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Unsupported training device: {requested!r}.") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot access a CUDA device. "
                "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
            )
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is unavailable; "
                f"detected {torch.cuda.device_count()} device(s)."
            )
        return torch.device(f"cuda:{index}")
    if device.type != "cpu":
        raise ValueError("Training device must be 'auto', 'cpu', or a CUDA device.")
    return device


def _configure_accelerator(device: torch.device, allow_tf32: bool) -> dict:
    metadata = {
        "resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.type != "cuda":
        return metadata
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(torch.initial_seed())
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    properties = torch.cuda.get_device_properties(device)
    metadata.update(
        {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": int(properties.total_memory),
            "allow_tf32": bool(allow_tf32),
        }
    )
    return metadata


def _training_episode_seed(
    config: Phase3TrainingConfig, episode_index: int
) -> tuple[int, int | None, int]:
    """Return a reproducible episode seed and its round-robin provenance."""
    if episode_index < 0:
        raise ValueError("episode_index must not be negative.")
    if not config.training_seed_groups:
        return config.seed + episode_index, None, episode_index
    group_count = len(config.training_seed_groups)
    group = config.training_seed_groups[episode_index % group_count]
    offset = episode_index // group_count
    return group * 10_000 + offset, group, offset


def _validate_training_seed_groups(seed_groups: tuple[int, ...]) -> None:
    """Reject ambiguous seed rotations before any artifacts are written."""
    if not seed_groups:
        return
    if any(not isinstance(group, int) or group < 0 for group in seed_groups):
        raise ValueError("training_seed_groups must contain non-negative integers.")
    if len(set(seed_groups)) != len(seed_groups):
        raise ValueError("training_seed_groups must not contain duplicates.")


def _training_budget_available(
    config: Phase3TrainingConfig, episodes: int, steps: int
) -> bool:
    if episodes >= config.episodes:
        return False
    if config.environment_step_budget is not None:
        return steps < config.environment_step_budget
    return True


def _writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(path))
    except ImportError:
        return None


def _make_training_env(config: Phase3TrainingConfig) -> Phase2Warehouse:
    return Phase2Warehouse(
        n_agents=config.n_agents,
        max_steps=config.max_steps,
        env_id=config.env_id,
        charge_threshold=config.charge_threshold,
        charge_release_threshold=config.charge_release_threshold,
        battery_cost_scale=config.battery_cost_scale,
        waypoint_reward=config.waypoint_reward,
        oracle_interaction_mask=config.oracle_interaction_mask,
        deadlock_steps=config.deadlock_steps,
        priority_schedule=config.priority_schedule,
        batch_interval=config.batch_interval,
        batch_size_range=config.batch_size_range,
        initial_priority_label=config.initial_priority_label,
        request_queue_size=config.request_queue_size,
        task_completion_target=config.task_completion_target,
        include_priority_features=True,
    )


def _engagement_targets(env: Phase2Warehouse) -> np.ndarray:
    active_letters = sorted(
        {task.label[0] for task in env.env.task_queue.active_tasks}
    )
    values = []
    for agent in env.env.agents:
        task = env.env.task_queue.task_for_agent(agent.id)
        if agent.dead or agent.picking_lock_steps or task is None:
            values.append(0.1)
        elif env.priority_schedule is None:
            rank = active_letters.index(task.label[0])
            count = len(active_letters)
            values.append(0.1 + 0.7 * (1.0 - (rank + 1) / count))
        elif task.label.startswith("A"):
            values.append(0.8)
        elif task.label.startswith("B"):
            values.append(0.5)
        else:
            values.append(0.3)
    return np.asarray(values, dtype=np.float32)


def _environment_worker(connection, config: Phase3TrainingConfig) -> None:
    """Own one environment and A* teacher for the lifetime of a worker."""
    env = None
    expert = None
    try:
        env = _make_training_env(config)
        if config.astar_kl_enabled:
            expert = AStarExpert()
        env.reset(seed=config.seed)
        connection.send(
            (
                "ok",
                {
                    "actor_observation_dim": env.actor_observation_dim,
                    "n_agents": env.n_agents,
                },
            )
        )
        while True:
            command, payload = connection.recv()
            response, should_close = _handle_environment_command(
                env, expert, command, payload
            )
            connection.send(("ok", response))
            if should_close:
                break
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


def _handle_environment_command(env, expert, command: str, payload):
    if command == "reset":
        return env.reset(seed=payload), False
    if command == "snapshot":
        masks = env.action_masks()
        preferences = expert.act(env, masks)[1] if expert else None
        return (masks, preferences, _engagement_targets(env)), False
    if command == "step":
        return env.step(payload), False
    if command == "stats":
        return _expert_statistics(expert), False
    if command == "close":
        return _expert_statistics(expert), True
    raise ValueError(f"Unknown environment-worker command: {command}")


def _expert_statistics(expert: AStarExpert | None) -> dict:
    if expert is None:
        return {}
    return expert.statistics()


class _EnvironmentPool:
    """Synchronous vector environments backed by persistent spawned processes."""

    def __init__(self, config: Phase3TrainingConfig, count: int):
        self._context = mp.get_context("spawn")
        self._connections = []
        self._processes = []
        self._closed = False
        for index in range(count):
            parent, child = self._context.Pipe()
            process = self._context.Process(
                target=_environment_worker,
                args=(child, config),
                name=f"phase3-env-{index:02d}",
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)
        ready = [self._receive(index) for index in range(count)]
        dimensions = {item["actor_observation_dim"] for item in ready}
        agent_counts = {item["n_agents"] for item in ready}
        if len(dimensions) != 1 or agent_counts != {config.n_agents}:
            self.close()
            raise RuntimeError("Environment workers returned incompatible spaces.")
        self.actor_observation_dim = dimensions.pop()

    def _receive(self, index: int):
        try:
            status, payload = self._connections[index].recv()
        except EOFError as exc:
            process = self._processes[index]
            raise RuntimeError(
                f"Environment worker {index} exited unexpectedly "
                f"with code {process.exitcode}."
            ) from exc
        if status != "ok":
            raise RuntimeError(f"Environment worker {index} failed:\n{payload}")
        return payload

    def _broadcast(self, indices, command: str, payloads=None):
        selected = list(indices)
        values = [None] * len(selected) if payloads is None else list(payloads)
        for index, payload in zip(selected, values):
            self._connections[index].send((command, payload))
        return [self._receive(index) for index in selected]

    def reset(self, index: int, seed: int):
        self._connections[index].send(("reset", seed))
        return self._receive(index)

    def snapshots(self, indices):
        return self._broadcast(indices, "snapshot")

    def step(self, indices, actions):
        return self._broadcast(indices, "step", actions)

    def close(self) -> list[dict]:  # noqa: C901
        if self._closed:
            return []
        self._closed = True
        live = []
        for index, process in enumerate(self._processes):
            if process.is_alive():
                try:
                    self._connections[index].send(("close", None))
                    live.append(index)
                except (BrokenPipeError, EOFError, OSError):
                    pass
        statistics = []
        for index in live:
            try:
                statistics.append(self._receive(index))
            except (BrokenPipeError, EOFError, RuntimeError):
                statistics.append({})
        for connection in self._connections:
            connection.close()
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        return statistics


def _engagement_label_definition(
    config: Phase3TrainingConfig,
) -> Dict[str, float | str]:
    """Describe the supervision semantics persisted beside a checkpoint."""
    if config.priority_schedule is None:
        return {
            "active_letter_rank": "0.1 + 0.7 * (1 - (rank + 1) / n)",
            "idle_or_inactive": 0.1,
        }
    return {
        "A": 0.8,
        "B": 0.5,
        "C": 0.3,
        "idle_or_inactive": 0.1,
    }


def _save_checkpoint(path: Path, policy, config, episodes: int, steps: int) -> None:
    torch.save(
        {
            "model_state": policy.state_dict(),
            "config": asdict(config),
            "actor_observation_dim": policy.actor.motion_encoder[0].in_features,
            "semantic_dim": policy.actor.semantic_dim,
            "semantic_features_enabled": policy.actor.semantic_features_enabled,
            "episodes": episodes,
            "steps": steps,
            "phase": config.phase,
        },
        path,
    )


def _write_csv(path: Path, rows: List[dict]) -> None:
    """Atomically write heterogeneous episode records for live plot readers."""
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(6):
        try:
            temporary_path.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * 2**attempt)


def _append_csv(path: Path, row: dict) -> None:
    """Append one fixed-schema metric row without rewriting prior updates."""
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(row)
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as stream:
            fieldnames = next(csv.reader(stream))
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _completion_durations(tasks: List[dict]) -> Dict[str, List[float]]:
    """Group completed-task latencies by priority label."""
    durations: Dict[str, List[float]] = {}
    for task in tasks:
        if task["status"] != "completed":
            continue
        label = task["label"][0]
        durations.setdefault(label, []).append(
            float(task["completed_step"] - task["arrival_step"])
        )
    return durations


def _mean_durations(durations: Dict[str, List[float]]) -> Dict[str, float]:
    return {
        label: float(np.mean(values))
        for label, values in sorted(durations.items())
        if values
    }


def _reservation_coefficient(
    hyperparameters: PPOHyperparameters, initial: float, episodes: int
) -> float:
    """Schedule the Phase 3b A* KL weight without affecting Phase 3a."""
    if initial <= 0.0:
        return 0.0
    if hyperparameters.reservation_kl_decay_interval < 1:
        raise ValueError("reservation_kl_decay_interval must be positive.")
    if not 0.0 < hyperparameters.reservation_kl_decay_factor <= 1.0:
        raise ValueError("reservation_kl_decay_factor must be in (0, 1].")
    return max(
        hyperparameters.reservation_kl_minimum,
        initial
        * hyperparameters.reservation_kl_decay_factor
        ** (episodes // hyperparameters.reservation_kl_decay_interval),
    )


def _engagement_coefficient(
    hyperparameters: PPOHyperparameters, initial: float, episodes: int
) -> float:
    """Decay the offline semantic teacher weight as the policy becomes autonomous."""
    if initial <= 0.0:
        return 0.0
    if hyperparameters.engagement_decay_interval < 1:
        raise ValueError("engagement_decay_interval must be positive.")
    if not 0.0 < hyperparameters.engagement_decay_factor <= 1.0:
        raise ValueError("engagement_decay_factor must be in (0, 1].")
    return max(
        hyperparameters.engagement_minimum,
        initial
        * hyperparameters.engagement_decay_factor
        ** (episodes // hyperparameters.engagement_decay_interval),
    )


def train_phase3(config: Phase3TrainingConfig) -> Dict[str, object]:  # noqa: C901
    """Train one Phase 3 architecture ablation on the fixed medium/3-AGV scale."""
    if config.phase in {"3a", "3b"} and config.n_agents != 3:
        raise ValueError("Phase 3a and 3b are fixed to the medium three-AGV setting.")
    if config.phase == "4" and config.n_agents != 5:
        raise ValueError("Phase 4 is fixed to the controlled medium five-AGV setting.")
    if config.phase not in {"3a", "3b", "4"}:
        raise ValueError("Phase must be '3a', '3b', or '4'.")
    if config.phase == "3a" and config.ppo.reservation_kl_coefficient != 0.0:
        raise ValueError("Phase 3a must not enable A* path distillation.")
    if config.phase == "4":
        if config.offline_llm_enabled and not config.offline_semantic_dataset:
            raise ValueError("LLM distillation requires an offline_semantic_dataset.")
        if config.offline_llm_enabled != (config.ppo.engagement_coefficient > 0.0):
            raise ValueError(
                "Phase 4 LLM teacher and engagement coefficient must be "
                "enabled together."
            )
        if config.astar_kl_enabled != (
            config.ppo.reservation_kl_coefficient > 0.0
        ):
            raise ValueError(
                "Phase 4 A* teacher and reservation KL coefficient must be "
                "enabled together."
            )
    elif config.phase == "3b" and config.ppo.reservation_kl_coefficient <= 0.0:
        raise ValueError("Phase 3b requires a positive A* KL coefficient.")
    if config.offline_semantic_neighbours < 1:
        raise ValueError("offline_semantic_neighbours must be positive.")
    if config.parallel_envs < 1:
        raise ValueError("parallel_envs must be positive.")
    if config.episodes < 1:
        raise ValueError("episodes must be positive.")
    if (
        config.environment_step_budget is not None
        and config.environment_step_budget < 1
    ):
        raise ValueError("environment_step_budget must be positive when provided.")
    if config.checkpoint_interval < 1 or config.metrics_write_interval < 1:
        raise ValueError("Phase 3 intervals must be positive.")
    _validate_training_seed_groups(config.training_seed_groups)
    _set_seed(config.seed, config.torch_num_threads)
    resolved_device = _resolve_device(config.device)
    accelerator = _configure_accelerator(
        resolved_device, config.cuda_allow_tf32
    )
    accelerator["requested"] = config.device
    run_dir = Path(config.output_dir) / f"seed_{config.seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    (run_dir / "runtime.json").write_text(
        json.dumps(accelerator, indent=2), encoding="utf-8"
    )
    offline_teacher = (
        OfflineSemanticTeacher.from_jsonl(config.offline_semantic_dataset)
        if config.offline_llm_enabled
        else None
    )
    environment_count = min(config.parallel_envs, config.episodes)
    environment_pool = _EnvironmentPool(config, environment_count)
    if offline_teacher is not None and (
        offline_teacher.observation_dim != environment_pool.actor_observation_dim
    ):
        environment_pool.close()
        raise ValueError("Offline semantic dataset observation size is incompatible.")
    observations = []
    episode_metadata = []
    next_episode_index = 0
    for index in range(environment_count):
        metadata = _training_episode_seed(config, next_episode_index)
        observations.append(environment_pool.reset(index, metadata[0]))
        episode_metadata.append(metadata)
        next_episode_index += 1
    semantic_dim = 2 if config.phase == "4" else 1
    policy = DualHeadMAPPOPolicy(
        environment_pool.actor_observation_dim,
        ACTION_COUNT,
        str(resolved_device),
        semantic_dim=semantic_dim,
        semantic_features_enabled=(
            config.offline_llm_enabled if config.phase == "4" else True
        ),
    )
    updater = MAPPOUpdater(policy, config.ppo)
    buffer = RolloutBuffer(config.n_agents, semantic_dim=semantic_dim)
    writer = _writer(run_dir / "tensorboard")
    reservation_kl_initial = config.ppo.reservation_kl_coefficient
    engagement_initial = config.ppo.engagement_coefficient
    episodes = 0
    steps = 0
    episode_records: List[dict] = []
    update_count = 0
    priority_durations: Dict[str, List[float]] = {}
    next_checkpoint = config.checkpoint_interval
    active = [True] * environment_count
    updates_path = run_dir / "updates.csv"
    updates_path.unlink(missing_ok=True)
    training_started = time.perf_counter()
    last_update_time = training_started
    last_update_steps = 0
    worker_statistics: List[dict] = []
    if writer:
        writer.add_scalar("config/episodes", config.episodes, 0)
        writer.add_scalar(
            "config/training_seed_group_count", len(config.training_seed_groups), 0
        )
        writer.add_scalar("config/n_agents", config.n_agents, 0)
        writer.add_scalar("config/parallel_envs", environment_count, 0)
        writer.add_text("config/requested_device", config.device, 0)
        writer.add_text("config/resolved_device", str(resolved_device), 0)
        writer.add_scalar(
            "config/cuda_enabled", int(resolved_device.type == "cuda"), 0
        )
        writer.add_scalar("config/max_steps", config.max_steps, 0)
        writer.add_scalar(
            "config/dynamic_ingress_enabled",
            int(config.batch_interval is not None),
            0,
        )
        if config.batch_interval is not None:
            writer.add_scalar("config/batch_interval", config.batch_interval, 0)
        if config.batch_size_range is not None:
            writer.add_scalar("config/batch_size_min", config.batch_size_range[0], 0)
            writer.add_scalar("config/batch_size_max", config.batch_size_range[1], 0)
        if config.request_queue_size is not None:
            writer.add_scalar(
                "config/request_queue_size", config.request_queue_size, 0
            )
        if config.task_completion_target is not None:
            writer.add_scalar(
                "config/task_completion_target", config.task_completion_target, 0
            )
        writer.add_scalar(
            "config/engagement_coefficient",
            engagement_initial,
            0,
        )
        writer.add_scalar(
            "config/offline_engagement_teacher", int(offline_teacher is not None), 0
        )
        writer.add_scalar(
            "config/astar_kl_teacher", int(config.astar_kl_enabled), 0
        )
        writer.add_scalar(
            "config/semantic_features_enabled",
            int(policy.actor.semantic_features_enabled),
            0,
        )
        if offline_teacher is not None:
            writer.add_scalar(
                "config/offline_engagement_records", offline_teacher.size, 0
            )
        writer.add_scalar(
            "config/reservation_kl_coefficient",
            reservation_kl_initial,
            0,
        )
        writer.add_scalar("config/separate_engagement_encoder", 1, 0)
        writer.add_scalar("config/engagement_detached_for_motion", 1, 0)
        writer.flush()
    try:
        while _training_budget_available(config, episodes, steps):
            active_indices = [index for index, enabled in enumerate(active) if enabled]
            if config.environment_step_budget is not None:
                remaining_steps = config.environment_step_budget - steps
                active_indices = active_indices[:remaining_steps]
            if not active_indices:
                break
            snapshots = environment_pool.snapshots(active_indices)
            masks = np.stack([snapshot[0] for snapshot in snapshots])
            observation_batch = np.stack(
                [observations[index] for index in active_indices]
            )
            if offline_teacher is not None:
                flat_observations = observation_batch.reshape(
                    -1, observation_batch.shape[-1]
                )
                engagement_targets = offline_teacher.targets(
                    flat_observations, config.offline_semantic_neighbours
                ).reshape(len(active_indices), config.n_agents, semantic_dim)
            elif config.phase != "4":
                engagement_targets = np.stack(
                    [snapshot[2] for snapshot in snapshots]
                )
            else:
                engagement_targets = None
            reservation_preferences = (
                np.stack([snapshot[1] for snapshot in snapshots])
                if config.astar_kl_enabled
                else None
            )
            actions, log_probs, values, _ = policy.act(observation_batch, masks)
            transitions = environment_pool.step(active_indices, actions)
            for offset, index in enumerate(active_indices):
                transition = transitions[offset]
                done = (
                    transition.terminated
                    or transition.truncated
                    or transition.metrics.deadlocked
                )
                buffer.add(
                    observations[index],
                    actions[offset],
                    log_probs[offset],
                    transition.team_reward,
                    done,
                    values[offset],
                    masks[offset],
                    reservation_preferences=(
                        reservation_preferences[offset]
                        if reservation_preferences is not None
                        else None
                    ),
                    engagement_targets=(
                        engagement_targets[offset]
                        if engagement_targets is not None
                        else None
                    ),
                    stream_id=index,
                )
                observations[index] = transition.observations
                steps += 1
                if not done:
                    continue
                episodes += 1
                environment_seed, seed_group, seed_offset = episode_metadata[index]
                record = {
                    "episode": episodes,
                    "environment_steps": steps,
                    "environment_index": index,
                    "environment_seed": environment_seed,
                    "training_seed_offset": seed_offset,
                    **transition.metrics.as_dict(),
                }
                if seed_group is not None:
                    record["training_seed_group"] = seed_group
                by_priority = _completion_durations(transition.info["tasks"])
                for label, durations in by_priority.items():
                    priority_durations.setdefault(label, []).extend(durations)
                    record[f"priority_{label}_mean_completion_steps"] = float(
                        np.mean(durations)
                    )
                episode_records.append(record)
                if writer:
                    for key, metric in record.items():
                        if key != "episode" and isinstance(metric, (int, float, bool)):
                            writer.add_scalar(f"episode/{key}", metric, episodes)
                if episodes % config.metrics_write_interval == 0:
                    _write_csv(run_dir / "episodes.csv", episode_records)
                if _training_budget_available(config, episodes, steps) and (
                    next_episode_index < config.episodes
                ):
                    metadata = _training_episode_seed(config, next_episode_index)
                    episode_metadata[index] = metadata
                    observations[index] = environment_pool.reset(index, metadata[0])
                    next_episode_index += 1
                else:
                    active[index] = False
                if episodes % config.metrics_write_interval == 0:
                    elapsed = max(time.perf_counter() - training_started, 1e-9)
                    message = {
                        "episodes": episodes,
                        "target_episodes": config.episodes,
                        "steps": steps,
                        "env_steps_per_second": round(steps / elapsed, 2),
                        "elapsed_seconds": round(elapsed, 1),
                    }
                    print(json.dumps(message), file=sys.stderr, flush=True)
                    if writer:
                        writer.flush()
            if (
                len(buffer) >= config.rollout_steps
                or not any(active)
                or not _training_budget_available(config, episodes, steps)
            ):
                active_indices = [
                    index for index, enabled in enumerate(active) if enabled
                ]
                last_values = {index: 0.0 for index in range(environment_count)}
                if active_indices:
                    bootstrap_observations = torch.as_tensor(
                        np.stack([observations[index] for index in active_indices]),
                        dtype=torch.float32,
                        device=policy.device,
                    )
                    with torch.no_grad():
                        bootstrap_values = policy.values(
                            bootstrap_observations
                        ).cpu().numpy()
                    last_values.update(
                        zip(active_indices, bootstrap_values.astype(float))
                    )
                config.ppo.reservation_kl_coefficient = _reservation_coefficient(
                    config.ppo, reservation_kl_initial, episodes
                )
                config.ppo.engagement_coefficient = _engagement_coefficient(
                    config.ppo, engagement_initial, episodes
                )
                losses = updater.update(buffer, last_values)
                losses["engagement_coefficient"] = config.ppo.engagement_coefficient
                update_finished = time.perf_counter()
                interval_seconds = max(update_finished - last_update_time, 1e-9)
                total_seconds = max(update_finished - training_started, 1e-9)
                losses["rollout_env_steps_per_second"] = (
                    steps - last_update_steps
                ) / interval_seconds
                losses["overall_env_steps_per_second"] = steps / total_seconds
                losses["elapsed_seconds"] = total_seconds
                last_update_time = update_finished
                last_update_steps = steps
                update_count += 1
                update = {"update": update_count, "steps": steps, **losses}
                _append_csv(updates_path, update)
                if writer:
                    for key, metric in losses.items():
                        writer.add_scalar(f"training/{key}", metric, steps)
                    writer.flush()
            while episodes >= next_checkpoint:
                _save_checkpoint(
                    run_dir / f"checkpoint_ep_{next_checkpoint:05d}.pt",
                    policy,
                    config,
                    episodes,
                    steps,
                )
                next_checkpoint += config.checkpoint_interval
    finally:
        worker_statistics = environment_pool.close()
        if writer:
            writer.close()
    final = run_dir / "checkpoint_final.pt"
    _save_checkpoint(final, policy, config, episodes, steps)
    _write_csv(run_dir / "episodes.csv", episode_records)
    training_elapsed_seconds = time.perf_counter() - training_started
    summary = {
        "phase": config.phase,
        "seed": config.seed,
        "training_seed_schedule": {
            "mode": (
                "round_robin_seed_groups"
                if config.training_seed_groups
                else "legacy_contiguous"
            ),
            "seed_groups": list(config.training_seed_groups),
            "seed_formula": "group * 10000 + round_offset",
        },
        "episodes": episodes,
        "steps": steps,
        "environment_step_budget": config.environment_step_budget,
        "training_elapsed_seconds": training_elapsed_seconds,
        "env_steps_per_second": (
            steps / training_elapsed_seconds if training_elapsed_seconds > 0 else 0.0
        ),
        "parallel_envs": environment_count,
        "accelerator": accelerator,
        "checkpoint": str(final),
        "last_episode": episode_records[-1] if episode_records else None,
        "priority_mean_completion_steps": _mean_durations(priority_durations),
        "a_star_distillation": config.astar_kl_enabled,
        "offline_llm_semantic_distillation": offline_teacher is not None,
        "semantic_architecture": (
            "two_dimensional_fixed_zero_semantics_for_motion"
            if not policy.actor.semantic_features_enabled
            else (
                "task_commitment_and_local_assertiveness_detached_for_motion"
                if semantic_dim == 2
                else "single_engagement_detached_for_motion"
            )
        ),
    }
    if config.phase != "4":
        summary["rule_engagement_labels"] = _engagement_label_definition(config)
    if offline_teacher is not None:
        summary["offline_semantic_teacher"] = {
            "dataset": config.offline_semantic_dataset,
            "records": offline_teacher.size,
            "models": list(offline_teacher.model_names),
            "neighbours": config.offline_semantic_neighbours,
            "api_calls_during_training": 0,
        }
    if config.astar_kl_enabled:
        reservation_teacher = {
            key: sum(statistics.get(key, 0) for statistics in worker_statistics)
            for key in (
                "path_livelocks",
                "state_deadlocks",
                "cache_hits",
                "cache_misses",
                "reached_goal_plans",
                "partial_paths",
                "terminal_conflicts",
                "reservation_false_no_paths",
                "explicit_waits",
                "replans",
                "expanded_nodes",
                "planning_time_count",
                "planning_time_ms_total",
            )
        }
        planning_count = reservation_teacher["planning_time_count"]
        reservation_teacher["planning_time_ms_mean"] = (
            reservation_teacher["planning_time_ms_total"] / planning_count
            if planning_count
            else 0.0
        )
        reservation_teacher["planning_time_ms_p95_worker_max"] = max(
            (
                statistics.get("planning_time_ms_p95", 0.0)
                for statistics in worker_statistics
            ),
            default=0.0,
        )
        summary["reservation_teacher"] = reservation_teacher
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _checkpoint_metadata_semantic_dims(
    checkpoint: Mapping[str, object],
) -> Dict[str, int]:
    candidates: Dict[str, int] = {}
    explicit = checkpoint.get("semantic_dim")
    if explicit is not None:
        try:
            candidates["semantic_dim metadata"] = int(explicit)
        except (TypeError, ValueError) as error:
            raise ValueError("Checkpoint semantic_dim must be an integer.") from error

    phase = str(checkpoint.get("phase", ""))
    if phase == "4":
        candidates["phase metadata"] = 2
    elif phase.startswith("3"):
        candidates["phase metadata"] = 1
    return candidates


def _model_state_semantic_dims(model_state: Mapping[str, object]) -> Dict[str, int]:
    candidates: Dict[str, int] = {}
    semantic_weight = model_state.get("actor.engagement_head.0.weight")
    if isinstance(semantic_weight, torch.Tensor) and semantic_weight.ndim == 2:
        candidates["semantic-head weight"] = int(semantic_weight.shape[0])
    semantic_bias = model_state.get("actor.engagement_head.0.bias")
    if isinstance(semantic_bias, torch.Tensor) and semantic_bias.ndim == 1:
        candidates["semantic-head bias"] = int(semantic_bias.shape[0])

    motion_weight = model_state.get("actor.motion_head.weight")
    encoder_weight = model_state.get("actor.motion_encoder.2.weight")
    if (
        isinstance(motion_weight, torch.Tensor)
        and motion_weight.ndim == 2
        and isinstance(encoder_weight, torch.Tensor)
        and encoder_weight.ndim == 2
    ):
        candidates["motion-head input"] = int(
            motion_weight.shape[1] - encoder_weight.shape[0]
        )
    return candidates


def _checkpoint_semantic_dim(checkpoint: Mapping[str, object]) -> int:
    """Infer and validate the semantic width of Phase 3/4 checkpoints.

    Phase 4 checkpoints created before ``semantic_dim`` was persisted must be
    inferred from their model tensors.  Cross-checking every available source
    turns an otherwise opaque ``load_state_dict`` shape error into a clear
    checkpoint-compatibility error.
    """
    candidates = _checkpoint_metadata_semantic_dims(checkpoint)
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("Checkpoint model_state must be a parameter mapping.")
    candidates.update(_model_state_semantic_dims(model_state))

    if not candidates:
        raise ValueError("Checkpoint does not describe a semantic policy dimension.")
    invalid = {
        source: width for source, width in candidates.items() if width not in (1, 2)
    }
    if invalid:
        raise ValueError(
            f"Checkpoint contains an invalid semantic dimension: {invalid}."
        )
    widths = set(candidates.values())
    if len(widths) != 1:
        raise ValueError(f"Checkpoint semantic dimensions disagree: {candidates}.")
    return widths.pop()


def load_phase3_policy(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    semantic_dim = _checkpoint_semantic_dim(checkpoint)
    config = checkpoint.get("config", {})
    feature_flag = checkpoint.get("semantic_features_enabled")
    if feature_flag is None and isinstance(config, Mapping):
        feature_flag = config.get("use_offline_llm_teacher")
    if feature_flag is None:
        feature_flag = True
    policy = DualHeadMAPPOPolicy(
        checkpoint["actor_observation_dim"],
        ACTION_COUNT,
        device=device,
        semantic_dim=semantic_dim,
        semantic_features_enabled=bool(feature_flag),
    )
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    return policy, checkpoint["config"], checkpoint


def _charging_metrics_summary(records: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = list(records)
    count_keys = (
        "low_battery_triggers",
        "charging_target_steps",
        "charger_arrivals",
        "charged_events",
        "charging_wait_steps",
        "task_recoveries",
        "energy_deaths",
    )
    summary = {
        f"mean_{key}_per_episode": float(
            np.mean([float(row.get(key, 0.0)) for row in rows])
        )
        if rows
        else 0.0
        for key in count_keys
    }
    summary["mean_charging_exposure_rate"] = (
        float(np.mean([float(row.get("charging_exposure_rate", 0.0)) for row in rows]))
        if rows
        else 0.0
    )
    summary["episodes_with_low_battery_rate"] = (
        float(np.mean([float(row.get("low_battery_triggers", 0)) > 0 for row in rows]))
        if rows
        else 0.0
    )
    summary["episodes_with_charging_rate"] = (
        float(np.mean([float(row.get("charged_events", 0)) > 0 for row in rows]))
        if rows
        else 0.0
    )
    summary["minimum_battery"] = (
        min(float(row.get("minimum_battery", 1.0)) for row in rows) if rows else 1.0
    )
    return summary


def _episode_metrics_summary(records: Iterable[Dict[str, object]]) -> Dict[str, float]:
    rows = list(records)
    count_keys = (
        "completed_tasks",
        "created_tasks",
        "steps",
        "reward",
        "collisions",
        "agent_deaths",
        "picked_tasks",
        "blocked_forwards",
    )
    summary = {
        f"mean_{key}_per_episode": float(
            np.mean([float(row.get(key, 0.0)) for row in rows])
        )
        if rows
        else 0.0
        for key in count_keys
    }
    total_steps = sum(float(row.get("steps", 0.0)) for row in rows)
    total_completed = sum(float(row.get("completed_tasks", 0.0)) for row in rows)
    summary["completed_tasks_per_1000_steps"] = (
        1000.0 * total_completed / total_steps if total_steps > 0.0 else 0.0
    )
    return summary


def evaluate_phase3(  # noqa: C901
    policy,
    config: Dict[str, object],
    seeds: Iterable[int],
    episodes_per_seed: int = 20,
    collect_engagement: bool = False,
    engagement_sample_rate: int = 10,
) -> Dict[str, object]:
    """Evaluate a Phase 3 checkpoint on deterministic seeds.

    When ``collect_engagement`` is true the function also records
    ``(priority_label, engagement_value)`` samples every
    ``engagement_sample_rate`` steps so the diagnostic
    :func:`~llm_mappo.plotting.plot_engagement_by_priority` figure can be drawn.
    The samples are returned under the ``"engagement_samples"`` key as a list of
    ``(label, value)`` tuples.
    """
    if engagement_sample_rate < 1:
        raise ValueError("engagement_sample_rate must be positive.")
    priority_schedule = config.get("priority_schedule", ("A", "B", "C"))
    batch_size_range = config.get("batch_size_range")
    env = Phase2Warehouse(
        n_agents=int(config.get("n_agents", 3)),
        max_steps=config["max_steps"],
        env_id=config["env_id"],
        charge_threshold=config.get("charge_threshold", 0.2),
        charge_release_threshold=config.get("charge_release_threshold", 0.8),
        battery_cost_scale=config.get("battery_cost_scale", 1.0),
        waypoint_reward=config.get("waypoint_reward", 0.01),
        oracle_interaction_mask=config.get("oracle_interaction_mask", True),
        deadlock_steps=config.get("deadlock_steps", 180),
        priority_schedule=(
            tuple(priority_schedule) if priority_schedule else None
        ),
        batch_interval=config.get("batch_interval"),
        batch_size_range=(
            tuple(batch_size_range) if batch_size_range is not None else None
        ),
        initial_priority_label=config.get("initial_priority_label", "B"),
        request_queue_size=config.get("request_queue_size"),
        task_completion_target=config.get("task_completion_target"),
        include_priority_features=True,
    )
    per_seed: List[dict] = []
    all_records: List[dict] = []
    engagement_samples: List[tuple[str, float]] = []
    try:
        for seed in seeds:
            records = []
            priority_durations: Dict[str, List[float]] = {}
            for offset in range(episodes_per_seed):
                observations = env.reset(seed=seed * 10_000 + offset)
                step = 0
                while True:
                    outputs = policy.act(
                        observations, env.action_masks(), deterministic=True
                    )
                    actions = outputs[0]
                    engagement = outputs[3] if len(outputs) > 3 else None
                    if (
                        collect_engagement
                        and engagement is not None
                        and step % engagement_sample_rate == 0
                    ):
                        labels = _engagement_label(env)
                        for label, value in zip(labels, engagement):
                            engagement_samples.append((label, float(value)))
                    transition = env.step(actions)
                    observations = transition.observations
                    step += 1
                    if (
                        transition.terminated
                        or transition.truncated
                        or transition.metrics.deadlocked
                    ):
                        record = transition.metrics.as_dict()
                        records.append(record)
                        all_records.append(record)
                        for label, values in _completion_durations(
                            transition.info["tasks"]
                        ).items():
                            priority_durations.setdefault(label, []).extend(values)
                        break
            completion = np.asarray(
                [record["task_completion_rate"] for record in records], dtype=float
            )
            collisions = np.asarray(
                [record["collisions"] for record in records], dtype=float
            )
            per_seed.append(
                {
                    "seed": seed,
                    "episodes": len(records),
                    "task_completion_rate": (
                        float(completion.mean()) if len(records) else 0.0
                    ),
                    "mean_collisions": float(collisions.mean()) if len(records) else 0.0,
                    "deadlock_rate": float(
                        np.mean([record["deadlocked"] for record in records])
                    )
                    if records
                    else 0.0,
                    "success_rate": float(
                        np.mean([record["success"] for record in records])
                    )
                    if records
                    else 0.0,
                    "priority_mean_completion_steps": _mean_durations(
                        priority_durations
                    ),
                    **_episode_metrics_summary(records),
                    **_charging_metrics_summary(records),
                }
            )
    finally:
        env.close()
    completion = np.asarray(
        [record["task_completion_rate"] for record in per_seed], dtype=float
    )
    collisions = np.asarray(
        [record["mean_collisions"] for record in per_seed], dtype=float
    )
    deadlock_rates = np.asarray(
        [record["deadlock_rate"] for record in per_seed], dtype=float
    )
    success_rates = np.asarray(
        [record["success_rate"] for record in per_seed], dtype=float
    )
    priority_durations: Dict[str, List[float]] = {}
    for seed_result in per_seed:
        for label, value in seed_result["priority_mean_completion_steps"].items():
            priority_durations.setdefault(label, []).append(value)
    priority_means = _mean_durations(priority_durations)
    priority_ordering = (
        priority_means["A"] < priority_means["C"]
        if "A" in priority_means and "C" in priority_means
        else None
    )
    result: Dict[str, object] = {
        "seeds": per_seed,
        "episodes": int(sum(record["episodes"] for record in per_seed)),
        "task_completion_rate": float(completion.mean()) if len(per_seed) else 0.0,
        "mean_collisions_per_episode": (
            float(collisions.mean()) if len(per_seed) else 0.0
        ),
        "deadlock_rate": float(deadlock_rates.mean()) if len(per_seed) else 0.0,
        "success_rate_mean": float(success_rates.mean()) if len(per_seed) else 0.0,
        "success_rate_std": float(success_rates.std()) if len(per_seed) else 0.0,
        "priority_mean_completion_steps": priority_means,
        "high_priority_faster_than_low": priority_ordering,
        **_episode_metrics_summary(all_records),
        **_charging_metrics_summary(all_records),
    }
    if collect_engagement:
        result["engagement_samples"] = engagement_samples
    return result


def _engagement_label(env: Phase2Warehouse) -> List[str]:
    """Return the current priority label per agent for engagement sampling."""
    labels: List[str] = []
    for agent in env.env.agents:
        task = env.env.task_queue.task_for_agent(agent.id)
        if task is None:
            labels.append("none")
        else:
            labels.append(task.label[0])
    return labels
