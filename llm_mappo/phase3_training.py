"""Local Phase 3a training: dual-head MAPPO with rule priority labels."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

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
    waypoint_reward: float = 0.01
    oracle_interaction_mask: bool = True
    deadlock_steps: int = 180
    episodes: int = 800
    rollout_steps: int = 512
    checkpoint_interval: int = 200
    metrics_write_interval: int = 20
    output_dir: str = "artifacts/phase3a_dual_head"
    offline_semantic_dataset: str | None = None
    offline_semantic_neighbours: int = 3
    ppo: PPOHyperparameters = field(
        default_factory=lambda: PPOHyperparameters(
            reservation_kl_coefficient=0.0,
            engagement_coefficient=0.1,
        )
    )

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
            waypoint_reward=environment.get("waypoint_reward", cls.waypoint_reward),
            oracle_interaction_mask=environment.get(
                "oracle_interaction_mask", cls.oracle_interaction_mask
            ),
            deadlock_steps=environment.get("deadlock_steps", cls.deadlock_steps),
            episodes=training.get("episodes", cls.episodes),
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
            ppo=PPOHyperparameters(**ppo_values),
        )


def _set_seed(seed: int, threads: int) -> None:
    if threads < 1:
        raise ValueError("torch_num_threads must be positive.")
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def _writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(path))
    except ImportError:
        return None


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
    temporary_path.replace(path)


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
    if config.phase == "4" and not config.offline_semantic_dataset:
        raise ValueError("Phase 4 requires an offline_semantic_dataset.")
    if (
        config.phase in {"3b", "4"}
        and config.ppo.reservation_kl_coefficient <= 0.0
    ):
        raise ValueError("Phase 3b and Phase 4 require a positive A* KL coefficient.")
    if config.offline_semantic_neighbours < 1:
        raise ValueError("offline_semantic_neighbours must be positive.")
    if config.checkpoint_interval < 1 or config.metrics_write_interval < 1:
        raise ValueError("Phase 3 intervals must be positive.")
    _validate_training_seed_groups(config.training_seed_groups)
    _set_seed(config.seed, config.torch_num_threads)
    run_dir = Path(config.output_dir) / f"seed_{config.seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    env = Phase2Warehouse(
        n_agents=config.n_agents,
        max_steps=config.max_steps,
        env_id=config.env_id,
        charge_threshold=config.charge_threshold,
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
    environment_seed, seed_group, seed_offset = _training_episode_seed(config, 0)
    observations = env.reset(seed=environment_seed)
    semantic_dim = 2 if config.phase == "4" else 1
    policy = DualHeadMAPPOPolicy(
        env.actor_observation_dim,
        ACTION_COUNT,
        config.device,
        semantic_dim=semantic_dim,
    )
    updater = MAPPOUpdater(policy, config.ppo)
    offline_teacher = (
        OfflineSemanticTeacher.from_jsonl(config.offline_semantic_dataset)
        if config.phase == "4"
        else None
    )
    if offline_teacher is not None and (
        offline_teacher.observation_dim != env.actor_observation_dim
    ):
        raise ValueError("Offline semantic dataset observation size is incompatible.")
    buffer = RolloutBuffer(config.n_agents, semantic_dim=semantic_dim)
    writer = _writer(run_dir / "tensorboard")
    reservation_expert = AStarExpert() if config.phase in {"3b", "4"} else None
    reservation_kl_initial = config.ppo.reservation_kl_coefficient
    engagement_initial = config.ppo.engagement_coefficient
    episodes = 0
    steps = 0
    episode_records: List[dict] = []
    update_records: List[dict] = []
    priority_durations: Dict[str, List[float]] = {}
    next_checkpoint = config.checkpoint_interval
    if writer:
        writer.add_scalar("config/episodes", config.episodes, 0)
        writer.add_scalar(
            "config/training_seed_group_count", len(config.training_seed_groups), 0
        )
        writer.add_scalar("config/n_agents", config.n_agents, 0)
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
        while episodes < config.episodes:
            masks = env.action_masks()
            engagement_targets = (
                offline_teacher.targets(
                    observations, config.offline_semantic_neighbours
                )
                if offline_teacher is not None
                else _engagement_targets(env)
            )
            reservation_preferences = None
            if reservation_expert is not None:
                _, reservation_preferences = reservation_expert.act(env, masks)
            actions, log_probs, value, engagement = policy.act(observations, masks)
            transition = env.step(actions)
            done = (
                transition.terminated
                or transition.truncated
                or transition.metrics.deadlocked
            )
            buffer.add(
                observations,
                actions,
                log_probs,
                transition.team_reward,
                done,
                value,
                masks,
                reservation_preferences=reservation_preferences,
                engagement_targets=engagement_targets,
            )
            observations = transition.observations
            steps += 1
            if done:
                episodes += 1
                record = {
                    "episode": episodes,
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
                    writer.flush()
                if episodes % config.metrics_write_interval == 0:
                    _write_csv(run_dir / "episodes.csv", episode_records)
                environment_seed, seed_group, seed_offset = _training_episode_seed(
                    config, episodes
                )
                observations = env.reset(seed=environment_seed)
            if len(buffer) >= config.rollout_steps or (
                episodes == config.episodes and len(buffer)
            ):
                last_value = (
                    0.0
                    if done
                    else policy.act(observations, env.action_masks())[2]
                )
                config.ppo.reservation_kl_coefficient = _reservation_coefficient(
                    config.ppo, reservation_kl_initial, episodes
                )
                config.ppo.engagement_coefficient = _engagement_coefficient(
                    config.ppo, engagement_initial, episodes
                )
                losses = updater.update(buffer, last_value)
                losses["engagement_coefficient"] = config.ppo.engagement_coefficient
                update = {"update": len(update_records) + 1, "steps": steps, **losses}
                update_records.append(update)
                _write_csv(run_dir / "updates.csv", update_records)
                if writer:
                    for key, metric in losses.items():
                        writer.add_scalar(f"training/{key}", metric, steps)
                    writer.flush()
            if episodes >= next_checkpoint:
                _save_checkpoint(
                    run_dir / f"checkpoint_ep_{next_checkpoint:05d}.pt",
                    policy,
                    config,
                    episodes,
                    steps,
                )
                next_checkpoint += config.checkpoint_interval
    finally:
        env.close()
        if writer:
            writer.close()
    final = run_dir / "checkpoint_final.pt"
    _save_checkpoint(final, policy, config, episodes, steps)
    _write_csv(run_dir / "episodes.csv", episode_records)
    _write_csv(run_dir / "updates.csv", update_records)
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
        "checkpoint": str(final),
        "last_episode": episode_records[-1] if episode_records else None,
        "priority_mean_completion_steps": _mean_durations(priority_durations),
        "a_star_distillation": config.phase in {"3b", "4"},
        "offline_llm_semantic_distillation": offline_teacher is not None,
        "semantic_architecture": (
            "task_commitment_and_local_assertiveness_detached_for_motion"
            if semantic_dim == 2
            else "single_engagement_detached_for_motion"
        ),
        "rule_engagement_labels": _engagement_label_definition(config),
    }
    if offline_teacher is not None:
        summary["offline_semantic_teacher"] = {
            "dataset": config.offline_semantic_dataset,
            "records": offline_teacher.size,
            "models": list(offline_teacher.model_names),
            "neighbours": config.offline_semantic_neighbours,
            "api_calls_during_training": 0,
        }
    if reservation_expert is not None:
        summary["reservation_teacher"] = {
            "path_livelocks": reservation_expert.path_livelocks,
            "state_deadlocks": reservation_expert.state_deadlocks,
        }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def load_phase3_policy(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = DualHeadMAPPOPolicy(
        checkpoint["actor_observation_dim"], ACTION_COUNT, device=device
    )
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    return policy, checkpoint["config"], checkpoint


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
