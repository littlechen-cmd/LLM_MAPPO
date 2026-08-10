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


@dataclass
class Phase3TrainingConfig:
    """Reproducible Phase 3a configuration; A* distillation is disabled."""

    seed: int = 7
    device: str = "cpu"
    torch_num_threads: int = 1
    n_agents: int = 3
    max_steps: int = 400
    env_id: str = "llm-mappo-medium-3ag-v1"
    priority_schedule: tuple[str, ...] = ("A", "B", "C")
    charge_threshold: float = 0.2
    waypoint_reward: float = 0.01
    oracle_interaction_mask: bool = True
    deadlock_steps: int = 180
    episodes: int = 800
    rollout_steps: int = 512
    checkpoint_interval: int = 200
    output_dir: str = "artifacts/phase3a_dual_head"
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
        ppo_values.setdefault("reservation_kl_coefficient", 0.0)
        ppo_values.setdefault("engagement_coefficient", 0.1)
        return cls(
            seed=training.get("seed", cls.seed),
            device=training.get("device", cls.device),
            torch_num_threads=training.get("torch_num_threads", cls.torch_num_threads),
            n_agents=environment.get("n_agents", cls.n_agents),
            max_steps=environment.get("max_steps", cls.max_steps),
            env_id=environment.get("id", cls.env_id),
            priority_schedule=tuple(
                environment.get("priority_schedule", cls.priority_schedule)
            ),
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
            output_dir=training.get("output_dir", cls.output_dir),
            ppo=PPOHyperparameters(**ppo_values),
        )


def _set_seed(seed: int, threads: int) -> None:
    if threads < 1:
        raise ValueError("torch_num_threads must be positive.")
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(path))
    except ImportError:
        return None


def _engagement_targets(env: Phase2Warehouse) -> np.ndarray:
    values = []
    for agent in env.env.agents:
        task = env.env.task_queue.task_for_agent(agent.id)
        if task is None:
            values.append(0.3)
        elif task.label.startswith("A"):
            values.append(0.8)
        elif task.label.startswith("B"):
            values.append(0.5)
        else:
            values.append(0.3)
    return np.asarray(values, dtype=np.float32)


def _save_checkpoint(path: Path, policy, config, episodes: int, steps: int) -> None:
    torch.save(
        {
            "model_state": policy.state_dict(),
            "config": asdict(config),
            "actor_observation_dim": policy.actor.encoder[0].in_features,
            "episodes": episodes,
            "steps": steps,
            "phase": "3a",
        },
        path,
    )


def _write_csv(path: Path, rows: List[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def train_phase3(config: Phase3TrainingConfig) -> Dict[str, object]:  # noqa: C901
    """Train Phase 3a on the unchanged medium/3-AGV environment scale."""
    if config.n_agents != 3:
        raise ValueError("Phase 3a is fixed to the medium three-AGV setting.")
    if config.ppo.reservation_kl_coefficient != 0.0:
        raise ValueError("Phase 3a must not enable A* or reservation KL distillation.")
    _set_seed(config.seed, config.torch_num_threads)
    run_dir = Path(config.output_dir) / f"seed_{config.seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=config.max_steps,
        env_id=config.env_id,
        charge_threshold=config.charge_threshold,
        waypoint_reward=config.waypoint_reward,
        oracle_interaction_mask=config.oracle_interaction_mask,
        deadlock_steps=config.deadlock_steps,
        priority_schedule=config.priority_schedule,
    )
    observations = env.reset(seed=config.seed)
    policy = DualHeadMAPPOPolicy(env.actor_observation_dim, ACTION_COUNT, config.device)
    updater = MAPPOUpdater(policy, config.ppo)
    buffer = RolloutBuffer(config.n_agents)
    writer = _writer(run_dir / "tensorboard")
    episodes = 0
    steps = 0
    episode_records: List[dict] = []
    update_records: List[dict] = []
    try:
        while episodes < config.episodes:
            masks = env.action_masks()
            engagement_targets = _engagement_targets(env)
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
                engagement_targets=engagement_targets,
            )
            observations = transition.observations
            steps += 1
            if done:
                episodes += 1
                record = {"episode": episodes, **transition.metrics.as_dict()}
                completed = [
                    task for task in transition.info["tasks"]
                    if task["status"] == "completed"
                ]
                by_priority = {}
                for task in completed:
                    by_priority.setdefault(task["label"][0], []).append(
                        task["completed_step"] - task["arrival_step"]
                    )
                for label, durations in by_priority.items():
                    record[f"priority_{label}_mean_completion_steps"] = float(
                        np.mean(durations)
                    )
                episode_records.append(record)
                if writer:
                    for key, metric in record.items():
                        if key != "episode" and isinstance(metric, (int, float, bool)):
                            writer.add_scalar(f"episode/{key}", metric, episodes)
                observations = env.reset(seed=config.seed + episodes)
            if len(buffer) >= config.rollout_steps or (
                episodes == config.episodes and len(buffer)
            ):
                last_value = (
                    0.0
                    if done
                    else policy.act(observations, env.action_masks())[2]
                )
                losses = updater.update(buffer, last_value)
                update = {"update": len(update_records) + 1, "steps": steps, **losses}
                update_records.append(update)
                if writer:
                    for key, metric in losses.items():
                        writer.add_scalar(f"training/{key}", metric, steps)
                    writer.flush()
            if episodes and episodes % config.checkpoint_interval == 0:
                _save_checkpoint(
                    run_dir / f"checkpoint_ep_{episodes:05d}.pt",
                    policy,
                    config,
                    episodes,
                    steps,
                )
    finally:
        env.close()
        if writer:
            writer.close()
    final = run_dir / "checkpoint_final.pt"
    _save_checkpoint(final, policy, config, episodes, steps)
    _write_csv(run_dir / "episodes.csv", episode_records)
    _write_csv(run_dir / "updates.csv", update_records)
    summary = {
        "phase": "3a",
        "seed": config.seed,
        "episodes": episodes,
        "steps": steps,
        "checkpoint": str(final),
        "last_episode": episode_records[-1] if episode_records else None,
        "a_star_distillation": False,
        "rule_engagement_labels": {"A": 0.8, "B": 0.5, "C_or_lower": 0.3},
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


def evaluate_phase3(
    policy,
    config: Dict[str, object],
    seeds: Iterable[int],
    episodes_per_seed: int = 20,
) -> Dict[str, object]:
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=config["max_steps"],
        env_id=config["env_id"],
        charge_threshold=config.get("charge_threshold", 0.2),
        waypoint_reward=config.get("waypoint_reward", 0.01),
        oracle_interaction_mask=config.get("oracle_interaction_mask", True),
        deadlock_steps=config.get("deadlock_steps", 180),
        priority_schedule=tuple(config.get("priority_schedule", ("A", "B", "C"))),
    )
    records = []
    try:
        for seed in seeds:
            for offset in range(episodes_per_seed):
                observations = env.reset(seed=seed * 10_000 + offset)
                while True:
                    actions = policy.act(
                        observations, env.action_masks(), deterministic=True
                    )[0]
                    transition = env.step(actions)
                    observations = transition.observations
                    if (
                        transition.terminated
                        or transition.truncated
                        or transition.metrics.deadlocked
                    ):
                        records.append(transition.metrics.as_dict())
                        break
    finally:
        env.close()
    completion = np.asarray([record["task_completion_rate"] for record in records])
    collisions = np.asarray([record["collisions"] for record in records])
    return {
        "episodes": len(records),
        "task_completion_rate": float(completion.mean()) if len(records) else 0.0,
        "mean_collisions_per_episode": float(collisions.mean()) if len(records) else 0.0,
        "deadlock_rate": float(np.mean([record["deadlocked"] for record in records]))
        if records else 0.0,
    }
