"""Training and evaluation workflow for the Phase 2 MAPPO baseline."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import yaml

from llm_mappo.mappo import MAPPOPolicy, MAPPOUpdater, PPOHyperparameters, RolloutBuffer
from llm_mappo.phase2 import ACTION_COUNT, Phase2Warehouse


@dataclass
class Phase2TrainingConfig:
    """All configuration required to reproduce a Phase 2 run."""

    seed: int = 7
    device: str = "cpu"
    n_agents: int = 3
    max_steps: int = 400
    episodes: int = 5000
    rollout_steps: int = 512
    checkpoint_interval: int = 250
    output_dir: str = "artifacts/phase2"
    ppo: PPOHyperparameters = field(default_factory=PPOHyperparameters)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Phase2TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            source = yaml.safe_load(stream) or {}
        environment = source.get("environment", {})
        training = source.get("training", {})
        ppo = PPOHyperparameters(**source.get("ppo", {}))
        return cls(
            seed=training.get("seed", cls.seed),
            device=training.get("device", cls.device),
            n_agents=environment.get("n_agents", cls.n_agents),
            max_steps=environment.get("max_steps", cls.max_steps),
            episodes=training.get("episodes", cls.episodes),
            rollout_steps=training.get("rollout_steps", cls.rollout_steps),
            checkpoint_interval=training.get(
                "checkpoint_interval", cls.checkpoint_interval
            ),
            output_dir=training.get("output_dir", cls.output_dir),
            ppo=ppo,
        )


def set_seed(seed: int) -> None:
    """Seed every local RNG used by the baseline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _writer(path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(path))
    except ImportError:
        return None


def train_phase2(config: Phase2TrainingConfig) -> Dict[str, object]:
    """Train MAPPO and persist metrics, configuration, and checkpoints."""
    set_seed(config.seed)
    run_dir = Path(config.output_dir) / f"seed_{config.seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    env = Phase2Warehouse(n_agents=config.n_agents, max_steps=config.max_steps)
    observations = env.reset(seed=config.seed)
    policy = MAPPOPolicy(env.actor_observation_dim, ACTION_COUNT, config.device)
    updater = MAPPOUpdater(policy, config.ppo)
    buffer = RolloutBuffer(config.n_agents)
    writer = _writer(run_dir / "tensorboard")
    episode_records: List[dict] = []
    update_records: List[dict] = []
    episodes = 0
    steps = 0
    next_checkpoint = config.checkpoint_interval

    try:
        while episodes < config.episodes:
            actions, log_probs, value = policy.act(observations)
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
            )
            observations = transition.observations
            steps += 1
            if done:
                episodes += 1
                record = {"episode": episodes, **transition.metrics.as_dict()}
                episode_records.append(record)
                if writer is not None:
                    for key, metric in record.items():
                        if key not in {"episode", "deadlocked", "success"}:
                            writer.add_scalar(f"episode/{key}", metric, episodes)
                    writer.add_scalar(
                        "episode/deadlocked", record["deadlocked"], episodes
                    )
                    writer.add_scalar("episode/success", record["success"], episodes)
                observations = env.reset(seed=config.seed + episodes)
            if len(buffer) >= config.rollout_steps or (
                episodes == config.episodes and len(buffer)
            ):
                last_value = 0.0 if done else policy.act(observations)[2]
                losses = updater.update(buffer, last_value)
                update = {"update": len(update_records) + 1, "steps": steps, **losses}
                update_records.append(update)
                if writer is not None:
                    for key, loss in losses.items():
                        writer.add_scalar(f"training/{key}", loss, steps)
            next_checkpoint = _checkpoint_if_due(
                episodes,
                next_checkpoint,
                run_dir,
                policy,
                config,
                steps,
            )
    finally:
        env.close()
        if writer is not None:
            writer.close()

    final_checkpoint = run_dir / "checkpoint_final.pt"
    _save_checkpoint(final_checkpoint, policy, config, episodes, steps)
    _write_csv(run_dir / "episodes.csv", episode_records)
    _write_csv(run_dir / "updates.csv", update_records)
    convergence = _loss_convergence(update_records)
    summary = {
        "seed": config.seed,
        "episodes": episodes,
        "steps": steps,
        "checkpoint": str(final_checkpoint),
        "convergence": convergence,
        "last_episode": episode_records[-1] if episode_records else None,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _save_checkpoint(
    path: Path,
    policy: MAPPOPolicy,
    config: Phase2TrainingConfig,
    episodes: int,
    steps: int,
) -> None:
    torch.save(
        {
            "model_state": policy.state_dict(),
            "config": asdict(config),
            "actor_observation_dim": policy.actor.encoder[0].in_features,
            "episodes": episodes,
            "steps": steps,
        },
        path,
    )


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _loss_convergence(updates: Sequence[dict]) -> Dict[str, float | bool | None]:
    """Report stable loss only after enough updates to compare two windows."""
    values = np.asarray([entry["value_loss"] for entry in updates], dtype=np.float64)
    if len(values) < 20:
        return {"loss_stable": False, "recent_value_loss_std": None}
    window = max(5, len(values) // 10)
    recent = values[-window:]
    prior = values[-2 * window:-window]
    relative_shift = abs(recent.mean() - prior.mean()) / max(abs(prior.mean()), 1e-6)
    return {
        "loss_stable": bool(relative_shift <= 0.1),
        "recent_value_loss_std": float(recent.std()),
        "relative_value_loss_shift": float(relative_shift),
    }


def _checkpoint_if_due(
    episodes: int,
    next_checkpoint: int,
    run_dir: Path,
    policy: MAPPOPolicy,
    config: Phase2TrainingConfig,
    steps: int,
) -> int:
    if episodes < next_checkpoint:
        return next_checkpoint
    _save_checkpoint(
        run_dir / f"checkpoint_ep_{episodes:05d}.pt",
        policy,
        config,
        episodes,
        steps,
    )
    return next_checkpoint + config.checkpoint_interval


def load_policy(checkpoint_path: str | Path, device: str = "cpu"):
    """Load a final or intermediate Phase 2 policy checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    policy = MAPPOPolicy(
        checkpoint["actor_observation_dim"], ACTION_COUNT, device=device
    )
    policy.load_state_dict(checkpoint["model_state"])
    policy.eval()
    return policy, config, checkpoint


def evaluate_policy(
    policy: MAPPOPolicy,
    n_agents: int,
    max_steps: int,
    seeds: Iterable[int],
    episodes_per_seed: int,
) -> Dict[str, object]:
    """Evaluate one deterministic policy across independently seeded episodes."""
    env = Phase2Warehouse(n_agents=n_agents, max_steps=max_steps)
    per_seed: List[dict] = []
    try:
        for seed in seeds:
            records = []
            for offset in range(episodes_per_seed):
                observations = env.reset(seed=seed * 10_000 + offset)
                while True:
                    actions, _, _ = policy.act(observations, deterministic=True)
                    transition = env.step(actions)
                    observations = transition.observations
                    if (
                        transition.terminated
                        or transition.truncated
                        or transition.metrics.deadlocked
                    ):
                        records.append(transition.metrics.as_dict())
                        break
            per_seed.append(_aggregate_seed(seed, records))
    finally:
        env.close()

    completion_rates = np.asarray(
        [entry["task_completion_rate"] for entry in per_seed], dtype=np.float64
    )
    success_rates = np.asarray(
        [entry["success_rate"] for entry in per_seed], dtype=np.float64
    )
    collision_counts = np.asarray(
        [entry["mean_collisions"] for entry in per_seed], dtype=np.float64
    )
    deadlock_rates = np.asarray(
        [entry["deadlock_rate"] for entry in per_seed], dtype=np.float64
    )
    summary = {
        "seeds": per_seed,
        "task_completion_rate": float(completion_rates.mean()),
        "mean_collisions_per_episode": float(collision_counts.mean()),
        "deadlock_rate": float(deadlock_rates.mean()),
        "success_rate_mean": float(success_rates.mean()),
        "success_rate_std": float(success_rates.std()),
    }
    summary["go_no_go"] = {
        "task_completion_rate": summary["task_completion_rate"] >= 0.95,
        "collisions": summary["mean_collisions_per_episode"] <= 2.0,
        "deadlock_rate": summary["deadlock_rate"] <= 0.05,
        "success_rate_std": summary["success_rate_std"] <= 0.10,
    }
    summary["passed"] = all(summary["go_no_go"].values())
    return summary


def _aggregate_seed(seed: int, records: Sequence[dict]) -> Dict[str, float | int]:
    return {
        "seed": seed,
        "episodes": len(records),
        "task_completion_rate": float(
            np.mean([record["task_completion_rate"] for record in records])
        ),
        "mean_collisions": float(np.mean([record["collisions"] for record in records])),
        "deadlock_rate": float(np.mean([record["deadlocked"] for record in records])),
        "success_rate": float(np.mean([record["success"] for record in records])),
    }
