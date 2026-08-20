"""Train the preregistered QMIX-WP baseline under the frozen G3 contract."""

from __future__ import annotations

from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from llm_mappo.phase3_training import (
    Phase3TrainingConfig,
    _make_training_env,
    _resolve_device,
    _set_seed,
)
from llm_mappo.qmix import QMIXHyperparameters, QMIXLearner


def _sample_batch(replay: list[dict], size: int, rng: np.random.Generator) -> dict:
    indices = rng.choice(len(replay), size=size, replace=False)
    return {
        key: np.stack([replay[index][key] for index in indices])
        for key in replay[0]
    }


def train_qmix(config_path: str | Path, seed: int | None = None) -> dict:
    """Run QMIX with exactly the configured environment-step cap."""
    source = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    base = Phase3TrainingConfig.from_yaml(config_path)
    training = source["training"]
    qmix_values = source.get("qmix", {})
    qmix = QMIXHyperparameters(
        gamma=qmix_values.get("gamma", 0.99),
        learning_rate=qmix_values.get("learning_rate", 3e-4),
        target_update_interval=qmix_values.get("target_update_interval", 1000),
    )
    run_seed = int(training["seed"] if seed is None else seed)
    _set_seed(run_seed, int(training.get("torch_num_threads", 1)))
    device = _resolve_device(str(training.get("device", "cpu")))
    output = Path(training["output_dir"]) / f"seed_{run_seed:03d}"
    output.mkdir(parents=True, exist_ok=True)
    env = _make_training_env(base)
    rng = np.random.default_rng(run_seed)
    replay: list[dict] = []
    budget = base.environment_step_budget
    if budget is None:
        raise ValueError("QMIX comparison requires environment_step_budget.")
    observations = env.reset(seed=run_seed)
    learner = QMIXLearner(
        env.actor_observation_dim, env.action_masks().shape[1], base.n_agents,
        qmix, device,
    )
    steps, episodes, team_reward, completed = 0, 0, 0.0, 0
    losses: list[float] = []
    episode_rows: list[dict] = []
    try:
        while steps < budget:
            fraction = min(1.0, steps / max(1, qmix_values["epsilon_decay_steps"]))
            epsilon = qmix_values["epsilon_start"] + fraction * (
                qmix_values["epsilon_final"] - qmix_values["epsilon_start"]
            )
            masks = env.action_masks()
            actions = learner.select_actions(observations, masks, epsilon, rng)
            transition = env.step(actions)
            next_masks = env.action_masks()
            done = (
                transition.terminated or transition.truncated
                or transition.metrics.deadlocked
            )
            replay.append({
                "observations": observations.astype(np.float32), "actions": actions,
                "rewards": np.float32(transition.team_reward),
                "next_observations": transition.observations.astype(np.float32),
                "dones": np.float32(done), "masks": masks, "next_masks": next_masks,
            })
            if len(replay) > qmix_values["replay_capacity"]:
                replay.pop(0)
            steps += 1
            team_reward += transition.team_reward
            observations = transition.observations
            if len(replay) >= qmix_values["learning_starts"]:
                batch = _sample_batch(replay, qmix_values["batch_size"], rng)
                losses.append(learner.update(batch))
            if steps % qmix.target_update_interval == 0:
                learner.update_targets()
            if done:
                episodes += 1
                completed += transition.metrics.completed_tasks
                episode_rows.append({
                    "episode": episodes,
                    "environment_steps": steps,
                    "completed_tasks": transition.metrics.completed_tasks,
                    "steps": transition.metrics.steps,
                    "team_reward": transition.metrics.reward,
                })
                observations = env.reset(seed=run_seed + episodes)
    finally:
        env.close()
    torch.save(learner.checkpoint(), output / "checkpoint_final.pt")
    summary = {
        "algorithm": "QMIX-WP", "seed": run_seed, "environment_steps": steps,
        "episodes": episodes, "team_reward": team_reward,
        "completed_tasks": completed,
        "mean_qmix_loss": float(np.mean(losses)) if losses else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not episode_rows:
        episode_rows.append({
            "episode": 0, "environment_steps": steps, "completed_tasks": 0,
            "steps": max(1, steps), "team_reward": team_reward,
        })
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)
    return summary


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(train_qmix(args.config, args.seed), indent=2))


if __name__ == "__main__":
    main()
