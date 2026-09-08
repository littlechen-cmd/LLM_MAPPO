"""Owner-run training and evaluation for the teacher-free MAPPO-FP-DG baseline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from llm_mappo.mappo_baseline.fp_state import FPStateSchema, build_fp_state
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage
from llm_mappo.mappo_baseline.structured_observation import (
    StructuredObservationSchema,
    build_structured_observations,
)
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer
from llm_mappo.mappo_baseline.vector_env import (
    MAPPOVectorEnvironment,
    new_baseline_environment,
)


@dataclass(frozen=True)
class BaselineExperiment:
    schema: str
    method: str
    artifact_root: str
    environment: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    config_sha256: str


def load_baseline_experiment(path: str | Path) -> BaselineExperiment:
    source = Path(path)
    raw = source.read_bytes()
    values = yaml.safe_load(raw)
    required = {
        "schema", "method", "artifact_root", "environment", "training", "evaluation"
    }
    if set(values) != required:
        raise ValueError("MAPPO baseline configuration has unexpected top-level keys.")
    environment = dict(values["environment"])
    training = dict(values["training"])
    expected_environment = {
        "environment_id": "llm-mappo-medium-3ag-v1",
        "n_agents": 3,
        "max_steps": 1000,
        "dynamic_ingress_interval": 40,
        "batch_size_range": [1, 3],
        "queue_size": 4,
        "task_target": 9,
        "initial_priority_label": "A",
        "battery_cost_scale": 1.10,
        "charge_threshold": 0.30,
        "charge_release_threshold": 0.80,
        "deadlock_steps": 180,
        "observation_schema": "direct-goal-observation-v1",
        "goal_progress_reward": 0.10,
        "reward_version": "legacy-v1",
        "oracle_interaction_mask": True,
    }
    for name, expected in expected_environment.items():
        if environment.get(name) != expected:
            raise ValueError(f"MAPPO baseline environment field {name} is not frozen.")
    expected_training = {
        "environment_steps": 500_000,
        "num_env_workers": 16,
        "rollout_length": 64,
        "ppo_epochs": 5,
        "num_minibatches": 1,
    }
    for name, expected in expected_training.items():
        if training.get(name) != expected:
            raise ValueError(f"MAPPO baseline training field {name} is not frozen.")
    if values["schema"] != "r1-mappo-fp-dg-v1" or values["method"] != "MAPPO-FP-DG":
        raise ValueError("MAPPO baseline schema or method is incompatible.")
    return BaselineExperiment(
        schema=str(values["schema"]),
        method=str(values["method"]),
        artifact_root=str(values["artifact_root"]),
        environment=environment,
        training=training,
        evaluation=dict(values["evaluation"]),
        config_sha256=sha256(raw).hexdigest(),
    )


class BaselineRun:
    """Train, checkpoint and independently evaluate one frozen baseline seed."""

    UPDATE_FIELDS = (
        "global_environment_steps",
        "updates",
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "actor_grad_norm",
        "critic_grad_norm",
        "grad_norm",
        "actor_learning_rate",
        "critic_learning_rate",
        "learning_rate",
        "environment_steps_per_second",
        "rollout_seconds",
        "inference_seconds",
        "ppo_update_seconds",
        "total_elapsed_seconds",
        "peak_cuda_memory_allocated",
        "peak_cuda_memory_reserved",
    )
    EPISODE_FIELDS = (
        "global_environment_steps",
        "worker_index",
        "episode_index",
        "episode_seed",
        "completed_tasks",
        "created_tasks",
        "task_completion_target",
        "task_completion_rate",
        "reward",
        "collisions",
        "deadlocked",
        "agent_deaths",
        "picked_tasks",
        "blocked_forwards",
        "low_battery_triggers",
        "charging_target_steps",
        "charging_exposure_rate",
        "charger_arrivals",
        "charged_events",
        "charging_wait_steps",
        "task_recoveries",
        "energy_deaths",
        "minimum_battery",
        "steps",
        "success",
    )

    def __init__(
        self,
        experiment: BaselineExperiment,
        *,
        device: str,
        output_root: str | Path | None = None,
        resume: str | Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.device = torch.device(device)
        self.schema = FPStateSchema(
            int(experiment.environment["n_agents"]), n_shelves=144
        )
        self.actor_schema = StructuredObservationSchema()
        self.identity = {
            "schema": experiment.schema,
            "method": experiment.method,
            "code_commit": _commit(),
            "config_sha256": experiment.config_sha256,
            "seed": int(experiment.training["seed"]),
            "observation_schema": experiment.environment["observation_schema"],
            "actor_observation_schema": "structured-direct-goal-v1",
            "actor_observation_dimension": self.actor_schema.dimension,
            "fp_state_dimension": self.schema.dimension,
            "global_step_unit": "cumulative_joint_environment_transition",
        }
        if resume is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            root = Path(output_root or experiment.artifact_root)
            self.directory = root / f"seed_{self.identity['seed']:04d}_{timestamp}"
            self.directory.mkdir(parents=True, exist_ok=False)
            _write_json(self.directory / "run_manifest.json", {
                "identity": self.identity,
                "environment": dict(experiment.environment),
                "training": dict(experiment.training),
                "evaluation": dict(experiment.evaluation),
                "teacher_modules_enabled": False,
            })
            _write_json(self.directory / "state.json", {"status": "running"})
            self._create_csv(self.directory / "updates.csv", self.UPDATE_FIELDS)
            self._create_csv(self.directory / "episodes.csv", self.EPISODE_FIELDS)
        else:
            self.directory = Path(resume)
            manifest = json.loads(
                (self.directory / "run_manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("identity") != self.identity:
                raise ValueError("MAPPO baseline resume identity differs.")
            state = json.loads(
                (self.directory / "state.json").read_text(encoding="utf-8")
            )
            if state.get("status") != "running":
                raise ValueError("Only a running MAPPO baseline artifact can resume.")
        config = MAPPOConfig(
            observation_dim=self.actor_schema.dimension,
            state_dim=self.schema.dimension,
            action_dim=5,
            hidden_dim=int(experiment.training["hidden_dimension"]),
            actor_lr=float(experiment.training["actor_learning_rate"]),
            critic_lr=float(experiment.training["critic_learning_rate"]),
            clip_param=float(experiment.training["clip_param"]),
            ppo_epochs=int(experiment.training["ppo_epochs"]),
            num_minibatches=int(experiment.training["num_minibatches"]),
            entropy_coef=float(experiment.training["entropy_coefficient"]),
            value_loss_coef=float(experiment.training["value_loss_coefficient"]),
            max_grad_norm=float(experiment.training["max_gradient_norm"]),
            huber_delta=float(experiment.training["huber_delta"]),
        )
        self._seed(int(experiment.training["seed"]))
        self.trainer = MAPPOTrainer(config, self.device)
        self.pool = MAPPOVectorEnvironment(
            experiment.environment,
            int(experiment.training["num_env_workers"]),
            int(experiment.training["seed"]),
        )
        self.global_steps = 0
        self.updates = 0
        hidden_shape = (
            int(experiment.training["num_env_workers"]),
            self.schema.n_agents,
            config.hidden_dim,
        )
        self.actor_hidden_states = np.zeros(hidden_shape, dtype=np.float32)
        self.critic_hidden_states = np.zeros(hidden_shape, dtype=np.float32)
        self.started_at = time.perf_counter()
        self.elapsed_offset = 0.0
        self.inference_seconds = 0.0
        self.rollout_seconds = 0.0
        self.update_seconds = 0.0
        self.planner_query_count = 0
        self._board = self._tensorboard()
        if resume is not None:
            self._restore(self.directory / "checkpoint_latest.pt")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def train(self) -> dict[str, Any]:  # noqa: C901
        budget = int(self.experiment.training["environment_steps"])
        workers = int(self.experiment.training["num_env_workers"])
        rollout_length = int(self.experiment.training["rollout_length"])
        checkpoint_interval = int(
            self.experiment.training["checkpoint_interval_steps"]
        )
        last_checkpoint_bucket = self.global_steps // checkpoint_interval
        try:
            while self.global_steps < budget:
                steps_this_rollout = min(
                    rollout_length, (budget - self.global_steps) // workers
                )
                if steps_this_rollout < 1:
                    raise RuntimeError("Environment budget is not divisible by workers.")
                storage = MAPPORolloutStorage(
                    steps_this_rollout,
                    workers,
                    self.schema.n_agents,
                    self.actor_schema.dimension,
                    self.schema.dimension,
                    5,
                    self.trainer.config.hidden_dim,
                )
                episodes = []
                rollout_started = time.perf_counter()
                for _ in range(steps_this_rollout):
                    observations = self.pool.observations.copy()
                    fp_states = self.pool.fp_states.copy()
                    action_masks = self.pool.action_masks.copy()
                    active_masks = self.pool.active_masks.copy()
                    inference_started = time.perf_counter()
                    (
                        actions,
                        log_probs,
                        values,
                        value_predictions,
                        next_actor_hidden,
                        next_critic_hidden,
                    ) = self.trainer.act(
                        observations,
                        fp_states,
                        action_masks,
                        self.actor_hidden_states,
                        self.critic_hidden_states,
                        deterministic=False,
                    )
                    self.inference_seconds += time.perf_counter() - inference_started
                    response = self.pool.step(actions, global_step=self.global_steps)
                    self.global_steps += workers
                    self.planner_query_count = max(
                        self.planner_query_count, response.planner_query_count
                    )
                    continuation = (
                        (~response.dones).astype(np.float32)[..., None]
                        * response.active_masks[..., None]
                    )
                    episodes.extend(response.episodes)
                    storage.insert(
                        observations=observations,
                        fp_states=fp_states,
                        actions=actions,
                        action_masks=action_masks,
                        log_probs=log_probs,
                        values=values,
                        value_predictions=value_predictions,
                        rewards=response.rewards,
                        dones=response.dones,
                        active_masks=active_masks,
                        actor_hidden_states=self.actor_hidden_states,
                        critic_hidden_states=self.critic_hidden_states,
                    )
                    self.actor_hidden_states = next_actor_hidden * continuation
                    self.critic_hidden_states = next_critic_hidden * continuation
                self.rollout_seconds += time.perf_counter() - rollout_started
                next_values, _ = self.trainer.values(
                    self.pool.fp_states, self.critic_hidden_states
                )
                batch = storage.compute_returns(
                    next_values,
                    gamma=float(self.experiment.training["gamma"]),
                    gae_lambda=float(self.experiment.training["gae_lambda"]),
                )
                self.trainer.linear_lr_decay(self.global_steps, budget)
                update_started = time.perf_counter()
                metrics = self.trainer.update(batch)
                self.update_seconds += time.perf_counter() - update_started
                self.updates += 1
                row = self._update_row(metrics)
                self._append_csv(self.directory / "updates.csv", row, self.UPDATE_FIELDS)
                for episode in episodes:
                    self._append_csv(
                        self.directory / "episodes.csv", episode, self.EPISODE_FIELDS
                    )
                    self._write_episode_board(episode)
                self._write_update_board(row)
                current_bucket = self.global_steps // checkpoint_interval
                if current_bucket > last_checkpoint_bucket:
                    self._checkpoint(self.directory / "checkpoint_latest.pt")
                    last_checkpoint_bucket = current_bucket
            if self.planner_query_count != 0:
                raise RuntimeError("DirectGoal baseline observed a planner query.")
            self._checkpoint(self.directory / "checkpoint_final.pt")
            evaluation = self.evaluate()
            summary = self._summary(evaluation)
            _write_json(self.directory / "summary.json", summary)
            _write_json(self.directory / "state.json", {"status": "complete"})
            return summary
        except BaseException as error:
            _write_json(self.directory / "state.json", {
                "status": "failed", "reason": type(error).__name__
            })
            raise
        finally:
            self.close()

    def evaluate(self) -> dict[str, Any]:
        results = []
        self.trainer.actor.eval()
        for seed in self.experiment.evaluation["seeds"]:
            results.append(self._evaluation_episode(int(seed)))
        mean_completion = float(np.mean([
            row["task_completion_rate"] for row in results
        ]))
        replay_root = self.directory / "evaluation" / "replays"
        replay_root.mkdir(parents=True, exist_ok=True)
        fixed_seed = int(self.experiment.evaluation["seeds"][0])
        lowest_seed = int(min(
            results, key=lambda row: (row["task_completion_rate"], row["seed"])
        )["seed"])
        fixed_path = replay_root / f"fixed_seed_{fixed_seed}.gif"
        lowest_path = replay_root / f"lowest_seed_{lowest_seed}.gif"
        self._evaluation_episode(fixed_seed, replay_path=fixed_path)
        self._evaluation_episode(lowest_seed, replay_path=lowest_path)
        result = {
            "schema": "mappo-fp-dg-evaluation-v1",
            "policy": "deterministic_argmax",
            "checkpoint": "checkpoint_final.pt",
            "episodes": results,
            "mean_task_completion_rate": mean_completion,
            "mean_completed_tasks": float(np.mean([
                row["completed_tasks"] for row in results
            ])),
            "gate_target": 0.90,
            "gate_pass": mean_completion >= 0.90,
            "fixed_replay": str(fixed_path.relative_to(self.directory)),
            "lowest_completion_replay": str(
                lowest_path.relative_to(self.directory)
            ),
            "metrics_plot": "evaluation/evaluation_metrics.png",
        }
        output = self.directory / "evaluation"
        output.mkdir(exist_ok=True)
        _write_evaluation_plot(output / "evaluation_metrics.png", results)
        _write_json(output / "evaluation.json", result)
        return result

    def _evaluation_episode(
        self, seed: int, replay_path: Path | None = None
    ) -> dict[str, Any]:
        environment = new_baseline_environment(self.experiment.environment)
        frames = []
        try:
            environment.reset(seed=seed)
            actor_hidden = np.zeros((1, environment.n_agents, 64), dtype=np.float32)
            critic_hidden = np.zeros((1, environment.n_agents, 64), dtype=np.float32)
            transition = None
            stride = int(self.experiment.evaluation["replay_frame_stride"])
            for step in range(int(self.experiment.environment["max_steps"])):
                if replay_path is not None and step % stride == 0:
                    from llm_mappo.visualization import render_warehouse_frame

                    frames.append(render_warehouse_frame(environment.env, cell_size=16))
                observations = build_structured_observations(
                    environment, self.actor_schema
                )
                states = build_fp_state(environment, self.schema)
                masks = environment.action_masks()
                actions, _, _, _, next_actor, next_critic = self.trainer.act(
                    observations[None],
                    states[None],
                    masks[None],
                    actor_hidden,
                    critic_hidden,
                    deterministic=True,
                )
                transition = environment.step(actions[0])
                continuation = 0.0 if (
                    transition.terminated or transition.truncated or transition.metrics.deadlocked
                ) else 1.0
                actor_hidden = next_actor * continuation
                critic_hidden = next_critic * continuation
                if (
                    transition.terminated
                    or transition.truncated
                    or transition.metrics.deadlocked
                ):
                    break
            if transition is None:
                raise RuntimeError(
                    "MAPPO evaluation did not execute an environment step."
                )
            if replay_path is not None:
                from llm_mappo.visualization import render_warehouse_frame
                from PIL import Image

                frames.append(render_warehouse_frame(environment.env, cell_size=16))
                images = [Image.fromarray(frame) for frame in frames]
                images[0].save(
                    replay_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(
                        self.experiment.evaluation["replay_frame_duration_ms"]
                    ),
                    loop=0,
                    optimize=False,
                )
            return {"seed": seed, **transition.metrics.as_dict()}
        finally:
            environment.close()

    def _checkpoint(self, path: Path) -> None:
        payload = {
            "schema": "mappo-fp-dg-checkpoint-v2",
            "identity": self.identity,
            "trainer": self.trainer.state_dict(),
            "global_environment_steps": self.global_steps,
            "updates": self.updates,
            "planner_query_count": self.planner_query_count,
            "worker_states": self.pool.snapshot(),
            "recurrent_states": {
                "actor": self.actor_hidden_states,
                "critic": self.critic_hidden_states,
            },
            "timing": {
                "rollout_seconds": self.rollout_seconds,
                "inference_seconds": self.inference_seconds,
                "ppo_update_seconds": self.update_seconds,
                "total_elapsed_seconds": self._elapsed(),
            },
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None,
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    def _restore(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "mappo-fp-dg-checkpoint-v2":
            raise ValueError("MAPPO baseline checkpoint schema differs.")
        if payload.get("identity") != self.identity:
            raise ValueError("MAPPO baseline checkpoint identity differs.")
        self.trainer.load_state_dict(payload["trainer"])
        self.pool.restore(payload["worker_states"])
        recurrent = payload["recurrent_states"]
        self.actor_hidden_states = np.asarray(recurrent["actor"], dtype=np.float32)
        self.critic_hidden_states = np.asarray(recurrent["critic"], dtype=np.float32)
        self.global_steps = int(payload["global_environment_steps"])
        self.updates = int(payload["updates"])
        self.planner_query_count = int(payload["planner_query_count"])
        timing = payload["timing"]
        self.rollout_seconds = float(timing["rollout_seconds"])
        self.inference_seconds = float(timing["inference_seconds"])
        self.update_seconds = float(timing["ppo_update_seconds"])
        self.elapsed_offset = float(timing["total_elapsed_seconds"])
        self.started_at = time.perf_counter()
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch_cpu"])
        if torch.cuda.is_available() and payload["rng"]["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["torch_cuda"])
        self._truncate_evidence()

    def _update_row(self, metrics) -> dict[str, Any]:
        elapsed = self._elapsed()
        peak_allocated = 0
        peak_reserved = 0
        if self.device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
        return {
            "global_environment_steps": self.global_steps,
            "updates": self.updates,
            **{name: metrics[name] for name in (
                "policy_loss", "value_loss", "entropy", "total_loss", "approx_kl",
                "clip_fraction", "explained_variance", "actor_grad_norm",
                "critic_grad_norm", "actor_learning_rate", "critic_learning_rate",
            )},
            "grad_norm": max(
                float(metrics["actor_grad_norm"]),
                float(metrics["critic_grad_norm"]),
            ),
            "learning_rate": float(metrics["actor_learning_rate"]),
            "environment_steps_per_second": self.global_steps / max(elapsed, 1e-9),
            "rollout_seconds": self.rollout_seconds,
            "inference_seconds": self.inference_seconds,
            "ppo_update_seconds": self.update_seconds,
            "total_elapsed_seconds": elapsed,
            "peak_cuda_memory_allocated": peak_allocated,
            "peak_cuda_memory_reserved": peak_reserved,
        }

    def _summary(self, evaluation) -> dict[str, Any]:
        episodes = self._read_csv(self.directory / "episodes.csv")
        window = episodes[-100:]
        return {
            "identity": self.identity,
            "global_environment_steps": self.global_steps,
            "updates": self.updates,
            "completed_training_episodes": len(episodes),
            "planner_query_count": self.planner_query_count,
            "training_episode_window": _episode_window(window),
            "evaluation": evaluation,
            "total_elapsed_seconds": self._elapsed(),
        }

    def _tensorboard(self):
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(self.directory / "tensorboard"))

    def _write_update_board(self, row) -> None:
        step = int(row["global_environment_steps"])
        for name in (
            "policy_loss", "value_loss", "entropy", "total_loss", "approx_kl",
            "clip_fraction", "explained_variance", "actor_grad_norm",
            "critic_grad_norm", "grad_norm", "actor_learning_rate",
            "critic_learning_rate", "learning_rate",
        ):
            self._board.add_scalar(f"train/{name}", float(row[name]), step)
        for name in (
            "environment_steps_per_second", "rollout_seconds", "inference_seconds",
            "ppo_update_seconds",
        ):
            self._board.add_scalar(f"performance/{name}", float(row[name]), step)
        self._board.add_scalar(
            "performance/gpu_memory_mb",
            float(row["peak_cuda_memory_allocated"]) / (1024.0 * 1024.0),
            step,
        )
        self._board.flush()

    def _write_episode_board(self, row) -> None:
        step = int(row["global_environment_steps"])
        mapping = {
            "team_reward": "reward",
            "completed_tasks": "completed_tasks",
            "task_completion_rate": "task_completion_rate",
            "episode_length": "steps",
            "collisions": "collisions",
            "blocked_forward": "blocked_forwards",
            "deadlocked": "deadlocked",
            "energy_deaths": "energy_deaths",
            "charging_exposure_rate": "charging_exposure_rate",
        }
        for tag, name in mapping.items():
            self._board.add_scalar(f"episode/{tag}", float(row[name]), step)
        rate = 1000.0 * float(row["completed_tasks"]) / max(float(row["steps"]), 1.0)
        self._board.add_scalar("episode/tasks_per_1000_steps", rate, step)
        self._board.flush()

    def close(self) -> None:
        if getattr(self, "_board", None) is not None:
            self._board.close()
            self._board = None
        if getattr(self, "pool", None) is not None:
            self.pool.close()
            self.pool = None

    def _elapsed(self) -> float:
        return self.elapsed_offset + time.perf_counter() - self.started_at

    def _truncate_evidence(self) -> None:
        """Drop rows written after the latest durable checkpoint boundary."""

        for name, fields in (
            ("updates.csv", self.UPDATE_FIELDS),
            ("episodes.csv", self.EPISODE_FIELDS),
        ):
            path = self.directory / name
            rows = self._read_csv(path)
            kept = [
                row for row in rows
                if int(row["global_environment_steps"]) <= self.global_steps
            ]
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)
            temporary.replace(path)

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _create_csv(path: Path, fields) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    @staticmethod
    def _append_csv(path: Path, row, fields) -> None:
        if set(row) != set(fields):
            raise ValueError(f"Evidence row for {path.name} has incompatible fields.")
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
            handle.flush()

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def _episode_window(rows) -> dict[str, float | int]:
    if not rows:
        return {"window_size": 0}
    return {
        "window_size": len(rows),
        "mean_task_completion_rate": float(np.mean([
            float(row["task_completion_rate"]) for row in rows
        ])),
        "mean_completed_tasks": float(np.mean([
            float(row["completed_tasks"]) for row in rows
        ])),
        "mean_team_reward": float(np.mean([float(row["reward"]) for row in rows])),
    }


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _write_evaluation_plot(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Create a compact fixed-seed completion chart without GUI dependencies."""

    from PIL import Image, ImageDraw

    width, height = 920, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 16), "MAPPO-FP-DG deterministic evaluation", fill="black")
    left, top, right, bottom = 60, 60, 890, 310
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    target_y = bottom - int(0.90 * (bottom - top))
    draw.line((left, target_y, right, target_y), fill="#d62728", width=2)
    draw.text((right - 95, target_y - 18), "90% Gate", fill="#d62728")
    bar_width = max((right - left) // max(len(rows), 1) - 18, 12)
    for index, row in enumerate(rows):
        value = float(row["task_completion_rate"])
        x0 = left + 12 + index * ((right - left) // len(rows))
        x1 = x0 + bar_width
        y0 = bottom - int(min(max(value, 0.0), 1.0) * (bottom - top))
        color = "#2ca02c" if value >= 0.90 else "#1f77b4"
        draw.rectangle((x0, y0, x1, bottom), fill=color)
        draw.text((x0, bottom + 8), str(row["seed"]), fill="black")
        draw.text((x0, max(y0 - 18, top)), f"{value:.0%}", fill="black")
    image.save(path)
