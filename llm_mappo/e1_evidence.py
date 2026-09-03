"""Atomic, fail-closed checkpoints for E1 exploratory noisy-teacher runs."""

from pathlib import Path
import random
import csv
import json
import os
from typing import Any, Mapping

import numpy as np
import torch


_SCHEMA = "e1-checkpoint-v2"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


class E1EvidenceWriter:
    """Compact formal artifact writer; full state/teacher arrays are prohibited."""

    _fields = {
        "updates.csv": ("real_env_steps", "policy_loss", "value_loss", "entropy", "total_loss",
                        "approx_kl", "clip_fraction", "explained_variance", "grad_norm",
                        "learning_rate", "astar_loss", "astar_valid_rate", "lambda_a",
                        "calibration_sample_rate", "delta_g_mean", "delta_g_positive_rate",
                        "rc_confidence_mean", "semantic_loss", "semantic_valid_rate",
                        "semantic_reliability_mean", "lambda_l", "semantic_valid_denominator",
                        "num_env_workers", "rollout_length", "global_environment_steps",
                        "environment_steps_per_second", "rollout_wall_time", "policy_inference_time",
                        "ppo_update_time", "total_elapsed_time", "peak_cuda_memory_allocated",
                        "peak_cuda_memory_reserved"),
        "episodes.csv": (
            "real_env_steps", "worker_index", "episode_index", "episode_seed",
            "completed_tasks", "created_tasks", "task_completion_target",
            "task_completion_rate", "reward", "collisions", "deadlocked",
            "agent_deaths", "picked_tasks", "blocked_forwards",
            "low_battery_triggers", "charging_target_steps",
            "charging_exposure_rate", "charger_arrivals", "charged_events",
            "charging_wait_steps", "task_recoveries", "energy_deaths",
            "minimum_battery", "steps", "success",
        ),
        "teacher_step_counts.csv": ("real_env_steps", "teacher_queries", "shadow_calls", "ema_updates",
                                    "semantic_valid_slots", "semantic_total_slots", "planner_query_count"),
    }

    @classmethod
    def create(cls, directory: str | Path, manifest: Mapping[str, Any]):
        path = Path(directory)
        if path.exists():
            raise FileExistsError("E1 artifact directory already exists.")
        path.mkdir(parents=True)
        _atomic_json(path / "run_manifest.json", manifest); _atomic_json(path / "state.json", {"status": "running"})
        for name, fields in cls._fields.items():
            with (path / name).open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
        (path / "teacher_events.jsonl").touch(); (path / "resource_windows.csv").write_text("real_env_steps,rss_bytes,cuda_reserved_bytes\n", encoding="utf-8")
        return cls(path)

    @classmethod
    def open_existing(cls, directory: str | Path):
        path = Path(directory)
        if json.loads((path / "state.json").read_text(encoding="utf-8")).get("status") != "running":
            raise ValueError("Only running E1 artifacts can resume.")
        return cls(path)

    def __init__(self, directory: Path): self.directory = directory

    def append(self, filename: str, row: Mapping[str, Any]) -> None:
        fields = self._fields[filename]
        if set(row) - set(fields): raise ValueError("E1 evidence row contains an unregistered field.")
        with (self.directory / filename).open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    def _rows(self, filename: str) -> list[dict[str, str]]:
        with (self.directory / filename).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _append_once(
        self,
        filename: str,
        row: Mapping[str, Any],
        *,
        identity_fields: tuple[str, ...],
    ) -> bool:
        fields = self._fields[filename]
        if set(row) != set(fields):
            raise ValueError(f"E1 {filename} row does not match the frozen schema.")
        encoded = {name: str(row[name]) for name in fields}
        existing = self._rows(filename)
        identity = tuple(encoded[name] for name in identity_fields)
        for current in existing:
            if tuple(current[name] for name in identity_fields) != identity:
                continue
            if current != encoded:
                raise ValueError(f"E1 {filename} evidence conflicts with its checkpoint.")
            return False
        if existing and int(existing[-1]["real_env_steps"]) > int(row["real_env_steps"]):
            raise ValueError(f"E1 {filename} evidence is out of order.")
        self.append(filename, row)
        return True

    def reconcile_checkpoint_evidence(
        self, evidence: Mapping[str, Any]
    ) -> dict[str, int | bool]:
        """Idempotently materialize the evidence batch stored in a checkpoint."""
        if set(evidence) != {"update_row", "episode_rows"}:
            raise ValueError("E1 checkpoint evidence state is incompatible.")
        update_appended = self._append_once(
            "updates.csv",
            evidence["update_row"],
            identity_fields=("real_env_steps",),
        )
        episodes_appended = 0
        for row in evidence["episode_rows"]:
            episodes_appended += int(self._append_once(
                "episodes.csv",
                row,
                identity_fields=("worker_index", "episode_index", "episode_seed"),
            ))
        return {
            "update_appended": update_appended,
            "episodes_appended": episodes_appended,
        }

    def event(self, row: Mapping[str, Any]) -> None:
        forbidden = {"observations", "semantic_observations", "teacher_preferences", "action_masks"} & set(row)
        if forbidden: raise ValueError("E1 evidence event contains a forbidden full array.")
        encoded = json.dumps(dict(row), sort_keys=True)
        if len(encoded.encode("utf-8")) > 8192: raise ValueError("E1 evidence event is oversized.")
        with (self.directory / "teacher_events.jsonl").open("a", encoding="utf-8") as handle: handle.write(encoded + "\n")

    def complete(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        episodes = [_typed_episode(row) for row in self._rows("episodes.csv")]
        manifest = json.loads(
            (self.directory / "run_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("requires_completed_episodes") and not episodes:
            raise ValueError("E1 run contains no completed episode evidence.")
        final = dict(summary)
        if episodes:
            window = episodes[-100:]
            final["completed_episodes"] = len(episodes)
            final["latest_episode_metrics"] = episodes[-1]
            final["episode_window"] = {
                "window_size": len(window),
                "mean_task_completion_rate": _mean(window, "task_completion_rate"),
                "mean_completed_tasks": _mean(window, "completed_tasks"),
                "mean_reward": _mean(window, "reward"),
                "mean_collisions": _mean(window, "collisions"),
                "deadlock_rate": _mean(window, "deadlocked"),
                "success_rate": _mean(window, "success"),
            }
        _atomic_json(self.directory / "summary.json", final)
        _atomic_json(self.directory / "state.json", {"status": "complete"})
        return final

    def fail(self, reason: str) -> None: _atomic_json(self.directory / "state.json", {"status": "failed", "reason": reason})


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


_EPISODE_INTEGER_FIELDS = {
    "real_env_steps", "worker_index", "episode_index", "episode_seed",
    "completed_tasks", "created_tasks", "task_completion_target", "collisions",
    "agent_deaths", "picked_tasks", "blocked_forwards", "low_battery_triggers",
    "charging_target_steps", "charger_arrivals", "charged_events",
    "charging_wait_steps", "task_recoveries", "energy_deaths", "steps",
}
_EPISODE_BOOLEAN_FIELDS = {"deadlocked", "success"}


def _typed_episode(row: Mapping[str, str]) -> dict[str, Any]:
    typed: dict[str, Any] = {}
    for name, value in row.items():
        if name in _EPISODE_INTEGER_FIELDS:
            typed[name] = int(value)
        elif name in _EPISODE_BOOLEAN_FIELDS:
            if value not in {"True", "False"}:
                raise ValueError("E1 episode evidence contains an invalid boolean.")
            typed[name] = value == "True"
        else:
            typed[name] = float(value)
    return typed


def _mean(rows: list[Mapping[str, Any]], name: str) -> float:
    return float(sum(float(row[name]) for row in rows) / len(rows))


class E1TensorBoardWriter:
    """R1-A scalar view keyed only by cumulative environment transitions."""

    def __init__(self, directory: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=str(directory))

    def add_update(self, row: Mapping[str, Any]) -> None:
        step = int(row["real_env_steps"])
        tags = {
            "train/policy_loss": "policy_loss",
            "train/value_loss": "value_loss",
            "train/entropy": "entropy",
            "train/total_loss": "total_loss",
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/explained_variance": "explained_variance",
            "train/grad_norm": "grad_norm",
            "train/learning_rate": "learning_rate",
            "teacher/astar_loss": "astar_loss",
            "teacher/astar_valid_rate": "astar_valid_rate",
            "teacher/astar_lambda": "lambda_a",
            "teacher/calibration_sample_rate": "calibration_sample_rate",
            "teacher/delta_g_mean": "delta_g_mean",
            "teacher/delta_g_positive_rate": "delta_g_positive_rate",
            "teacher/rc_confidence_mean": "rc_confidence_mean",
            "teacher/semantic_loss": "semantic_loss",
            "teacher/semantic_valid_rate": "semantic_valid_rate",
            "teacher/semantic_reliability_mean": "semantic_reliability_mean",
            "teacher/semantic_lambda": "lambda_l",
            "train/astar_loss": "astar_loss",
            "train/semantic_loss": "semantic_loss",
            "performance/environment_steps_per_second": "environment_steps_per_second",
            "performance/rollout_seconds": "rollout_wall_time",
            "performance/inference_seconds": "policy_inference_time",
            "performance/ppo_update_seconds": "ppo_update_time",
            "performance/rollout_wall_time": "rollout_wall_time",
            "performance/policy_inference_time": "policy_inference_time",
            "performance/ppo_update_time": "ppo_update_time",
        }
        for tag, name in tags.items():
            self._writer.add_scalar(tag, float(row[name]), step)
        self._writer.add_scalar(
            "performance/gpu_memory_mb",
            float(row["peak_cuda_memory_allocated"]) / (1024.0 * 1024.0),
            step,
        )
        self._writer.flush()

    def add_episode(self, row: Mapping[str, Any]) -> None:
        step = int(row["real_env_steps"])
        tags = {
            "episode/task_completion_rate": "task_completion_rate",
            "episode/completed_tasks": "completed_tasks",
            "episode/team_reward": "reward",
            "episode/collisions": "collisions",
            "episode/deadlocked": "deadlocked",
            "episode/charging_exposure_rate": "charging_exposure_rate",
            "episode/tasks_per_1000_steps": None,
            "episode/episode_length": "steps",
            "episode/blocked_forward": "blocked_forwards",
            "episode/energy_deaths": "energy_deaths",
        }
        for tag, name in tags.items():
            value = (1000.0 * float(row["completed_tasks"]) / max(float(row["steps"]), 1.0)
                     if name is None else float(row[name]))
            self._writer.add_scalar(tag, value, step)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def save_e1_checkpoint(path: str | Path, *, identity: Mapping[str, Any], actor, critic,
                       optimizer, schedule_state: Mapping[str, Any],
                       calibration_state: Mapping[str, Any] | None,
                       trainer_state: Mapping[str, Any],
                       evidence_state: Mapping[str, Any] | None = None) -> None:
    """Write a complete checkpoint through a same-directory temporary file."""
    if not identity.get("raw_records_sha256"):
        raise ValueError("E1 checkpoint requires raw evidence identity.")
    if schedule_state.get("schedule_version") != "linear-env-step-v1":
        raise ValueError("E1 checkpoint requires the frozen schedule.")
    if trainer_state.get("schema") not in {"e1-runtime-v1", "e1-runtime-v2"}:
        raise ValueError("E1 checkpoint requires a resumable runtime state.")
    destination = Path(path)
    payload = {"schema": _SCHEMA, "identity": dict(identity), "actor": actor.state_dict(),
               "critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
               "schedule": dict(schedule_state), "calibration": None if calibration_state is None else dict(calibration_state),
               "trainer_state": dict(trainer_state),
               "evidence_state": None if evidence_state is None else dict(evidence_state),
               "rng": _rng_state()}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_e1_checkpoint(path: str | Path, *, expected_identity: Mapping[str, Any], actor,
                       critic, optimizer) -> dict[str, Any]:
    """Restore only an exact formal identity with all optimizer/RNG state."""
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("E1 checkpoint is unreadable.") from error
    if payload.get("schema") != _SCHEMA or payload.get("identity") != dict(expected_identity):
        raise ValueError("E1 checkpoint identity is incompatible.")
    rng = payload.get("rng")
    if not isinstance(rng, Mapping) or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("E1 checkpoint RNG state is incomplete.")
    if not expected_identity.get("raw_records_sha256"):
        raise ValueError("E1 expected identity lacks raw evidence hash.")
    if payload.get("schedule", {}).get("schedule_version") != "linear-env-step-v1":
        raise ValueError("E1 checkpoint schedule is incompatible.")
    if payload.get("trainer_state", {}).get("schema") not in {"e1-runtime-v1", "e1-runtime-v2"}:
        raise ValueError("E1 checkpoint trainer state is incompatible.")
    try:
        actor.load_state_dict(payload["actor"]); critic.load_state_dict(payload["critic"])
        optimizer.load_state_dict(payload["optimizer"])
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("E1 checkpoint training state is incomplete.") from error
    return {"schedule_state": payload["schedule"], "calibration_state": payload["calibration"],
            "trainer_state": payload["trainer_state"],
            "evidence_state": payload.get("evidence_state")}
