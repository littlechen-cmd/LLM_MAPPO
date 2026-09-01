"""Atomic, fail-closed checkpoints for E1 exploratory noisy-teacher runs."""

from pathlib import Path
import random
import csv
import json
from typing import Any, Mapping

import numpy as np
import torch


_SCHEMA = "e1-checkpoint-v1"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


class E1EvidenceWriter:
    """Compact formal artifact writer; full state/teacher arrays are prohibited."""

    _fields = {
        "updates.csv": ("real_env_steps", "policy_loss", "value_loss", "astar_loss",
                        "semantic_loss", "semantic_valid_denominator", "lambda_a", "lambda_l",
                        "num_env_workers", "rollout_length", "global_environment_steps",
                        "environment_steps_per_second", "rollout_wall_time", "policy_inference_time",
                        "ppo_update_time", "total_elapsed_time", "peak_cuda_memory_allocated",
                        "peak_cuda_memory_reserved"),
        "episodes.csv": ("real_env_steps", "completed_tasks", "reward", "collisions", "deadlocked"),
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

    def event(self, row: Mapping[str, Any]) -> None:
        forbidden = {"observations", "semantic_observations", "teacher_preferences", "action_masks"} & set(row)
        if forbidden: raise ValueError("E1 evidence event contains a forbidden full array.")
        encoded = json.dumps(dict(row), sort_keys=True)
        if len(encoded.encode("utf-8")) > 8192: raise ValueError("E1 evidence event is oversized.")
        with (self.directory / "teacher_events.jsonl").open("a", encoding="utf-8") as handle: handle.write(encoded + "\n")

    def complete(self, summary: Mapping[str, Any]) -> None:
        _atomic_json(self.directory / "summary.json", summary); _atomic_json(self.directory / "state.json", {"status": "complete"})

    def fail(self, reason: str) -> None: _atomic_json(self.directory / "state.json", {"status": "failed", "reason": reason})


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_e1_checkpoint(path: str | Path, *, identity: Mapping[str, Any], actor, critic,
                       optimizer, schedule_state: Mapping[str, Any],
                       calibration_state: Mapping[str, Any] | None,
                       trainer_state: Mapping[str, Any]) -> None:
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
               "trainer_state": dict(trainer_state), "rng": _rng_state()}
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
            "trainer_state": payload["trainer_state"]}
