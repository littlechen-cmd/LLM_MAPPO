"""Compact, atomic and resumable evidence primitives for O2 runs."""

import csv
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch


_FORBIDDEN_EVENT_FIELDS = {
    "action_masks",
    "astar_preferences",
    "observations",
    "semantic_observations",
    "states",
    "teacher_arrays",
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


class O2EvidenceWriter:
    """Write only the compact, pre-registered O2 evidence schema."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._closed = False

    @classmethod
    def create(
        cls, directory: str | Path, run_manifest: Mapping[str, Any]
    ) -> "O2EvidenceWriter":
        path = Path(directory)
        if path.exists():
            raise FileExistsError(
                "O2 run directory already exists and cannot overwrite."
            )
        path.mkdir(parents=True)
        _atomic_json(path / "run_manifest.json", run_manifest)
        _atomic_json(path / "state.json", {"status": "running"})
        for filename, fields in {
            "teacher_step_counts.csv": (
                "real_env_steps", "teacher_queries", "shadow_calls", "ema_updates"
            ),
            "updates.csv": (
                "real_env_steps", "policy_loss", "value_loss", "astar_loss"
            ),
            "episodes.csv": (
                "real_env_steps", "cumulative_completed_tasks",
                "cumulative_episode_steps"
            ),
        }.items():
            with (path / filename).open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
        (path / "teacher_events.jsonl").touch()
        return cls(path)

    def write_teacher_step_count(self, row: Mapping[str, Any]) -> None:
        self._append_csv("teacher_step_counts.csv", row)

    def write_update(self, row: Mapping[str, Any]) -> None:
        self._append_csv("updates.csv", row)

    def write_episode(self, row: Mapping[str, Any]) -> None:
        self._append_csv("episodes.csv", row)

    def write_event(self, event: Mapping[str, Any]) -> None:
        forbidden = _FORBIDDEN_EVENT_FIELDS & set(event)
        if forbidden:
            name = sorted(forbidden)[0]
            raise ValueError(f"O2 event cannot contain full-array field {name}.")
        serialized = json.dumps(dict(event), sort_keys=True)
        if len(serialized.encode("utf-8")) > 8192:
            raise ValueError("O2 event cannot contain a full-array payload.")
        event_file = self.directory / "teacher_events.jsonl"
        with event_file.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")

    def close(self, *, summary: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("O2 evidence writer is already closed.")
        _atomic_json(self.directory / "summary.json", summary)
        _atomic_json(self.directory / "state.json", {"status": "complete"})
        self._closed = True

    def fail(self, *, reason: str) -> None:
        _atomic_json(
            self.directory / "state.json", {"status": "failed", "reason": reason}
        )
        self._closed = True

    def _append_csv(self, filename: str, row: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to closed O2 evidence.")
        path = self.directory / filename
        with path.open("r", newline="", encoding="utf-8") as handle:
            fields = next(csv.reader(handle))
        unknown = set(row) - set(fields)
        if unknown:
            raise ValueError(f"Unknown compact O2 CSV field: {sorted(unknown)[0]}.")
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(dict(row))


def save_o2_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    actor,
    critic,
    optimizer,
    schedule_state: Mapping[str, Any],
    calibration_state: Mapping[str, Any] | None,
    trainer_state: Mapping[str, Any],
    rollout_empty: bool,
) -> None:
    """Atomically save only an update-boundary checkpoint."""
    if not rollout_empty:
        raise ValueError("O2 checkpoint requires an empty rollout boundary.")
    destination = Path(path)
    payload = {
        "schema": "o2-checkpoint-v1",
        "identity": dict(identity),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedule": dict(schedule_state),
        "calibration": None if calibration_state is None else dict(calibration_state),
        "trainer_state": dict(trainer_state),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_o2_checkpoint(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any],
    actor,
    critic,
    optimizer,
) -> dict[str, Any]:
    """Fail closed unless a checkpoint belongs to this exact O2 run."""
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("O2 checkpoint is unreadable.") from error
    if payload.get("schema") != "o2-checkpoint-v1":
        raise ValueError("O2 checkpoint schema is incompatible.")
    if payload.get("identity") != dict(expected_identity):
        raise ValueError("O2 checkpoint identity does not match this run.")
    actor.load_state_dict(payload["actor"])
    critic.load_state_dict(payload["critic"])
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    return {
        "schedule_state": payload["schedule"],
        "calibration_state": payload["calibration"],
        "trainer_state": payload["trainer_state"],
    }


def compute_throughput_grid(
    episodes: list[Mapping[str, Any]], grid: list[int]
) -> list[dict[str, float | int]]:
    """Sample completed-task throughput on a fixed real-step grid."""
    rows = sorted(episodes, key=lambda row: int(row["real_env_steps"]))
    sampled: list[dict[str, float | int]] = []
    cursor = 0
    current: Mapping[str, Any] | None = None
    for step in grid:
        while cursor < len(rows) and int(rows[cursor]["real_env_steps"]) <= step:
            current = rows[cursor]
            cursor += 1
        throughput = 0.0
        if current is not None:
            episode_steps = float(current["cumulative_episode_steps"])
            if episode_steps > 0.0:
                throughput = 1000.0 * float(
                    current["cumulative_completed_tasks"]
                ) / episode_steps
        sampled.append({"real_env_steps": int(step), "throughput": throughput})
    return sampled
