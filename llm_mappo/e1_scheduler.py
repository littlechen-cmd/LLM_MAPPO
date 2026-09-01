"""File-lock based four-slot scheduler primitives for owner-run E2 launchers."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
import time


@dataclass(frozen=True)
class E1Slot:
    physical_gpu: int
    slot: int
    @property
    def name(self): return f"gpu{self.physical_gpu}-slot{self.slot}"


SLOTS = tuple(E1Slot(gpu, slot) for gpu in (0, 1) for slot in (0, 1))


class E1SlotLocks:
    """Formal-only four-slot locks, intentionally distinct from P1/O1/O2 leases."""

    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def acquire(self, slot: E1Slot, payload: Mapping) -> Path:
        path = self.root / f"{slot.name}.json"
        if path.exists(): raise RuntimeError(f"E1 formal slot is occupied: {slot.name}")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"schema": "e1-formal-slot-v1", "slot": slot.name, **dict(payload)}, sort_keys=True), encoding="utf-8")
        try: temporary.replace(path)
        except FileExistsError as error: raise RuntimeError(f"E1 formal slot is occupied: {slot.name}") from error
        return path

    def release(self, slot: E1Slot, run_identity: str) -> None:
        path = self.root / f"{slot.name}.json"
        if not path.is_file(): raise RuntimeError("E1 formal slot lock is absent.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("run_identity") != run_identity: raise RuntimeError("E1 formal slot ownership mismatch.")
        path.unlink()


class SeedBlockScheduler:
    """Deterministically bind every paired run for one seed to one physical GPU."""

    def __init__(self, state_path: str | Path):
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def assign(self, *, seed: int, available_gpus: tuple[int, ...]) -> int:
        key = str(int(seed)); bindings = self.state["seed_bindings"]
        if key in bindings: return int(bindings[key])
        if not available_gpus: raise RuntimeError("No E1 GPU slot is available.")
        counts = {gpu: sum(value == gpu for value in bindings.values()) for gpu in available_gpus}
        selected = min(available_gpus, key=lambda gpu: (counts[gpu], gpu))
        bindings[key] = selected; self._write(); return selected

    def mark(self, run_identity: str, status: str, **details) -> None:
        if status not in {"pending", "running", "complete", "failed"}:
            raise ValueError("E1 matrix status is incompatible.")
        self.state["runs"][run_identity] = {"status": status, **details}; self._write()

    def summary(self) -> dict:
        values = list(self.state["runs"].values())
        return {"schema": "e1-matrix-state-v1", "seed_bindings": dict(self.state["seed_bindings"]),
                "counts": {status: sum(row.get("status") == status for row in values)
                           for status in ("pending", "running", "complete", "failed")}}

    def _load(self):
        if not self.path.exists(): return {"schema": "e1-matrix-state-v1", "seed_bindings": {}, "runs": {}}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("schema") != "e1-matrix-state-v1": raise ValueError("E1 matrix state is incompatible.")
        return value

    def _write(self):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


def admit_slot_memory(*, free_memory_mib: int, max_family_peak_mib: int) -> int:
    """Return frozen M_slot or fail closed before a second worker is started."""
    if max_family_peak_mib < 1: raise ValueError("E1 peak memory must be positive.")
    required = int((1.5 * max_family_peak_mib + 1024 + .999999) // 1)
    if int(free_memory_mib) < required:
        raise RuntimeError("E1 GPU free memory is below the frozen slot requirement.")
    return required


def write_heartbeat(path: str | Path, *, run_identity: str, pid: int, gpu: int,
                    free_memory_mib: int) -> None:
    """Atomically record the project worker's latest read-only resource sample."""
    destination = Path(path); temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps({"schema": "e1-heartbeat-v1", "run_identity": run_identity,
        "pid": int(pid), "physical_gpu": int(gpu), "free_memory_mib": int(free_memory_mib),
        "timestamp_unix": time.time()}, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
