"""Fail-closed inspection and waiting primitives for the P1 Linux server."""

import csv
import io
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Optional, Tuple

import psutil


@dataclass(frozen=True)
class GpuInfo:
    """One physical GPU reported by the host-level NVIDIA inventory."""

    physical_index: int
    uuid: str
    pci_bus_id: str
    name: str
    total_memory_mib: int
    free_memory_mib: int
    utilization_percent: float
    driver_version: str = "unknown"
    compute_pids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class MachineSnapshot:
    """Read-only evidence collected before a CUDA task can begin."""

    os_name: str
    architecture: str
    cpu_model: str
    cpu_logical_count: int
    cpu_percent: float
    available_ram_gib: float
    free_disk_gib: float
    git_clean: bool
    gpus: Tuple[GpuInfo, ...]

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ServerPolicy:
    """The immutable resource eligibility requirements for P1/O1."""

    physical_gpu_index: int
    expected_gpu_name: str
    minimum_total_gpu_memory_mib: int
    minimum_available_ram_gib: float
    minimum_free_disk_gib: float
    maximum_cpu_percent: float
    minimum_free_gpu_fraction: float
    poll_seconds: int
    required_consecutive_free_samples: int
    wait_timeout_hours: float
    require_clean_git: bool
    required_python: str
    required_torch: str
    require_no_external_compute_processes: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ServerPolicy":
        return cls(**values)


@dataclass(frozen=True)
class PreflightResult:
    """A resource decision plus its machine-readable reasons."""

    passed: bool
    reasons: Tuple[str, ...]
    snapshot: Optional[MachineSnapshot] = None
    consecutive_samples: int = 0

    def to_dict(self) -> Mapping[str, object]:
        payload = asdict(self)
        return payload


CommandRunner = Callable[[Tuple[str, ...]], str]


def parse_gpu_inventory(text: str) -> Tuple[GpuInfo, ...]:
    """Parse a no-header/no-units `nvidia-smi --query-gpu` CSV response."""
    rows = []
    try:
        for row in csv.reader(io.StringIO(text)):
            if not row:
                continue
            if len(row) != 8:
                raise ValueError("invalid nvidia-smi GPU inventory column count")
            rows.append(
                GpuInfo(
                    physical_index=int(row[0].strip()),
                    uuid=row[1].strip(),
                    pci_bus_id=row[2].strip(),
                    name=row[3].strip(),
                    total_memory_mib=int(row[4].strip()),
                    free_memory_mib=int(row[5].strip()),
                    utilization_percent=float(row[6].strip()),
                    driver_version=row[7].strip(),
                )
            )
    except (TypeError, ValueError) as error:
        raise ValueError("invalid nvidia-smi GPU inventory") from error
    if not rows or any(not gpu.uuid or gpu.total_memory_mib <= 0 for gpu in rows):
        raise ValueError("incomplete nvidia-smi GPU inventory")
    if len({gpu.physical_index for gpu in rows}) != len(rows):
        raise ValueError("duplicate physical GPU index")
    return tuple(sorted(rows, key=lambda gpu: gpu.physical_index))


def apply_compute_processes(
    gpus: Iterable[GpuInfo], compute_processes: str
) -> Tuple[GpuInfo, ...]:
    """Attach compute PIDs by UUID; unknown/malformed records fail closed."""
    indexed = {gpu.uuid: gpu for gpu in gpus}
    pids = {uuid: [] for uuid in indexed}
    text = compute_processes.strip()
    if not text or text == "No running processes found":
        return tuple(indexed[uuid] for uuid in sorted(
            indexed, key=lambda item: indexed[item].physical_index
        ))
    try:
        for row in csv.reader(io.StringIO(text)):
            if len(row) != 2:
                raise ValueError("invalid nvidia-smi compute-process column count")
            pid, uuid = int(row[0].strip()), row[1].strip()
            if uuid not in indexed:
                raise ValueError("compute process references an unknown GPU UUID")
            pids[uuid].append(pid)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid nvidia-smi compute-process inventory") from error
    return tuple(
        GpuInfo(**{**asdict(gpu), "compute_pids": tuple(sorted(pids[gpu.uuid]))})
        for gpu in sorted(indexed.values(), key=lambda item: item.physical_index)
    )


def _cpu_model(lscpu_output: str) -> str:
    for line in lscpu_output.splitlines():
        if ":" in line:
            label, value = line.split(":", 1)
        elif "：" in line:
            label, value = line.split("：", 1)
        else:
            continue
        if label.strip().lower() == "model name" or label.strip() == "型号名称":
            return value.strip()
    raise ValueError("lscpu output does not contain a model name")


def _default_command_runner(command: Tuple[str, ...]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def collect_machine_snapshot(
    command_runner: CommandRunner, paths: Mapping[str, Path]
) -> MachineSnapshot:
    """Collect a snapshot using read-only host commands and local capacity probes."""
    repo = Path(paths["repo"])
    gpu_rows = command_runner((
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,"
        "utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ))
    compute_rows = command_runner((
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits",
    ))
    gpus = apply_compute_processes(parse_gpu_inventory(gpu_rows), compute_rows)
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(repo)
    git_status = command_runner(("git", "-C", str(repo), "status", "--porcelain"))
    return MachineSnapshot(
        os_name=platform.system(),
        architecture=platform.machine(),
        cpu_model=_cpu_model(command_runner(("lscpu",))),
        cpu_logical_count=psutil.cpu_count(logical=True) or 0,
        cpu_percent=float(psutil.cpu_percent(interval=None)),
        available_ram_gib=memory.available / (1024 ** 3),
        free_disk_gib=disk.free / (1024 ** 3),
        git_clean=not git_status.strip(),
        gpus=gpus,
    )


def _target_gpu(snapshot: MachineSnapshot, index: int) -> Optional[GpuInfo]:
    return next((gpu for gpu in snapshot.gpus if gpu.physical_index == index), None)


def _target_gpu_reasons(
    target: Optional[GpuInfo], policy: ServerPolicy
) -> Tuple[str, ...]:
    if target is None:
        return ("target_gpu_missing",)
    reasons = []
    if target.name != policy.expected_gpu_name:
        reasons.append("target_gpu_name")
    if target.total_memory_mib < policy.minimum_total_gpu_memory_mib:
        reasons.append("target_gpu_total_memory")
    if target.compute_pids and policy.require_no_external_compute_processes:
        reasons.append("external_compute_processes")
    if target.free_memory_mib < (
        target.total_memory_mib * policy.minimum_free_gpu_fraction
    ):
        reasons.append("target_gpu_free_memory")
    return tuple(reasons)


def _host_reasons(snapshot: MachineSnapshot, policy: ServerPolicy) -> Tuple[str, ...]:
    reasons = []
    if snapshot.os_name != "Linux":
        reasons.append("operating_system")
    if snapshot.available_ram_gib < policy.minimum_available_ram_gib:
        reasons.append("available_ram")
    if snapshot.free_disk_gib < policy.minimum_free_disk_gib:
        reasons.append("free_disk")
    if snapshot.cpu_percent > policy.maximum_cpu_percent:
        reasons.append("cpu_utilization")
    if policy.require_clean_git and not snapshot.git_clean:
        reasons.append("git_dirty")
    return tuple(reasons)


def evaluate_preflight(
    snapshot: MachineSnapshot, policy: ServerPolicy
) -> PreflightResult:
    """Evaluate the fixed P1 policy without changing resource state."""
    target = _target_gpu(snapshot, policy.physical_gpu_index)
    reasons = list(_host_reasons(snapshot, policy))
    reasons.extend(_target_gpu_reasons(target, policy))
    return PreflightResult(not reasons, tuple(reasons), snapshot=snapshot)


def wait_for_resources(
    sample: Callable[[], MachineSnapshot],
    policy: ServerPolicy,
    clock: Callable[[], float],
    sink: Callable[[PreflightResult], None],
    sleep: Callable[[float], None] = time.sleep,
) -> PreflightResult:
    """Wait for fixed consecutive clean samples; never relax the supplied policy."""
    started = clock()
    consecutive = 0
    while True:
        try:
            snapshot = sample()
        except Exception:
            result = PreflightResult(False, ("inventory_collection_error",))
            sink(result)
            return result
        result = evaluate_preflight(snapshot, policy)
        consecutive = consecutive + 1 if result.passed else 0
        sampled = PreflightResult(
            passed=result.passed,
            reasons=result.reasons,
            snapshot=result.snapshot,
            consecutive_samples=consecutive,
        )
        sink(sampled)
        if consecutive >= policy.required_consecutive_free_samples:
            return sampled
        if clock() - started >= policy.wait_timeout_hours * 3600:
            return PreflightResult(False, ("wait_timeout",), snapshot=result.snapshot)
        sleep(policy.poll_seconds)


@contextmanager
def gpu_lease(lock_path: Path) -> Iterator[None]:
    """Hold a non-blocking project-only Linux lock for the selected GPU."""
    if not platform.system().lower().startswith("linux"):
        raise RuntimeError("P1 GPU leases require Linux fcntl.flock")
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("P1 GPU lease is already held by this project") from error
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def write_new_atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create an immutable JSON artifact and refuse an overwrite."""
    if path.exists():
        raise FileExistsError("immutable artifact already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError("immutable artifact already exists: " + str(path))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def replace_state_atomic(
    path: Path, payload: Mapping[str, object], identity: Mapping[str, str]
) -> None:
    """Replace only a same-identity `state.json` with an fsync-backed write."""
    if path.name != "state.json":
        raise ValueError("only state.json permits atomic replacement")
    expected_identity = dict(identity)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("run_identity") != expected_identity:
            raise ValueError("state.json run identity does not match")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {**payload, "run_identity": expected_identity},
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
