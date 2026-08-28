from dataclasses import replace
from pathlib import Path

import pytest

from llm_mappo.linux_server_runtime import (
    MachineSnapshot,
    ServerPolicy,
    apply_compute_processes,
    evaluate_preflight,
    gpu_lease,
    parse_gpu_inventory,
    replace_state_atomic,
    wait_for_resources,
    write_new_atomic_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _policy():
    return ServerPolicy(
        physical_gpu_index=0,
        expected_gpu_name="NVIDIA GeForce RTX 4090",
        minimum_total_gpu_memory_mib=48000,
        minimum_available_ram_gib=64,
        minimum_free_disk_gib=200,
        maximum_cpu_percent=50,
        minimum_free_gpu_fraction=0.95,
        poll_seconds=60,
        required_consecutive_free_samples=5,
        wait_timeout_hours=48,
        require_clean_git=True,
        required_python="3.10.19",
        required_torch="2.10.0+cu128",
    )


def _snapshot():
    rows = (ROOT / "tests/fixtures/nvidia_smi_p1_inventory.csv").read_text(
        encoding="utf-8"
    )
    gpus = apply_compute_processes(
        parse_gpu_inventory(rows), "No running processes found"
    )
    return MachineSnapshot(
        os_name="Linux",
        architecture="x86_64",
        cpu_model="AMD EPYC 7542",
        cpu_logical_count=64,
        cpu_percent=10.0,
        available_ram_gib=100.0,
        free_disk_gib=670.0,
        git_clean=True,
        gpus=gpus,
    )


def test_inventory_maps_uuid_and_external_compute_pid():
    rows = (ROOT / "tests/fixtures/nvidia_smi_p1_inventory.csv").read_text(
        encoding="utf-8"
    )
    gpus = apply_compute_processes(
        parse_gpu_inventory(rows), "8123, GPU-4090-uuid\n"
    )

    assert [gpu.physical_index for gpu in gpus] == [0, 1]
    assert gpus[0].compute_pids == (8123,)
    assert gpus[1].compute_pids == ()


def test_preflight_fails_closed_for_busy_or_ineligible_server():
    snapshot = _snapshot()
    assert evaluate_preflight(snapshot, _policy()).passed is True

    busy = replace(snapshot, gpus=apply_compute_processes(
        snapshot.gpus, "8123, GPU-4090-uuid\n"
    ))
    result = evaluate_preflight(busy, _policy())
    assert result.passed is False
    assert "external_compute_processes" in result.reasons

    under_memory = replace(snapshot, available_ram_gib=10.0, git_clean=False)
    result = evaluate_preflight(under_memory, _policy())
    assert result.passed is False
    assert {"available_ram", "git_dirty"} <= set(result.reasons)


def test_wait_requires_five_consecutive_eligible_samples():
    eligible = _snapshot()
    busy = replace(eligible, cpu_percent=70.0)
    samples = iter([eligible, busy, eligible, eligible, eligible, eligible, eligible])
    ticks = iter(range(0, 10000, 60))
    recorded = []

    result = wait_for_resources(
        sample=lambda: next(samples),
        policy=_policy(),
        clock=lambda: next(ticks),
        sink=recorded.append,
        sleep=lambda _: None,
    )

    assert result.passed is True
    assert result.consecutive_samples == 5
    assert len(recorded) == 7


def test_wait_times_out_without_weakening_policy():
    snapshot = replace(_snapshot(), cpu_percent=90.0)
    ticks = iter([0, 48 * 3600 + 1])

    result = wait_for_resources(
        sample=lambda: snapshot,
        policy=_policy(),
        clock=lambda: next(ticks),
        sink=lambda _: None,
        sleep=lambda _: None,
    )

    assert result.passed is False
    assert result.reasons == ("wait_timeout",)


def test_atomic_new_json_refuses_overwrite(tmp_path):
    output = tmp_path / "manifest.json"
    write_new_atomic_json(output, {"pass": True})

    with pytest.raises(FileExistsError):
        write_new_atomic_json(output, {"pass": False})


def test_atomic_state_replacement_requires_the_same_run_identity(tmp_path):
    state = tmp_path / "state.json"
    identity = {"commit": "abc", "config": "def"}
    replace_state_atomic(state, {"status": "waiting"}, identity)
    replace_state_atomic(state, {"status": "running"}, identity)

    assert "running" in state.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        replace_state_atomic(state, {"status": "wrong"}, {"commit": "other"})
    with pytest.raises(ValueError, match="state.json"):
        replace_state_atomic(tmp_path / "summary.json", {}, identity)


def test_gpu_lease_is_linux_only(tmp_path):
    if __import__("sys").platform.startswith("linux"):
        pytest.skip("Windows runner validates the Linux-only rejection path.")
    with pytest.raises(RuntimeError, match="Linux"):
        with gpu_lease(tmp_path / "gpu.lock"):
            pass


def test_once_cli_writes_a_versioned_preflight_report(tmp_path):
    from scripts.check_optimization_server import run

    exit_code = run(
        [
            "--config",
            str(ROOT / "configs/optimization/p1_linux_server.yaml"),
            "--once",
            "--output",
            str(tmp_path),
        ],
        collector=lambda: _snapshot(),
    )

    reports = list(tmp_path.glob("preflight_*.json"))
    assert exit_code == 0
    assert len(reports) == 1


def test_once_cli_records_a_collection_failure_without_process_control(tmp_path):
    from scripts.check_optimization_server import run

    exit_code = run(
        [
            "--config",
            str(ROOT / "configs/optimization/p1_linux_server.yaml"),
            "--once",
            "--output",
            str(tmp_path),
        ],
        collector=lambda: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")),
    )

    report = next(tmp_path.glob("preflight_*.json"))
    assert exit_code == 2
    assert "inventory_collection_error" in report.read_text(encoding="utf-8")
