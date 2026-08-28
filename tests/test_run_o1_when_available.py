import json
from pathlib import Path
from contextlib import contextmanager

from llm_mappo.linux_server_runtime import (
    GpuInfo,
    MachineSnapshot,
    PreflightResult,
)
from scripts.run_o1_when_available import (
    build_child_environment,
    build_gate_command,
    route_gate_result,
    run,
)


def _preflight():
    return PreflightResult(
        passed=True,
        reasons=(),
        snapshot=MachineSnapshot(
            os_name="Linux",
            architecture="x86_64",
            cpu_model="AMD EPYC 7542",
            cpu_logical_count=64,
            cpu_percent=0.0,
            available_ram_gib=100.0,
            free_disk_gib=600.0,
            git_clean=True,
            gpus=(GpuInfo(
                physical_index=0,
                uuid="GPU-4090-uuid",
                pci_bus_id="00000000:01:00.0",
                name="NVIDIA GeForce RTX 4090",
                total_memory_mib=49140,
                free_memory_mib=49000,
                utilization_percent=0.0,
                driver_version="580.173.02",
            ),),
        ),
        consecutive_samples=5,
    )


def test_gate_child_is_pinned_before_the_benchmark_module_can_import_torch(tmp_path):
    command = build_gate_command(
        python_executable="/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python",
        gate_config=Path("configs/optimization/o1_reward_calibration_smoke.yaml"),
        preflight_report=tmp_path / "preflight.json",
        environment_report=tmp_path / "environment.json",
        output=tmp_path / "o1",
    )

    assert command[0].endswith("/python")
    assert command[1:4] == ["scripts/benchmark_reward_calibration.py", "gate", "--config"]
    assert build_child_environment({"CUDA_VISIBLE_DEVICES": "1"})[
        "CUDA_VISIBLE_DEVICES"
    ] == "0"
    assert "O2" not in command


def test_route_requires_passing_summary_and_receipt_before_naming_o2(tmp_path):
    summary = tmp_path / "summary.json"
    receipt = tmp_path / "o1_gate_receipt.json"
    summary.write_text(json.dumps({"gate_pass": True}), encoding="utf-8")
    receipt.write_text(json.dumps({"next_required_phase": "O2"}), encoding="utf-8")

    result = route_gate_result(0, summary, receipt)
    assert result["next_required_phase"] == "O2"

    receipt.unlink()
    result = route_gate_result(0, summary, receipt)
    assert result["next_required_phase"] == "O0"


def test_route_sends_nonzero_or_no_go_to_o0(tmp_path):
    summary = tmp_path / "summary.json"
    receipt = tmp_path / "o1_gate_receipt.json"
    summary.write_text(json.dumps({"gate_pass": False}), encoding="utf-8")

    assert route_gate_result(0, summary, receipt)["next_required_phase"] == "O0"
    assert route_gate_result(130, summary, receipt)["next_required_phase"] == "O0"


def test_launcher_holds_the_lease_and_never_invokes_o2(tmp_path):
    environment = tmp_path / "p1_linux_server" / "environment_report.json"
    environment.parent.mkdir()
    environment.write_text(
        json.dumps({"pass": True, "freeze_sha256": "environment"}),
        encoding="utf-8",
    )
    events = []

    @contextmanager
    def lease(_):
        events.append("lease-enter")
        yield
        events.append("lease-exit")

    class Child:
        returncode = 0

    def child(command, **kwargs):
        events.append("child")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert all("O2" not in str(item) for item in command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps({"gate_pass": True}), encoding="utf-8"
        )
        (output / "o1_gate_receipt.json").write_text(
            json.dumps({"next_required_phase": "O2"}), encoding="utf-8"
        )
        return Child()

    result = run(
        [
            "--server-config", str(Path("configs/optimization/p1_linux_server.yaml")),
            "--gate-config", "configs/optimization/o1_reward_calibration_smoke.yaml",
            "--output-root", str(tmp_path),
        ],
        collector=lambda: _preflight().snapshot,
        process_runner=child,
        lease_factory=lease,
        canonical_check=lambda: None,
        run_id_factory=lambda: "test-run",
        sleep=lambda _: None,
    )

    assert result["next_required_phase"] == "O2"
    assert events == ["lease-enter", "child", "lease-exit"]
