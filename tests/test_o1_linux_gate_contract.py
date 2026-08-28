from llm_mappo.linux_server_runtime import GpuInfo, MachineSnapshot
import pytest

from scripts.benchmark_reward_calibration import (
    validate_cuda_binding,
    validate_cuda_visibility,
)


def _snapshot():
    return MachineSnapshot(
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
        ),),
    )


def test_cuda_binding_requires_logical_cuda_zero_on_the_frozen_gpu():
    binding = validate_cuda_binding(
        0,
        _snapshot(),
        lambda: {
            "available": True,
            "logical_device": "cuda:0",
            "device_name": "NVIDIA GeForce RTX 4090",
        },
    )

    assert binding.physical_gpu_index == 0
    assert binding.logical_device == "cuda:0"
    assert binding.uuid == "GPU-4090-uuid"


def test_cuda_binding_rejects_an_unavailable_or_wrong_logical_device():
    for probe in (
        {"available": False, "logical_device": "cuda:0", "device_name": "x"},
        {"available": True, "logical_device": "cuda:1", "device_name": "x"},
    ):
        try:
            validate_cuda_binding(0, _snapshot(), lambda: probe)
        except RuntimeError:
            continue
        raise AssertionError("invalid CUDA binding was accepted")


def test_cuda_visibility_exposes_only_the_frozen_physical_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    validate_cuda_visibility()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=0"):
        validate_cuda_visibility()
