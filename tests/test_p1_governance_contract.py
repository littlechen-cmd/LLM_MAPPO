from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_p1_governance_freezes_linux_o1_to_o2_sequence():
    roadmap = (ROOT / "specs/roadmap.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    protocol = (ROOT / "plan/experiment-protocol.md").read_text(encoding="utf-8")
    manifest = yaml.safe_load(
        (ROOT / "configs/g3_experiment_manifest.yaml").read_text(encoding="utf-8")
    )

    assert "O1 本地 ──> P1 ──> O1 CUDA Gate ──> O2" in roadmap
    assert "## P1 — 优化路线 Linux 服务器执行基础设施" in tasks
    assert "P1→O1→O2" in protocol
    assert "A600" not in protocol

    assert manifest["schema_version"] == 8
    assert manifest["status"] == "p1_and_o1_gate_passed_o2_implementation_in_progress"
    assert manifest["route_profiles"]["optimization"]["prerequisites"] == [
        "O0",
        "O1",
        "P1",
        "O2",
        "O3",
    ]
    linux = manifest["python_environment"]["linux_optimization"]
    assert linux["interpreter"] == (
        "/home/lzx/.conda/envs/llm-a-mappo-py310/bin/python"
    )
    assert linux["creation"] == "owner_only_user_prefix"
    gate = manifest["o1_cuda_gate"]
    assert gate["execution_environment"] == "linux_optimization_server"
    assert gate["physical_gpu_index"] == 0
    assert gate["expected_gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert gate["o2_launch_requires_o1_gate_pass"] is True
    assert gate["status"] == "passed"
