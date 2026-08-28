"""The benchmark interface preserves H=12 as the sole formal horizon."""

import pytest

from scripts.benchmark_reward_calibration import (
    _Worker,
    _rho,
    BenchmarkConfig,
    REQUIRED_ARTIFACTS,
    analyze_memory_rows,
    parse_arguments,
)
from llm_mappo.optimization_training import OptimizationTrainingConfig


def test_benchmark_config_rejects_noncanonical_formal_horizon():
    assert BenchmarkConfig(condition="h12").horizon == 12
    assert BenchmarkConfig(condition="h4").horizon == 4
    with pytest.raises(ValueError, match="condition"):
        BenchmarkConfig(condition="h8")


def test_owner_gate_parser_accepts_only_baseline_and_h12(tmp_path):
    arguments = parse_arguments(
        [
            "gate",
            "--config", "configs/optimization/o1_reward_calibration_smoke.yaml",
            "--preflight-report", str(tmp_path / "preflight.json"),
            "--environment-report", str(tmp_path / "environment.json"),
            "--modes", "baseline", "h12", "--workers", "12",
            "--repeats", "5", "--warmup-vector-steps", "16",
            "--measure-vector-steps", "128", "--memory-warmup-windows", "2",
            "--memory-measure-windows", "10", "--output", str(tmp_path),
        ]
    )
    assert arguments.workers == 12
    assert arguments.modes == ["baseline", "h12"]
    assert arguments.command == "gate"

    with pytest.raises(SystemExit):
        parse_arguments(
            [
                "gate", "--config", "config.yaml", "--preflight-report", "p.json",
                "--environment-report", "e.json", "--modes", "baseline", "h4",
                "--workers", "12", "--repeats", "5", "--warmup-vector-steps", "16",
                "--measure-vector-steps", "128", "--memory-warmup-windows", "2",
                "--memory-measure-windows", "10", "--output", str(tmp_path),
            ]
        )


def test_h4_diagnostic_requires_a_failed_normal_gate(tmp_path):
    failed = tmp_path / "failed.json"
    failed.write_text('{"gate_pass": false}', encoding="utf-8")
    arguments = parse_arguments(
        [
            "diagnose-h4", "--config", "configs/optimization/o1_reward_calibration_smoke.yaml",
            "--preflight-report", str(tmp_path / "preflight.json"),
            "--environment-report", str(tmp_path / "environment.json"),
            "--failed-gate-summary", str(failed), "--workers", "12", "--repeats", "5",
            "--warmup-vector-steps", "16", "--measure-vector-steps", "128",
            "--memory-warmup-windows", "2", "--memory-measure-windows", "10",
            "--output", str(tmp_path),
        ]
    )
    assert arguments.command == "diagnose-h4"


def test_normal_gate_accepts_only_an_explicit_matching_resume_directory(tmp_path):
    run_dir = tmp_path / "run"
    arguments = parse_arguments(
        [
            "gate", "--config", "configs/optimization/o1_reward_calibration_smoke.yaml",
            "--preflight-report", str(tmp_path / "preflight.json"),
            "--environment-report", str(tmp_path / "environment.json"),
            "--modes", "baseline", "h12", "--workers", "12", "--repeats", "5",
            "--warmup-vector-steps", "16", "--measure-vector-steps", "128",
            "--memory-warmup-windows", "2", "--memory-measure-windows", "10",
            "--output", str(run_dir), "--resume", str(run_dir),
        ]
    )

    assert arguments.resume == str(run_dir)


def test_gate_requires_all_frozen_artifacts():
    assert REQUIRED_ARTIFACTS == {
        "manifest.json", "runtime.csv", "memory.csv",
        "branch_objects.csv", "summary.json",
    }


def test_memory_analysis_detects_only_large_monotonic_growth():
    stable = [
        {"window": index, "cpu_rss_bytes": 100_000_000 + index,
         "cuda_reserved_bytes": 200_000_000, "branch_objects": 24,
         "teacher_cache_entries": 50}
        for index in range(10)
    ]
    assert analyze_memory_rows(stable)["persistent_growth"] is False
    growing = [
        {"window": index, "cpu_rss_bytes": 100_000_000 + index * 10_000_000,
         "cuda_reserved_bytes": 200_000_000 + index * 10_000_000,
         "branch_objects": 24 + index, "teacher_cache_entries": 50 + index}
        for index in range(10)
    ]
    analysis = analyze_memory_rows(growing)
    assert analysis["persistent_growth"] is True
    assert analysis["branch_object_growth"] is True


def test_memory_trend_uses_spearman_rank_correlation():
    assert _rho([1, 2, 4, 8]) == pytest.approx(1.0)


def test_worker_runs_the_shared_optimizer_update_every_32_steps(tmp_path):
    config = OptimizationTrainingConfig.from_yaml(
        "configs/optimization/o1_functional_smoke.yaml"
    )
    worker = _Worker(config, config.seed, tmp_path)
    for _ in range(32):
        worker.step("baseline")
    assert worker.update_count == 1
