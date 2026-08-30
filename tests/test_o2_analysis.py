"""O2 aggregation must fail closed on coverage and paired-AUC degradation."""

import pytest


def test_normalized_auc_uses_the_fixed_real_step_grid():
    from scripts.analyze_o2_calibration import normalized_throughput_auc

    rows = [
        {"real_env_steps": step, "throughput": step / 1500.0}
        for step in range(0, 150001, 10000)
    ]
    assert normalized_throughput_auc(rows) == pytest.approx(50.0)


def test_o2_gate_requires_each_rc_coverage_and_median_paired_auc():
    from scripts.analyze_o2_calibration import aggregate_o2

    runs = []
    for seed in (107, 117, 127):
        runs.append({
            "group": "MAPPO-DG", "seed": seed, "coverage": None, "auc": 100.0
        })
        runs.append({
            "group": "RC-AStarKD", "seed": seed, "coverage": 0.25, "auc": 92.0
        })
    result = aggregate_o2(runs)
    assert result["gate_pass"] is True
    assert result["median_relative_degradation"] == pytest.approx(0.08)

    runs[-1]["coverage"] = 0.249
    result = aggregate_o2(runs)
    assert result["gate_pass"] is False
    assert "coverage" in result["reasons"]
