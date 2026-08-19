import json

import pytest

from llm_mappo.formal_results import (
    METRICS,
    collect_seed_rows,
    compare_factorial_effects,
    compare_full_to_baseline,
    exact_sign_flip_pvalue,
    holm_adjust,
    summarize_groups,
)


GROUPS = {
    "MAPPO-WP": "mappo_wp",
    "MAPPO-WP+A*KD": "mappo_wp_astar_kd",
    "MAPPO-WP+LLMKD": "mappo_wp_llm_kd",
    "MAPPO-WP+A*KD+LLMKD": "mappo_wp_astar_llm_kd",
}


def _manifest():
    return {
        "core_groups": {
            name: {"artifact_slug": slug} for name, slug in GROUPS.items()
        },
        "training": {"policy_initialization_seeds": [7, 17]},
        "evaluation": {
            "held_out_seeds": [0, 1],
            "episodes_per_seed": 2,
        },
    }


def _write_matrix(root):
    for group_index, slug in enumerate(GROUPS.values()):
        for policy_seed in (7, 17):
            output = root / slug / f"seed_{policy_seed:03d}" / "evaluation.json"
            output.parent.mkdir(parents=True)
            result = {
                "seeds": [
                    {"seed": seed, "episodes": 2} for seed in (0, 1)
                ],
                **{
                    metric: float(group_index + policy_seed / 1000)
                    for metric in METRICS
                },
            }
            output.write_text(json.dumps(result), encoding="utf-8")


def test_formal_aggregation_uses_paired_training_seeds(tmp_path):
    _write_matrix(tmp_path)
    rows = collect_seed_rows(_manifest(), tmp_path)

    assert len(rows) == 4 * 2 * len(METRICS)
    summaries = summarize_groups(rows, bootstrap_samples=200)
    assert len(summaries) == 4 * len(METRICS)
    assert {row["n_training_seeds"] for row in summaries} == {2}

    comparisons = compare_full_to_baseline(rows, bootstrap_samples=200)
    assert len(comparisons) == len(METRICS)
    assert {row["n_pairs"] for row in comparisons} == {2}
    assert all(
        row["mean_paired_difference"] == pytest.approx(3.0)
        for row in comparisons
    )
    assert all(0.0 <= row["p_value_holm"] <= 1.0 for row in comparisons)

    factorial = compare_factorial_effects(rows, bootstrap_samples=200)
    assert len(factorial) == 4 * len(METRICS)
    by_contrast = {}
    for row in factorial:
        by_contrast.setdefault(row["contrast"], set()).add(row["mean_contrast"])
    expected = {
        "full_vs_baseline": 3.0,
        "astar_kd_main_effect": 1.0,
        "llm_kd_main_effect": 2.0,
        "astar_llm_interaction": 0.0,
    }
    for contrast, target in expected.items():
        assert all(value == pytest.approx(target) for value in by_contrast[contrast])


def test_formal_aggregation_rejects_incomplete_matrix(tmp_path):
    _write_matrix(tmp_path)
    missing = tmp_path / "mappo_wp" / "seed_007" / "evaluation.json"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="matrix is incomplete"):
        collect_seed_rows(_manifest(), tmp_path)


def test_exact_sign_flip_and_holm_are_deterministic():
    assert exact_sign_flip_pvalue([1.0, 1.0]) == pytest.approx(0.5)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
