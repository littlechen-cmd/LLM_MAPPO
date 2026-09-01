import json

from llm_mappo.e1_smoke import aggregate_cuda_smoke, frozen_smoke_waves


def test_e1_smoke_waves_are_exactly_preregistered():
    waves = frozen_smoke_waves()
    assert len(waves) == 2
    assert all(len(wave) == 4 for wave in waves)
    assert {item[2] for wave in waves for item in wave} == {0}
    assert {item[:2] for wave in waves for item in wave} == {
        ("MAPPO-DG", 9001), ("Fixed-AStarKD+LLMKD", 9002),
        ("RuleKD-v3", 9003), ("NoOOD-v1", 9004),
        ("RC-AStarKD+LLMKD", 9001), ("QMIX-DG", 9002),
        ("ShuffleKD-v3", 9003), ("NoGoalHint-v1", 9004),
    }


def test_e1_cuda_smoke_aggregator_requires_exact_frozen_matrix(tmp_path):
    rows = [
        ("MAPPO-DG", 9001, 0), ("Fixed-AStarKD+LLMKD", 9002, 0),
        ("RuleKD-v3", 9003, 0), ("NoOOD-v1", 9004, 0),
        ("RC-AStarKD+LLMKD", 9001, 0), ("QMIX-DG", 9002, 0),
        ("ShuffleKD-v3", 9003, 0), ("NoGoalHint-v1", 9004, 0),
    ]
    for index, (group, seed, gpu) in enumerate(rows):
        directory = tmp_path / str(index); directory.mkdir()
        (directory / "smoke_receipt.json").write_text(json.dumps({
            "group": group, "seed": seed, "physical_gpu": gpu,
            "device": "cuda:0", "steps_before_resume": 128,
            "steps_after_resume": 256, "planner_query_count": 0,
            "online_llm_calls": 0, "finite": True,
        }), encoding="utf-8")
    result = aggregate_cuda_smoke(tmp_path)
    assert result["pass"] is True
    assert result["total_environment_steps"] == 2048
    assert result["gpu_run_counts"] == {0: 8}


def test_e1_cuda_runner_uses_only_gpu_zero_and_four_slots():
    script = open("scripts/run_e1_cuda_smoke.sh", encoding="utf-8").read()
    assert "admit_gpu 0" in script
    assert "admit_gpu 1" not in script
    assert "2 * M_SLOT_MIB" not in script
    assert "4 * M_SLOT_MIB" in script
