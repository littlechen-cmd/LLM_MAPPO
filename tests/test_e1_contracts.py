"""E1 method-contract parity regressions."""

import pytest


def test_fixed_rc_contract_allows_only_reward_confidence():
    from llm_mappo.e1_contracts import compare_method_contracts

    fixed = {"method": "Fixed-AStarKD+LLMKD", "astar": "fixed",
             "semantic": "raw-llm-v3", "schedule": "linear-env-step-v1",
             "reward_confidence": "one"}
    calibrated = {**fixed, "method": "RC-AStarKD+LLMKD", "astar": "reward-calibrated",
                  "reward_confidence": "c_A_reward"}
    report = compare_method_contracts(fixed, calibrated)

    assert report["pass"] is True
    assert report["optimization_difference"] == ["reward_confidence"]


def test_fixed_rc_contract_rejects_any_extra_difference():
    from llm_mappo.e1_contracts import compare_method_contracts

    with pytest.raises(ValueError, match="unexpected"):
        compare_method_contracts(
            {"method": "Fixed-AStarKD+LLMKD", "reward_confidence": "one", "network": "v1"},
            {"method": "RC-AStarKD+LLMKD", "reward_confidence": "c_A_reward", "network": "v2"},
        )


def test_qmix_dg_rejects_waypoint_or_teacher_fallback():
    from llm_mappo.e1_qmix import validate_qmix_dg_contract

    run = type("Run", (), {"group": "QMIX-DG", "algorithm": "qmix",
                            "observation_schema": "direct-goal-observation-v1",
                            "astar_kd": "disabled", "semantic_teacher": "disabled"})()
    validate_qmix_dg_contract(run, {"observation_schema": "direct-goal-observation-v1"})
    with pytest.raises(ValueError, match="waypoint"):
        validate_qmix_dg_contract(run, {"observation_schema": "legacy-waypoint-v1"})


def test_seed_block_scheduler_keeps_paired_runs_on_one_gpu(tmp_path):
    from llm_mappo.e1_scheduler import SLOTS, SeedBlockScheduler

    assert [(slot.physical_gpu, slot.slot) for slot in SLOTS] == [
        (0, 0), (0, 1), (0, 2), (0, 3),
    ]

    scheduler = SeedBlockScheduler(tmp_path / "matrix.json")
    assert scheduler.assign(seed=7, available_gpus=(0,)) == 0
    assert scheduler.assign(seed=7, available_gpus=(0,)) == 0
    assert scheduler.assign(seed=17, available_gpus=(0,)) == 0
    scheduler.mark("MAPPO-DG:seed007", "failed", reason="RuntimeError")
    assert scheduler.summary()["counts"]["failed"] == 1


def test_e1_slot_memory_and_heartbeat_are_fail_closed(tmp_path):
    from llm_mappo.e1_scheduler import admit_slot_memory, write_heartbeat

    assert admit_slot_memory(free_memory_mib=2524, max_family_peak_mib=1000) == 2524
    with pytest.raises(RuntimeError, match="free memory"):
        admit_slot_memory(free_memory_mib=2523, max_family_peak_mib=1000)
    heartbeat = tmp_path / "heartbeat.json"
    write_heartbeat(heartbeat, run_identity="MAPPO-DG:seed007", pid=1, gpu=0,
                    free_memory_mib=48000)
    assert heartbeat.is_file()


def test_e1_qmix_runtime_state_is_resumable(tmp_path):
    from llm_mappo.e1_qmix import E1QMIXDGTrainer

    run = type("Run", (), {"group": "QMIX-DG", "algorithm": "qmix", "seed": 9002,
        "real_environment_steps": 256, "observation_schema": "direct-goal-observation-v1",
        "astar_kd": "disabled", "semantic_teacher": "disabled"})()
    environment = {"environment_id": "llm-mappo-medium-3ag-v1", "n_agents": 2,
        "max_steps": 16, "charge_threshold": .3, "charge_release_threshold": .8,
        "battery_cost_scale": 1.1, "deadlock_steps": 12, "dynamic_ingress_interval": 40,
        "batch_size_range": [4, 8], "queue_size": 8, "task_target": 9,
        "observation_schema": "direct-goal-observation-v1"}
    first = E1QMIXDGTrainer(run=run, environment=environment, device="cpu")
    try:
        first.run_prefix(8); state = first.runtime_state()
    finally: first.close()
    resumed = E1QMIXDGTrainer(run=run, environment=environment, device="cpu")
    try:
        resumed.restore_runtime_state(state)
        assert resumed.run_prefix(12)["real_env_steps"] == 12
    finally: resumed.close()
