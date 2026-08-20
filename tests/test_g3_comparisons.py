import numpy as np
import torch
import yaml
from pathlib import Path

from llm_mappo.llm_teacher import EngagementScenario, LabelledScenario
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase3_training import Phase3TrainingConfig
from llm_mappo.qmix import QMIXHyperparameters, QMIXLearner
from llm_mappo.semantic_controls import derive_rule_kd, derive_shuffle_kd
from llm_mappo.types import SemanticPreferenceLabel


def _record(index: int, scenario_type: str = "normal_transport") -> LabelledScenario:
    scenario = EngagementScenario(
        scenario_id=f"scenario-{index}", observation_version="phase4-semantic-v2",
        scenario_type=scenario_type, observation=(float(index),), agent_id=index,
        battery=0.9, loaded=index == 0, priority_label="A1", target_kind="delivery",
        nearby_agents=(),
    )
    return LabelledScenario(
        scenario, SemanticPreferenceLabel(
            scenario.scenario_id, scenario.observation_version, 0.1 + index / 10,
            0.2 + index / 10, "source", "2026-08-20T00:00:00Z"
        ), "source", "source"
    )


def test_g3_rule_and_shuffle_controls_are_deterministic_and_disjoint():
    records = [_record(0), _record(1), _record(2, "narrow_corridor_yield")]
    rules = derive_rule_kd(records)
    shuffled = derive_shuffle_kd(records, 20260820)
    assert rules[0].label.task_commitment == 0.9
    assert rules[2].label.local_assertiveness == 0.2
    assert [item.label.model for item in shuffled] == ["shuffled-semantic-kd-v1"] * 3
    assert all(
        (item.label.task_commitment, item.label.local_assertiveness)
        != (source.label.task_commitment, source.label.local_assertiveness)
        for item, source in zip(shuffled, records)
    )
    assert np.allclose(
        sorted(item.label.task_commitment for item in shuffled), [0.1, 0.2, 0.3]
    )


def test_nowp_keeps_observation_contract_without_planning_calls(monkeypatch):
    env = Phase2Warehouse(n_agents=2, max_steps=8, include_waypoint_features=False)
    try:
        monkeypatch.setattr(env._planner, "plan", lambda *_: (_ for _ in ()).throw(AssertionError))
        observations = env.reset(seed=3)
        assert observations.shape == (2, env.actor_observation_dim)
        assert np.isfinite(observations).all()
    finally:
        env.close()


def test_qmix_learner_respects_masks_and_updates():
    learner = QMIXLearner(4, 5, 2, QMIXHyperparameters(), torch.device("cpu"))
    observations = np.zeros((2, 4), dtype=np.float32)
    masks = np.asarray([[False, True, False, False, False], [True] * 5])
    actions = learner.select_actions(observations, masks, 0.0, np.random.default_rng(3))
    assert actions[0] == 1
    batch = {
        "observations": np.zeros((2, 2, 4), dtype=np.float32),
        "actions": np.asarray([[1, 2], [1, 2]], dtype=np.int64),
        "rewards": np.zeros(2, dtype=np.float32),
        "next_observations": np.zeros((2, 2, 4), dtype=np.float32),
        "dones": np.zeros(2, dtype=np.float32),
        "masks": np.ones((2, 2, 5), dtype=bool),
        "next_masks": np.ones((2, 2, 5), dtype=bool),
    }
    assert np.isfinite(learner.update(batch))


def test_g3_comparison_configs_share_the_frozen_environment_contract():
    paths = [
        "configs/g3_q2_rule_kd.yaml", "configs/g3_q2_shuffle_kd.yaml",
        "configs/g3_q2_mappo_no_wp.yaml", "configs/g3_q2_qmix_wp.yaml",
    ]
    sources = [yaml.safe_load(Path(path).read_text(encoding="utf-8")) for path in paths]
    reference = sources[0]["environment"]
    for source in sources:
        environment = source["environment"]
        for key in ("id", "n_agents", "battery_cost_scale", "charge_threshold",
                    "charge_release_threshold", "batch_interval", "task_completion_target"):
            assert environment[key] == reference[key]
        assert source["training"]["environment_step_budget"] == 150_000
    nowp = Phase3TrainingConfig.from_yaml(paths[2])
    assert nowp.include_waypoint_features is False
    assert nowp.waypoint_reward == 0.0
