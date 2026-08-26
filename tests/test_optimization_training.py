"""O1 composition and frozen-config contracts."""

from pathlib import Path

import pytest

from llm_mappo.optimization_training import (
    OptimizationTrainer,
    OptimizationTrainingConfig,
)


def test_optimization_config_accepts_only_the_frozen_functional_smoke():
    config = OptimizationTrainingConfig.from_yaml(
        Path("configs/optimization/o1_functional_smoke.yaml")
    )
    assert config.real_env_steps == 128
    assert config.h_reward == 12
    assert config.fixture_only
    with pytest.raises(ValueError, match="H_reward"):
        OptimizationTrainingConfig.from_mapping({**config.as_dict(), "h_reward": 4})
    with pytest.raises(ValueError, match="online LLM"):
        OptimizationTrainingConfig.from_mapping({**config.as_dict(), "online_llm": True})


def test_environment_teacher_adapter_normalizes_runtime_grid_coordinates(tmp_path):
    config = OptimizationTrainingConfig.from_yaml(
        Path("configs/optimization/o1_functional_smoke.yaml")
    )
    trainer = OptimizationTrainer(config, tmp_path)
    trainer.environment.reset(seed=config.seed)
    preferences, valid = trainer._teacher_batch(trainer.environment)
    assert preferences.shape == (config.n_agents, 3)
    assert valid.shape == (config.n_agents,)
    assert valid.any()


def test_shadow_teacher_queries_do_not_write_real_teacher_cache(tmp_path):
    config = OptimizationTrainingConfig.from_yaml(
        Path("configs/optimization/o1_functional_smoke.yaml")
    )
    trainer = OptimizationTrainer(config, tmp_path)
    trainer.environment.reset(seed=config.seed)
    before = len(trainer.teacher._cache)
    trainer._shadow_teacher_batch(trainer.environment)
    assert len(trainer.teacher._cache) == before
