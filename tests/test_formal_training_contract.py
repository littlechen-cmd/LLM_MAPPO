"""Regression checks for the provisional three-scene MAPPO contract."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from llm_mappo.formal_training import (
    EVALUATION_SEEDS, SCENES, make_diagnostic_experiment,
)
from llm_mappo.formal_layout import FORMAL_MAP_SHA256, FORMAL_MAP_PATH
from llm_mappo.mappo_baseline.fp_state import FPStateSchema, build_fp_state
from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment
from llm_mappo.mappo_baseline.vector_env import new_baseline_environment


CONFIG_ROOT = Path("configs/formal_environment_training")


def test_formal_map_and_all_twelve_provisional_configs() -> None:
    from hashlib import sha256

    assert sha256(FORMAL_MAP_PATH.read_bytes()).hexdigest() == FORMAL_MAP_SHA256
    paths = sorted(CONFIG_ROOT.glob("*.yaml"))
    assert len(paths) == 12
    contracts = [load_baseline_experiment(path) for path in paths]
    assert {item.environment["scene_id"] for item in contracts} == set(SCENES)
    assert all(item.approval_status == "provisional" for item in contracts)
    assert all(item.training["environment_steps"] is None for item in contracts)
    assert all(item.training["num_env_workers"] == 16 for item in contracts)
    assert all(item.training["rollout_length"] == 64 for item in contracts)
    assert all(tuple(item.evaluation["seeds"]) == EVALUATION_SEEDS
               for item in contracts)
    assert len(set(EVALUATION_SEEDS)) == 200


@pytest.mark.parametrize("scene_id", tuple(SCENES))
def test_scene_reset_and_fp_dimensions(scene_id: str) -> None:
    scene = SCENES[scene_id]
    experiment = load_baseline_experiment(CONFIG_ROOT / f"{scene_id}-baseline.yaml")
    environment = new_baseline_environment(experiment.environment)
    try:
        first = environment.reset(seed=9300)
        initial = (
            [(agent.x, agent.y, agent.dir) for agent in environment.env.agents],
            [(task.shelf_id, task.label) for task in
             environment.env.task_queue.active_tasks],
        )
        second = environment.reset(seed=9300)
        assert np.array_equal(first, second)
        assert (
            [(agent.x, agent.y, agent.dir) for agent in environment.env.agents],
            [(task.shelf_id, task.label) for task in
             environment.env.task_queue.active_tasks],
        ) == initial
        assert len(environment.env.shelfs) == scene["n_shelves"]
        assert len(environment.env.task_queue.active_tasks) == scene["n_agents"]
        assert tuple(environment.env.charging_stations) == scene["charging_stations"]
        assert tuple(environment.env.picking_stations) == scene["picking_stations"]
        schema = FPStateSchema(
            scene["n_agents"], scene["n_shelves"],
            len(scene["charging_stations"]), len(scene["picking_stations"]),
        )
        assert build_fp_state(environment, schema).shape == (
            scene["n_agents"], schema.dimension
        )
        for _ in range(3):
            environment.step(np.zeros(scene["n_agents"], dtype=int))
        assert not environment.env._dynamic_ingress_enabled()
    finally:
        environment.close()


def test_provisional_config_refuses_long_training_before_artifacts(tmp_path) -> None:
    experiment = load_baseline_experiment(CONFIG_ROOT / "tiny-3ag-baseline.yaml")
    with pytest.raises(ValueError, match="provisional configuration"):
        BaselineRun(experiment, device="cpu", output_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_diagnostic_contract_is_bounded() -> None:
    experiment = load_baseline_experiment(CONFIG_ROOT / "tiny-3ag-baseline.yaml")
    diagnostic = make_diagnostic_experiment(experiment, environment_steps=50_000)
    assert diagnostic.training["environment_steps"] == 50_000
    assert diagnostic.evaluation["seeds"] == [9300, 9301, 9302]
    with pytest.raises(ValueError, match="up to 50k"):
        make_diagnostic_experiment(experiment, environment_steps=50_001)
    with pytest.raises(ValueError, match="up to 50k"):
        make_diagnostic_experiment(
            replace(experiment, training={**experiment.training, "seed": 17}),
            environment_steps=1000,
        )
