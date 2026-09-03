import copy

import numpy as np

from llm_mappo.e1_vector_env import _agent_state, _new_environment
from llm_mappo.r1_diagnostics import (
    load_r1c_diagnostic,
    r1c_identity,
    r1c_trend,
)
from llm_mappo.r1_evaluation import _write_evaluation_plot
from llm_mappo.semantic_v3 import SemanticViewV3


_CONFIG = "configs/optimization/r1_4agv_lowload.yaml"


def test_r1c_config_expands_the_frozen_four_arms():
    expected = {
        "legacy-r128": ("legacy-v1", 128),
        "reward-v2-r128": ("reward-v2", 128),
        "legacy-r32": ("legacy-v1", 32),
        "reward-v2-r32": ("reward-v2", 32),
    }
    for arm, (reward, rollout) in expected.items():
        diagnostic = load_r1c_diagnostic(_CONFIG, arm)
        assert diagnostic.environment["n_agents"] == 4
        assert diagnostic.environment["batch_size_range"] == [2, 4]
        assert diagnostic.environment["reward_version"] == reward
        assert diagnostic.training["num_env_workers"] == 16
        assert diagnostic.training["rollout_length"] == rollout
        assert diagnostic.run.astar_kd == "disabled"
        assert diagnostic.run.semantic_control == "none"


def test_r1c_identity_binds_rollout_and_environment_profile():
    diagnostic = load_r1c_diagnostic(_CONFIG, "reward-v2-r128")
    first = r1c_identity(code_commit="a" * 40, diagnostic=diagnostic,
        raw_records_sha256="b" * 64, layout_hash="layout", initial_parameter_sha256="c" * 64)
    altered = copy.deepcopy(diagnostic)
    altered_training = {**altered.training, "rollout_length": 32}
    second = r1c_identity(code_commit="a" * 40,
        diagnostic=type(diagnostic)(arm=altered.arm, environment=altered.environment,
            training=altered_training, evaluation_seeds=altered.evaluation_seeds,
            artifact_root=altered.artifact_root), raw_records_sha256="b" * 64,
        layout_hash="layout", initial_parameter_sha256="c" * 64)
    assert first["training_sha256"] != second["training_sha256"]


def test_r1c_trend_requires_40_complete_episodes_and_positive_change():
    insufficient = r1c_trend([{"task_completion_rate": 0.5}] * 39)
    assert insufficient == {"complete_episode_count": 39, "trend_available": False,
                            "trend_pass": False}
    rows = ([{"task_completion_rate": 0.2}] * 20 +
            [{"task_completion_rate": 0.7}] * 20)
    evidence = r1c_trend(rows)
    assert evidence["trend_available"] and evidence["trend_pass"]
    assert evidence["first_20_mean_completion_rate"] < evidence["last_20_mean_completion_rate"]


def test_r1c_four_agents_have_direct_goal_observations_and_three_real_peers():
    diagnostic = load_r1c_diagnostic(_CONFIG, "reward-v2-r128")
    environment = _new_environment(diagnostic.environment, diagnostic.run)
    try:
        observations = environment.reset(seed=9107)
        assert observations.shape == (4, 613)
        assert environment.action_masks().shape == (4, 5)
        warehouse = environment.env
        width, height = warehouse.grid_size[1], warehouse.grid_size[0]
        for agent in warehouse.agents:
            peers = [_agent_state(environment, peer,
                environment._target_for_agent(peer.id)[1])
                for peer in warehouse.agents if peer.id != agent.id]
            view = SemanticViewV3.from_state(warehouse.shadow_layout_hash(), width, height,
                _agent_state(environment, agent, environment._target_for_agent(agent.id)[1]), peers)
            assert len(peers) == 3
            assert view.vector.shape == (61,)
        environment.step(np.zeros(4, dtype=np.int64))
        assert environment.planner_query_counter.count == 0
    finally:
        environment.close()


def test_r1c_evaluation_plot_is_written_without_a_plotting_dependency(tmp_path):
    output = tmp_path / "evaluation_metrics.png"
    _write_evaluation_plot(output, [{"seed": 9300, "metrics": {
        "task_completion_rate": 0.2, "completed_tasks": 4, "collisions": 1,
    }}])
    assert output.is_file()
