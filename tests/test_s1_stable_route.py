"""Regression and freeze tests for the S1 stable route A* restoration."""

from pathlib import Path

import numpy as np
import yaml

from llm_mappo.phase2_expert import AStarExpert
from llm_mappo.phase3_training import Phase3TrainingConfig
from llm_mappo.planner import ReservationTable
from rware.warehouse import Action, Direction


_S1_CONFIG = "configs/s1_phase3_dynamic_ingress.yaml"


def test_s1_config_freezes_stable_environment_contract():
    environment = yaml.safe_load(
        Path(_S1_CONFIG).read_text(encoding="utf-8")
    )["environment"]
    assert environment["n_agents"] == 3
    assert environment["task_completion_target"] == 9
    assert environment["batch_interval"] == 40
    assert environment["batch_size_range"] == [1, 3]
    assert environment["initial_priority_label"] == "A"
    assert environment["priority_schedule"] is None
    assert environment["request_queue_size"] == 4
    assert environment["battery_cost_scale"] == 1.1
    assert environment["charge_threshold"] == 0.3
    assert environment["charge_release_threshold"] == 0.8

    config = Phase3TrainingConfig.from_yaml(_S1_CONFIG)
    assert config.n_agents == 3
    assert config.task_completion_target == 9
    assert config.battery_cost_scale == 1.1
    assert config.charge_threshold == 0.3
    assert config.charge_release_threshold == 0.8


def test_s1_legacy_terminal_reservation_persists_full_horizon():
    horizon = 6
    path = [(0, 0), (1, 0)]

    legacy = ReservationTable(horizon)
    legacy.reserve(path, persistent=True)
    assert not legacy.is_available((1, 0), horizon)

    bounded = ReservationTable(horizon)
    bounded.reserve(path, terminal_hold_steps=2)
    assert not bounded.is_available((1, 0), 3)
    assert bounded.is_available((1, 0), horizon)


def test_s1_coordinator_yield_action_right_restores_legacy_turn():
    agents = [_FakeAgent(5, 5, Direction.RIGHT), _FakeAgent(6, 5, Direction.LEFT)]
    env = _FakeEnv(agents)
    actions = np.asarray([Action.FORWARD.value, Action.FORWARD.value])

    right = AStarExpert._coordinate_actions(env, actions, yield_action="right")
    noop = AStarExpert._coordinate_actions(env, actions, yield_action="noop")

    assert list(right) == [Action.RIGHT.value, Action.RIGHT.value]
    assert list(noop) == [Action.NOOP.value, Action.NOOP.value]


def test_s2_predecision_config_freezes_contract_with_astar_llm_kd():
    config = Phase3TrainingConfig.from_yaml(
        "configs/s2_phase3b_dynamic_ingress_astar_kl.yaml"
    )
    assert config.n_agents == 3
    assert config.task_completion_target == 9
    assert config.battery_cost_scale == 1.1
    assert config.charge_threshold == 0.3
    assert config.charge_release_threshold == 0.8
    assert config.phase == "3b"
    assert config.astar_kl_enabled is True
    assert config.ppo.engagement_coefficient == 0.1


class _FakeAgent:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.dir = direction


class _FakeEnv:
    def __init__(self, agents):
        self.env = _FakeWarehouse(agents)


class _FakeWarehouse:
    def __init__(self, agents):
        self.agents = agents

    def _forward_target(self, agent):
        x, y = agent.x, agent.y
        if agent.dir == Direction.UP:
            return x, y - 1
        if agent.dir == Direction.DOWN:
            return x, y + 1
        if agent.dir == Direction.LEFT:
            return x - 1, y
        return x + 1, y
