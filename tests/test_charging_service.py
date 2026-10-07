"""Regression coverage for the opt-in full-charge rule-layer service."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from llm_mappo.mappo_baseline.vector_env import new_baseline_environment
from llm_mappo.mappo_baseline.runner import _validate_environment
from llm_mappo.warehouse_environment import Phase2Warehouse
from rware.warehouse import Action


@pytest.fixture
def service():
    environment = Phase2Warehouse(
        n_agents=1, max_steps=30, deadlock_steps=1,
        charging_contract="goal-full-v1", charging_rate=0.1,
        charge_release_threshold=1.0,
    )
    environment.reset(seed=7)
    yield environment
    environment.close()


def _park(environment, battery):
    agent = environment.env.agents[0]
    agent.x, agent.y = environment.env.charging_stations[0]
    agent.battery = battery
    environment._charging_active.add(agent.id)
    environment.env._recalc_grid()
    return agent


@pytest.mark.parametrize("battery,steps", [(0.3, 7), (0.8, 2), (0.95, 1)])
def test_service_locks_until_full_without_an_extra_float_rounding_step(
    service, battery, steps,
):
    agent = _park(service, battery)
    position = agent.x, agent.y
    for index in range(steps):
        assert service.action_masks()[0].tolist() == [True, False, False, False, False]
        transition = service.step([Action.FORWARD])
        assert (agent.x, agent.y) == position
        assert agent.req_action == Action.NOOP
        assert agent.battery == pytest.approx(min(1.0, battery + 0.1 * (index + 1)))
        assert not transition.metrics.deadlocked
        assert any(event["type"] == "charged" for event in transition.info["events"])
    assert agent.battery == 1.0
    assert service.action_masks()[0, Action.FORWARD.value]
    assert agent.id not in service._charging_targets()


def test_direct_step_enforces_service_without_requesting_a_mask(service):
    agent = _park(service, 0.15)
    service.step([Action.RIGHT])
    assert agent.req_action == Action.NOOP
    assert agent.battery == pytest.approx(0.25)
    assert service.previous_action(agent.id) == Action.NOOP.value


def test_passer_and_agent_en_route_are_not_locked(service):
    agent = _park(service, 0.5)
    service._charging_active.clear()
    assert service.action_masks()[0, Action.RIGHT.value]
    service.step([Action.RIGHT])
    assert agent.req_action == Action.RIGHT
    agent.x, agent.y = 0, 1
    agent.battery = 0.15
    service.env._recalc_grid()
    assert service.action_masks()[0, Action.RIGHT.value]


def test_service_restores_reservations_and_active_goal(service):
    agent = _park(service, 0.15)
    service.step([Action.NOOP])
    warehouse_state = service.env.export_shadow_state()
    adapter_state = service.export_shadow_state()
    expected = service.step([Action.RIGHT])
    service.env.import_shadow_state(warehouse_state)
    service.import_shadow_state(adapter_state)
    restored = service.step([Action.RIGHT])
    assert agent.battery == pytest.approx(0.35)
    assert restored.info["events"] == expected.info["events"]
    assert np.array_equal(restored.observations, expected.observations)


def test_actual_station_must_match_the_reserved_charging_goal(service):
    agent = _park(service, 0.3)
    service.env.charging_reservations = {(0, 1): agent.id}
    direction = agent.dir
    service.env.step([Action.RIGHT])
    assert agent.req_action == Action.RIGHT
    assert agent.dir != direction


def test_vector_factory_applies_full_charge_contract_and_config_identity():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/core_integration/medium-5ag-baseline.yaml"
    )
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))["environment"]
    values.update(charging_contract="goal-full-v1", charging_rate=0.1,
                  charge_release_threshold=1.0)
    environment = new_baseline_environment(values)
    try:
        environment.reset(seed=7)
        agent = _park(environment, 0.15)
        assert environment.action_masks()[0].tolist() == [
            True, False, False, False, False,
        ]
        transition = environment.step([Action.NOOP.value] * environment.n_agents)
        assert agent.battery == pytest.approx(0.25)
        payload = environment.env.shadow_config_payload()
        assert payload["charging_contract"] == "goal-full-v1"
        assert payload["charging_rate"] == 0.1
        assert not transition.metrics.deadlocked
    finally:
        environment.close()


def test_legacy_default_keeps_passive_charging_and_releases_at_eighty_percent():
    environment = Phase2Warehouse(n_agents=1, max_steps=10)
    try:
        environment.reset(seed=7)
        agent = _park(environment, 0.79)
        assert environment.action_masks()[0, Action.RIGHT.value]
        environment.step([Action.NOOP])
        assert agent.battery == pytest.approx(0.81)
        assert agent.id not in environment._charging_targets()
        assert "charging_contract" not in environment.env.shadow_config_payload()
    finally:
        environment.close()


@pytest.mark.parametrize("overrides", [
    {"charging_rate": 0.02}, {"charge_release_threshold": 0.8},
    {"charging_contract": "unknown"},
])
def test_new_contract_rejects_incompatible_service_settings(overrides):
    settings = dict(
        charging_contract="goal-full-v1", charging_rate=0.1,
        charge_release_threshold=1.0,
    )
    settings.update(overrides)
    with pytest.raises(ValueError, match="charging"):
        Phase2Warehouse(**settings)


@pytest.mark.parametrize("overrides", [
    {}, {"charging_rate": 0.02}, {"charge_release_threshold": 0.8},
    {"charging_contract": "unknown"},
])
def test_runner_validates_explicit_full_charge_contract(overrides):
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs/llm_kd/phase_20261007/medium-5ag-baseline-goal-full.yaml"
    )
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))["environment"]
    values.update(overrides)
    if overrides:
        with pytest.raises(ValueError, match="charging"):
            _validate_environment(values)
    else:
        _validate_environment(values)
