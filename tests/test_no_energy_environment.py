"""No-energy diagnostic preserves task dynamics and neutralizes energy roles."""
import numpy as np
import pytest

from llm_mappo.warehouse_environment import Phase2Warehouse
from llm_mappo.mappo_baseline.structured_observation import (
    StructuredObservationSchema, build_structured_observations,
)
from llm_mappo.mappo_baseline.fp_state import FPStateSchema, build_fp_state
from rware.warehouse import Action


def make_env(**kwargs):
    return Phase2Warehouse(
        n_agents=5, max_steps=800, deadlock_steps=180,
        charging_stations=((0, 0), (19, 0), (0, 23), (19, 23), (9, 0)),
        picking_stations=((9, 21), (10, 21)),
        replenishment_mode="rware_delivery",
        delivery_contract="balanced-random-v1", request_queue_size=5,
        priority_schedule=("A", "B", "C"),
        initial_ingress_batches=1, terminate_on_task_target=False,
        observation_schema="direct-goal-observation-v1", **kwargs,
    )


def test_disabled_energy_freezes_battery_and_charging_metrics():
    env = make_env(energy_enabled=False, charge_event_reward=2.0)
    try:
        env.reset(seed=7)
        w = env.env
        a = w.agents[0]
        a.x, a.y = w.charging_stations[0]
        a.battery = 0.00001
        w._recalc_grid()
        env._charging_active.add(a.id)
        w.charging_reservations[(a.x, a.y)] = a.id
        result = env.step([Action.NOOP] * 5)
        assert all(agent.battery == 1.0 and not agent.dead for agent in w.agents)
        assert env._charging_targets() == {}
        assert not w.charging_reservations
        metrics = env.metrics.as_dict()
        for key in ("charged_events", "energy_deaths", "charger_arrivals",
                    "charging_target_steps", "low_battery_triggers",
                    "charging_wait_steps", "task_recoveries", "reward_charging",
                    "reward_energy"):
            assert metrics[key] == 0, key
        assert metrics["minimum_battery"] == 1.0
        assert not any(e["type"] in {"charged", "agent_dead"}
                       for e in w.last_events)
    finally:
        env.close()


def test_disabled_energy_actor_and_critic_are_neutral_without_shape_change():
    env = make_env(energy_enabled=False)
    try:
        env.reset(seed=7)
        w = env.env
        w.agents[0].x, w.agents[0].y = w.charging_stations[0]
        w._recalc_grid()
        for a in w.agents:
            a.battery = 0.2
        actor_schema = StructuredObservationSchema(
            observation_schema="direct-goal-observation-v2")
        obs = build_structured_observations(env, actor_schema)
        assert obs.shape == (5, 209)
        for i, name in enumerate(actor_schema.feature_names):
            if (name.endswith("_charger") or name.endswith("_phase_charging")
                    or name == "goal_type_charging"):
                assert np.all(obs[:, i] == 0), name
            if name.endswith("_battery"):
                assert np.all(obs[:, i] == 1)
        schema = FPStateSchema(5, 144, 5, 2)
        state = build_fp_state(env, schema)
        assert state.shape == (5, 689)
        for i, name in enumerate(schema.feature_names):
            if name.endswith("_battery"):
                assert np.all(state[:, i] == 1), name
            if (name.startswith("charger_") or name == "occupied_charger_fraction"
                    or name.endswith("_target_charging")):
                assert np.all(state[:, i] == 0), name
    finally:
        env.close()


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_energy_flag_requires_boolean(value):
    with pytest.raises(ValueError, match="energy_enabled"):
        make_env(energy_enabled=value)


def test_default_enabled_matches_explicit_enabled_seeded_transitions():
    a, b = make_env(), make_env(energy_enabled=True)
    try:
        assert np.array_equal(a.reset(seed=7), b.reset(seed=7))
        for _ in range(3):
            sa, sb = a.step([0] * 5), b.step([0] * 5)
            assert np.array_equal(sa.observations, sb.observations)
            assert sa.team_reward == sb.team_reward
        assert a.env.agents[0].battery < 1
        assert "energy_enabled" not in a.env.shadow_config_payload()
        assert "energy_enabled" not in b.env.shadow_config_payload()
    finally:
        a.close()
        b.close()


def test_disabled_energy_still_terminates_for_deadlock():
    env = make_env(energy_enabled=False)
    try:
        env.reset(seed=7)
        for _ in range(180):
            result = env.step([0] * 5)
        assert env.metrics.deadlocked
        assert not any(a.dead for a in env.env.agents)
    finally:
        env.close()


def test_no_energy_reservations_do_not_block_assignment_at_low_battery():
    env = make_env(energy_enabled=False)
    try:
        env.reset(seed=7)
        w = env.env
        a = w.agents[0]
        w.task_queue.release_agent(a.id)
        a.task_id = None
        a.battery = 0
        w.charging_reservations[w.charging_stations[0]] = a.id
        w._assign_available_tasks()
        assert a.task_id is not None
        assert a.battery == 1
        assert not w.charging_reservations
    finally:
        env.close()


def test_no_energy_hard_collision_rejection_remains_active():
    from rware.warehouse import Direction
    env = make_env(energy_enabled=False)
    try:
        env.reset(seed=7)
        w = env.env
        a, b = w.agents[:2]
        a.x, a.y, a.dir = 0, 0, Direction.RIGHT
        b.x, b.y = 1, 0
        w._recalc_grid()
        env.step([Action.FORWARD, 0, 0, 0, 0])
        assert (a.x, a.y) == (0, 0)
        assert (b.x, b.y) == (1, 0)
        assert env.metrics.collisions == 1
        assert len({(p.x, p.y) for p in w.agents}) == 5
    finally:
        env.close()


def test_no_energy_can_complete_replenished_tasks_after_returning_shelves():
    from llm_mappo.planner import AStarPlanner
    env = Phase2Warehouse(
        n_agents=1, max_steps=3000, deadlock_steps=500,
        energy_enabled=False, replenishment_mode="rware_delivery",
        delivery_contract="balanced-random-v1", priority_schedule=("A", "B", "C"),
        initial_ingress_batches=1, request_queue_size=1,
        terminate_on_task_target=False, observation_schema="direct-goal-observation-v1",
    )
    try:
        env.reset(seed=9300)
        w, planner = env.env, AStarPlanner()
        a = w.agents[0]
        def follow(target):
            for _ in range(400):
                if (a.x, a.y) == target:
                    return
                plan = planner.plan(w, a.id, target)
                assert plan.reached_goal
                result = env.step([int(np.argmax(plan.action_preferences))])
                assert not result.truncated
            pytest.fail("Oracle could not reach task-cycle target")
        for expected in (1, 2, 3):
            task = w.task_queue.task_for_agent(a.id)
            assert task is not None
            shelf = w.shelfs[task.shelf_id - 1]
            home = w._shelf_home[shelf.id]
            follow((shelf.x, shelf.y))
            env.step([Action.TOGGLE_LOAD])
            assert a.carrying_shelf is shelf
            target, phase = env._target_for_agent(a.id)
            assert phase == "delivery"
            follow(target)
            assert w._completed_task_count() == expected
            assert len(w.task_queue.active_tasks) == 1
            follow(home)
            while a.picking_lock_steps:
                env.step([Action.NOOP])
            env.step([Action.TOGGLE_LOAD])
            assert a.carrying_shelf is None
            assert (shelf.x, shelf.y) == home
            assert a.battery == 1 and not a.dead
        assert env.metrics.charged_events == env.metrics.energy_deaths == 0
    finally:
        env.close()


def test_disabled_shadow_roundtrip_preserves_energy_identity_and_transitions():
    first = make_env(energy_enabled=False)
    restored = make_env(energy_enabled=False)
    try:
        first.reset(seed=7)
        first.step([0] * 5)
        restored.reset(seed=99)
        restored.env.import_shadow_state(first.env.export_shadow_state())
        restored.import_shadow_state(first.export_shadow_state())
        assert first.env.shadow_config_payload()["energy_enabled"] is False
        assert restored.env.shadow_config_payload()["energy_enabled"] is False
        a = first.step([0] * 5)
        b = restored.step([0] * 5)
        assert np.array_equal(a.observations, b.observations)
        assert a.team_reward == b.team_reward
        assert all(agent.battery == 1 for agent in restored.env.agents)
    finally:
        first.close()
        restored.close()


def test_energy_switch_does_not_shift_existing_positional_parameters():
    from inspect import Parameter, signature
    parameters = signature(Phase2Warehouse).parameters
    assert parameters["energy_enabled"].kind is Parameter.KEYWORD_ONLY
