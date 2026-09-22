"""Deterministic tests for the balanced-random delivery contract."""

from collections import Counter
from pathlib import Path

import pytest

from llm_mappo.warehouse_environment import Phase2Warehouse
from llm_mappo.mappo_baseline.runner import (
    checkpoint_schema_for_experiment,
    load_baseline_experiment,
)
from llm_mappo.types import PriorityAdjustment
from rware.warehouse import Action, Direction


BALANCED_RANDOM = "balanced-random-v1"
ROOT = Path(__file__).resolve().parents[1]


def _environment(**overrides):
    values = {
        "n_agents": 1,
        "max_steps": 32,
        "deadlock_steps": 31,
        "batch_interval": 40,
        "batch_size_range": (6, 6),
        "initial_ingress_batches": 1,
        "request_queue_size": 6,
        "task_completion_target": 12,
        "terminate_on_task_target": False,
        "replenishment_mode": "rware_delivery",
        "priority_schedule": ("A", "B", "C"),
        "delivery_contract": BALANCED_RANDOM,
    }
    values.update(overrides)
    return Phase2Warehouse(**values)


def test_priority_bag_balances_every_triplet_and_weights_stay_fixed():
    """Catch batch-level labels or weights derived from the active pool."""

    environment = _environment()
    try:
        environment.reset(seed=17)
        tasks = sorted(
            environment.env.task_queue.active_tasks,
            key=lambda task: task.task_id,
        )
        letters = [task.label[0] for task in tasks]

        assert Counter(letters[:3]) == Counter({"A": 1, "B": 1, "C": 1})
        assert Counter(letters[3:6]) == Counter({"A": 1, "B": 1, "C": 1})

        for task in tasks:
            if task.label.startswith("A"):
                environment.env.task_queue.complete(task.task_id, completed_step=1)
        assert environment.env.task_queue.priority_weight("B1") == pytest.approx(1.5)
        assert environment.env.task_queue.priority_weight("C1") == pytest.approx(1.0)
    finally:
        environment.close()


def test_task_assignment_is_seeded_random_not_agent_order_fifo():
    """Catch fixed agent-ID/FIFO matching while preserving replayability."""

    assignments = []
    first_agent_priorities = []
    for seed in range(1, 7):
        environment = _environment(
            n_agents=3,
            batch_size_range=(3, 3),
            request_queue_size=3,
        )
        replay = _environment(
            n_agents=3,
            batch_size_range=(3, 3),
            request_queue_size=3,
        )
        try:
            environment.reset(seed=seed)
            replay.reset(seed=seed)
            assignment = tuple(
                environment.env.task_queue.task_for_agent(agent_id).task_id
                for agent_id in (1, 2, 3)
            )
            replayed = tuple(
                replay.env.task_queue.task_for_agent(agent_id).task_id
                for agent_id in (1, 2, 3)
            )
            assert replayed == assignment
            assignments.append(assignment)
            first_agent_priorities.append(
                environment.env.task_queue.task_for_agent(1).label[0]
            )
        finally:
            environment.close()
            replay.close()

    assert len(set(assignments)) > 1
    assert len(set(first_agent_priorities)) > 1


def test_completed_delivery_returns_shelf_home_then_receives_replacement_task():
    """Catch an AGV remaining permanently loaded after its first completion."""

    environment = _environment(
        batch_size_range=(1, 1),
        request_queue_size=1,
    )
    try:
        environment.reset(seed=3)
        warehouse = environment.env
        first_task = warehouse.task_queue.task_for_agent(1)
        shelf = warehouse.shelfs[first_task.shelf_id - 1]
        home = warehouse._shelf_home[shelf.id]
        agent = warehouse.agents[0]
        goal_x, goal_y = warehouse.goals[0]
        agent.x, agent.y, agent.dir = goal_x, goal_y - 1, Direction.DOWN
        agent.carrying_shelf = shelf
        shelf.x, shelf.y = agent.x, agent.y
        warehouse._recalc_grid()

        transition = environment.step([Action.FORWARD])

        assert transition.metrics.completed_tasks == 1
        assert len(warehouse.task_queue.active_tasks) == 1
        assert environment._target_for_agent(1) == (home, "delivery")
        assert warehouse.task_queue.task_for_agent(1) is None

        agent.picking_lock_steps = 0
        agent.x, agent.y = home
        shelf.x, shelf.y = home
        warehouse._recalc_grid()
        masks = environment.action_masks()[0]
        assert masks.tolist() == [False, False, False, False, True]

        environment.step([Action.TOGGLE_LOAD])

        replacement = warehouse.task_queue.task_for_agent(1)
        assert agent.carrying_shelf is None
        assert replacement is not None
        assert replacement.task_id != first_task.task_id
    finally:
        environment.close()


def test_idle_low_battery_agent_does_not_lock_a_pending_task():
    """Catch task assignment to an idle AGV already committed to charging."""

    environment = _environment(
        batch_size_range=(1, 1),
        request_queue_size=1,
        charge_threshold=0.3,
    )
    try:
        environment.reset(seed=5)
        warehouse = environment.env
        agent = warehouse.agents[0]
        warehouse.task_queue.release_agent(agent.id)
        agent.task_id = None
        agent.battery = 0.2

        warehouse._assign_available_tasks()

        assert warehouse.task_queue.task_for_agent(agent.id) is None
        assert warehouse.task_queue.active_tasks[0].assigned_agent_id is None
    finally:
        environment.close()


def test_idle_agent_in_charge_hysteresis_does_not_lock_a_pending_task():
    """Catch assignment before an active charging detour reaches release level."""

    environment = _environment(
        batch_size_range=(1, 1),
        request_queue_size=1,
        charge_threshold=0.3,
        charge_release_threshold=0.8,
    )
    try:
        environment.reset(seed=6)
        warehouse = environment.env
        agent = warehouse.agents[0]
        warehouse.task_queue.release_agent(agent.id)
        agent.task_id = None
        agent.battery = 0.5
        environment._charging_active.add(agent.id)
        environment._charging_targets()

        warehouse._assign_available_tasks()

        assert warehouse.task_queue.task_for_agent(agent.id) is None
    finally:
        environment.close()


def test_shadow_restore_recovers_all_delivery_random_streams_and_priority_bag():
    """Catch resume drift in replacement, priority, or assignment randomness."""

    source = _environment(
        n_agents=3,
        batch_size_range=(3, 3),
        request_queue_size=3,
    )
    restored = _environment(
        n_agents=3,
        batch_size_range=(3, 3),
        request_queue_size=3,
    )
    try:
        source.reset(seed=29)
        snapshot = source.env.export_shadow_state()
        restored.reset(seed=999)
        restored.env.import_shadow_state(snapshot)

        source_delivered = source.env.request_queue[0]
        restored_delivered = restored.env.request_queue[0]
        source.env._renew_delivered_request(source_delivered)
        restored.env._renew_delivered_request(restored_delivered)
        assert [shelf.id for shelf in restored.env.request_queue] == [
            shelf.id for shelf in source.env.request_queue
        ]

        source_created = source.env._create_batch((140, 141, 142))
        restored_created = restored.env._create_batch((140, 141, 142))
        assert [task.label for task in restored_created] == [
            task.label for task in source_created
        ]

        for environment in (source, restored):
            for agent in environment.env.agents:
                environment.env.task_queue.release_agent(agent.id)
                agent.task_id = None
            environment.env._assign_available_tasks()
        assert [
            restored.env.task_queue.task_for_agent(agent_id).task_id
            for agent_id in (1, 2, 3)
        ] == [
            source.env.task_queue.task_for_agent(agent_id).task_id
            for agent_id in (1, 2, 3)
        ]
    finally:
        source.close()
        restored.close()


def test_balanced_delivery_rejects_runtime_priority_adjustments():
    """Catch any path that lets an online component rewrite task urgency."""

    environment = _environment(
        batch_size_range=(1, 1),
        request_queue_size=1,
    )
    try:
        environment.reset(seed=7)
        task = environment.env.task_queue.active_tasks[0]
        adjustment = PriorityAdjustment(
            task=task.label,
            new_label=f"B{task.label[1:]}",
            reason="online rewrite must be rejected",
        )

        with pytest.raises(ValueError, match="fixed"):
            environment.env.apply_priority_adjustments((adjustment,))
    finally:
        environment.close()


def test_all_core_configs_share_versioned_balanced_delivery_contract():
    """Catch mixed train/eval replenishment or reuse of the old namespace."""

    configs = sorted((ROOT / "configs" / "core_integration").glob("*.yaml"))
    assert len(configs) == 4
    for config in configs:
        experiment = load_baseline_experiment(config)
        assert experiment.schema == "core-mappo-209d-delivery-v2"
        assert experiment.approval_status == "provisional"
        assert experiment.environment["priority_schedule"] == ["A", "B", "C"]
        assert experiment.environment["delivery_contract"] == BALANCED_RANDOM
        assert (
            experiment.environment["training_replenishment_mode"]
            == "rware_delivery"
        )
        assert experiment.evaluation["replenishment_mode"] == "rware_delivery"
        assert checkpoint_schema_for_experiment(experiment).endswith(
            "209d-delivery-checkpoint-v2"
        )
