"""Contract tests for O0's branch-local warehouse snapshot."""

import numpy as np
import pytest

from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.shadow_state import EventAddressedRandomness, ShadowStateAdapter


def _environment() -> Phase2Warehouse:
    environment = Phase2Warehouse(
        n_agents=3,
        max_steps=80,
        batch_interval=8,
        batch_size_range=(1, 2),
        request_queue_size=3,
        task_completion_target=9,
        observation_schema=ObservationSchema.DIRECT_GOAL_V1,
    )
    environment.reset(seed=19)
    return environment


def test_snapshot_restores_two_isolated_branches_and_real_continuation():
    real = _environment()
    real.step([0, 1, 2])
    real_adapter = ShadowStateAdapter(real, code_commit="test-commit")
    snapshot = real_adapter.capture(
        run_seed=7,
        episode_index=2,
        episode_seed=19,
        environment_index=0,
        real_global_step=3,
        episode_step=1,
    )
    branch_one = _environment()
    branch_two = _environment()
    first = ShadowStateAdapter(branch_one, code_commit="test-commit")
    second = ShadowStateAdapter(branch_two, code_commit="test-commit")
    first.restore(snapshot)
    second.restore(snapshot)

    assert first.state_hash() == snapshot.state_hash
    assert second.state_hash() == snapshot.state_hash
    assert branch_one.env.agents[0] is not branch_two.env.agents[0]
    assert branch_one.env.task_queue is not branch_two.env.task_queue

    first_step = branch_one.step([0, 1, 2])
    second_step = branch_two.step([0, 1, 2])
    assert np.array_equal(first_step.observations, second_step.observations)
    assert first_step.team_reward == second_step.team_reward
    assert first_step.terminated == second_step.terminated
    assert first_step.info == second_step.info
    assert real_adapter.state_hash() == snapshot.state_hash


def test_snapshot_rejects_config_mismatch_and_restores_from_bytes():
    source = _environment()
    source_adapter = ShadowStateAdapter(source, code_commit="test-commit")
    snapshot = source_adapter.capture(
        run_seed=1,
        episode_index=0,
        episode_seed=19,
        environment_index=0,
        real_global_step=0,
        episode_step=0,
    )
    target = _environment()
    target_adapter = ShadowStateAdapter(target, code_commit="test-commit")
    target_adapter.restore_bytes(snapshot.to_bytes())
    assert target_adapter.state_hash() == snapshot.state_hash

    incompatible = Phase2Warehouse(
        n_agents=2,
        max_steps=80,
        observation_schema=ObservationSchema.DIRECT_GOAL_V1,
    )
    incompatible.reset(seed=19)
    with pytest.raises(ValueError, match="config hash"):
        ShadowStateAdapter(incompatible, code_commit="test-commit").restore(snapshot)


def test_event_addressed_randomness_has_no_branch_local_counter():
    randomness = EventAddressedRandomness()
    first = randomness.integer(
        episode_seed=19,
        real_global_step=3,
        shadow_offset=2,
        event_type="dynamic_ingress_batch_size",
        event_slot=0,
        low=1,
        high=4,
    )
    randomness.integer(
        episode_seed=19,
        real_global_step=3,
        shadow_offset=2,
        event_type="unrelated_event",
        event_slot=0,
        low=1,
        high=4,
    )
    again = randomness.integer(
        episode_seed=19,
        real_global_step=3,
        shadow_offset=2,
        event_type="dynamic_ingress_batch_size",
        event_slot=0,
        low=1,
        high=4,
    )
    assert first == again
