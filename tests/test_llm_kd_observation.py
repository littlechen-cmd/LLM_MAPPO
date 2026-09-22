from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from llm_mappo.mappo_baseline.runner import (
    BaselineRun,
    load_baseline_experiment,
)
from llm_mappo.mappo_baseline.structured_observation import (
    StructuredObservationSchema,
    build_structured_observations,
)
from llm_mappo.mappo_baseline.vector_env import MAPPOVectorEnvironment
from llm_mappo.warehouse_environment import Phase2Warehouse
from llm_mappo.warehouse_observation import ObservationSchema


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "core_integration" / "medium-5ag-baseline.yaml"
V1 = "direct-goal-observation-v1"
V2 = "direct-goal-observation-v2"


def _environment() -> Phase2Warehouse:
    environment = Phase2Warehouse(
        n_agents=3,
        max_steps=20,
        env_id="llm-mappo-medium-3ag-v1",
        batch_interval=40,
        batch_size_range=(3, 3),
        initial_priority_label="A",
        priority_schedule=("A", "B", "C", "D"),
        request_queue_size=3,
        task_completion_target=3,
        observation_schema=ObservationSchema.DIRECT_GOAL_V1,
    )
    environment.reset(seed=7)
    return environment


def _experiment(observation_schema: str):
    source = load_baseline_experiment(CONFIG)
    environment = {**source.environment, "observation_schema": observation_schema}
    training = {**source.training, "num_env_workers": 1, "rollout_length": 1}
    return replace(
        source,
        environment=environment,
        training=training,
        approval_status="approved",
    )


def test_v2_adds_only_own_priority_at_feature_index_ten() -> None:
    """A misplaced/new extra feature must break the exact v1/v2 ledger."""
    environment = _environment()
    try:
        v1 = StructuredObservationSchema(observation_schema=V1)
        v2 = StructuredObservationSchema(observation_schema=V2)
        v1_observations = build_structured_observations(environment, v1)
        v2_observations = build_structured_observations(environment, v2)

        assert v1.dimension == 208
        assert v2.dimension == 209
        assert v2.feature_names[10] == "self_priority_weight"
        assert v2.feature_names[:10] == v1.feature_names[:10]
        assert v2.feature_names[11:] == v1.feature_names[10:]
        assert v1_observations.shape == (3, 208)
        assert v2_observations.shape == (3, 209)

        for row, agent in zip(v2_observations, environment.env.agents):
            task = environment.env.task_queue.task_for_agent(agent.id)
            assert task is not None
            expected = environment.env.task_queue.priority_weight(task.label) / 2.0
            assert row[10] == pytest.approx(expected)
    finally:
        environment.close()


def test_v2_uses_zero_own_priority_when_the_focal_agent_has_no_task() -> None:
    """An idle focal AGV must not inherit another task's priority."""
    environment = _environment()
    try:
        focal_id = environment.env.agents[0].id
        environment.env.task_queue.release_agent(focal_id)
        schema = StructuredObservationSchema(observation_schema=V2)

        observation = build_structured_observations(environment, schema)[0]

        assert environment.env.task_queue.task_for_agent(focal_id) is None
        assert observation[10] == 0.0
    finally:
        environment.close()


def test_vector_worker_selects_the_v2_actor_observation_from_config() -> None:
    """Ignoring the configured schema would silently train a 208D Actor."""
    experiment = _experiment(V2)
    pool = MAPPOVectorEnvironment(experiment.environment, count=1, base_seed=23)
    try:
        assert pool.observations.shape == (1, 5, 209)
    finally:
        pool.close()


def test_v1_checkpoint_is_rejected_by_a_v2_run_before_loading_weights(
    tmp_path: Path,
) -> None:
    """A v1 trainer checkpoint must never migrate into the v2 Actor."""
    source = BaselineRun(
        _experiment(V1), device="cpu", output_root=tmp_path / "v1"
    )
    checkpoint = source.directory / "checkpoint.pt"
    try:
        source._checkpoint(checkpoint)
    finally:
        source.close()

    target = BaselineRun(
        _experiment(V2), device="cpu", output_root=tmp_path / "v2"
    )
    try:
        assert target.actor_schema.dimension == 209
        with pytest.raises(ValueError, match="identity differs"):
            target._restore(checkpoint)
    finally:
        target.close()
