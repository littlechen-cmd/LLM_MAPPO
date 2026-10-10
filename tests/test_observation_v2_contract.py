"""Native 209D Actor-observation integration contract."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment
from llm_mappo.mappo_baseline import structured_observation as observation
from llm_mappo.rules import TaskQueue
from llm_mappo.warehouse_environment import Phase2Warehouse


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = ROOT / "configs/core_integration/medium-5ag-baseline.yaml"


def test_v2_inserts_only_focal_priority_after_v1_self_features():
    v1 = observation.StructuredObservationSchema(
        observation_schema=observation.DIRECT_GOAL_OBSERVATION_V1
    )
    v2 = observation.StructuredObservationSchema(
        observation_schema=observation.DIRECT_GOAL_OBSERVATION_V2
    )

    assert v1.dimension == 208
    assert v2.dimension == 209
    assert v2.feature_names[10] == "self_priority_weight"
    assert v2.feature_names[:10] == v1.feature_names[:10]
    assert v2.feature_names[11:] == v1.feature_names[10:]

    with pytest.raises(ValueError, match="Unknown MAPPO Actor observation schema"):
        observation.StructuredObservationSchema(observation_schema="unsupported")


def test_v2_deletes_exactly_to_v1_and_encodes_focal_priority():
    environment = Phase2Warehouse(n_agents=3, max_steps=8)
    environment.reset(seed=7)
    try:
        queue = TaskQueue()
        queue.create_batch((1,), batch_id=1, letter="B", arrival_step=0)
        queue.assign_next(agent_id=1, battery=1.0)
        for batch_id, label in enumerate(("A", "C", "D"), start=2):
            queue.create_batch(
                (batch_id,), batch_id=batch_id, letter=label, arrival_step=0
            )
        environment.env.task_queue = queue
        v1 = observation.build_structured_observations(
            environment,
            observation.StructuredObservationSchema(
                observation_schema=observation.DIRECT_GOAL_OBSERVATION_V1
            ),
        )
        v2 = observation.build_structured_observations(
            environment,
            observation.StructuredObservationSchema(
                observation_schema=observation.DIRECT_GOAL_OBSERVATION_V2
            ),
        )
        np.testing.assert_array_equal(np.delete(v2, 10, axis=1), v1)
        np.testing.assert_array_equal(v2[:, 10], [0.75, 0.0, 0.0])
    finally:
        environment.close()


def test_core_baseline_propagates_native_209d(tmp_path):
    experiment = load_baseline_experiment(BASELINE_CONFIG)
    experiment = replace(experiment, approval_status="approved")
    run = BaselineRun(experiment, device="cpu", output_root=tmp_path)
    try:
        assert experiment.environment["observation_schema"] == (
            observation.DIRECT_GOAL_OBSERVATION_V2
        )
        assert run.actor_schema.dimension == 209
        assert run.trainer.config.observation_dim == 209
        assert run.pool.observations.shape[-1] == 209
    finally:
        run.close()
