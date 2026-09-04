from pathlib import Path

import numpy as np

from llm_mappo.mappo_baseline.runner import load_baseline_experiment
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer
from llm_mappo.mappo_baseline.vector_env import MAPPOVectorEnvironment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "optimization" / "r1_mappo_fp_dg_3ag_lowload.yaml"


def test_frozen_baseline_contract_is_exact_and_teacher_free():
    experiment = load_baseline_experiment(CONFIG)
    assert experiment.environment["n_agents"] == 3
    assert experiment.environment["observation_schema"] == (
        "direct-goal-observation-v1"
    )
    assert experiment.environment["goal_progress_reward"] == 0.10
    assert experiment.training["environment_steps"] == 500_000
    assert experiment.training["num_env_workers"] == 16
    assert experiment.training["rollout_length"] == 64
    assert experiment.training["ppo_epochs"] == 5
    assert experiment.training["num_minibatches"] == 1
    assert experiment.method == "MAPPO-FP-DG"
    for forbidden in ("astar", "semantic", "calibration"):
        assert forbidden not in experiment.training
        assert forbidden not in experiment.environment


def test_vector_worker_returns_fp_state_and_independent_agent_axes():
    experiment = load_baseline_experiment(CONFIG)
    values = dict(experiment.environment)
    pool = MAPPOVectorEnvironment(values, count=2, base_seed=91)
    try:
        assert pool.observations.shape == (2, 3, 613)
        assert pool.fp_states.shape[:2] == (2, 3)
        assert pool.action_masks.shape == (2, 3, 5)
        responses = pool.step(np.zeros((2, 3), dtype=np.int64), global_step=0)
        assert responses.rewards.shape == (2, 3)
        assert responses.dones.shape == (2, 3)
        assert responses.active_masks.shape == (2, 3)
        assert pool.observations.shape == (2, 3, 613)
    finally:
        pool.close()


def test_two_worker_rollout_reaches_one_canonical_update():
    experiment = load_baseline_experiment(CONFIG)
    pool = MAPPOVectorEnvironment(experiment.environment, count=2, base_seed=101)
    trainer = MAPPOTrainer(
        MAPPOConfig(
            observation_dim=613,
            state_dim=pool.fp_states.shape[-1],
            ppo_epochs=1,
            num_minibatches=1,
        ),
        device="cpu",
    )
    storage = MAPPORolloutStorage(
        rollout_length=2,
        num_workers=2,
        num_agents=3,
        observation_dim=613,
        state_dim=pool.fp_states.shape[-1],
        action_dim=5,
    )
    try:
        for global_step in (0, 2):
            observations = pool.observations.copy()
            states = pool.fp_states.copy()
            masks = pool.action_masks.copy()
            active = pool.active_masks.copy()
            actions, log_probs, values, predictions = trainer.act(
                observations, states, masks, deterministic=False
            )
            result = pool.step(actions, global_step=global_step)
            storage.insert(
                observations=observations,
                fp_states=states,
                actions=actions,
                action_masks=masks,
                log_probs=log_probs,
                values=values,
                value_predictions=predictions,
                rewards=result.rewards,
                dones=result.dones,
                active_masks=active,
            )
        batch = storage.compute_returns(
            trainer.values(pool.fp_states), gamma=0.99, gae_lambda=0.95
        )
        metrics = trainer.update(batch)
        assert metrics["optimizer_steps"] == 1
        assert result.planner_query_count == 0
    finally:
        pool.close()


def test_worker_snapshot_restore_recovers_observation_and_fp_state():
    experiment = load_baseline_experiment(CONFIG)
    pool = MAPPOVectorEnvironment(experiment.environment, count=1, base_seed=211)
    try:
        observations = pool.observations.copy()
        fp_states = pool.fp_states.copy()
        states = pool.snapshot()
        pool.step(np.zeros((1, 3), dtype=np.int64), global_step=0)
        pool.restore(states)
        np.testing.assert_array_equal(pool.observations, observations)
        np.testing.assert_array_equal(pool.fp_states, fp_states)
    finally:
        pool.close()
