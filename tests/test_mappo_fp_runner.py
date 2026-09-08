from pathlib import Path
from dataclasses import replace

import numpy as np

from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer
from llm_mappo.mappo_baseline.vector_env import MAPPOVectorEnvironment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase1" / "medium-3ag-low-workload.yaml"


def test_phase1_baseline_contract_is_teacher_free_and_cpu_executable():
    experiment = load_baseline_experiment(CONFIG)
    assert experiment.environment["n_agents"] == 3
    assert experiment.environment["observation_schema"] == (
        "direct-goal-observation-v1"
    )
    assert experiment.environment["reward_version"] == "reward-v2"
    assert experiment.training["environment_steps"] == 300_000
    assert experiment.training["num_env_workers"] == 6
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
        assert pool.observations.shape == (2, 3, 208)
        assert pool.fp_states.shape[:2] == (2, 3)
        assert pool.action_masks.shape == (2, 3, 5)
        responses = pool.step(np.zeros((2, 3), dtype=np.int64), global_step=0)
        assert responses.rewards.shape == (2, 3)
        assert responses.dones.shape == (2, 3)
        assert responses.active_masks.shape == (2, 3)
        assert pool.observations.shape == (2, 3, 208)
    finally:
        pool.close()


def test_vector_worker_exposes_compact_structured_actor_observations():
    experiment = load_baseline_experiment(CONFIG)
    pool = MAPPOVectorEnvironment(experiment.environment, count=1, base_seed=92)
    try:
        # self (16) + DirectGoal (7) + 5x5x5 local map (125) + 3 peers (20)
        assert pool.observations.shape == (1, 3, 208)
    finally:
        pool.close()


def test_two_worker_rollout_reaches_one_canonical_update():
    experiment = load_baseline_experiment(CONFIG)
    pool = MAPPOVectorEnvironment(experiment.environment, count=2, base_seed=101)
    trainer = MAPPOTrainer(
        MAPPOConfig(
            observation_dim=208,
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
        observation_dim=208,
        state_dim=pool.fp_states.shape[-1],
        action_dim=5,
    )
    actor_hidden = np.zeros((2, 3, 64), dtype=np.float32)
    critic_hidden = np.zeros((2, 3, 64), dtype=np.float32)
    try:
        for global_step in (0, 2):
            observations = pool.observations.copy()
            states = pool.fp_states.copy()
            masks = pool.action_masks.copy()
            active = pool.active_masks.copy()
            actions, log_probs, values, predictions, next_actor, next_critic = trainer.act(
                observations, states, masks, actor_hidden, critic_hidden,
                deterministic=False
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
                actor_hidden_states=actor_hidden,
                critic_hidden_states=critic_hidden,
            )
            continuation = (
                (~result.dones).astype(np.float32)[..., None]
                * result.active_masks[..., None]
            )
            actor_hidden = next_actor * continuation
            critic_hidden = next_critic * continuation
        batch = storage.compute_returns(
            trainer.values(pool.fp_states, critic_hidden)[0], gamma=0.99,
            gae_lambda=0.95
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


def test_checkpoint_resume_restores_actor_and_critic_hidden_states(tmp_path):
    original = load_baseline_experiment(CONFIG)
    training = dict(original.training)
    training.update({"num_env_workers": 1, "rollout_length": 1})
    experiment = replace(original, training=training)
    run = BaselineRun(experiment, device="cpu", output_root=tmp_path)
    try:
        run.actor_hidden_states.fill(0.25)
        run.critic_hidden_states.fill(-0.5)
        run._checkpoint(run.directory / "checkpoint_latest.pt")
        directory = run.directory
    finally:
        run.close()

    resumed = BaselineRun(experiment, device="cpu", resume=directory)
    try:
        np.testing.assert_array_equal(
            resumed.actor_hidden_states, np.full((1, 3, 64), 0.25)
        )
        np.testing.assert_array_equal(
            resumed.critic_hidden_states, np.full((1, 3, 64), -0.5)
        )
    finally:
        resumed.close()
