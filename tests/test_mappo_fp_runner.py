from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage
from llm_mappo.mappo_baseline.trainer import MAPPOConfig, MAPPOTrainer
from llm_mappo.mappo_baseline.vector_env import MAPPOVectorEnvironment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "core_integration" / "medium-5ag-baseline.yaml"


def test_baseline_contract_is_teacher_free_and_cpu_executable():
    experiment = load_baseline_experiment(CONFIG)
    assert experiment.environment["n_agents"] == 5
    assert experiment.environment["observation_schema"] == (
        "direct-goal-observation-v2"
    )
    assert experiment.environment["reward_version"] == "reward-v2"
    assert experiment.training["environment_steps"] == 450_000
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
    agents = int(experiment.environment["n_agents"])
    values = dict(experiment.environment)
    pool = MAPPOVectorEnvironment(values, count=2, base_seed=91)
    try:
        assert pool.observations.shape == (2, agents, 209)
        assert pool.fp_states.shape[:2] == (2, agents)
        assert pool.action_masks.shape == (2, agents, 5)
        responses = pool.step(np.zeros((2, agents), dtype=np.int64), global_step=0)
        assert responses.rewards.shape == (2, agents)
        assert responses.dones.shape == (2, agents)
        assert responses.active_masks.shape == (2, agents)
        assert pool.observations.shape == (2, agents, 209)
    finally:
        pool.close()


def test_vector_worker_exposes_compact_structured_actor_observations():
    experiment = load_baseline_experiment(CONFIG)
    pool = MAPPOVectorEnvironment(experiment.environment, count=1, base_seed=92)
    try:
        # self (17) + DirectGoal (7) + 5x5x5 local map (125) + 3 peers (20)
        assert pool.observations.shape == (1, 5, 209)
    finally:
        pool.close()


def test_vector_worker_uses_waypoint_geometry_only_when_selected():
    experiment = load_baseline_experiment(CONFIG)
    direct_values = dict(experiment.environment)
    waypoint_values = {**direct_values, "goal_interface": "waypoint"}
    direct_pool = MAPPOVectorEnvironment(direct_values, count=1, base_seed=93)
    waypoint_pool = MAPPOVectorEnvironment(waypoint_values, count=1, base_seed=93)
    try:
        assert direct_pool.observations.shape == waypoint_pool.observations.shape
        assert not np.array_equal(
            direct_pool.observations[:, :, 16:19],
            waypoint_pool.observations[:, :, 16:19],
        )
    finally:
        direct_pool.close()
        waypoint_pool.close()


def test_waypoint_method_requires_a_waypoint_goal_interface(tmp_path):
    values = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    values["method"] = "MAPPO-FP-WP"
    values["environment"]["goal_interface"] = "direct_goal"
    config = tmp_path / "invalid_wp.yaml"
    config.write_text(yaml.safe_dump(values), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible"):
        load_baseline_experiment(config)


def test_two_worker_rollout_reaches_one_canonical_update():
    experiment = load_baseline_experiment(CONFIG)
    agents = int(experiment.environment["n_agents"])
    pool = MAPPOVectorEnvironment(experiment.environment, count=2, base_seed=101)
    trainer = MAPPOTrainer(
        MAPPOConfig(
            observation_dim=pool.observations.shape[-1],
            state_dim=pool.fp_states.shape[-1],
            ppo_epochs=1,
            num_minibatches=1,
        ),
        device="cpu",
    )
    storage = MAPPORolloutStorage(
        rollout_length=2,
        num_workers=2,
        num_agents=agents,
        observation_dim=pool.observations.shape[-1],
        state_dim=pool.fp_states.shape[-1],
        action_dim=5,
    )
    actor_hidden = np.zeros((2, agents, 64), dtype=np.float32)
    critic_hidden = np.zeros((2, agents, 64), dtype=np.float32)
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
        pool.step(
            np.zeros((1, int(experiment.environment["n_agents"])), dtype=np.int64),
            global_step=0,
        )
        pool.restore(states)
        np.testing.assert_array_equal(pool.observations, observations)
        np.testing.assert_array_equal(pool.fp_states, fp_states)
    finally:
        pool.close()


def test_checkpoint_resume_restores_actor_and_critic_hidden_states(tmp_path):
    original = load_baseline_experiment(CONFIG)
    training = dict(original.training)
    training.update({"num_env_workers": 1, "rollout_length": 1})
    experiment = replace(
        original, training=training, approval_status="approved"
    )
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
            resumed.actor_hidden_states, np.full((1, 5, 64), 0.25)
        )
        np.testing.assert_array_equal(
            resumed.critic_hidden_states, np.full((1, 5, 64), -0.5)
        )
    finally:
        resumed.close()


def test_checkpoint_evaluation_writes_a_separate_workload_receipt(tmp_path):
    source_experiment = load_baseline_experiment(CONFIG)
    source_experiment = replace(
        source_experiment,
        approval_status="approved",
        training={**source_experiment.training, "num_env_workers": 1},
    )
    source = BaselineRun(source_experiment, device="cpu", output_root=tmp_path)
    checkpoint = source.directory / "checkpoint_final.pt"
    try:
        source._checkpoint(checkpoint)
    finally:
        source.close()

    target_experiment = load_baseline_experiment(CONFIG)
    target_experiment = replace(
        target_experiment,
        training={**target_experiment.training, "num_env_workers": 1},
        evaluation={
            **target_experiment.evaluation,
            "environment_overrides": {
                "max_steps": 2,
                "dynamic_ingress_interval": 77,
            },
        },
    )
    target = BaselineRun(
        target_experiment,
        device="cpu",
        output_root=tmp_path,
        evaluation_only=True,
    )
    result = target.evaluate_checkpoint(checkpoint)

    manifest = json.loads(
        (target.directory / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "complete"
    assert manifest["schema"] == "mappo-fp-checkpoint-evaluation-v1"
    assert manifest["source_checkpoint_sha256"] == sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert manifest["evaluation_environment"]["dynamic_ingress_interval"] == 77
    assert manifest["evaluation_environment"]["max_steps"] == 2
    assert json.loads((target.directory / "state.json").read_text()) == {
        "status": "complete"
    }
