"""Dataset isolation and deterministic splitting for Phase 3 LLM KD."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from hashlib import sha256

import numpy as np
import pytest
from types import SimpleNamespace
import torch

from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.dataset import (
    ASTAR_DIAGNOSTIC_DISTRIBUTION,
    MAIN_BASELINE_DISTRIBUTION,
    DistributionTrajectory,
    build_partition_windows,
    deterministic_stratified_sample,
    split_complete_episodes,
)
from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_llm_kd.collector import (
    FrozenMAPPOPolicy,
    collect_complete_episode,
    load_phase3_collection_policy,
)
from llm_mappo.mappo_llm_kd.trajectory import ActorObservationTrajectory


SCHEMA = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
)
INDEX = {name: index for index, name in enumerate(SCHEMA.feature_names)}
ROOT = Path(__file__).resolve().parents[1]
COLLECTION_CONFIG = (
    ROOT
    / "configs"
    / "llm_kd"
    / "medium-5ag-low-workload-collection-baseline.yaml"
)


def test_collection_baseline_is_native_209d_and_isolated():
    experiment = load_baseline_experiment(COLLECTION_CONFIG)
    assert experiment.schema == "mappo-collection-209d-v1"
    assert experiment.method == "MAPPO-FP-DG"
    assert experiment.environment["observation_schema"] == "direct-goal-observation-v2"
    assert experiment.environment["n_agents"] == 5
    assert experiment.environment["profile_id"] == (
        "medium-5ag-low-workload-collection"
    )
    assert experiment.environment["max_steps"] == 800
    assert experiment.environment["dynamic_ingress_interval"] == 120
    assert experiment.environment["batch_size_range"] == [1, 1]
    assert experiment.environment["initial_ingress_batches"] == 2
    assert experiment.environment["task_target"] == 8
    assert experiment.environment["terminate_on_task_target"] is True
    assert experiment.training["environment_steps"] == 300000
    assert experiment.artifact_root == (
        "artifacts/llm_kd/collection_baseline/medium_5ag_low_workload"
    )
    assert experiment.evaluation["environment_overrides"] == {
        "max_steps": 800,
        "dynamic_ingress_interval": 120,
        "batch_size_range": [1, 1],
        "initial_ingress_batches": 2,
        "queue_size": 5,
        "task_target": 8,
        "terminate_on_task_target": True,
    }
    assert "resume" not in experiment.training


def test_collection_baseline_builds_a_native_209d_actor_with_its_own_checkpoint(
    tmp_path,
):
    experiment = load_baseline_experiment(COLLECTION_CONFIG)
    experiment = replace(
        experiment,
        training={**experiment.training, "num_env_workers": 1},
    )
    run = BaselineRun(experiment, device="cpu", output_root=tmp_path)
    try:
        assert run.actor_schema.dimension == 209
        assert run.trainer.actor.observation_dim == 209
        assert run._checkpoint_schema == (
            "mappo-fp-collection-209d-checkpoint-v1"
        )
        assert run.identity["actor_observation_dimension"] == 209
    finally:
        run.close()


def _trajectory(episode: str, agent: int, *, frames: int = 5):
    rows = np.zeros((frames, SCHEMA.dimension), dtype=np.float32)
    rows[:, INDEX["self_orientation_up"]] = 1.0
    rows[:, INDEX["self_phase_task"]] = 1.0
    rows[:, INDEX["goal_type_task"]] = 1.0
    rows[:, INDEX["self_previous_action_0"]] = 1.0
    rows[:, INDEX["self_priority_weight"]] = 0.5
    rows[:, INDEX["goal_normalized_distance"]] = np.linspace(0.8, 0.4, frames)
    return ActorObservationTrajectory(
        episode_id=episode,
        agent_id=agent,
        timesteps=np.arange(frames),
        observations=rows,
        reset_masks=np.asarray([True] + [False] * (frames - 1)),
        behavior_policy_id="MAPPO-FP-DG:checkpoint",
        environment_sha256="a" * 64,
        config_sha256="b" * 64,
    )


def _main_records(episodes=10):
    return [
        DistributionTrajectory(MAIN_BASELINE_DISTRIBUTION, _trajectory(f"e{i}", agent))
        for i in range(episodes)
        for agent in (1, 2, 3, 4, 5)
    ]


def test_split_is_deterministic_episode_disjoint_and_keeps_agents_together():
    records = _main_records()
    first = split_complete_episodes(
        records, {"train": 0.6, "validation": 0.2, "test": 0.2}, salt="d1"
    )
    second = split_complete_episodes(
        list(reversed(records)),
        {"train": 0.6, "validation": 0.2, "test": 0.2},
        salt="d1",
    )

    assert {
        name: [item.trajectory.episode_id for item in values]
        for name, values in first.items()
    } == {
        name: [item.trajectory.episode_id for item in values]
        for name, values in second.items()
    }
    episode_sets = {
        name: {item.trajectory.episode_id for item in values}
        for name, values in first.items()
    }
    assert len(episode_sets["train"]) == 6
    assert len(episode_sets["validation"]) == 2
    assert len(episode_sets["test"]) == 2
    assert episode_sets["train"].isdisjoint(episode_sets["validation"])
    assert episode_sets["train"].isdisjoint(episode_sets["test"])
    assert episode_sets["validation"].isdisjoint(episode_sets["test"])
    for values in first.values():
        grouped = {}
        for item in values:
            grouped.setdefault(item.trajectory.episode_id, set()).add(
                item.trajectory.agent_id
            )
        assert all(agents == {1, 2, 3, 4, 5} for agents in grouped.values())


def test_windows_are_created_after_split_and_never_cross_episode_or_agent():
    split = split_complete_episodes(
        _main_records(),
        {"train": 0.6, "validation": 0.2, "test": 0.2},
        salt="d1",
    )
    windows = build_partition_windows(split, history_length=3)

    for partition, values in windows.items():
        allowed = {
            (item.trajectory.episode_id, item.trajectory.agent_id)
            for item in split[partition]
        }
        assert values
        assert all((value.episode_id, value.agent_id) in allowed for value in values)
        assert all(value.end_timestep - value.start_timestep < 3 for value in values)


def test_stratified_sampling_is_deterministic_and_balances_available_strata():
    split = split_complete_episodes(
        _main_records(),
        {"train": 0.6, "validation": 0.2, "test": 0.2},
        salt="d1",
    )
    windows = build_partition_windows(split, history_length=3)["train"]
    alternate = list(windows)
    alternate[0] = type(alternate[0])(
        **{**alternate[0].__dict__, "stratum": "rare"}
    )

    first = deterministic_stratified_sample(alternate, sample_budget=4, salt="s")
    second = deterministic_stratified_sample(
        list(reversed(alternate)), sample_budget=4, salt="s"
    )

    identity = lambda value: (
        value.episode_id, value.agent_id, value.end_timestep, value.stratum
    )
    assert [identity(item) for item in first] == [identity(item) for item in second]
    assert {item.stratum for item in first} == {windows[0].stratum, "rare"}


def test_observable_strata_are_deterministic_and_encode_declared_axes():
    record = _main_records(1)[0]
    record.trajectory.observations.flags.writeable = True
    row = record.trajectory.observations[-1]
    row[INDEX["self_previous_blocked"]] = 1.0
    row[INDEX["map_f0_r1_other_agv"]] = 1.0
    record.trajectory.observations.flags.writeable = False
    split = {"train": [record], "validation": [], "test": []}

    window = build_partition_windows(split, history_length=3)["train"][-1]

    assert window.stratum == (
        "phase=task|priority=medium|blocked_or_no_progress=true|congestion=present"
    )


def test_astar_distribution_cannot_enter_main_split_or_pretraining_windows():
    astar = DistributionTrajectory(
        ASTAR_DIAGNOSTIC_DISTRIBUTION, _trajectory("astar-e", 1)
    )
    with pytest.raises(ValueError, match=r"A\*"):
        split_complete_episodes(
            [*_main_records(), astar],
            {"train": 0.6, "validation": 0.2, "test": 0.2},
            salt="d1",
        )


@pytest.mark.parametrize(
    "ratios",
    [
        {"train": 0.7, "validation": 0.3},
        {"train": 0.7, "validation": 0.2, "test": 0.2},
        {"train": 1, "validation": 0, "test": 0},
    ],
)
def test_split_rejects_noncanonical_or_invalid_ratios(ratios):
    with pytest.raises(ValueError, match="split ratios"):
        split_complete_episodes(_main_records(), ratios, salt="d1")


class _Policy:
    hidden_dim = 4
    behavior_policy_id = "MAPPO-FP-DG:" + "c" * 64

    def __init__(self):
        self.received_dimensions = []
        self.received_rows = []
        self.received_hidden = []

    def act(self, observations, action_masks, hidden):
        self.received_dimensions.append(observations.shape)
        self.received_rows.append(observations.copy())
        self.received_hidden.append(hidden.copy())
        return np.zeros(5, dtype=np.int64), hidden + 1.0


class _Environment:
    n_agents = 5

    def __init__(self):
        self.steps = 0

    def reset(self, *, seed):
        self.seed = seed
        self.steps = 0

    def action_masks(self):
        return np.ones((5, 5), dtype=np.bool_)

    def step(self, actions):
        assert actions.shape == (5,)
        self.steps += 1
        return SimpleNamespace(
            terminated=False,
            truncated=self.steps == 3,
            metrics=SimpleNamespace(deadlocked=False),
        )


def test_collector_feeds_and_records_the_same_native_209d_rows(monkeypatch):
    environment = _Environment()
    policy = _Policy()
    calls = []

    def observations(_environment, schema, **kwargs):
        assert kwargs == {"goal_interface": "direct_goal", "planner": None}
        calls.append(schema.observation_schema)
        width = schema.dimension
        rows = np.zeros((5, width), dtype=np.float32)
        rows[:, :10] = environment.steps
        if width == 209:
            rows[:, 10] = (0.25, 0.75, 0.0, 0.5, 1.0)
        rows[:, 10 + (width == 209)] = 1.0
        return rows

    monkeypatch.setattr(
        "llm_mappo.mappo_llm_kd.collector.build_structured_observations",
        observations,
    )
    records = collect_complete_episode(
        environment,
        policy,
        seed=17,
        episode_id="seed-17-episode-0",
        environment_sha256="a" * 64,
        config_sha256="b" * 64,
        max_steps=3,
    )

    assert calls == ["direct-goal-observation-v2"] * 3
    assert policy.received_dimensions == [(5, 209)] * 3
    assert np.array_equal(
        policy.received_rows[0][0],
        records[0].trajectory.observations[0],
    )
    assert np.count_nonzero(policy.received_hidden[0]) == 0
    assert len(records) == 5
    assert all(item.distribution == MAIN_BASELINE_DISTRIBUTION for item in records)
    assert all(item.trajectory.observations.shape == (3, 209) for item in records)
    assert records[0].trajectory.observations[:, 10].tolist() == [0.25] * 3
    assert records[1].trajectory.observations[:, 10].tolist() == [0.75] * 3
    assert all(item.trajectory.reset_masks.tolist() == [True, False, False]
               for item in records)


def test_collector_rejects_incomplete_or_non_medium_episode(monkeypatch):
    monkeypatch.setattr(
        "llm_mappo.mappo_llm_kd.collector.build_structured_observations",
        lambda _environment, schema, **kwargs: np.zeros(
            (5, schema.dimension), dtype=np.float32
        ),
    )
    environment = _Environment()
    environment.n_agents = 2
    with pytest.raises(ValueError, match="five AGVs"):
        collect_complete_episode(
            environment, _Policy(), seed=1, episode_id="e",
            environment_sha256="a" * 64, config_sha256="b" * 64,
            max_steps=2,
        )


def _checkpoint_payload():
    actor = MAPPOActor(209, 5, 64)
    return {
        "schema": "mappo-collection-actor-209d-v1",
        "identity": {
            "schema": "mappo-collection-209d-v1",
            "method": "MAPPO-FP-DG",
            "observation_schema": DIRECT_GOAL_OBSERVATION_V2,
            "actor_observation_schema": "structured-direct_goal-v2",
            "actor_observation_dimension": 209,
            "reward_version": "reward-v2",
            "profile_id": "medium-5ag-phase2-train",
            "environment_id": "llm-mappo-medium-3ag-v1",
            "n_agents": 5,
            "goal_interface": "direct_goal",
            "environment_sha256": (
                "6031952c1a872bcab3a25243ecd29280b4b350b43d6edea1d55a38386da7da04"
            ),
        },
        "actor_config": {"observation_dim": 209, "action_dim": 5, "hidden_dim": 64},
        "actor_state_dict": actor.state_dict(),
    }


def _load(path):
    return load_phase3_collection_policy(
        path, device="cpu", expected_artifact_sha256=sha256(path.read_bytes()).hexdigest()
    )


def test_checkpoint_loader_accepts_only_native_phase3_collection_baseline(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint_payload(), path)

    policy = _load(path)

    assert isinstance(policy, FrozenMAPPOPolicy)
    assert policy.hidden_dim == 64
    assert policy.behavior_policy_id.startswith("MAPPO-FP-DG:")
    assert policy.usage_scope == "offline_behavior_collection_only"
    assert "initialization" not in policy.behavior_policy_id


@pytest.mark.parametrize(
    "mutation",
    [
        "checkpoint_schema",
        "identity_schema",
        "method",
        "observation_schema",
        "actor_observation_dimension",
        "reward_version",
        "profile_id",
        "environment_id",
        "n_agents",
        "goal_interface",
        "environment_sha256",
        "missing_actor",
        "missing_weight",
        "extra_weight",
    ],
)
def test_checkpoint_loader_rejects_every_non_native_phase3_contract(tmp_path, mutation):
    payload = _checkpoint_payload()
    if mutation == "checkpoint_schema":
        payload["schema"] = "mappo-fp-phase2-baseline-checkpoint-v1"
    elif mutation == "identity_schema":
        payload["identity"]["schema"] = "phase2-mappo-baseline-v1"
    elif mutation == "method":
        payload["identity"]["method"] = "MAPPO-FP-ASTARKD"
    elif mutation == "observation_schema":
        payload["identity"]["observation_schema"] = "direct-goal-observation-v1"
    elif mutation == "actor_observation_dimension":
        payload["identity"]["actor_observation_dimension"] = 208
    elif mutation == "reward_version":
        payload["identity"]["reward_version"] = "reward-v1"
    elif mutation == "profile_id":
        payload["identity"]["profile_id"] = "wrong-profile"
    elif mutation == "environment_id":
        payload["identity"]["environment_id"] = "wrong-environment"
    elif mutation == "n_agents":
        payload["identity"]["n_agents"] = 4
    elif mutation == "goal_interface":
        payload["identity"]["goal_interface"] = "waypoint"
    elif mutation == "environment_sha256":
        payload["identity"]["environment_sha256"] = None
    elif mutation == "missing_actor":
        del payload["actor_state_dict"]
    elif mutation == "missing_weight":
        del payload["actor_state_dict"]["action_head.bias"]
    elif mutation == "extra_weight":
        payload["actor_state_dict"]["unexpected"] = torch.zeros(1)
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    with pytest.raises(
        (RuntimeError, ValueError), match="Phase 3 collection|Missing|Unexpected"
    ):
        _load(path)


def test_loader_requires_the_explicitly_approved_artifact_digest(tmp_path):
    path = tmp_path / "actor.pt"
    torch.save(_checkpoint_payload(), path)
    with pytest.raises(ValueError, match="not approved"):
        load_phase3_collection_policy(
            path, device="cpu", expected_artifact_sha256="0" * 64
        )


def test_runner_exports_weights_only_safe_actor_consumed_by_collector(tmp_path):
    experiment = replace(
        load_baseline_experiment(COLLECTION_CONFIG),
        training={**load_baseline_experiment(COLLECTION_CONFIG).training,
                  "num_env_workers": 1},
    )
    run = BaselineRun(experiment, device="cpu", output_root=tmp_path)
    try:
        destination = run.directory / "collection_actor_test.pt"
        run._export_collection_actor(destination)
        policy = _load(destination)
        assert policy.hidden_dim == 64
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        assert payload["identity"]["environment_sha256"] == run.identity[
            "environment_sha256"
        ]
    finally:
        run.close()


@pytest.mark.parametrize("mutation", ["missing_agent", "duplicate_agent", "provenance", "timesteps", "length"])
def test_split_rejects_incomplete_or_misaligned_episode_groups(mutation):
    records = _main_records(1)
    if mutation == "missing_agent":
        records.pop()
    elif mutation == "duplicate_agent":
        records[-1] = DistributionTrajectory(
            MAIN_BASELINE_DISTRIBUTION, replace(records[-1].trajectory, agent_id=4)
        )
    elif mutation == "provenance":
        records[-1] = DistributionTrajectory(
            MAIN_BASELINE_DISTRIBUTION,
            replace(records[-1].trajectory, config_sha256="e" * 64),
        )
    elif mutation == "timesteps":
        records[-1] = DistributionTrajectory(
            MAIN_BASELINE_DISTRIBUTION,
            replace(records[-1].trajectory, timesteps=np.arange(1, 6)),
        )
    elif mutation == "length":
        records[-1] = DistributionTrajectory(
            MAIN_BASELINE_DISTRIBUTION, _trajectory("e0", 5, frames=4)
        )
    with pytest.raises(ValueError, match="agents 1..5|provenance|timesteps"):
        split_complete_episodes(
            records, {"train": 0.6, "validation": 0.2, "test": 0.2}, salt="d1"
        )


def test_frozen_policy_sampling_replays_from_its_private_generator(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint_payload(), path)
    policy = _load(path)
    observations = np.zeros((5, 209), dtype=np.float32)
    action_masks = np.ones((5, 5), dtype=np.bool_)
    hidden = np.zeros((1, 5, 64), dtype=np.float32)

    policy.start_episode(17)
    first_actions, _ = policy.act(observations, action_masks, hidden)
    first_state = policy._generator.get_state().clone()
    policy.start_episode(17)
    second_actions, _ = policy.act(observations, action_masks, hidden)
    second_state = policy._generator.get_state().clone()
    policy.start_episode(18)
    third_state = policy._generator.get_state().clone()

    assert np.array_equal(first_actions, second_actions)
    assert torch.equal(first_state, second_state)
    assert not torch.equal(second_state, third_state)
