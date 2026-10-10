"""Synthetic v3 collection/selection contracts; never real policy evaluation."""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)
from llm_mappo.mappo_llm_kd.goal_full_dataset_run import (
    collect_goal_full_episode, run_goal_full_collection, select_goal_full_windows,
)


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT = ROOT / "configs/llm_kd/phase_20261007/medium-5ag-baseline-goal-full.yaml"
LEDGER = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2,
).feature_names
INDEX = {name: index for index, name in enumerate(LEDGER)}


class Policy:
    hidden_dim = 64
    behavior_policy_id = "synthetic-policy-not-research-evidence"

    def start_episode(self, seed):
        self.seed = seed

    def act(self, observations, masks, hidden):
        assert hidden.shape == (1, observations.shape[0], 64)
        assert masks.shape == (observations.shape[0], 5)
        return np.zeros(observations.shape[0], dtype=np.int64), hidden + 1


class Environment:
    def __init__(self, agents=5, horizon=24, interrupted=False):
        self.n_agents, self.horizon = agents, horizon
        self.interrupted = interrupted
        self.calls = 0

    def reset(self, seed):
        self.seed, self.calls = seed, 0

    def action_masks(self):
        return np.ones((self.n_agents, 5), dtype=np.bool_)

    def step(self, actions):
        self.calls += 1
        if self.interrupted:
            raise RuntimeError("synthetic interruption")
        return SimpleNamespace(terminated=False, truncated=self.calls == self.horizon,
                               metrics=SimpleNamespace(deadlocked=False))

    def close(self):
        pass


def observations(environment):
    rows = np.zeros((environment.n_agents, 209), dtype=np.float32)
    for name in ("self_orientation_up", "self_phase_task",
                 "self_previous_action_0", "goal_type_task"):
        rows[:, INDEX[name]] = 1.0
    rows[:, INDEX["self_priority_weight"]] = 0.5
    rows[:, INDEX["self_battery"]] = 1.0
    rows[:, INDEX["goal_normalized_distance"]] = (
        environment.calls * 0.001 + np.arange(environment.n_agents) * 0.01
        + (environment.seed - 10000) * 0.02
    )
    return rows


@pytest.mark.parametrize("agents", [3, 5, 12])
def test_synthetic_collection_arrays_follow_actual_agent_count(agents):
    episode = collect_goal_full_episode(
        Environment(agents), Policy(), seed=10000, max_steps=800,
        observation_builder=observations,
    )
    assert episode["observations"].shape == (agents, 24, 209)
    assert episode["reset_masks"].shape == (agents, 24)
    assert np.all(episode["reset_masks"][:, 0])
    assert not np.any(episode["reset_masks"][:, 1:])
    np.testing.assert_allclose(
        episode["observations"][:, 0, INDEX["goal_normalized_distance"]],
        np.arange(agents, dtype=np.float32) * 0.01, atol=1e-8,
    )
    assert episode["boundary"] == "truncated"


def test_unfinished_episode_cannot_be_saved_as_complete():
    with pytest.raises(RuntimeError, match="complete episode"):
        collect_goal_full_episode(Environment(horizon=30), Policy(), seed=10000,
                                  max_steps=24, observation_builder=observations)


def source_loader(**_):
    return Policy(), {"source_config_sha256": "a" * 64,
                      "source_environment_sha256": "b" * 64,
                      "actor_sha256": "c" * 64, "verified_source": False}


def _collect(tmp_path, count, **overrides):
    options = dict(
        source_config="synthetic", source_checkpoint="synthetic",
        actor_artifact="synthetic", actor_sha256="c" * 64,
        rollout_config=ROLLOUT, output_root=tmp_path, episode_count=count,
        source_loader=source_loader, environment_factory=lambda _: Environment(),
        observation_builder=observations,
    )
    options.update(overrides)
    return run_goal_full_collection(**options)


def test_source_and_rollout_remain_distinct_and_resume_skips_complete_episodes(tmp_path):
    first = _collect(tmp_path, 2)
    directory = Path(first["run_directory"])
    identity = json.loads((directory / "collection_identity.json").read_text("utf-8"))
    assert identity["source_actor_contract"]["source_config_sha256"] == "a" * 64
    assert identity["rollout_environment_contract"]["config_sha256"] != "a" * 64
    before = {p.name: p.read_bytes() for p in (directory / "episodes").glob("*.npz")}
    starts = []

    def factory(_):
        starts.append(True)
        return Environment()

    _collect(tmp_path, 3, resume=True, environment_factory=factory)
    assert len(starts) == 1
    for name, content in before.items():
        assert (directory / "episodes" / name).read_bytes() == content


def test_failed_episode_receipt_is_preserved_and_retry_can_resume(tmp_path):
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _collect(tmp_path, 1, environment_factory=lambda _: Environment(interrupted=True))
    directory = next(tmp_path.glob("goal-full-v3-*"))
    failures = (directory / "failures.jsonl").read_bytes()
    _collect(tmp_path, 1, resume=True)
    assert (directory / "failures.jsonl").read_bytes() == failures
    assert len(list((directory / "episodes").glob("*.npz"))) == 1


def test_resume_repairs_receipt_after_complete_array_commit_without_rerun(
    tmp_path, monkeypatch,
):
    import llm_mappo.mappo_llm_kd.goal_full_dataset_run as dataset_run

    atomic = dataset_run._atomic_json

    def interrupt(path, value):
        if Path(path).name == "seed_10000.json":
            raise RuntimeError("receipt interruption")
        atomic(path, value)

    monkeypatch.setattr(dataset_run, "_atomic_json", interrupt)
    with pytest.raises(RuntimeError, match="receipt interruption"):
        _collect(tmp_path, 1)
    monkeypatch.setattr(dataset_run, "_atomic_json", atomic)

    def must_not_run(_):
        pytest.fail("A complete persisted episode must not be rerun")

    _collect(tmp_path, 1, resume=True, environment_factory=must_not_run)


def test_selection_splits_episodes_before_nonoverlapping_content_unique_windows(tmp_path):
    result = _collect(tmp_path, 10)
    selected = select_goal_full_windows(
        result["run_directory"], budgets={"train": 20, "validation": 5, "test": 5},
    )
    directory = Path(result["run_directory"])
    rows = [json.loads(line) for line in (directory / "windows.jsonl").read_text("utf-8").splitlines()]
    assert selected["manifest"]["schema"] == "llm-dataset-build-run-v3"
    assert selected["manifest"]["pretraining_eligible"] is False
    assert selected["manifest"]["episode_split_counts"] == {
        "train": 7, "validation": 2, "test": 1,
    }
    seen = {}
    endpoints = {}
    content = set()
    for row in rows:
        episode = row["episode_id"]
        assert seen.setdefault(episode, row["partition"]) == row["partition"]
        key = (episode, row["agent_id"])
        end = row["end_timestep"]
        assert all(abs(end - previous) >= 8 for previous in endpoints.get(key, []))
        endpoints.setdefault(key, []).append(end)
        assert end - row["start_timestep"] == 7
        encoded = json.dumps(row["observations"])
        assert encoded not in content
        content.add(encoded)
        assert len(row) == 8
    assert len(rows) == 30
    assert (directory / "candidate_index.sqlite").is_file()
    assert selected["manifest"]["coverage_missing"]["train"]["charging_service"] > 0
    again = select_goal_full_windows(directory, budgets={"train": 20, "validation": 5, "test": 5})
    assert again["manifest"]["windows_sha256"] == selected["manifest"]["windows_sha256"]


def test_collection_rejects_unapproved_extension_before_loading_source(tmp_path):
    with pytest.raises(ValueError, match="120"):
        _collect(tmp_path, 121)
    assert not list(tmp_path.iterdir())


def test_trial_review_repeat_and_pro_are_frozen_subsets(tmp_path):
    result = _collect(tmp_path, 10, environment_factory=lambda _: Environment(horizon=48))
    select_goal_full_windows(result["run_directory"],
                             budgets=dict(train=130, validation=10, test=10))
    lists = json.loads((Path(result["run_directory"]) /
                        "label_selection_lists.json").read_text("utf-8"))
    ids = {key: {row["view_sha256"] for row in value} for key, value in lists.items()}
    assert len(ids["trial"]) == 120
    assert len(ids["review"]) == 60
    assert ids["review"] <= ids["trial"]
    assert len(ids["repeat"]) == 40
    assert ids["repeat"] == ids["pro"]
    assert ids["repeat"] <= ids["trial"]
    from llm_mappo.mappo_llm_kd.dataset_labeling import _window_id
    for rows in lists.values():
        for row in rows:
            assert row["view_id"] == _window_id(dict(partition="train", **row))


def freeze_fixture(directory, manifest, budgets):
    from llm_mappo.mappo_llm_kd.goal_full_dataset_run import _read, _receipts
    from llm_mappo.mappo_llm_kd.source_actor_v3 import canonical_sha256, file_sha256

    frozen = dict(
        budgets=budgets, windows_sha256=manifest["windows_sha256"],
        selection_manifest_sha256=manifest["selection_manifest_sha256"],
        manifest_sha256=file_sha256(directory / "manifest.json"),
        label_lists_sha256=file_sha256(directory / "label_selection_lists.json"),
        episode_partitions_sha256=file_sha256(directory / "episode_partitions.json"),
        episode_receipts_sha256=canonical_sha256(
            _receipts(directory, _read(directory / "collection_identity.json"))),
    )
    (directory / "selection_frozen.json").write_text(json.dumps(frozen), "utf-8")


@pytest.mark.parametrize("mutation", ["none", "budgets", "windows", "manifest", "receipt"])
def test_frozen_selection_is_verified_without_reselection(tmp_path, monkeypatch, mutation):
    import llm_mappo.mappo_llm_kd.goal_full_dataset_run as dataset_run

    result = _collect(tmp_path, 3)
    directory = Path(result["run_directory"])
    budgets = dict(train=5, validation=2, test=2)
    selected = select_goal_full_windows(directory, budgets=budgets)
    freeze_fixture(directory, selected["manifest"], budgets)
    if mutation == "budgets":
        budgets["train"] += 1
    elif mutation in {"windows", "manifest"}:
        path = directory / ("windows.jsonl" if mutation == "windows" else "manifest.json")
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "receipt":
        path = directory / "episodes/seed_10000.json"
        receipt = json.loads(path.read_text("utf-8"))
        receipt["boundary"] = "terminated"
        path.write_text(json.dumps(receipt), "utf-8")
    monkeypatch.setattr(dataset_run, "_index_candidates",
                        lambda *args: pytest.fail("Frozen selection must not be rebuilt"))
    if mutation == "none":
        assert select_goal_full_windows(directory, budgets=budgets) == selected
    else:
        with pytest.raises(ValueError, match="[Ff]rozen"):
            select_goal_full_windows(directory, budgets=budgets)


@pytest.mark.parametrize("status", ["complete", "running"])
def test_source_requires_historical_completed_final_manifest(tmp_path, monkeypatch, status):
    import llm_mappo.mappo_llm_kd.source_actor_v3 as source_run
    from llm_mappo.medium5_contract import CORE_MEDIUM5_C1_SCHEMA

    experiment = SimpleNamespace(
        schema=CORE_MEDIUM5_C1_SCHEMA, method="MAPPO-FP-DG",
        approval_status="approved", training={"seed": 7}, environment={"n_agents": 5},
        config_sha256="a" * 64, reward_definition_sha256="b" * 64,
    )
    manifest = dict(environment=experiment.environment, training=experiment.training,
                    identity=dict(schema=experiment.schema, method=experiment.method,
                                  seed=7, config_sha256="a" * 64,
                                  reward_definition_sha256="b" * 64,
                                  code_commit="c" * 40))
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), "utf-8")
    (tmp_path / "state.json").write_text(json.dumps(dict(status=status)), "utf-8")
    checkpoint = tmp_path / "checkpoint_final.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    monkeypatch.setattr(source_run, "load_baseline_experiment", lambda _: experiment)
    monkeypatch.setattr(source_run, "load_medium5_collection_policy", lambda *a, **k: Policy())
    options = dict(source_config=tmp_path / "source.yaml", source_checkpoint=checkpoint,
                   actor_artifact=tmp_path / "actor.pt", actor_sha256="d" * 64)
    if status == "running":
        with pytest.raises(ValueError, match="completed final"):
            source_run.load_pinned_medium_source(**options)
    else:
        _, identity = source_run.load_pinned_medium_source(**options)
        assert identity["verified_source"] is True


def test_resume_binds_implementation_but_head_is_metadata_only(tmp_path, monkeypatch):
    import llm_mappo.mappo_llm_kd.goal_full_dataset_run as dataset_run

    result = _collect(tmp_path, 1)
    directory = Path(result["run_directory"])
    (directory / "launch_metadata.json").write_text(
        json.dumps(dict(code_commit="unrelated-doc-change")), "utf-8")
    _collect(tmp_path, 1, resume=True,
             environment_factory=lambda _: pytest.fail("Persisted episode rerun"))
    hashes = dataset_run._implementation_hashes()
    hashes["mappo_llm_kd/goal_full_dataset_run.py"] = "0" * 64
    monkeypatch.setattr(dataset_run, "_implementation_hashes", lambda: hashes)
    with pytest.raises(ValueError, match="identity differs"):
        _collect(tmp_path, 1, resume=True)


@pytest.mark.parametrize("direct", [True, False])
def test_competitive_low_uses_raw_direct_and_weak_relation(direct):
    from llm_mappo.mappo_llm_kd.goal_full_dataset_run import (
        _observable_flags, _window_view,
    )

    environment = Environment()
    environment.reset(10000)
    rows = np.repeat(observations(environment)[0:1], 8, axis=0)
    value = _window_view(rows, np.zeros(8, dtype=np.bool_))
    frame = value["history"][-1]
    frame["self"].update(previous_requested_action="FORWARD", previous_blocked=direct)
    target = 1 if direct else 2
    for cell in frame["local"]["cells"]:
        cell["other_agv"] = (cell["forward"], cell["right"]) == (target, 0)
        cell["static_obstacle"] = (
            not direct and cell["forward"] in (0, 1, 2) and cell["right"] in (-1, 1))
    frame["peers"][0] = dict(
        valid=True, relative_forward=0.02, relative_right=0.0, orientation="DOWN",
        carrying=False, battery=0.8, task_phase="task", priority_weight=0.0,
        previous_requested_action="FORWARD", previous_blocked=False,
    )
    _, flags = _observable_flags(value)
    assert flags["competition"] and flags["competitive_low"]
    if not direct:
        frame["peers"][0]["orientation"] = "UP"
        _, flags = _observable_flags(value)
        assert not flags["competition"] and not flags["competitive_low"]


def test_content_deduplication_is_global_across_agents_and_episodes(tmp_path):
    def duplicate_observations(environment):
        rows = observations(environment)
        rows[:, INDEX["goal_normalized_distance"]] = 0.1
        return rows

    result = _collect(tmp_path, 3, observation_builder=duplicate_observations)
    selected = select_goal_full_windows(result["run_directory"],
                                        budgets=dict(train=10, validation=5, test=5))
    # Reset flags distinguish the initial window from later identical windows.
    assert sum(selected["manifest"]["window_counts"].values()) <= 2
    assert not selected["manifest"]["raw_candidate_coverage_passed"]


def test_fixed_list_preflight_reports_unknown_pair_and_missing_review_strata():
    from llm_mappo.mappo_llm_kd.goal_full_dataset_run import _label_preflight
    from llm_mappo.mappo_llm_kd.semantic_v3 import SEMANTIC_DIMENSIONS_V3

    rows = [dict(view_sha256=str(i), flags=dict(ordinary=True,
                 **{f"{dim}_known": i < 19 for dim in SEMANTIC_DIMENSIONS_V3[:3]}))
            for i in range(120)]
    lists = dict(trial=rows, repeat=rows[:40], pro=rows[:40], review=rows[:60])
    report = _label_preflight(lists, rows)
    assert report["passed"] is False
    assert report["coverage_missing"]["paired_goal-priority_known"] == 1
    assert report["coverage_missing"]["review_charging_service"] == 1
    assert report["trial_known_rates"]["goal-priority"] == 19 / 120


def test_collection_records_single_inference_thread_and_runtime_versions(tmp_path):
    import torch
    result = _collect(tmp_path, 1)
    directory = Path(result["run_directory"])
    identity = json.loads((directory / "collection_identity.json").read_text("utf-8"))
    metadata = json.loads((directory / "launch_metadata.json").read_text("utf-8"))
    assert identity["collection_contract"]["learner_inference_threads"] == 1
    assert torch.get_num_threads() == 1
    assert set(metadata["runtime_versions"]) == {"python", "numpy", "torch"}


@pytest.mark.parametrize("prior_state", [None, "insufficient_coverage"])
@pytest.mark.parametrize("eligible", [False, True])
def test_frozen_selection_repairs_state_from_manifest_without_reselection(
    tmp_path, monkeypatch, prior_state, eligible,
):
    import llm_mappo.mappo_llm_kd.goal_full_dataset_run as dataset_run

    result = _collect(tmp_path, 3)
    directory = Path(result["run_directory"])
    budgets = dict(train=5, validation=2, test=2)
    selected = select_goal_full_windows(directory, budgets=budgets)
    # A persisted frozen manifest is the recovery authority; this fixture tests
    # both eligibility values without claiming synthetic data passed real gates.
    manifest = selected["manifest"]
    manifest["main_pretraining_eligible"] = eligible
    manifest["pretraining_eligible"] = eligible
    dataset_run._atomic_json(directory / "manifest.json", manifest)
    freeze_fixture(directory, manifest, budgets)
    immutable = ["windows.jsonl", "manifest.json", "selection_manifest.json",
                 "label_selection_lists.json", "selection_frozen.json"]
    before = {name: (directory / name).read_bytes() for name in immutable}
    if prior_state is None:
        (directory / "state.json").unlink()
    else:
        dataset_run._atomic_json(directory / "state.json", dict(status=prior_state))
    monkeypatch.setattr(dataset_run, "_index_candidates",
                        lambda *args: pytest.fail("Frozen selection must not be rebuilt"))
    select_goal_full_windows(directory, budgets=budgets)
    state = dataset_run._read(directory / "state.json")
    assert state == dict(status="completed" if eligible else "insufficient_coverage",
                         main_pretraining_eligible=eligible,
                         coverage_missing=manifest["coverage_missing"])
    assert before == {name: (directory / name).read_bytes() for name in immutable}
