"""Incremental raw-only selection preserves lineage and frozen holdout."""

import json
from copy import deepcopy
import sqlite3

import numpy as np
import pytest

from llm_mappo.mappo_llm_kd import medium_extension_dataset as extension
from llm_mappo.mappo_llm_kd import goal_full_dataset_run as old


def database():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE candidates (episode TEXT,agent INTEGER,"
                       "end INTEGER,partition TEXT,stratum TEXT,flags TEXT,"
                       "view_hash TEXT,rank TEXT,PRIMARY KEY(episode,agent,end))")
    extension._create_tracking_tables(connection)
    return connection


def insert(connection, episode, end, digest, flags, rank="a", stratum="ordinary"):
    connection.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)",
                       (episode, 1, end, "train", stratum,
                        json.dumps(flags), digest, rank))


def valid_arrays(steps=10):
    observations = np.zeros((5, steps, 209), dtype=np.float32)
    ledger = {name: index for index, name in enumerate(old.SCHEMA.feature_names)}
    for name in ("self_orientation_up", "self_phase_task", "goal_type_task",
                 "self_previous_action_0"):
        observations[:, :, ledger[name]] = 1
    reset = np.zeros((5, steps), dtype=bool)
    reset[:, 0] = True
    return observations, reset


def test_quota_priority_reserves_heldout_content_and_spacing():
    connection = database()
    insert(connection, "heldout-copy", 7, "heldout", {"rare": True}, "0", "z")
    insert(connection, "rare", 7, "r1", {"rare": True}, "1", "z")
    insert(connection, "rare", 8, "overlap", {"rare": True}, "2", "z")
    insert(connection, "rare", 15, "r2", {"rare": True}, "3", "z")
    insert(connection, "duplicate", 7, "r1", {"rare": True}, "4", "z")
    for index in range(12):
        insert(connection, f"common-{index}", 7, f"c{index}", {}, str(index),
               f"common-{index:02d}")
    _, original_coverage = old._choose_candidates(
        connection, dict(train=5, validation=0, test=0))
    assert original_coverage["train"].get("rare", 0) == 0
    rows, coverage = extension._choose_train(
        connection, 5, {"heldout"}, requirements={"rare": 2}, order=("rare",))
    assert coverage["rare"] == 2
    assert [row["view_sha256"] for row in rows[:2]] == ["r1", "r2"]
    assert len(rows) == 5
    assert len({row["view_sha256"] for row in rows}) == 5
    assert "heldout" not in {row["view_sha256"] for row in rows}


def test_atomic_episode_index_resume_and_source_stamp(tmp_path, monkeypatch):
    connection = database()
    (tmp_path / "episodes").mkdir()
    path = tmp_path / "episodes/seed_10120.npz"
    observations, reset = valid_arrays()
    np.savez(path, observations=observations, reset_masks=reset)
    receipt = {"episode_id": "new", "seed": 10120, "steps": 10,
               "npz_sha256": old.file_sha256(path)}
    source = {"directory": str(tmp_path), "identity_sha256": "identity"}
    original = old._iter_window_views

    def broken(*args):
        yield 7, {}
        raise RuntimeError("interrupted")

    monkeypatch.setattr(old, "_iter_window_views", broken)
    monkeypatch.setattr(old, "_observable_flags", lambda view: ("ordinary", {}))
    with pytest.raises(RuntimeError, match="interrupted"):
        extension._index_episode(connection, source, receipt, "implementation")
    assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM episode_stamps").fetchone()[0] == 0
    monkeypatch.setattr(old, "_iter_window_views", original)
    assert extension._index_episode(connection, source, receipt, "implementation")
    assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 15
    monkeypatch.setattr(old, "_iter_window_views", broken)
    assert not extension._index_episode(connection, source, receipt, "implementation")
    with pytest.raises(ValueError, match="stamp"):
        extension._index_episode(connection, source, receipt, "changed")


def test_holdout_bytes_and_content_are_retained_and_raw_failure_not_frozen(tmp_path):
    parent = tmp_path / "parent.jsonl"
    validation = b'{ "partition": "validation", "id": 1 }\n'
    test = b'{"partition":"test","id":2}\n'
    parent.write_bytes(b'{"partition":"train"}\n' + validation + test)
    rows = extension._holdout_lines(parent)
    assert rows == [validation, test]
    target = tmp_path / "windows.jsonl"
    extension._write_windows(target, [], {}, rows)
    assert target.read_bytes() == validation + test
    assert not extension._raw_eligible({"train": {"rare": 1}}, True)
    assert extension._raw_eligible({"train": {"rare": 0}}, True)


def test_attempt_seal_rejects_rewriting_or_fake_freeze(tmp_path):
    (tmp_path / "manifest.json").write_text('{"eligible":false}', encoding="utf-8")
    extension._seal_attempt(tmp_path, {"parent": "hash"}, ["manifest.json"])
    extension._verify_attempt_seal(tmp_path, {"parent": "hash"})
    (tmp_path / "selection_frozen.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact"):
        extension._verify_attempt_seal(tmp_path, {"parent": "hash"})


def test_index_backup_is_independent_and_resume_does_not_open_parent(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    original = parent / "candidate_index.sqlite"
    source = database()
    insert(source, "original", 7, "original-content", {})
    source.commit()
    target = sqlite3.connect(original)
    source.backup(target)
    target.close()
    source.close()
    reference = dict(run_directory=str(parent), candidate_index_sha256=old.file_sha256(original))
    workspace = tmp_path / "workspace"
    connection = extension._open_index(workspace, reference, {"code": "frozen"})
    insert(connection, "new", 7, "new-content", {})
    connection.commit()
    connection.close()
    assert old.file_sha256(original) == reference["candidate_index_sha256"]
    assert old.file_sha256(workspace / "candidate_index.sqlite") != old.file_sha256(original)
    original_connect = sqlite3.connect

    def no_parent(*args, **kwargs):
        assert "mode=ro" not in str(args[0]), "Resume must not recopy the original index"
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", no_parent)
    connection = extension._open_index(workspace, reference, {"code": "frozen"})
    assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 2
    connection.close()


def test_batch_bounds_and_completeness_are_strict():
    contract = dict(batch_index=1, seed_start=10120, seed_end=10159,
                    partition="train", initial_episode_target=40,
                    maximum_episodes=40, extension_maximum_episodes=240,
                    extension_seed_bounds=[10120, 10359], n_agents=5,
                    max_steps=800, history_length=8)
    receipts = {str(seed): {"seed": seed} for seed in range(10120, 10160)}
    extension._validate_batch_bounds(contract, receipts)
    with pytest.raises(ValueError, match="complete"):
        extension._validate_batch_bounds(contract, dict(list(receipts.items())[:-1]))
    contract["seed_start"] = 10119
    with pytest.raises(ValueError, match="bounds"):
        extension._validate_batch_bounds(contract, receipts)


def test_relocation_preserves_recorded_windows_path_and_rejects_escape(tmp_path):
    recorded = r"D:\codeProject\project\.worktrees\core\artifacts\dataset\pool"
    assert extension._locate(recorded, tmp_path) == tmp_path / "artifacts/dataset/pool"
    with pytest.raises(ValueError, match="relocation"):
        extension._locate(r"D:\artifacts\..\secrets", tmp_path)


@pytest.mark.parametrize("wrong", ["seed", "episode_id"])
def test_source_receipt_rejects_filename_seed_and_episode_identity(tmp_path, wrong):
    (tmp_path / "episodes").mkdir()
    path = tmp_path / "episodes/seed_10120.npz"
    observations, reset = valid_arrays()
    np.savez(path, observations=observations, reset_masks=reset)
    identity = {"collection_contract": {"n_agents": 5}}
    receipt = dict(schema="goal-full-complete-episode-v3", seed=10120,
                   episode_id="collection-seed-10120-episode-0", steps=10,
                   boundary="truncated", npz_sha256=old.file_sha256(path),
                   collection_identity_sha256=old.canonical_sha256(identity),
                   charging_validation={"violations": 0})
    receipt[wrong] = 10121 if wrong == "seed" else "fake-episode"
    path.with_suffix(".json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="filename/seed/episode"):
        extension._receipt_source(tmp_path, identity)


def test_derived_failure_is_immutable_resumable_and_preserves_holdout(tmp_path, monkeypatch):
    """Bounded orchestration fixture; production source validation is separately tested."""
    parent, batch = tmp_path / "parent", tmp_path / "batch"
    ledger = {name: index for index, name in enumerate(old.SCHEMA.feature_names)}
    sources, holdouts, raw_lines = [], {}, []
    for directory, episodes, identity_sha in (
            (parent, [("old", 10000), ("v", 10001), ("t", 10002)], "old-identity"),
            (batch, [("new", 10120)], "new-identity")):
        (directory / "episodes").mkdir(parents=True)
        receipts = {}
        for number, (episode, seed) in enumerate(episodes):
            observations, resets = valid_arrays()
            observations[:, :, ledger["self_battery"]] = .1 + number * .2
            for agent in range(5):
                observations[agent, :, ledger["self_priority_weight"]] = .75
                observations[agent, :, ledger["self_battery"]] += agent * .01
            if episode == "new":
                observations[:, :, ledger["self_battery"]] += .7
            path = directory / "episodes" / f"seed_{seed:05d}.npz"
            np.savez(path, observations=observations, reset_masks=resets)
            receipts[episode] = dict(episode_id=episode, seed=seed, steps=10,
                                     npz_sha256=old.file_sha256(path))
            if episode in {"v", "t"}:
                partition = "validation" if episode == "v" else "test"
                view = old._window_view(observations[0, :8], resets[0, :8])
                stratum, flags = old._observable_flags(view)
                holdouts[partition] = [dict(episode_id=episode, agent_id=1,
                                            end_timestep=7, stratum=stratum,
                                            view_sha256=old.canonical_sha256(view), flags=flags)]
                window = dict(partition=partition, episode_id=episode, agent_id=1,
                              start_timestep=0, end_timestep=7, stratum=stratum,
                              observations=observations[0, :8].tolist(),
                              reset_masks=resets[0, :8].tolist())
                raw_lines.append((json.dumps(window, separators=(", ", ": "))
                                  + "\n").encode())
        sources.append(dict(directory=str(directory), identity_sha256=identity_sha,
                            receipts=receipts))
    (parent / "windows.jsonl").write_bytes(b"".join(raw_lines))
    old._atomic_json(parent / "selection_manifest.json", {"selected": holdouts})
    reference = {"run_directory": str(parent), "candidate_index_sha256": "fixture"}
    base_identity = dict(source_actor_contract={"verified_source": True},
                         rollout_environment_contract={"environment_sha256": "environment"},
                         native_runtime={})
    partitions = dict(train=["old"], validation=["v"], test=["t"])
    batches = [{"run_directory": str(batch), "batch_index": 1}]

    def inputs(*args):
        return base_identity, reference, sources, deepcopy(partitions), batches

    def index(*args):
        connection = database()
        receipt = sources[0]["receipts"]["old"]
        extension._index_episode(connection, sources[0], receipt, "fixture")
        return connection

    monkeypatch.setattr(extension, "_inputs", inputs)
    monkeypatch.setattr(extension, "_open_index", index)
    result = extension.select_medium_train_extension(parent, [batch], tmp_path / "out")
    directory = old.Path(result["run_directory"])
    assert result["manifest"]["main_pretraining_eligible"] is False
    assert not (directory / "selection_frozen.json").exists()
    assert extension._holdout_lines(directory / "windows.jsonl") == raw_lines
    assert extension.verify_medium_extension_dataset(directory, require_frozen=False) is None
    before = (directory / "version_receipt.json").read_bytes()

    def forbidden(*args):
        pytest.fail("Completed derived attempt must not reopen or reindex a source pool")

    monkeypatch.setattr(extension, "_open_index", forbidden)
    extension.select_medium_train_extension(parent, [batch], tmp_path / "out")
    assert (directory / "version_receipt.json").read_bytes() == before
    with pytest.raises(ValueError, match="insufficient"):
        extension.verify_medium_extension_dataset(directory)
