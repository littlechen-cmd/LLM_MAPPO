"""Immutable train-only derivations with resumable, source-bound candidate indexing.

This module does not collect trajectories, call a provider, or edit a source pool.
Raw eligibility is a prerequisite for labeling, never a label quality result.
"""

from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
import json
import os
from pathlib import Path, PureWindowsPath
import sqlite3

import numpy as np

from . import goal_full_dataset_run as old
from .source_actor_v3 import canonical_sha256 as canonical, file_sha256 as digest


LINEAGE_SCHEMA = "medium-train-extension-selection-v1"
IDENTITY_SCHEMA = "medium-train-extension-dataset-v1"
BASE_ID = "3fb225003be8bc9d45e525fe5d72d2775e43d67cbd40289329c6c216d9bf85bb"
QUOTA_ORDER = (
    "yielding-preference_high", "competitive_low", "charging_service",
    "stagnation_positive", "stagnation_zero", "goal-priority_high",
    "goal-priority_low", "yielding-preference_low", "local-constraint_high",
    "local-constraint_low", "ordinary", "competition", "structural",
)
ARTIFACTS = (
    "collection_identity.json", "episode_sources.json", "episode_partitions.json",
    "windows.jsonl", "label_selection_lists.json", "selection_manifest.json",
    "manifest.json", "state.json",
)


def _implementation():
    root = Path(__file__).resolve().parents[2]
    return dict(selector=digest(__file__),
                entrypoint=digest(root / "scripts/select_medium_train_extension.py"),
                original=old._implementation_hashes())


def _locate(recorded, artifact_root=None):
    """Map only a recorded artifacts/... suffix under an explicit repository root."""
    if artifact_root is None:
        return Path(recorded).resolve()
    parts = PureWindowsPath(str(recorded)).parts
    if parts.count("artifacts") != 1 or any(p in {"..", "."} for p in parts):
        raise ValueError("Source relocation requires an unambiguous artifacts suffix.")
    suffix = parts[parts.index("artifacts"):]
    root = Path(artifact_root).resolve()
    path = root.joinpath(*suffix).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source relocation would escape its explicit root.")
    return path


def _holdout_lines(path):
    with Path(path).open("rb") as stream:
        return [line for line in stream if json.loads(line)["partition"] != "train"]


def _receipt_source(directory, identity, recorded_directory=None):
    for path in (directory / "episodes").glob("seed_*.json"):
        if path.name.endswith(".intent.json"):
            continue
        receipt = old._read(path)
        seed = receipt.get("seed")
        if (type(seed) is not int or path.name != f"seed_{seed:05d}.json"
                or receipt.get("episode_id") != f"collection-seed-{seed}-episode-0"):
            raise ValueError("Source receipt filename/seed/episode identity differs.")
    receipts = old._receipts(directory, identity)
    for receipt in receipts.values():
        if receipt.get("charging_validation", {}).get("violations") != 0:
            raise ValueError("Source hard-charge receipt has failed validation.")
    return dict(directory=(str(directory.resolve()) if recorded_directory is None
                           else recorded_directory), identity_sha256=canonical(identity),
                receipts=receipts)


def _parent(directory, recorded_directory=None):
    directory = Path(directory).resolve()
    identity = old._read(directory / "collection_identity.json")
    if (canonical(identity) != BASE_ID
            or identity.get("synthetic_execution") is not False
            or identity.get("implementation_sha256") != old._implementation_hashes()
            or identity.get("source_actor_contract", {}).get("verified_source")
            is not True):
        raise ValueError("Original verified parent identity or implementation differs.")
    root = Path(__file__).resolve().parents[2]
    if digest(root / "scripts/run_hard_charge_dataset_snapshot.py") != (
            identity["native_runtime"]["collector_sha256"]):
        raise ValueError("Original native collector implementation differs.")
    source = _receipt_source(directory, identity, recorded_directory)
    receipts = source["receipts"]
    state = old._read(directory / "collection_state.json")
    partitions = old._read(directory / "episode_partitions.json")
    if (state.get("status") != "completed" or state.get("completed_count") != 120
            or len(receipts) != 120
            or {r["seed"] for r in receipts.values()} != set(range(10000, 10120))
            or {p: len(partitions.get(p, [])) for p in old.PARTITIONS}
            != dict(train=84, validation=18, test=18)
            or len({e for rows in partitions.values() for e in rows}) != 120
            or {e for rows in partitions.values() for e in rows} != set(receipts)):
        raise ValueError("Parent requires the original complete 120 and 84/18/18 split.")
    names = ("collection_identity.json", "collection_state.json", "manifest.json",
             "selection_manifest.json", "episode_partitions.json", "windows.jsonl",
             "label_selection_lists.json", "candidate_index.sqlite")
    files = {name: digest(directory / name) for name in names}
    manifest = old._read(directory / "manifest.json")
    if (manifest.get("windows_sha256") != files["windows.jsonl"]
            or manifest.get("selection_manifest_sha256")
            != files["selection_manifest.json"]
            or manifest.get("collection_identity_sha256") != BASE_ID
            or manifest.get("window_counts") != old.MEDIUM_BUDGETS):
        raise ValueError("Parent manifest is not bound to original windows/selection.")
    assignments = {e: p for p, episodes in partitions.items() for e in episodes}
    with sqlite3.connect((directory / "candidate_index.sqlite").as_uri() + "?mode=ro",
                         uri=True) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Original candidate SQLite integrity has failed.")
        actual = connection.execute(
            "SELECT episode,agent,partition,COUNT(*),MIN(end),MAX(end) "
            "FROM candidates GROUP BY episode,agent,partition").fetchall()
    expected = sorted((e, a, assignments[e], r["steps"] - 7, 7, r["steps"] - 1)
                      for e, r in receipts.items() for a in range(1, 6))
    if sorted(actual) != expected or sum(row[3] for row in actual) != 475800:
        raise ValueError("Original complete 475800 candidate index is required.")
    reference = dict(run_directory=source["directory"],
                     collection_identity_sha256=BASE_ID,
                     collection_identity_file_sha256=files["collection_identity.json"],
                     episode_receipts_sha256=canonical(receipts),
                     candidate_index_sha256=files["candidate_index.sqlite"],
                     episode_partitions_sha256=files["episode_partitions.json"],
                     windows_sha256=files["windows.jsonl"],
                     selection_manifest_sha256=files["selection_manifest.json"],
                     manifest_sha256=files["manifest.json"], files_sha256=files)
    return identity, reference, source, partitions


def _validate_batch_bounds(contract, receipts):
    index = contract.get("batch_index")
    if type(index) is not int or not 1 <= index <= 6:
        raise ValueError("Extension batch bounds differ.")
    start = 10120 + 40 * (index - 1)
    expected = dict(seed_start=start, seed_end=start + 39, partition="train",
                    initial_episode_target=40, maximum_episodes=40,
                    extension_maximum_episodes=240,
                    extension_seed_bounds=[10120, 10359], n_agents=5,
                    max_steps=800, history_length=8)
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Extension batch bounds or train-only contract differ.")
    if (len(receipts) != 40 or {r["seed"] for r in receipts.values()}
            != set(range(start, start + 40))):
        raise ValueError("Selection requires a complete 40-episode batch.")


def _inputs(parent_directory, batch_directories, artifact_root=None):
    recorded_parent = (str(Path(parent_directory).resolve()) if artifact_root is None
                       else parent_directory)
    identity, reference, parent_source, partitions = _parent(
        _locate(parent_directory, artifact_root), recorded_parent)
    sources, batches, seen_indices = [parent_source], [], set()
    seen_episodes = set(parent_source["receipts"])
    root = Path(__file__).resolve().parents[2]
    for recorded in batch_directories:
        directory = _locate(recorded, artifact_root)
        recorded = str(directory) if artifact_root is None else recorded
        batch = old._read(directory / "collection_identity.json")
        if (batch.get("schema") != old.COLLECTION_V3
                or batch.get("parent_pool") != reference
                or batch.get("synthetic_execution") is not False
                or any(batch.get(key) != identity.get(key) for key in (
                    "source_actor_contract", "rollout_environment_contract",
                    "native_runtime", "implementation_sha256"))
                or batch.get("extension_implementation_sha256") != dict(
                    entrypoint=digest(root / "scripts/run_medium_train_extension.py"),
                    frozen_native_helpers=digest(
                        root / "scripts/run_hard_charge_dataset_snapshot.py"))):
            raise ValueError("Extension original source/runtime/parent binding differs.")
        source = _receipt_source(directory, batch, recorded)
        _validate_batch_bounds(batch["collection_contract"], source["receipts"])
        state = old._read(directory / "collection_state.json")
        index = batch["collection_contract"]["batch_index"]
        if (state.get("status") != "completed" or state.get("completed_count") != 40
                or index in seen_indices
                or seen_episodes.intersection(source["receipts"])):
            raise ValueError("Duplicate or incomplete extension batch is prohibited.")
        seen_indices.add(index)
        seen_episodes.update(source["receipts"])
        sources.append(source)
        batches.append(dict(run_directory=recorded, batch_index=index,
                            collection_identity_sha256=canonical(batch),
                            collection_identity_file_sha256=digest(
                                directory / "collection_identity.json"),
                            episode_receipts_sha256=canonical(source["receipts"]),
                            collection_state_sha256=digest(
                                directory / "collection_state.json")))
    if not batches or len(batches) > 6:
        raise ValueError("One to six complete extension batches are required.")
    batches.sort(key=lambda item: item["batch_index"])
    if [b["batch_index"] for b in batches] != list(range(1, len(batches) + 1)):
        raise ValueError("Extension batches must form the consecutive approved prefix.")
    return identity, reference, sources, partitions, batches


def _create_tracking_tables(connection):
    connection.execute("CREATE TABLE IF NOT EXISTS extension_metadata "
                       "(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS episode_stamps "
                       "(episode TEXT PRIMARY KEY,binding TEXT NOT NULL,"
                       "candidate_count INTEGER NOT NULL)")
    connection.commit()


def _open_index(workspace, parent_reference, implementation):
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "candidate_index.sqlite"
    binding = dict(parent_pool=parent_reference, implementation=implementation)
    encoded = json.dumps(binding, sort_keys=True)
    if not path.exists():
        temporary = path.with_suffix(".sqlite.tmp")
        if temporary.exists():
            temporary.unlink()
        parent_path = Path(parent_reference["run_directory"]) / "candidate_index.sqlite"
        with closing(sqlite3.connect(
                parent_path.as_uri() + "?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(temporary)) as target:
                source.backup(target)
                _create_tracking_tables(target)
                target.execute("INSERT INTO extension_metadata VALUES (?,?)",
                               ("binding", encoded))
                target.commit()
        os.replace(temporary, path)
        if digest(parent_path) != parent_reference["candidate_index_sha256"]:
            raise ValueError("Original candidate index changed during backup.")
    connection = sqlite3.connect(path)
    row = connection.execute("SELECT value FROM extension_metadata WHERE key='binding'")
    if row.fetchone() != (encoded,):
        connection.close()
        raise ValueError("Incremental index parent or implementation binding differs.")
    return connection


def _index_episode(connection, source, receipt, implementation):
    episode = receipt["episode_id"]
    path = Path(source["directory"]) / "episodes" / f"seed_{receipt['seed']:05d}.npz"
    if digest(path) != receipt["npz_sha256"]:
        raise ValueError("Incremental episode raw SHA differs.")
    binding = json.dumps(dict(raw_sha256=receipt["npz_sha256"],
                              receipt_sha256=canonical(receipt),
                              collection_identity_sha256=source["identity_sha256"],
                              implementation_sha256=implementation), sort_keys=True)
    count = 5 * max(0, receipt["steps"] - 7)
    stamp = connection.execute("SELECT binding,candidate_count FROM episode_stamps "
                               "WHERE episode=?", (episode,)).fetchone()
    actual = connection.execute("SELECT COUNT(*) FROM candidates WHERE episode=?",
                                (episode,)).fetchone()[0]
    if stamp:
        if stamp != (binding, count) or actual != count:
            raise ValueError("Completed episode stamp or candidate count differs.")
        return False
    if actual:
        raise ValueError("Unstamped partial candidate episode cannot be trusted.")
    observations, resets = old._episode_arrays(path, 5)
    if observations.shape[1] != receipt["steps"]:
        raise ValueError("Episode step receipt differs from raw arrays.")
    with connection:
        for agent in range(5):
            for end, view in old._iter_window_views(observations[agent], resets[agent]):
                stratum, flags = old._observable_flags(view)
                connection.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)", (
                    episode, agent + 1, end, "train", stratum,
                    json.dumps(flags, sort_keys=True), canonical(view),
                    canonical([old.SALT, episode, agent + 1, end]),
                ))
        connection.execute("INSERT INTO episode_stamps VALUES (?,?,?)",
                           (episode, binding, count))
    return True


def _requirements(budget):
    empty = {p: [] for p in old.PARTITIONS}
    missing = old._coverage_missing({"train": {}}, dict(
        train=budget, validation=0, test=0), empty)
    return {k: v for k, v in missing["train"].items() if k != "quota"}


def _choose_train(connection, budget, reserved, *, requirements=None, order=QUOTA_ORDER):
    requirements = _requirements(budget) if requirements is None else requirements
    rows, coverage, seen, used = [], Counter(), set(reserved), defaultdict(list)

    def take(candidate):
        episode, agent, end, stratum, encoded, view_hash = candidate
        if view_hash in seen or any(abs(end - n) < 8 for n in used[episode, agent]):
            return
        flags = json.loads(encoded)
        seen.add(view_hash)
        used[episode, agent].append(end)
        coverage.update({key: int(value) for key, value in flags.items()})
        rows.append(dict(episode_id=episode, agent_id=agent, end_timestep=end,
                         stratum=stratum, view_sha256=view_hash, flags=flags))

    columns = "SELECT episode,agent,end,stratum,flags,view_hash FROM candidates "
    for key in order:
        if coverage[key] >= requirements.get(key, 0):
            continue
        cursor = connection.execute(columns + "WHERE partition='train' "
                                    "AND json_extract(flags,?)=1 "
                                    "ORDER BY rank,episode,agent,end", (f'$."{key}"',))
        for candidate in cursor:
            if coverage[key] >= requirements[key] or len(rows) == budget:
                break
            take(candidate)
        cursor.close()
    strata = [row[0] for row in connection.execute(
        "SELECT DISTINCT stratum FROM candidates "
        "WHERE partition='train' ORDER BY stratum")]
    cursors = {s: connection.execute(
        columns + "WHERE partition='train' AND stratum=? "
        "ORDER BY rank,episode,agent,end", (s,)) for s in strata}
    index = 0
    while strata and len(rows) < budget:
        stratum = strata[index % len(strata)]
        candidate = cursors[stratum].fetchone()
        if candidate is None:
            strata.remove(stratum)
            index = 0
            continue
        take(candidate)
        index += 1
    for cursor in cursors.values():
        cursor.close()
    return rows, dict(coverage)


def _episode_sources(sources):
    return {episode: dict(directory=source["directory"],
                          identity_sha256=source["identity_sha256"], receipt=receipt)
            for source in sources for episode, receipt in source["receipts"].items()}


def _array_reader(episode_sources, artifact_root=None):
    @lru_cache(maxsize=2)
    def arrays(episode):
        source = episode_sources[episode]
        path = _locate(source["directory"], artifact_root) / "episodes" / (
            f"seed_{source['receipt']['seed']:05d}.npz")
        return old._episode_arrays(path, 5)
    return arrays


def _write_windows(path, train_rows, episode_sources, holdout_lines):
    arrays = _array_reader(episode_sources)
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("wb") as stream:
        for row in sorted(train_rows, key=lambda r: (
                r["episode_id"], r["agent_id"], r["end_timestep"])):
            observations, resets = arrays(row["episode_id"])
            agent, end = row["agent_id"] - 1, row["end_timestep"]
            window = dict(partition="train", episode_id=row["episode_id"],
                          agent_id=agent + 1, start_timestep=end - 7, end_timestep=end,
                          stratum=row["stratum"],
                          observations=observations[agent, end - 7:end + 1].tolist(),
                          reset_masks=resets[agent, end - 7:end + 1].tolist())
            stream.write((json.dumps(window, sort_keys=True, separators=(",", ":"))
                          + "\n").encode("utf-8"))
        stream.writelines(holdout_lines)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _raw_eligible(missing, list_complete):
    return list_complete and not any(v for row in missing.values() for v in row.values())


def _seal_attempt(directory, identity, names):
    old._atomic_json(directory / "version_receipt.json", dict(
        identity_sha256=canonical(identity),
        artifacts_sha256={name: digest(directory / name) for name in names}))


def _verify_attempt_seal(directory, identity):
    seal = old._read(directory / "version_receipt.json")
    hashes = seal.get("artifacts_sha256", {})
    if (seal.get("identity_sha256") != canonical(identity)
            or any(not (directory / name).is_file()
                   or digest(directory / name) != expected
                   for name, expected in hashes.items())
            or (directory / "selection_frozen.json").exists()
            != ("selection_frozen.json" in hashes)):
        raise ValueError("Immutable attempt artifact hashes differ.")
    return seal


def _fixed_lists(train):
    trial = old._fixed_subset(train, 120, "trial-v3")
    ids = {(r["episode_id"], r["agent_id"], r["end_timestep"]) for r in trial}
    trial_rows = [r for r in train if (
        r["episode_id"], r["agent_id"], r["end_timestep"]) in ids]
    repeat = old._fixed_subset(trial_rows, 40, "repeat-v3")
    return dict(trial=trial, repeat=repeat, pro=list(repeat),
                review=old._fixed_subset(trial_rows, 60, "review-v3"))


def _holdout_selected(parent):
    selection = old._read(Path(parent) / "selection_manifest.json")["selected"]
    return {p: selection[p] for p in ("validation", "test")}


def select_medium_train_extension(  # noqa: C901
    parent_pool, batch_directories, output_root,
):
    """Index only appended episodes and write an immutable raw selection attempt."""
    parent, reference, sources, partitions, batches = _inputs(
        parent_pool, batch_directories)
    implementation = _implementation()
    identity = dict(schema=IDENTITY_SCHEMA, lineage_schema=LINEAGE_SCHEMA,
                    parent_pool=reference, batches=batches,
                    implementation=implementation,
                    source_actor_contract=parent["source_actor_contract"],
                    rollout_environment_contract=parent["rollout_environment_contract"],
                    native_runtime=parent["native_runtime"], synthetic_execution=False,
                    budgets=old.MEDIUM_BUDGETS, quota_order=list(QUOTA_ORDER))
    root = Path(output_root).resolve()
    directory = root / f"medium-extension-{canonical(identity)[:16]}"
    if (directory / "version_receipt.json").exists():
        verify_medium_extension_dataset(directory, require_frozen=False)
        return dict(run_directory=str(directory),
                    manifest=old._read(directory / "manifest.json"))
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "selection_frozen.json").exists():
        raise ValueError("Unsealed frozen selection must not be overwritten.")
    workspace_key = canonical(dict(
        parent_pool=reference, implementation=implementation))[:16]
    connection = _open_index(root / f"index-{workspace_key}", reference, implementation)
    try:
        included = set()
        for source in sources[1:]:
            for episode, receipt in sorted(source["receipts"].items()):
                _index_episode(connection, source, receipt, canonical(implementation))
                included.add(episode)
        # Later batches in this workspace are outside the current attempt lineage.
        allowed = set(partitions["train"]) | included
        connection.execute("CREATE TEMP TABLE allowed_episodes "
                           "(episode TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO allowed_episodes VALUES (?)",
                               ((e,) for e in sorted(allowed)))
        connection.execute("CREATE TEMP VIEW attempt_candidates AS SELECT c.* FROM "
                           "main.candidates c JOIN allowed_episodes a "
                           "ON c.episode=a.episode")
        connection.execute("CREATE TEMP VIEW candidates AS "
                           "SELECT * FROM attempt_candidates")
        holdout = _holdout_selected(parent_pool)
        reserved = {r["view_sha256"] for rows in holdout.values() for r in rows}
        train, train_coverage = _choose_train(connection, 2100, reserved)
    finally:
        connection.close()
    selected = dict(train=train, **holdout)
    coverage = dict(train=train_coverage)
    for partition in ("validation", "test"):
        counts = Counter()
        for row in selected[partition]:
            counts.update({k: int(v) for k, v in row["flags"].items()})
        coverage[partition] = dict(counts)
    missing = old._coverage_missing(coverage, old.MEDIUM_BUDGETS, selected)
    lists = _fixed_lists(train)
    preflight = old._label_preflight(lists, train)
    missing["fixed_lists"] = preflight["coverage_missing"]
    complete = all(len(lists[k]) == n for k, n in (
        ("trial", 120), ("repeat", 40), ("pro", 40), ("review", 60)))
    eligible = _raw_eligible(missing, complete)
    episode_sources = _episode_sources(sources)
    partitions = dict(partitions)
    partitions["train"] = sorted(set(partitions["train"]) | included)
    old._atomic_json(directory / "collection_identity.json", identity)
    old._atomic_json(directory / "episode_sources.json", episode_sources)
    old._atomic_json(directory / "episode_partitions.json", partitions)
    _write_windows(directory / "windows.jsonl", train, episode_sources,
                   _holdout_lines(Path(parent_pool) / "windows.jsonl"))
    old._atomic_json(directory / "label_selection_lists.json", lists)
    old._atomic_json(directory / "selection_manifest.json", dict(
        selected=selected, coverage=coverage, coverage_missing=missing,
        raw_label_preflight=preflight, label_lists_complete=complete,
        actual_label_list_counts={k: len(v) for k, v in lists.items()},
        label_lists_sha256=digest(directory / "label_selection_lists.json")))
    manifest = dict(
        schema=old.DATASET_V3, distribution=old.DISTRIBUTION_V3,
        lineage_schema=LINEAGE_SCHEMA,
        semantic_contract_version=old.SEMANTIC_CONTRACT_V3,
        semantic_dimensions=list(old.SEMANTIC_DIMENSIONS_V3),
        semantic_sources=["llm", "llm", "llm", "observable-rule-v3"],
        config_sha256=canonical(identity),
        collection_identity_sha256=canonical(identity),
        source_actor_contract=parent["source_actor_contract"],
        rollout_environment_contract=parent["rollout_environment_contract"],
        environment_sha256=parent["rollout_environment_contract"]["environment_sha256"],
        episode_count=len(episode_sources), trajectory_count=len(episode_sources) * 5,
        added_train_episode_count=len(included),
        episode_split_counts={p: len(partitions[p]) for p in old.PARTITIONS},
        window_counts={p: len(selected[p]) for p in old.PARTITIONS},
        windows_sha256=digest(directory / "windows.jsonl"),
        selection_manifest_sha256=digest(directory / "selection_manifest.json"),
        coverage=coverage, coverage_missing=missing, raw_label_preflight=preflight,
        raw_candidate_coverage_passed=eligible, main_pretraining_eligible=eligible,
        pretraining_eligible=eligible, label_quality_passed=False,
        formal_performance_evidence=False)
    old._atomic_json(directory / "manifest.json", manifest)
    if eligible:
        frozen = dict(environment="medium-5ag", budgets=old.MEDIUM_BUDGETS,
                      total_samples=3000, frozen_before_labeling=True,
                      lineage_sha256=canonical(identity),
                      episode_receipts_sha256=canonical({
                          e: s["receipt"] for e, s in episode_sources.items()}))
        for key in ("trial", "repeat", "pro", "review"):
            frozen[f"{key}_ids"] = [r["view_id"] for r in lists[key]]
        for key, name in dict(windows_sha256="windows.jsonl",
                              manifest_sha256="manifest.json",
                              selection_manifest_sha256="selection_manifest.json",
                              label_lists_sha256="label_selection_lists.json",
                              episode_partitions_sha256="episode_partitions.json",
                              episode_sources_sha256="episode_sources.json").items():
            frozen[key] = digest(directory / name)
        old._atomic_json(directory / "selection_frozen.json", frozen)
    old._atomic_json(directory / "state.json", dict(
        status="completed" if eligible else "insufficient_coverage",
        main_pretraining_eligible=eligible, coverage_missing=missing))
    names = list(ARTIFACTS) + (["selection_frozen.json"] if eligible else [])
    _seal_attempt(directory, identity, names)
    verify_medium_extension_dataset(directory, require_frozen=False)
    return dict(run_directory=str(directory), manifest=manifest)


def verify_medium_extension_dataset(  # noqa: C901
    directory, *, require_frozen=True, artifact_root=None,
):
    """Recheck immutable lineage, raw windows, holdout, coverage, and fixed lists.

    Returns the original frozen-list shape for the labeling adapter. This does
    not grant label-quality Go or evaluate model output.
    """
    directory = Path(directory).resolve()
    identity = old._read(directory / "collection_identity.json")
    if (identity.get("schema") != IDENTITY_SCHEMA
            or identity.get("lineage_schema") != LINEAGE_SCHEMA
            or identity.get("implementation") != _implementation()):
        raise ValueError("Derived selection identity or implementation differs.")
    seal = _verify_attempt_seal(directory, identity)
    frozen_exists = (directory / "selection_frozen.json").exists()
    if set(seal["artifacts_sha256"]) != set(ARTIFACTS) | (
            {"selection_frozen.json"} if frozen_exists else set()):
        raise ValueError("Required immutable attempt artifact bindings are absent.")
    parent, reference, sources, partitions, batches = _inputs(
        identity["parent_pool"]["run_directory"],
        [b["run_directory"] for b in identity["batches"]], artifact_root)
    if (reference != identity["parent_pool"] or batches != identity["batches"]
            or identity.get("source_actor_contract") != parent["source_actor_contract"]
            or identity.get("rollout_environment_contract")
            != parent["rollout_environment_contract"]
            or identity.get("budgets") != old.MEDIUM_BUDGETS
            or identity.get("quota_order") != list(QUOTA_ORDER)):
        raise ValueError("Derived lineage parent/batch/source contracts differ.")
    episode_sources = _episode_sources(sources)
    if old._read(directory / "episode_sources.json") != episode_sources:
        raise ValueError("Original multi-pool receipt identities differ.")
    added = {e for s in sources[1:] for e in s["receipts"]}
    partitions["train"] = sorted(set(partitions["train"]) | added)
    if old._read(directory / "episode_partitions.json") != partitions:
        raise ValueError("Derived train-only partition membership differs.")
    if _holdout_lines(directory / "windows.jsonl") != _holdout_lines(
            _locate(reference["run_directory"], artifact_root) / "windows.jsonl"):
        raise ValueError("Original validation/test window bytes have changed.")
    arrays = _array_reader(episode_sources, artifact_root)
    seen, used = set(), defaultdict(list)
    selected = {p: [] for p in old.PARTITIONS}
    coverage = {p: Counter() for p in old.PARTITIONS}
    with (directory / "windows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            window = json.loads(line)
            partition, episode = window["partition"], window["episode_id"]
            agent, end = window["agent_id"], window["end_timestep"]
            if (partition not in partitions or episode not in partitions[partition]
                    or type(agent) is not int or not 1 <= agent <= 5
                    or type(end) is not int or window["start_timestep"] != end - 7):
                raise ValueError("Window coordinates or partition are invalid.")
            observations, resets = arrays(episode)
            rows, masks = observations[agent - 1, end - 7:end + 1], resets[
                agent - 1, end - 7:end + 1]
            if (end < 7 or end >= observations.shape[1]
                    or not np.array_equal(rows, np.asarray(
                        window["observations"], dtype=np.float32))
                    or not np.array_equal(masks, window["reset_masks"])):
                raise ValueError("Window differs from its original source raw arrays.")
            view = old._window_view(rows, masks)
            view_hash = canonical(view)
            if view_hash in seen or any(
                    abs(end - n) < 8 for n in used[episode, agent]):
                raise ValueError("Global content deduplication or frame spacing failed.")
            seen.add(view_hash)
            used[episode, agent].append(end)
            stratum, flags = old._observable_flags(view)
            if stratum != window["stratum"]:
                raise ValueError("Raw stratum differs from observable source evidence.")
            selected[partition].append(dict(episode_id=episode, agent_id=agent,
                                            end_timestep=end, stratum=stratum,
                                            flags=flags, view_sha256=view_hash))
            coverage[partition].update({k: int(v) for k, v in flags.items()})
    selection = old._read(directory / "selection_manifest.json")
    for partition in old.PARTITIONS:
        def sort_key(row):
            return row["episode_id"], row["agent_id"], row["end_timestep"]
        if sorted(selected[partition], key=sort_key) != sorted(
                selection["selected"][partition], key=sort_key):
            raise ValueError("Selected candidate hashes/flags differ from raw windows.")
    lists = _fixed_lists(selection["selected"]["train"])
    if old._read(directory / "label_selection_lists.json") != lists:
        raise ValueError("Fixed train-only labeling lists differ from frozen algorithm.")
    preflight = old._label_preflight(lists, selected["train"])
    missing = old._coverage_missing(coverage, old.MEDIUM_BUDGETS, selected)
    missing["fixed_lists"] = preflight["coverage_missing"]
    complete = all(len(lists[k]) == n for k, n in (
        ("trial", 120), ("repeat", 40), ("pro", 40), ("review", 60)))
    eligible = _raw_eligible(missing, complete)
    manifest = old._read(directory / "manifest.json")
    state = old._read(directory / "state.json")
    if (manifest.get("collection_identity_sha256") != canonical(identity)
            or manifest.get("lineage_schema") != LINEAGE_SCHEMA
            or manifest.get("main_pretraining_eligible") is not eligible
            or manifest.get("raw_candidate_coverage_passed") is not eligible
            or manifest.get("coverage_missing") != missing
            or manifest.get("coverage") != coverage
            or state.get("status")
            != ("completed" if eligible else "insufficient_coverage")
            or frozen_exists != eligible):
        raise ValueError("Derived eligibility/coverage/freeze differs from raw gates.")
    if require_frozen and not eligible:
        raise ValueError("Raw coverage or fixed-list observability is insufficient.")
    if not eligible:
        return None
    frozen = old._read(directory / "selection_frozen.json")
    bindings = dict(windows_sha256="windows.jsonl", manifest_sha256="manifest.json",
                    selection_manifest_sha256="selection_manifest.json",
                    label_lists_sha256="label_selection_lists.json",
                    episode_partitions_sha256="episode_partitions.json",
                    episode_sources_sha256="episode_sources.json")
    if (any(frozen.get(k) != digest(directory / n) for k, n in bindings.items())
            or frozen.get("lineage_sha256") != canonical(identity)
            or frozen.get("episode_receipts_sha256") != canonical({
                e: s["receipt"] for e, s in episode_sources.items()})
            or frozen.get("environment") != "medium-5ag"
            or frozen.get("budgets") != old.MEDIUM_BUDGETS
            or frozen.get("total_samples") != 3000
            or frozen.get("frozen_before_labeling") is not True
            or any(frozen.get(f"{k}_ids") != [r["view_id"] for r in lists[k]]
                   for k in ("trial", "repeat", "pro", "review"))):
        raise ValueError("Derived frozen lineage/list/receipt bindings differ.")
    return frozen
