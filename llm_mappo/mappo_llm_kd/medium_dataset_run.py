"""Bounded medium-5AG C1 dataset collection from an approved final Actor."""

from __future__ import annotations

from hashlib import sha256
from functools import lru_cache
import json
import os
from pathlib import Path
import sqlite3

import numpy as np

from llm_mappo.mappo_baseline.runner import load_baseline_experiment
from llm_mappo.mappo_baseline.vector_env import new_baseline_environment

from .collector import collect_complete_episode, load_medium5_collection_policy
from .dataset import _observable_stratum, _partition_counts
from .prompt import _canonical_json
from .temporal_view import serialize_temporal_semantic_view
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
)


MEDIUM5_CODE_COMMIT = "2e579fd621a0b134068adb2cfcf79510e7672f44"
MEDIUM5_SALT = "medium-5ag-llm-dataset-v1"
SEED_START = 10000
INITIAL_EPISODES = 120
MAX_EPISODES = 240
HISTORY_LENGTH = 8
WINDOW_BUDGETS = {"train": 4200, "validation": 900, "test": 900}
_FEATURE_LEDGER = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
).feature_names


def run_medium5_episode_collection(  # noqa: C901
    *, baseline_config: str | Path, source_checkpoint: str | Path,
    actor_artifact: str | Path, actor_sha256: str,
    output_root: str | Path = "artifacts/llm_kd/dataset/medium-5ag",
    resume: bool = False, episode_count: int = INITIAL_EPISODES,
    environment_factory=new_baseline_environment,
    policy_loader=load_medium5_collection_policy,
) -> dict[str, object]:
    """Collect with an explicit failed-state receipt on any interruption."""

    if _valid_sha256(actor_sha256):
        directory = Path(output_root) / f"main-{actor_sha256[:12]}"
        label_directory = directory / "labels"
        if label_directory.exists() and any(label_directory.iterdir()):
            raise ValueError("A dataset with label artifacts cannot be extended.")
    try:
        return _run_medium5_episode_collection(
            baseline_config=baseline_config,
            source_checkpoint=source_checkpoint,
            actor_artifact=actor_artifact,
            actor_sha256=actor_sha256,
            output_root=output_root,
            resume=resume,
            episode_count=episode_count,
            environment_factory=environment_factory,
            policy_loader=policy_loader,
        )
    except Exception as error:
        if _valid_sha256(actor_sha256):
            directory = Path(output_root) / f"main-{actor_sha256[:12]}"
            identity_path = directory / "collection_identity.json"
            if identity_path.exists():
                try:
                    identity = _read_json(identity_path)
                    if identity.get("actor_sha256") == actor_sha256:
                        completed = sorted(
                            int(path.stem.split("_")[1])
                            for path in (directory / "episodes").glob("seed_*.json")
                            if path.stem.split("_")[1].isdigit()
                            and path.stem.endswith(path.stem.split("_")[1])
                        )
                        first_missing = SEED_START
                        while first_missing in completed:
                            first_missing += 1
                        _atomic_json(directory / "collection_state.json", {
                            "status": "failed", "failed_seed": first_missing,
                            "error_type": type(error).__name__,
                            "completed_count": len(completed),
                        })
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
        raise


def _run_medium5_episode_collection(  # noqa: C901
    *, baseline_config: str | Path, source_checkpoint: str | Path,
    actor_artifact: str | Path, actor_sha256: str,
    output_root: str | Path = "artifacts/llm_kd/dataset/medium-5ag",
    resume: bool = False, episode_count: int = INITIAL_EPISODES,
    environment_factory=new_baseline_environment,
    policy_loader=load_medium5_collection_policy,
) -> dict[str, object]:
    """Persist one compressed 209D five-agent episode per seed; resume safely."""

    if type(episode_count) is not int or not 1 <= episode_count <= MAX_EPISODES:
        raise ValueError("Episode count must be within 1..240.")
    experiment = load_baseline_experiment(baseline_config)
    if (
        experiment.schema != "core-mappo-209d-delivery-c1-v3"
        or experiment.method != "MAPPO-FP-DG"
        or experiment.approval_status != "approved"
        or experiment.environment["n_agents"] != 5
        or experiment.environment["max_steps"] != 800
        or experiment.training["environment_steps"] != 600000
        or experiment.training["seed"] != 7
    ):
        raise ValueError("Collection requires the approved medium-5AG C1-v3 contract.")
    environment_sha256 = _json_sha256(dict(experiment.environment))
    checkpoint_sha256 = _file_sha256(Path(source_checkpoint))
    if not _valid_sha256(actor_sha256):
        raise ValueError("Actor artifact SHA-256 is invalid.")
    expected_actor_identity = {
        "schema": experiment.schema,
        "method": experiment.method,
        "code_commit": MEDIUM5_CODE_COMMIT,
        "config_sha256": experiment.config_sha256,
        "seed": 7,
        "reward_definition_sha256": experiment.reward_definition_sha256,
        "environment_sha256": environment_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
    }
    policy = policy_loader(
        actor_artifact, device="cpu", expected_artifact_sha256=actor_sha256,
        expected_identity=expected_actor_identity,
    )
    identity = {
        "schema": "medium5-c1-dataset-collection-v1",
        "baseline_config_sha256": experiment.config_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
        "actor_sha256": actor_sha256,
        "actor_identity": expected_actor_identity,
        "environment_sha256": environment_sha256,
        "behavior_policy_id": policy.behavior_policy_id,
        "seed_start": SEED_START,
        "max_episodes": MAX_EPISODES,
        "history_length": HISTORY_LENGTH,
        "split_salt": MEDIUM5_SALT,
        "window_budgets": WINDOW_BUDGETS,
    }
    directory = Path(output_root) / f"main-{actor_sha256[:12]}"
    identity_path = directory / "collection_identity.json"
    if resume:
        if not identity_path.exists() or _read_json(identity_path) != identity:
            raise ValueError("Collection resume identity differs.")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "episodes").mkdir()
        _atomic_json(identity_path, identity)
    episode_directory = directory / "episodes"
    episode_directory.mkdir(exist_ok=True)
    completed = []
    for seed in range(SEED_START, SEED_START + episode_count):
        episode_file = episode_directory / f"seed_{seed:05d}.npz"
        receipt_file = episode_directory / f"seed_{seed:05d}.json"
        intent_file = episode_directory / f"seed_{seed:05d}.intent.json"
        intent = {
            "schema": "medium5-c1-episode-intent-v1", "seed": seed,
            "collection_identity_sha256": _json_sha256(identity),
        }
        if episode_file.exists() and not receipt_file.exists():
            if not intent_file.is_file() or _read_json(intent_file) != intent:
                raise ValueError(f"Seed {seed} has an unclaimed episode artifact.")
            steps = _validate_episode_arrays(episode_file, seed=seed)
            _atomic_json(receipt_file, _episode_receipt(
                episode_file, seed=seed, steps=steps, identity=identity
            ))
        if episode_file.exists() or receipt_file.exists():
            _validate_episode_file(
                episode_file, receipt_file, seed=seed, identity=identity
            )
            completed.append(seed)
            continue
        if intent_file.exists():
            if _read_json(intent_file) != intent:
                raise ValueError(f"Seed {seed} collection intent differs.")
        else:
            _atomic_json(intent_file, intent)
        environment = environment_factory(
            experiment.environment,
            replenishment_mode=experiment.environment.get(
                "training_replenishment_mode", "rware_delivery"
            ),
        )
        try:
            trajectories = collect_complete_episode(
                environment, policy, seed=seed,
                episode_id=f"collection-seed-{seed}-episode-0",
                environment_sha256=environment_sha256,
                config_sha256=experiment.config_sha256,
                max_steps=800,
            )
        finally:
            close = getattr(environment, "close", None)
            if close is not None:
                close()
        rows = sorted(trajectories, key=lambda item: item.trajectory.agent_id)
        observations = np.stack(
            [item.trajectory.observations for item in rows]
        ).astype(np.float32, copy=False)
        masks = np.stack([item.trajectory.reset_masks for item in rows])
        if observations.shape[0] != 5 or masks.shape != observations.shape[:2]:
            raise ValueError("Collected episode is not five-agent aligned.")
        temporary = episode_file.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, observations=observations, reset_masks=masks
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, episode_file)
        finally:
            temporary.unlink(missing_ok=True)
        _atomic_json(receipt_file, _episode_receipt(
            episode_file, seed=seed, steps=int(observations.shape[1]),
            identity=identity,
        ))
        completed.append(seed)
    _atomic_json(directory / "collection_state.json", {
        "status": "collected" if episode_count >= INITIAL_EPISODES else "partial",
        "completed_seeds": completed,
        "episode_count": len(completed),
    })
    return {
        "run_directory": str(directory.resolve()),
        "episode_count": len(completed),
        "actor_sha256": actor_sha256,
    }


def _validate_episode_file(
    episode_file: Path, receipt_file: Path, *, seed: int,
    identity: dict[str, object],
) -> None:
    if not episode_file.is_file() or not receipt_file.is_file():
        raise ValueError(f"Seed {seed} has an incomplete episode artifact.")
    receipt = _read_json(receipt_file)
    if (
        receipt.get("schema") != "medium5-c1-episode-v1"
        or receipt.get("seed") != seed
        or receipt.get("episode_id") != f"collection-seed-{seed}-episode-0"
        or receipt.get("npz_sha256") != _file_sha256(episode_file)
        or receipt.get("collection_identity_sha256") != _json_sha256(identity)
    ):
        raise ValueError(f"Seed {seed} episode receipt differs.")
    steps = _validate_episode_arrays(episode_file, seed=seed)
    if receipt.get("steps") != steps:
        raise ValueError(f"Seed {seed} episode length differs.")


def _validate_episode_arrays(path: Path, *, seed: int) -> int:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"observations", "reset_masks"}:
            raise ValueError(f"Seed {seed} episode fields differ.")
        observations = payload["observations"]
        masks = payload["reset_masks"]
    steps = observations.shape[1] if observations.ndim == 3 else None
    if (
        type(steps) is not int or not 1 <= steps <= 800
        or observations.shape != (5, steps, 209)
        or observations.dtype != np.float32
        or not np.isfinite(observations).all()
        or masks.shape != (5, steps)
        or masks.dtype != np.bool_
        or not np.all(masks[:, 0])
        or np.any(masks[:, 1:])
    ):
        raise ValueError(f"Seed {seed} episode arrays differ.")
    return steps


def _episode_receipt(
    path: Path, *, seed: int, steps: int, identity: dict[str, object]
) -> dict[str, object]:
    return {
        "schema": "medium5-c1-episode-v1", "seed": seed,
        "episode_id": f"collection-seed-{seed}-episode-0",
        "steps": steps,
        "npz_sha256": _file_sha256(path),
        "collection_identity_sha256": _json_sha256(identity),
    }


def select_medium5_windows(  # noqa: C901
    run_directory: str | Path, *, minimum_episodes: int = INITIAL_EPISODES,
    budgets: dict[str, int] | None = None,
    minimum_coverage: dict[str, int] | None = None,
    source_run_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Select exact episode-disjoint, history-complete windows via disk index."""

    directory = Path(run_directory)
    label_directory = directory / "labels"
    if label_directory.exists() and any(label_directory.iterdir()):
        raise ValueError("A dataset with label artifacts cannot be reselected.")
    identity = _read_json(directory / "collection_identity.json")
    if identity.get("schema") != "medium5-c1-dataset-collection-v1":
        raise ValueError("Medium dataset collection identity differs.")
    source_evidence = _source_run_evidence(source_run_manifest, identity)
    budgets = dict(WINDOW_BUDGETS if budgets is None else budgets)
    minimum_coverage = dict(
        {"train": 100, "validation": 30, "test": 30}
        if minimum_coverage is None else minimum_coverage
    )
    partitions = ("train", "validation", "test")
    if set(budgets) != set(partitions) or any(
        type(value) is not int or value < 1 for value in budgets.values()
    ):
        raise ValueError("Window budgets are invalid.")
    if set(minimum_coverage) != set(partitions) or any(
        type(value) is not int or value < 0
        for value in minimum_coverage.values()
    ):
        raise ValueError("Coverage thresholds are invalid.")
    if type(minimum_episodes) is not int or minimum_episodes < 1:
        raise ValueError("Minimum episode count is invalid.")
    receipts = _complete_episode_receipts(directory, identity)
    if len(receipts) < minimum_episodes:
        raise ValueError("Insufficient complete episodes for window selection.")
    episode_ids = sorted(receipts)
    counts = _partition_counts(
        len(episode_ids), {"train": 0.70, "validation": 0.15, "test": 0.15}
    )
    ranked = sorted(episode_ids, key=lambda episode_id: (
        sha256(f"{MEDIUM5_SALT}\0{episode_id}".encode("utf-8")).digest(),
        episode_id,
    ))
    assignments = {}
    cursor = 0
    for partition in partitions:
        for episode_id in ranked[cursor:cursor + counts[partition]]:
            assignments[episode_id] = partition
        cursor += counts[partition]
    connection = sqlite3.connect(directory / "candidate_index.sqlite")
    try:
        _initialize_index(connection)
        _index_episodes(connection, directory, receipts)
        with connection:
            connection.execute("DELETE FROM assignments")
            connection.executemany(
                "INSERT INTO assignments(episode_id, partition) VALUES (?, ?)",
                sorted(assignments.items()),
            )
        selected, coverage = _select_candidates(
            connection, directory, budgets, minimum_coverage
        )
    finally:
        connection.close()
    selected_count = {name: len(selected[name]) for name in partitions}
    selected_strata_counts = {
        name: {
            stratum: sum(row[3] == stratum for row in selected[name])
            for stratum in sorted({row[3] for row in selected[name]})
        }
        for name in partitions
    }
    missing = {
        name: {
            "quota": max(0, budgets[name] - selected_count[name]),
            "congested": max(
                0, minimum_coverage[name] - coverage[name]["congested"]
            ),
            "blocked_or_no_progress": max(
                0, minimum_coverage[name] - coverage[name]["blocked_or_no_progress"]
            ),
        }
        for name in partitions
    }
    if any(any(value > 0 for value in values.values()) for values in missing.values()):
        _atomic_json(directory / "selection_state.json", {
            "status": "insufficient_coverage", "missing": missing,
            "episode_count": len(episode_ids), "selected_counts": selected_count,
        })
        raise ValueError(f"Window selection requires more episodes: {missing}")
    _atomic_json(directory / "episode_partitions.json", {
        name: sorted(
            episode_id for episode_id, assigned in assignments.items()
            if assigned == name
        ) for name in partitions
    })
    windows_path = directory / "windows.jsonl"
    _write_selected_windows(windows_path, directory, selected)
    manifest = {
        "schema": "llm-dataset-build-run-v1",
        "distribution": "native-209d-collection-main-v1",
        "config_sha256": _json_sha256(identity),
        "environment_sha256": identity["environment_sha256"],
        "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
        "source_actor_sha256": identity["actor_sha256"],
        "source_training_config_sha256": identity["baseline_config_sha256"],
        "source_training_code_commit": identity["actor_identity"]["code_commit"],
        "source_training_run": source_evidence,
        "behavior_policy_id": identity["behavior_policy_id"],
        "observation_schema": DIRECT_GOAL_OBSERVATION_V2,
        "observation_dimension": 209,
        "episode_count": len(episode_ids),
        "trajectory_count": 5 * len(episode_ids),
        "window_counts": selected_count,
        "episode_split_counts": counts,
        "selected_strata_counts": selected_strata_counts,
        "coverage": coverage,
        "windows_sha256": _file_sha256(windows_path),
        "collection_identity_sha256": _json_sha256(identity),
        "episode_receipts_sha256": _json_sha256(receipts),
        "selection_code_sha256": {
            name: _file_sha256(Path(__file__).with_name(name))
            for name in (
                "medium_dataset_run.py", "dataset.py", "temporal_view.py",
                "prompt.py",
            )
        },
        "collection_code_at_launch": "not_recorded_at_launch",
        "main_pretraining_eligible": (
            budgets == WINDOW_BUDGETS
            and minimum_episodes == INITIAL_EPISODES
            and len(episode_ids) >= INITIAL_EPISODES
            and source_evidence is not None
            and minimum_coverage == {
                "train": 100, "validation": 30, "test": 30
            }
        ),
        "formal_performance_evidence": False,
    }
    _atomic_json(directory / "manifest.json", manifest)
    _atomic_json(directory / "state.json", {"status": "completed"})
    _atomic_json(directory / "selection_state.json", {
        "status": "completed", "episode_count": len(episode_ids),
        "selected_counts": selected_count, "coverage": coverage,
    })
    return {"run_directory": str(directory.resolve()), "manifest": manifest}


def _source_run_evidence(
    source_run_manifest: str | Path | None, identity: dict[str, object]
) -> dict[str, str] | None:
    if source_run_manifest is None:
        return None
    path = Path(source_run_manifest)
    manifest = _read_json(path)
    state_path = path.with_name("state.json")
    state = _read_json(state_path)
    source_identity = manifest.get("identity")
    if (
        state.get("status") != "complete"
        or not isinstance(source_identity, dict)
        or source_identity.get("schema") != "core-mappo-209d-delivery-c1-v3"
        or source_identity.get("config_sha256")
        != identity["baseline_config_sha256"]
        or source_identity.get("code_commit")
        != identity["actor_identity"]["code_commit"]
        or source_identity.get("reward_definition_sha256")
        != identity["actor_identity"]["reward_definition_sha256"]
        or manifest.get("training", {}).get("environment_steps") != 600000
        or _json_sha256(manifest.get("environment"))
        != identity["environment_sha256"]
    ):
        raise ValueError("Source training run manifest differs from Actor identity.")
    return {
        "run_manifest_sha256": _file_sha256(path),
        "state_sha256": _file_sha256(state_path),
        "training_status": "complete",
    }


def _complete_episode_receipts(
    directory: Path, identity: dict[str, object]
) -> dict[str, dict[str, object]]:
    receipts = {}
    for path in sorted((directory / "episodes").glob("seed_*.json")):
        if path.name.endswith(".intent.json"):
            continue
        receipt = _read_json(path)
        seed = receipt.get("seed")
        if type(seed) is not int or not SEED_START <= seed < SEED_START + MAX_EPISODES:
            raise ValueError("Episode seed is outside reserved dataset range.")
        expected_name = f"seed_{seed:05d}.json"
        if path.name != expected_name:
            raise ValueError("Episode receipt filename differs.")
        episode_file = path.with_suffix(".npz")
        _validate_episode_file(
            episode_file, path, seed=seed, identity=identity
        )
        receipts[receipt["episode_id"]] = receipt
    if [value["seed"] for value in receipts.values()] != list(
        range(SEED_START, SEED_START + len(receipts))
    ):
        raise ValueError("Collected episode seeds must be a contiguous prefix.")
    return receipts


def _initialize_index(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS indexed_episodes (
            episode_id TEXT PRIMARY KEY, npz_sha256 TEXT NOT NULL,
            candidate_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidates (
            stratum TEXT NOT NULL, rank BLOB NOT NULL,
            episode_id TEXT NOT NULL, agent_id INTEGER NOT NULL,
            end_timestep INTEGER NOT NULL,
            PRIMARY KEY (episode_id, agent_id, end_timestep)
        );
        CREATE INDEX IF NOT EXISTS candidates_by_stratum_rank
            ON candidates(stratum, rank);
        CREATE TABLE IF NOT EXISTS assignments (
            episode_id TEXT PRIMARY KEY, partition TEXT NOT NULL
        );
    """)


def _index_episodes(
    connection: sqlite3.Connection, directory: Path,
    receipts: dict[str, dict[str, object]],
) -> None:
    indexed = {
        row[0]: (row[1], row[2])
        for row in connection.execute(
            "SELECT episode_id, npz_sha256, candidate_count FROM indexed_episodes"
        )
    }
    if set(indexed) - set(receipts):
        raise ValueError("Candidate index contains unknown episodes.")
    for episode_id, receipt in sorted(receipts.items()):
        expected = (receipt["npz_sha256"], 5 * max(0, receipt["steps"] - 7))
        if episode_id in indexed:
            if indexed[episode_id] != expected:
                raise ValueError("Candidate index episode identity differs.")
            continue
        path = directory / "episodes" / f"seed_{receipt['seed']:05d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            observations = payload["observations"]
        rows = []
        for agent_index in range(5):
            agent_rows = observations[agent_index]
            for end in range(7, agent_rows.shape[0]):
                stratum = _observable_stratum(agent_rows[end - 7:end + 1])
                rank_identity = (
                    f"{MEDIUM5_SALT}\0{stratum}\0{episode_id}"
                    f"\0{agent_index + 1}\0{end - 7}\0{end}"
                )
                rows.append((
                    stratum, sha256(rank_identity.encode("utf-8")).digest(),
                    episode_id, agent_index + 1, end,
                ))
        with connection:
            connection.executemany(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)", rows
            )
            connection.execute(
                "INSERT INTO indexed_episodes VALUES (?, ?, ?)",
                (episode_id, receipt["npz_sha256"], len(rows)),
            )


def _select_candidates(
    connection: sqlite3.Connection, directory: Path,
    budgets: dict[str, int], minimum_coverage: dict[str, int],
) -> tuple[dict[str, list[tuple]], dict[str, dict[str, int]]]:
    selected = {name: [] for name in budgets}
    coverage = {
        name: {"congested": 0, "blocked_or_no_progress": 0}
        for name in budgets
    }
    used_endpoints: dict[tuple[str, int], list[int]] = {}
    seen_views: set[str] = set()

    @lru_cache(maxsize=8)
    def episode_observations(episode_id: str) -> np.ndarray:
        seed = int(episode_id.split("-")[2])
        with np.load(
            directory / "episodes" / f"seed_{seed:05d}.npz", allow_pickle=False
        ) as payload:
            return payload["observations"]

    for partition in ("train", "validation", "test"):
        strata = [row[0] for row in connection.execute("""
            SELECT DISTINCT c.stratum FROM candidates AS c
            JOIN assignments AS a USING (episode_id)
            WHERE a.partition = ? ORDER BY c.stratum
        """, (partition,))]
        cursors = {}
        for stratum in strata:
            cursors[stratum] = connection.cursor().execute("""
                SELECT c.episode_id, c.agent_id, c.end_timestep
                FROM candidates AS c JOIN assignments AS a USING (episode_id)
                WHERE a.partition = ? AND c.stratum = ?
                ORDER BY c.rank, c.episode_id, c.agent_id, c.end_timestep
            """, (partition, stratum))
        index = 0
        while strata and len(selected[partition]) < budgets[partition]:
            stratum = strata[index % len(strata)]
            candidate = cursors[stratum].fetchone()
            if candidate is None:
                cursors[stratum].close()
                strata.remove(stratum)
                index = 0
                continue
            episode_id, agent_id, end = candidate
            key = (episode_id, agent_id)
            if any(
                abs(end - previous) < HISTORY_LENGTH
                for previous in used_endpoints.get(key, [])
            ):
                index += 1
                continue
            observations = episode_observations(episode_id)[
                agent_id - 1, end - 7:end + 1
            ]
            reset_masks = np.zeros(HISTORY_LENGTH, dtype=np.bool_)
            if end == 7:
                reset_masks[0] = True
            view = serialize_temporal_semantic_view(
                observations, reset_masks, _FEATURE_LEDGER, HISTORY_LENGTH
            )
            view_sha256 = sha256(_canonical_json(view).encode("utf-8")).hexdigest()
            if view_sha256 in seen_views:
                index += 1
                continue
            seen_views.add(view_sha256)
            used_endpoints.setdefault(key, []).append(end)
            selected[partition].append(
                (episode_id, agent_id, end, stratum, view_sha256)
            )
            coverage[partition]["congested"] += "congestion=present" in stratum
            coverage[partition]["blocked_or_no_progress"] += (
                "blocked_or_no_progress=true" in stratum
            )
            index += 1
        for cursor in cursors.values():
            cursor.close()
    return selected, coverage


def _write_selected_windows(
    path: Path, directory: Path, selected: dict[str, list[tuple]]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for partition in ("train", "validation", "test"):
                episode_groups: dict[str, list[tuple]] = {}
                for row in selected[partition]:
                    episode_groups.setdefault(row[0], []).append(row)
                for episode_id, rows in sorted(episode_groups.items()):
                    seed = int(episode_id.split("-")[2])
                    with np.load(
                        directory / "episodes" / f"seed_{seed:05d}.npz",
                        allow_pickle=False,
                    ) as payload:
                        observations = payload["observations"]
                        reset_masks = payload["reset_masks"]
                    for _, agent_id, end, stratum, _view_hash in rows:
                        start = end - HISTORY_LENGTH + 1
                        record = {
                            "partition": partition, "episode_id": episode_id,
                            "agent_id": agent_id, "start_timestep": start,
                            "end_timestep": end,
                            "observations": observations[
                                agent_id - 1, start:end + 1
                            ].tolist(),
                            "reset_masks": reset_masks[
                                agent_id - 1, start:end + 1
                            ].tolist(),
                            "stratum": stratum,
                        }
                        stream.write(json.dumps(
                            record, sort_keys=True, separators=(",", ":")
                        ) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object.")
    return value


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
