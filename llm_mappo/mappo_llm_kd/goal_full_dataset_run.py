"""Identity-separated v3 episode collection and bounded on-disk selection."""

from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import subprocess

import numpy as np
import torch

from llm_mappo.goal_full_contract import CORE_GOAL_FULL_SCHEMA
from llm_mappo.mappo_baseline.runner import load_baseline_experiment
from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
    build_structured_observations,
)
from llm_mappo.mappo_baseline.vector_env import new_baseline_environment
from .dataset import _partition_counts
from .dataset_labeling import _window_id
from .semantic_v3 import (
    SEMANTIC_CONTRACT_V3, SEMANTIC_DIMENSIONS_V3, reference_semantics_v3,
    _competition_candidate, _competing_peer, _weak_relation,
)
from .source_actor_v3 import (
    canonical_sha256, file_sha256, load_pinned_medium_source,
)
from .temporal_view import serialize_temporal_semantic_view


COLLECTION_V3 = "goal-full-dataset-collection-v3"
DATASET_V3 = "llm-dataset-build-run-v3"
DISTRIBUTION_V3 = "native-209d-goal-full-main-v3"
PARTITIONS = ("train", "validation", "test")
MEDIUM_BUDGETS = dict(train=2100, validation=450, test=450)
SALT = "goal-full-dataset-v3"
SCHEMA = StructuredObservationSchema(observation_schema=DIRECT_GOAL_OBSERVATION_V2)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _observations(environment):
    return build_structured_observations(
        environment, SCHEMA, goal_interface="direct_goal", planner=None,
    )


def collect_goal_full_episode(
    environment, policy, *, seed, max_steps=800, observation_builder=_observations,
):
    """Keep observations before each action through a real episode boundary."""
    agents = int(environment.n_agents)
    if (agents not in {3, 5, 12} or type(max_steps) is not int
            or not 1 <= max_steps <= 800):
        raise ValueError("Collection requires 3/5/12 agents and horizon at most 800.")
    environment.reset(seed=seed)
    policy.start_episode(seed)
    hidden = np.zeros((1, agents, policy.hidden_dim), dtype=np.float32)
    observations = np.empty((agents, max_steps, 209), dtype=np.float32)
    for index in range(max_steps):
        rows = np.asarray(observation_builder(environment), dtype=np.float32)
        masks = np.asarray(environment.action_masks(), dtype=np.bool_)
        if rows.shape != (agents, 209) or not np.isfinite(rows).all():
            raise ValueError("Collection Actor observations differ from native 209D.")
        if masks.shape != (agents, 5) or not np.all(masks.any(axis=-1)):
            raise ValueError("Collection action masks are invalid.")
        observations[:, index] = rows
        actions, hidden = policy.act(rows, masks, hidden)
        actions = np.asarray(actions)
        if (
            actions.shape != (agents,) or actions.dtype.kind not in "iu"
            or np.any(actions < 0) or np.any(actions >= 5)
            or not np.all(masks[np.arange(agents), actions])
        ):
            raise ValueError("Collection policy emitted an action outside its mask.")
        transition = environment.step(actions)
        boundary = (
            "terminated" if transition.terminated else
            "truncated" if transition.truncated else
            "deadlocked" if transition.metrics.deadlocked else None
        )
        if boundary:
            length = index + 1
            resets = np.zeros((agents, length), dtype=np.bool_)
            resets[:, 0] = True
            return dict(observations=observations[:, :length].copy(),
                        reset_masks=resets, boundary=boundary)
    raise RuntimeError("Collection stopped before a complete episode boundary.")


def _implementation_hashes():
    root = Path(__file__).resolve().parents[1]
    names = (
        "mappo_llm_kd/goal_full_dataset_run.py", "mappo_llm_kd/source_actor_v3.py",
        "mappo_llm_kd/collector.py", "mappo_llm_kd/semantic_v3.py",
        "mappo_llm_kd/temporal_view.py", "mappo_llm_kd/dataset.py",
        "mappo_llm_kd/dataset_labeling.py",
        "mappo_baseline/structured_observation.py", "mappo_baseline/networks.py",
        "mappo_baseline/vector_env.py", "mappo_baseline/runner.py",
        "environment.py", "warehouse_environment.py", "warehouse_observation.py",
        "goal_full_contract.py", "medium5_contract.py", "reward_v2.py",
        "rules.py", "types.py", "../rware/warehouse.py",
    )
    return {name: file_sha256(root / name) for name in names}


def _reject_labels(directory):
    labels = directory / "labels"
    if labels.exists() and any(labels.iterdir()):
        raise ValueError("A dataset with labels cannot be extended or reselected.")


def _episode_arrays(path, agents):
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"observations", "reset_masks"}:
            raise ValueError("Episode array fields differ.")
        observations, resets = payload["observations"], payload["reset_masks"]
    if (
        observations.ndim != 3 or observations.shape[0] != agents
        or observations.shape[2] != 209 or not 1 <= observations.shape[1] <= 800
        or observations.dtype != np.float32 or not np.isfinite(observations).all()
        or resets.shape != observations.shape[:2] or resets.dtype != np.bool_
        or not np.all(resets[:, 0]) or np.any(resets[:, 1:])
    ):
        raise ValueError("Episode arrays are misaligned or incomplete.")
    return observations, resets


def run_goal_full_collection(  # noqa: C901
    *, source_config, source_checkpoint, actor_artifact, actor_sha256,
    rollout_config, output_root, episode_count=120, seed_start=10000, resume=False,
    source_manifest=None, source_loader=load_pinned_medium_source,
    environment_factory=new_baseline_environment, observation_builder=_observations,
):
    """Append complete episodes under a fixed source/rollout/code contract."""
    if type(episode_count) is not int or not 1 <= episode_count <= 120:
        raise ValueError("Episode count exceeds the approved 120-episode maximum.")
    if type(seed_start) is not int or seed_start < 0:
        raise ValueError("Collection seed_start is invalid.")
    torch.set_num_threads(1)
    experiment = load_baseline_experiment(rollout_config)
    if (experiment.schema != CORE_GOAL_FULL_SCHEMA
            or experiment.approval_status != "approved"):
        raise ValueError("Real rollout requires an approved medium goal-full contract.")
    policy, source = source_loader(
        source_config=source_config, source_checkpoint=source_checkpoint,
        actor_artifact=actor_artifact, actor_sha256=actor_sha256,
        source_manifest=source_manifest,
    )
    identity = dict(
        schema=COLLECTION_V3, source_actor_contract=source,
        rollout_environment_contract=dict(
            schema=experiment.schema, config_sha256=experiment.config_sha256,
            environment=dict(experiment.environment),
            environment_sha256=canonical_sha256(dict(experiment.environment)),
            reward_definition_sha256=experiment.reward_definition_sha256,
        ),
        collection_contract=dict(
            seed_start=seed_start, initial_episode_target=120, maximum_episodes=120,
            max_steps=800, n_agents=int(experiment.environment["n_agents"]),
            history_length=8, split_ratios=dict(train=0.70, validation=0.15, test=0.15),
            split_salt=SALT, window_budgets=MEDIUM_BUDGETS,
            behavior_policy_id=policy.behavior_policy_id,
            sampling="seeded_stochastic_original_frozen_policy",
            learner_inference_threads=1,
        ),
        implementation_sha256=_implementation_hashes(),
        synthetic_execution=(environment_factory is not new_baseline_environment
                             or observation_builder is not _observations
                             or source_loader is not load_pinned_medium_source),
    )
    identity_sha = canonical_sha256(identity)
    directory = Path(output_root) / f"goal-full-v3-{identity_sha[:16]}"
    if resume:
        if (not directory.is_dir()
                or _read(directory / "collection_identity.json") != identity):
            raise ValueError("Collection resume source/rollout/code identity differs.")
        _reject_labels(directory)
        if (directory / "selection_frozen.json").exists():
            raise ValueError("A frozen selection cannot extend its raw collection.")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "episodes").mkdir()
        _atomic_json(directory / "collection_identity.json", identity)
        _atomic_json(directory / "launch_metadata.json", {
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                text=True,
            ).strip(), "head_is_metadata_only": True,
            "runtime_versions": dict(python=platform.python_version(),
                                     numpy=np.__version__, torch=torch.__version__),
        })
    episodes = directory / "episodes"
    completed = 0
    for seed in range(seed_start, seed_start + episode_count):
        path = episodes / f"seed_{seed:05d}.npz"
        receipt_path = path.with_suffix(".json")
        intent_path = path.with_suffix(".intent.json")
        temporary = path.with_suffix(".npz.tmp")
        intent = dict(seed=seed, collection_identity_sha256=identity_sha)
        try:
            if not receipt_path.exists() and intent_path.exists():
                pending = _read(intent_path)
                completion = pending.get("completion")
                candidate = path if path.exists() else temporary
                if completion is not None and candidate.exists():
                    if (
                        pending.get("seed") != seed
                        or pending.get("collection_identity_sha256") != identity_sha
                        or completion.get("npz_sha256") != file_sha256(candidate)
                        or completion.get("collection_identity_sha256") != identity_sha
                    ):
                        raise ValueError("Pending complete episode identity differs.")
                    _episode_arrays(
                        candidate, identity["collection_contract"]["n_agents"],
                    )
                    if candidate == temporary:
                        os.replace(temporary, path)
                    _atomic_json(receipt_path, completion)
            if receipt_path.exists():
                receipt = _read(receipt_path)
                if (receipt.get("collection_identity_sha256") != identity_sha
                        or receipt.get("npz_sha256") != file_sha256(path)
                        or receipt.get("boundary") not in {
                            "terminated", "truncated", "deadlocked"}):
                    raise ValueError("Completed episode receipt identity differs.")
                rows, _ = _episode_arrays(
                    path, identity["collection_contract"]["n_agents"],
                )
                if receipt.get("steps") != rows.shape[1] or receipt.get("seed") != seed:
                    raise ValueError("Completed episode receipt alignment differs.")
                completed += 1
                continue
            if path.exists():
                raise ValueError("Episode array lacks its complete-boundary receipt.")
            if temporary.exists():
                raise ValueError("Unclaimed temporary episode cannot be overwritten.")
            _atomic_json(intent_path, intent)
            environment = environment_factory(experiment.environment)
            try:
                result = collect_goal_full_episode(
                    environment, policy, seed=seed, max_steps=800,
                    observation_builder=observation_builder,
                )
            finally:
                environment.close()
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, observations=result["observations"],
                                    reset_masks=result["reset_masks"])
                stream.flush()
                os.fsync(stream.fileno())
            receipt = dict(
                schema="goal-full-complete-episode-v3", seed=seed,
                episode_id=f"collection-seed-{seed}-episode-0",
                steps=result["observations"].shape[1], boundary=result["boundary"],
                npz_sha256=file_sha256(temporary),
                collection_identity_sha256=identity_sha,
            )
            _atomic_json(intent_path, dict(**intent, completion=receipt))
            os.replace(temporary, path)
            _atomic_json(receipt_path, receipt)
            completed += 1
            _atomic_json(directory / "collection_state.json", dict(
                status="running", completed_count=completed,
                requested_count=episode_count,
            ))
        except Exception as error:
            failure = dict(seed=seed, error_type=type(error).__name__,
                           completed_count=completed,
                           collection_identity_sha256=identity_sha)
            with (directory / "failures.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, sort_keys=True) + "\n")
            _atomic_json(directory / "collection_state.json",
                         dict(status="failed", **failure))
            raise
    _atomic_json(directory / "collection_state.json", dict(
        status="completed", completed_count=completed, requested_count=episode_count,
    ))
    return dict(run_directory=str(directory.resolve()), completed_count=completed)


def _window_view(rows, resets):
    return serialize_temporal_semantic_view(rows, resets, SCHEMA.feature_names, 8)


def _observable_flags(view):
    """Reference-based candidate diagnostics, never accepted LLM labels."""
    reference = reference_semantics_v3(view)
    scores = reference["scores"]
    current = view["history"][-1]
    target, direct = _competition_candidate(current)
    competition = False
    if target is not None:
        peer, reason = _competing_peer(view, target)
        competition = reason is None and (direct or _weak_relation(current, peer))
    yielding = scores["yielding-preference"]
    constraint, stagnation = scores["local-constraint"], scores["motion-stagnation"]
    flags = dict(
        charging_service=(reference["unknown_reasons"]["yielding-preference"]
                          == "hard-service-lock"),
        competition=competition,
        structural=constraint is not None and constraint >= 0.5,
        competitive_low=competition and yielding == 0,
        stagnation_positive=stagnation is not None and stagnation >= 0.25,
        stagnation_zero=stagnation == 0,
    )
    flags["ordinary"] = not any(flags[key] for key in (
        "charging_service", "competition", "structural", "stagnation_positive",
    ))
    flags["structural_no_stagnation"] = flags["structural"] and stagnation == 0
    flags["open_stagnation"] = constraint == 0 and flags["stagnation_positive"]
    for dimension in SEMANTIC_DIMENSIONS_V3[:3]:
        value = scores[dimension]
        flags[f"{dimension}_known"] = value is not None
        flags[f"{dimension}_high"] = value is not None and value >= 0.5
        flags[f"{dimension}_low"] = value is not None and value <= 0.25
    stratum = "|".join(key for key in (
        "charging_service", "competition", "structural", "ordinary",
        "stagnation_positive",
    ) if flags[key]) or "other"
    return stratum, flags


def _receipts(directory, identity):
    result = {}
    identity_sha = canonical_sha256(identity)
    agents = identity["collection_contract"]["n_agents"]
    for path in sorted((directory / "episodes").glob("seed_*.json")):
        if path.name.endswith(".intent.json"):
            continue
        receipt = _read(path)
        arrays = path.with_suffix(".npz")
        rows, _ = _episode_arrays(arrays, agents)
        if (
            receipt.get("schema") != "goal-full-complete-episode-v3"
            or receipt.get("collection_identity_sha256") != identity_sha
            or receipt.get("npz_sha256") != file_sha256(arrays)
            or receipt.get("steps") != rows.shape[1]
            or receipt.get("boundary") not in {"terminated", "truncated", "deadlocked"}
            or receipt.get("episode_id") in result
        ):
            raise ValueError("Selection episode receipt differs from complete arrays.")
        result[receipt["episode_id"]] = receipt
    if not result:
        raise ValueError("Selection requires complete episodes.")
    return result


def _index_candidates(connection, directory, receipts, assignments):
    with connection:
        connection.execute("DROP TABLE IF EXISTS candidates")
        connection.execute("""CREATE TABLE candidates (
            episode TEXT, agent INTEGER, end INTEGER, partition TEXT,
            stratum TEXT, flags TEXT, view_hash TEXT, rank TEXT,
            PRIMARY KEY (episode, agent, end))""")
        for episode_id, receipt in sorted(receipts.items()):
            path = directory / "episodes" / f"seed_{receipt['seed']:05d}.npz"
            with np.load(path, allow_pickle=False) as payload:
                observations, resets = payload["observations"], payload["reset_masks"]
            for agent in range(observations.shape[0]):
                for end in range(7, observations.shape[1]):
                    view = _window_view(observations[agent, end - 7:end + 1],
                                        resets[agent, end - 7:end + 1])
                    stratum, flags = _observable_flags(view)
                    rank = canonical_sha256([SALT, episode_id, agent + 1, end])
                    connection.execute(
                        "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)", (
                            episode_id, agent + 1, end, assignments[episode_id], stratum,
                            json.dumps(flags, sort_keys=True), canonical_sha256(view),
                            rank,
                        ))
        connection.execute("CREATE INDEX candidate_order "
                           "ON candidates(partition,stratum,rank)")


def _choose_candidates(connection, budgets):
    selected, coverage = {}, {}
    used = defaultdict(list)
    seen_content = set()
    for partition in PARTITIONS:
        rows, counts = [], Counter()
        strata = [row[0] for row in connection.execute(
            "SELECT DISTINCT stratum FROM candidates WHERE partition=? ORDER BY stratum",
            (partition,),
        )]
        cursors = {stratum: connection.execute(
            "SELECT episode,agent,end,stratum,flags,view_hash FROM candidates "
            "WHERE partition=? AND stratum=? ORDER BY rank,episode,agent,end",
            (partition, stratum),
        ) for stratum in strata}
        cursor_index = 0
        while strata and len(rows) < budgets[partition]:
            stratum = strata[cursor_index % len(strata)]
            candidate = cursors[stratum].fetchone()
            if candidate is None:
                cursors[stratum].close()
                strata.remove(stratum)
                cursor_index = 0
                continue
            episode, agent, end, _, flags_json, view_hash = candidate
            key = (episode, agent)
            if view_hash not in seen_content and all(
                abs(end - previous) >= 8 for previous in used[key]
            ):
                seen_content.add(view_hash)
                used[key].append(end)
                flags = json.loads(flags_json)
                counts.update({name: int(value) for name, value in flags.items()})
                rows.append(dict(episode_id=episode, agent_id=agent, end_timestep=end,
                                 stratum=stratum, view_sha256=view_hash, flags=flags))
            cursor_index += 1
        for cursor in cursors.values():
            cursor.close()
        selected[partition], coverage[partition] = rows, dict(counts)
    return selected, coverage


def _coverage_missing(coverage, budgets, selected):
    missing = {partition: dict(quota=budgets[partition] - len(selected[partition]))
               for partition in PARTITIONS}
    count = budgets["train"]
    required = dict(charging_service=min(30, math.ceil(count * 0.02)),
                    competitive_low=math.ceil(count * 0.05),
                    stagnation_positive=max(10, math.ceil(count * 0.01)),
                    stagnation_zero=math.ceil(count * 0.10), ordinary=1,
                    competition=1, structural=1)
    for dimension in SEMANTIC_DIMENSIONS_V3[:3]:
        required[f"{dimension}_high"] = math.ceil(count * 0.05)
        required[f"{dimension}_low"] = math.ceil(count * 0.10)
    missing["train"].update({
        key: max(0, target - coverage["train"].get(key, 0))
        for key, target in required.items()
    })
    return missing


def _fixed_subset(rows, size, salt):
    """Deterministic stratum round robin, selected before any label requests."""
    groups = defaultdict(list)
    ordered = sorted(rows, key=lambda row: canonical_sha256([salt, row["view_sha256"]]))
    for row in ordered:
        groups[row["stratum"]].append(row)
    result = []
    while groups and len(result) < size:
        for stratum in sorted(list(groups)):
            result.append(groups[stratum].pop())
            if not groups[stratum]:
                del groups[stratum]
            if len(result) == size:
                break
    records = [dict(episode_id=row["episode_id"], agent_id=row["agent_id"],
                    start_timestep=row["end_timestep"] - 7,
                    end_timestep=row["end_timestep"], view_sha256=row["view_sha256"],
                    stratum=row["stratum"]) for row in result]
    for record in records:
        record["view_id"] = _window_id(dict(partition="train", **record))
    return records


def _label_preflight(lists, train_rows):
    """Necessary raw observability checks; never an LLM quality assessment."""
    flags = {row["view_sha256"]: row["flags"] for row in train_rows}
    paired_counts, trial_rates, missing = {}, {}, {}
    for dimension in SEMANTIC_DIMENSIONS_V3[:3]:
        key = f"{dimension}_known"
        count = sum(bool(flags[row["view_sha256"]].get(key))
                    for row in lists["repeat"])
        paired_counts[dimension] = count
        missing[f"paired_{dimension}_known"] = max(0, 20 - count)
        trial_rates[dimension] = (
            sum(bool(flags[row["view_sha256"]].get(key)) for row in lists["trial"])
            / len(lists["trial"]) if lists["trial"] else 0.0
        )
    review_counts = {}
    for stratum in ("ordinary", "competition", "structural", "charging_service"):
        count = sum(bool(flags[row["view_sha256"]].get(stratum))
                    for row in lists["review"])
        review_counts[stratum] = count
        missing[f"review_{stratum}"] = max(0, 1 - count)
    return dict(passed=not any(missing.values()), coverage_missing=missing,
                paired_known_counts=paired_counts, trial_known_rates=trial_rates,
                review_stratum_counts=review_counts,
                semantic_quality_passed=False)


def select_goal_full_windows(run_directory, *, budgets=None):  # noqa: C901
    """Freeze source-bound windows; raw coverage is only a necessary quality gate."""
    directory = Path(run_directory)
    identity = _read(directory / "collection_identity.json")
    if identity.get("schema") != COLLECTION_V3:
        raise ValueError("Selection requires the v3 collection identity.")
    if identity.get("implementation_sha256") != _implementation_hashes():
        raise ValueError("Selection implementation differs from collection launch.")
    budgets = dict(MEDIUM_BUDGETS if budgets is None else budgets)
    if (set(budgets) != set(PARTITIONS)
            or any(type(v) is not int or v < 1 for v in budgets.values())
            or sum(budgets.values()) > 8000):
        raise ValueError("Selection budgets must be positive and bounded at 8000.")
    receipts = _receipts(directory, identity)
    frozen_path = directory / "selection_frozen.json"
    if frozen_path.exists():
        frozen = _read(frozen_path)
        expected = {
            "windows_sha256": "windows.jsonl",
            "manifest_sha256": "manifest.json",
            "selection_manifest_sha256": "selection_manifest.json",
            "label_lists_sha256": "label_selection_lists.json",
            "episode_partitions_sha256": "episode_partitions.json",
        }
        if (frozen.get("budgets") != budgets
                or frozen.get("episode_receipts_sha256") != canonical_sha256(receipts)
                or any(not (directory / name).is_file()
                       or frozen.get(key) != file_sha256(directory / name)
                       for key, name in expected.items())):
            raise ValueError("Frozen selection budgets or persisted hashes differ.")
        manifest = _read(directory / "manifest.json")
        eligible = manifest["main_pretraining_eligible"] is True
        _atomic_json(directory / "state.json", dict(
            status="completed" if eligible else "insufficient_coverage",
            main_pretraining_eligible=eligible,
            coverage_missing=manifest["coverage_missing"],
        ))
        return dict(run_directory=str(directory.resolve()), manifest=manifest)
    _reject_labels(directory)
    counts = _partition_counts(len(receipts),
                               dict(train=0.70, validation=0.15, test=0.15))
    ranked = sorted(receipts,
                    key=lambda episode: (canonical_sha256([SALT, episode]), episode))
    assignments, start = {}, 0
    for partition in PARTITIONS:
        for episode in ranked[start:start + counts[partition]]:
            assignments[episode] = partition
        start += counts[partition]
    connection = sqlite3.connect(directory / "candidate_index.sqlite")
    try:
        _index_candidates(connection, directory, receipts, assignments)
        selected, coverage = _choose_candidates(connection, budgets)
    finally:
        connection.close()
    missing = _coverage_missing(coverage, budgets, selected)
    _atomic_json(directory / "episode_partitions.json", {
        partition: sorted(e for e, p in assignments.items() if p == partition)
        for partition in PARTITIONS
    })

    @lru_cache(maxsize=2)
    def arrays(episode):
        path = directory / "episodes" / f"seed_{receipts[episode]['seed']:05d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            return payload["observations"], payload["reset_masks"]

    windows = directory / "windows.jsonl"
    temporary = windows.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for partition in PARTITIONS:
            for row in sorted(selected[partition], key=lambda r: (
                r["episode_id"], r["agent_id"], r["end_timestep"],
            )):
                observations, resets = arrays(row["episode_id"])
                agent, end = row["agent_id"] - 1, row["end_timestep"]
                window = dict(partition=partition, episode_id=row["episode_id"],
                              agent_id=agent + 1, start_timestep=end - 7,
                              end_timestep=end, stratum=row["stratum"],
                              observations=observations[agent, end - 7:end + 1].tolist(),
                              reset_masks=resets[agent, end - 7:end + 1].tolist())
                encoded = json.dumps(window, sort_keys=True, separators=(",", ":"))
                stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, windows)
    lists = dict(trial=_fixed_subset(selected["train"], 120, "trial-v3"))
    trial_keys = {(r["episode_id"], r["agent_id"], r["end_timestep"])
                  for r in lists["trial"]}
    trial_rows = [r for r in selected["train"]
                  if (r["episode_id"], r["agent_id"], r["end_timestep"]) in trial_keys]
    lists["repeat"] = _fixed_subset(trial_rows, 40, "repeat-v3")
    lists["pro"] = list(lists["repeat"])
    lists["review"] = _fixed_subset(trial_rows, 60, "review-v3")
    preflight = _label_preflight(lists, selected["train"])
    missing["fixed_lists"] = preflight["coverage_missing"]
    list_complete = all(len(lists[key]) == target for key, target in (
        ("trial", 120), ("repeat", 40), ("pro", 40), ("review", 60),
    ))
    _atomic_json(directory / "label_selection_lists.json", lists)
    _atomic_json(directory / "selection_manifest.json", dict(
        selected=selected, coverage=coverage, coverage_missing=missing,
        label_lists_sha256=file_sha256(directory / "label_selection_lists.json"),
        actual_label_list_counts={key: len(value) for key, value in lists.items()},
        label_lists_complete=list_complete,
        raw_label_preflight=preflight,
    ))
    raw_passed = not any(any(v > 0 for v in row.values()) for row in missing.values())
    eligible = (raw_passed and list_complete and budgets == MEDIUM_BUDGETS
                and len(receipts)
                == identity["collection_contract"]["initial_episode_target"]
                and not identity["synthetic_execution"]
                and identity["source_actor_contract"].get("verified_source") is True)
    manifest = dict(
        schema=DATASET_V3, distribution=DISTRIBUTION_V3,
        semantic_contract_version=SEMANTIC_CONTRACT_V3,
        semantic_dimensions=list(SEMANTIC_DIMENSIONS_V3),
        semantic_sources=["llm", "llm", "llm", "observable-rule-v3"],
        config_sha256=canonical_sha256(identity),
        collection_identity_sha256=canonical_sha256(identity),
        environment_sha256=(identity["rollout_environment_contract"]
                            ["environment_sha256"]),
        source_actor_contract=identity["source_actor_contract"],
        rollout_environment_contract=identity["rollout_environment_contract"],
        episode_count=len(receipts), episode_split_counts=counts,
        trajectory_count=len(receipts) * identity["collection_contract"]["n_agents"],
        window_counts={p: len(selected[p]) for p in PARTITIONS},
        windows_sha256=file_sha256(windows),
        selection_manifest_sha256=file_sha256(directory / "selection_manifest.json"),
        coverage=coverage, coverage_missing=missing,
        raw_candidate_coverage_passed=raw_passed,
        main_pretraining_eligible=eligible, pretraining_eligible=eligible,
        label_quality_passed=False, formal_performance_evidence=False,
        raw_label_preflight=preflight,
    )
    _atomic_json(directory / "manifest.json", manifest)
    if eligible:
        _atomic_json(directory / "selection_frozen.json", dict(
            budgets=budgets,
            environment="medium-5ag",
            total_samples=sum(manifest["window_counts"].values()),
            frozen_before_labeling=True,
            trial_ids=[row["view_id"] for row in lists["trial"]],
            repeat_ids=[row["view_id"] for row in lists["repeat"]],
            pro_ids=[row["view_id"] for row in lists["pro"]],
            review_ids=[row["view_id"] for row in lists["review"]],
            windows_sha256=manifest["windows_sha256"],
            selection_manifest_sha256=manifest["selection_manifest_sha256"],
            manifest_sha256=file_sha256(directory / "manifest.json"),
            label_lists_sha256=file_sha256(directory / "label_selection_lists.json"),
            episode_partitions_sha256=file_sha256(directory / "episode_partitions.json"),
            episode_receipts_sha256=canonical_sha256(receipts),
        ))
    state = dict(status="completed" if eligible else "insufficient_coverage",
                 main_pretraining_eligible=eligible, coverage_missing=missing)
    _atomic_json(directory / "state.json", state)
    with (directory / "selection_attempts.jsonl").open("a", encoding="utf-8") as stream:
        attempt = dict(windows_sha256=manifest["windows_sha256"], **state)
        stream.write(json.dumps(attempt) + "\n")
    return dict(run_directory=str(directory.resolve()), manifest=manifest)
