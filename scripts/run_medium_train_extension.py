"""Collect approved train-only batches without changing the original raw pool."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE_ID = "3fb225003be8bc9d45e525fe5d72d2775e43d67cbd40289329c6c216d9bf85bb"
HELPER_PATH = ROOT / "scripts/run_hard_charge_dataset_snapshot.py"
_spec = spec_from_file_location("frozen_native_collection_helpers", HELPER_PATH)
helpers = module_from_spec(_spec)
_spec.loader.exec_module(helpers)
atomic, read, canonical = helpers.atomic, helpers.read, helpers.canonical
digest, verify_receipt = helpers.digest, helpers.verify_receipt
check_charging_holds = helpers.check_charging_holds


def batch_seed_start(batch_index):
    if type(batch_index) is not int or not 1 <= batch_index <= 6:
        raise ValueError("Extension batch index must be an integer in 1..6.")
    return 10120 + (batch_index - 1) * 40


def check_requested_prefix(directory, episode_count):
    directory = Path(directory)
    identity = read(directory / "collection_identity.json")
    identity_sha = canonical(identity)
    start = batch_seed_start(identity["collection_contract"]["batch_index"])
    found = set()
    for path in (directory / "episodes").glob("seed_*"):
        match = re.fullmatch(r"seed_(\d{5})\.(json|intent\.json|npz|npz\.tmp)",
                             path.name)
        if not match:
            raise ValueError("Unexpected extension episode filename.")
        seed = int(match.group(1))
        if not start <= seed < start + episode_count:
            raise ValueError("Requested prefix is smaller than existing episode seed.")
        if match.group(2) == "json":
            receipt = read(path)
            verify_extension_receipt(path.with_suffix(".npz"), receipt,
                                     identity_sha, seed)
            found.add(seed)
        elif match.group(2) == "intent.json":
            intent = read(path)
            if (intent.get("seed") != seed
                    or intent.get("collection_identity_sha256") != identity_sha):
                raise ValueError("Existing extension intent identity differs.")
    for seed in range(start, max(found, default=start - 1) + 1):
        if seed in found:
            continue
        path = directory / "episodes" / f"seed_{seed:05d}.npz"
        intent_path = path.with_suffix(".intent.json")
        intent = read(intent_path) if intent_path.exists() else {}
        completion = intent.get("completion")
        candidate = path if path.exists() else path.with_suffix(".npz.tmp")
        if not completion or not candidate.is_file():
            raise ValueError("Sparse prefix lacks a recoverable complete intent.")
        verify_extension_receipt(candidate, completion, identity_sha, seed)


def verify_extension_receipt(path, receipt, identity_sha, seed):
    verify_receipt(path, receipt, identity_sha, seed)
    if receipt.get("charging_validation", {}).get("violations") != 0:
        raise ValueError("Extension receipt hard-charge validation has failed.")


def check_parent_reference(reference):
    directory = Path(reference["run_directory"])
    if "episode_receipts_sha256" not in reference:
        raise ValueError("Immutable parent reference lacks complete receipt binding.")
    for name, expected in reference["files_sha256"].items():
        path = directory / name
        if (not path.resolve().is_relative_to(directory.resolve())
                or not path.is_file() or digest(path) != expected):
            raise ValueError("Immutable parent pool file differs from reference.")
    if "episode_receipts_sha256" in reference:
        identity_sha = reference["collection_identity_sha256"]
        expected_seeds = set(range(10000, 10120))
        expected_arrays = {f"seed_{seed:05d}.npz" for seed in expected_seeds}
        if {p.name for p in (directory / "episodes").glob("seed_*.npz")} != (
                expected_arrays):
            raise ValueError("Immutable parent episode array seed set differs.")
        receipt_paths = [p for p in (directory / "episodes").glob("seed_*.json")
                         if not p.name.endswith(".intent.json")]
        receipts = {}
        actual_seeds = set()
        for path in receipt_paths:
            match = re.fullmatch(r"seed_(\d{5})\.json", path.name)
            if not match:
                raise ValueError("Immutable parent episode filename differs.")
            seed = int(match.group(1))
            receipt = read(path)
            verify_receipt(path.with_suffix(".npz"), receipt, identity_sha, seed)
            if receipt.get("charging_validation", {}).get("violations") != 0:
                raise ValueError("Immutable parent hard-charge validation differs.")
            actual_seeds.add(seed)
            receipts[receipt["episode_id"]] = receipt
        if (actual_seeds != expected_seeds or len(receipt_paths) != 120
                or canonical(receipts) != reference["episode_receipts_sha256"]):
            raise ValueError("Immutable parent episode receipts differ from reference.")


def parent_reference(directory):
    directory = Path(directory).resolve()
    identity = read(directory / "collection_identity.json")
    if (canonical(identity) != BASE_ID
            or identity.get("synthetic_execution") is not False):
        raise ValueError("Only the approved original native pool may be extended.")
    for name, expected in identity["implementation_sha256"].items():
        if digest(ROOT / "llm_mappo" / name) != expected:
            raise ValueError("Original parent implementation bytes have changed.")
    if digest(HELPER_PATH) != identity["native_runtime"]["collector_sha256"]:
        raise ValueError("Original frozen native collector has changed.")
    state = read(directory / "collection_state.json")
    if state.get("status") != "completed" or state.get("completed_count") != 120:
        raise ValueError("Parent requires its original 120 complete episodes.")
    receipts = {}
    for seed in range(10000, 10120):
        path = directory / "episodes" / f"seed_{seed:05d}.npz"
        receipt = read(path.with_suffix(".json"))
        verify_receipt(path, receipt, BASE_ID, seed)
        if receipt.get("charging_validation", {}).get("violations") != 0:
            raise ValueError("Parent hard-charge receipt has not passed validation.")
        receipts[receipt["episode_id"]] = receipt
    names = ("collection_identity.json", "collection_state.json", "manifest.json",
             "selection_manifest.json", "episode_partitions.json", "windows.jsonl",
             "label_selection_lists.json", "candidate_index.sqlite")
    reference = dict(
        run_directory=str(directory), collection_identity_sha256=BASE_ID,
        collection_identity_file_sha256=digest(directory / "collection_identity.json"),
        episode_receipts_sha256=canonical(receipts),
        candidate_index_sha256=digest(directory / "candidate_index.sqlite"),
        episode_partitions_sha256=digest(directory / "episode_partitions.json"),
        windows_sha256=digest(directory / "windows.jsonl"),
        selection_manifest_sha256=digest(directory / "selection_manifest.json"),
        manifest_sha256=digest(directory / "manifest.json"),
        files_sha256={name: digest(directory / name) for name in names},
    )
    check_parent_reference(reference)
    return identity, reference


def build_identity(parent, reference, batch_index):
    start = batch_seed_start(batch_index)
    return dict(
        schema="goal-full-dataset-collection-v3",
        source_actor_contract=parent["source_actor_contract"],
        rollout_environment_contract=parent["rollout_environment_contract"],
        native_runtime=parent["native_runtime"],
        implementation_sha256=parent["implementation_sha256"],
        extension_implementation_sha256=dict(
            entrypoint=digest(__file__), frozen_native_helpers=digest(HELPER_PATH)),
        synthetic_execution=False, parent_pool=reference,
        collection_contract=dict(
            partition="train", batch_index=batch_index, seed_start=start,
            seed_end=start + 39, initial_episode_target=40, maximum_episodes=40,
            extension_maximum_episodes=240, extension_seed_bounds=[10120, 10359],
            n_agents=5, max_steps=800, history_length=8,
            behavior_policy_id=parent["collection_contract"]["behavior_policy_id"],
            sampling="seeded_stochastic_original_frozen_policy",
            learner_inference_threads=1,
            partition_policy="preserve-original-84-18-18-and-add-train-only",
        ),
    )


# Episode loop copied from the frozen native collector; only seed bounds differ.
def collect_extension_native(  # noqa: C901
    directory, experiment, actor, seed_start, episode_count,
):
    import numpy as np
    import torch
    from llm_mappo.mappo_baseline.vector_env import new_baseline_environment
    from llm_mappo.mappo_baseline.structured_observation import (
        DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
        build_structured_observations,
    )
    from llm_mappo.mappo_llm_kd.collector import FrozenMAPPOPolicy

    identity = read(directory / "collection_identity.json")
    identity_sha = canonical(identity)
    contract = identity["collection_contract"]
    if (seed_start != batch_seed_start(contract["batch_index"])
            or seed_start != contract["seed_start"]
            or type(episode_count) is not int or not 1 <= episode_count <= 40):
        raise ValueError("Extension native loop seed or batch count differs.")
    check_requested_prefix(directory, episode_count)
    schema = StructuredObservationSchema(observation_schema=DIRECT_GOAL_OBSERVATION_V2)
    policy = FrozenMAPPOPolicy(actor, identity["source_actor_contract"]
                               ["source_checkpoint_sha256"], "cpu")
    torch.set_num_threads(1)
    for number, seed in enumerate(range(seed_start, seed_start + episode_count), 1):
        path = directory / "episodes" / f"seed_{seed:05d}.npz"
        receipt_path = path.with_suffix(".json")
        intent_path = path.with_suffix(".intent.json")
        temporary = path.with_suffix(".npz.tmp")
        try:
            pending = read(intent_path) if intent_path.exists() else {}
            if pending and (
                    pending.get("seed") != seed
                    or pending.get("collection_identity_sha256") != identity_sha):
                raise ValueError("Pending episode intent identity differs.")
            completion = pending.get("completion")
            candidate = path if path.exists() else temporary
            if not receipt_path.exists() and completion and candidate.exists():
                verify_extension_receipt(candidate, completion, identity_sha, seed)
                if candidate == temporary:
                    os.replace(temporary, path)
                atomic(receipt_path, completion)
            if receipt_path.exists():
                verify_extension_receipt(path, read(receipt_path), identity_sha, seed)
                continue
            if path.exists() or temporary.exists():
                raise ValueError("Unclaimed episode array must not be overwritten.")
            atomic(intent_path, dict(seed=seed,
                                     collection_identity_sha256=identity_sha))
            environment = new_baseline_environment(experiment.environment)
            rows = np.empty((5, 800, 209), dtype=np.float32)
            reset = np.zeros((5, 800), dtype=np.bool_)
            reset[:, 0] = True
            hidden = np.zeros((1, 5, 64), dtype=np.float32)
            lock_steps, full_releases = 0, 0
            policy.start_episode(seed)
            try:
                environment.reset(seed=seed)
                for step in range(800):
                    observation = np.asarray(build_structured_observations(
                        environment, schema, goal_interface="direct_goal", planner=None),
                        dtype=np.float32)
                    mask = np.asarray(environment.action_masks(), dtype=np.bool_)
                    if (observation.shape != (5, 209)
                            or not np.isfinite(observation).all()
                            or mask.shape != (5, 5) or not mask.any(axis=-1).all()):
                        raise ValueError("Native observation or mask is invalid.")
                    rows[:, step] = observation
                    held = {
                        agent.id: ((agent.x, agent.y), agent.dir, float(agent.battery))
                        for agent in environment.env.agents
                        if agent.id in environment._charging_holds
                    }
                    actions, hidden = policy.act(observation, mask, hidden)
                    if not mask[np.arange(5), actions].all():
                        raise ValueError("Native policy emitted forbidden action.")
                    transition = environment.step(actions)
                    violations, releases = check_charging_holds(
                        held, environment, actions, mask)
                    lock_steps += len(held)
                    full_releases += releases
                    if violations:
                        atomic(directory / f"charging_violation_seed_{seed}.json",
                               dict(seed=seed, step=step, violations=violations))
                        raise ValueError("Natural hard-charge transition is invalid.")
                    boundary = ("terminated" if transition.terminated else
                                "truncated" if transition.truncated else
                                "deadlocked" if transition.metrics.deadlocked else None)
                    if boundary:
                        break
                else:
                    raise RuntimeError("No actual episode boundary within 800 steps.")
            finally:
                environment.close()
            steps = step + 1
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, observations=rows[:, :steps],
                                    reset_masks=reset[:, :steps])
                stream.flush()
                os.fsync(stream.fileno())
            receipt = dict(
                schema="goal-full-complete-episode-v3", seed=seed,
                episode_id=f"collection-seed-{seed}-episode-0", steps=steps,
                boundary=boundary, npz_sha256=digest(temporary),
                collection_identity_sha256=identity_sha,
                metrics=transition.metrics.as_dict(),
                charging_validation=dict(lock_steps=lock_steps,
                                         full_releases=full_releases, violations=0),
            )
            atomic(intent_path, dict(seed=seed, completion=receipt,
                                     collection_identity_sha256=identity_sha))
            os.replace(temporary, path)
            atomic(receipt_path, receipt)
            atomic(directory / "collection_state.json", dict(
                status="running", completed_count=number,
                requested_count=episode_count))
            print(json.dumps(dict(seed=seed, steps=steps, boundary=boundary)),
                  flush=True)
        except Exception as error:
            failure = dict(seed=seed, error_type=type(error).__name__,
                           collection_identity_sha256=identity_sha)
            failure_path = directory / "failures.jsonl"
            with failure_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure) + "\n")
            atomic(directory / "collection_state.json", dict(status="failed", **failure))
            raise
    atomic(directory / "collection_state.json", dict(
        status="completed", completed_count=episode_count,
        requested_count=episode_count))
def worker(input_path):
    request = read(input_path)
    directory = Path(request["directory"])
    identity = read(directory / "collection_identity.json")
    if canonical(identity) != request["identity_sha256"]:
        raise ValueError("Extension worker identity differs from launch.")
    check_parent_reference(identity["parent_pool"])
    if identity["extension_implementation_sha256"] != dict(
            entrypoint=digest(__file__), frozen_native_helpers=digest(HELPER_PATH)):
        raise ValueError("Extension worker implementation differs from launch.")
    snapshot = Path(request["snapshot"])
    sys.path.insert(0, str(snapshot))
    experiment, actor, source = helpers.inspect_source(
        snapshot, Path(request["source_config"]), Path(request["source_run"]),
        request["feature_names"])
    if source != identity["source_actor_contract"]:
        raise ValueError("Extension source differs from immutable parent Actor.")
    seed_start = batch_seed_start(identity["collection_contract"]["batch_index"])
    if (identity["collection_contract"]["seed_start"] != seed_start
            or not 1 <= request["episode_count"] <= 40):
        raise ValueError("Extension worker seed contract differs.")
    collect_extension_native(directory, experiment, actor, seed_start,
                             request["episode_count"])
    check_parent_reference(identity["parent_pool"])


def run(args):
    batch_seed_start(args.batch_index)
    if type(args.episode_count) is not int or not 1 <= args.episode_count <= 40:
        raise ValueError("Only the fixed 40-episode batch prefix is allowed.")
    parent, reference = parent_reference(args.parent_pool)
    identity = build_identity(parent, reference, args.batch_index)
    snapshot = (ROOT / "artifacts/llm_kd/runtime_snapshots"
                / (helpers.SOURCE_COMMIT + "-collection-lf"))
    stamp = snapshot.parent / (snapshot.name + ".json")
    snapshot_hashes = read(stamp)
    expected_snapshot = parent["native_runtime"]["snapshot_sha256"]
    if (helpers.snapshot_identity(snapshot) != snapshot_hashes
            or canonical(snapshot_hashes) != expected_snapshot):
        raise ValueError("Immutable native runtime snapshot differs from parent.")
    inspection = read(Path(args.parent_pool) / "worker_input.json")
    request = {key: inspection[key] for key in (
        "snapshot", "source_config", "source_run", "feature_names")}
    if Path(request["snapshot"]).resolve() != snapshot.resolve():
        raise ValueError("Parent worker runtime path differs from pinned snapshot.")
    if canonical(request["feature_names"]) != parent["native_runtime"][
            "feature_names_sha256"]:
        raise ValueError("Parent feature order differs from frozen semantic interface.")
    leaf = f"train-extension-batch-{args.batch_index:02d}-{canonical(identity)[:16]}"
    directory = Path(args.output_root).resolve() / leaf
    if args.inspect_only:
        print(json.dumps(dict(run_directory=str(directory), identity=identity,
                              created=False), sort_keys=True, indent=2))
        return
    if args.resume:
        if read(directory / "collection_identity.json") != identity:
            raise ValueError("Extension resume identity differs.")
        if (directory / "selection_frozen.json").exists():
            raise ValueError("Frozen extension cannot append raw episodes.")
        check_requested_prefix(directory, args.episode_count)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "episodes").mkdir()
        atomic(directory / "collection_identity.json", identity)
    request.update(directory=str(directory), episode_count=args.episode_count,
                   identity_sha256=canonical(identity))
    atomic(directory / "worker_input.json", request)
    print(json.dumps(dict(run_directory=str(directory)), sort_keys=True), flush=True)
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_worker",
         "--input", str(directory / "worker_input.json")], cwd=snapshot,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        parser.add_argument("_worker")
        parser.add_argument("--input", required=True)
        worker(parser.parse_args().input)
    else:
        parser.add_argument("--parent-pool", required=True)
        parser.add_argument("--batch-index", required=True, type=int)
        parser.add_argument("--episode-count", type=int, default=40)
        parser.add_argument("--output-root",
                            default="artifacts/llm_kd/phase_20261007/train-extension")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--inspect-only", action="store_true")
        run(parser.parse_args())
