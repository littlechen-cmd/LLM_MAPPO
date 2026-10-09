"""Collect native hard-charge trajectories using the source's immutable code."""

import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


SOURCE_COMMIT = "3798ec6b593f4c92f982685ba133e7acdb184a5f"
CONFIG_SHA = "6e127306367adcd3204f4e8696ef0efdb2ec1504c36164bc106dcc7ce47999e4"
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                  .encode("utf-8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_scalar(value):
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unexpected non-JSON value: {type(value).__name__}")


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False,
                  default=json_scalar)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def extract_snapshot(content, directory):
    directory = Path(directory).resolve()
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        for member in archive.getmembers():
            target = (directory / member.name).resolve()
            if (not target.is_relative_to(directory)
                    or not (member.isdir() or member.isfile())):
                raise ValueError("Unsafe source snapshot member.")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())


def snapshot_identity(snapshot):
    return {p.relative_to(snapshot).as_posix(): digest(p)
            for p in sorted(snapshot.rglob("*")) if p.is_file()
            and "__pycache__" not in p.parts}


def verify_receipt(path, receipt, identity_sha, seed):
    if (receipt.get("schema") != "goal-full-complete-episode-v3"
            or receipt.get("seed") != seed
            or receipt.get("collection_identity_sha256") != identity_sha
            or receipt.get("npz_sha256") != digest(path)
            or receipt.get("boundary") not in {
                "terminated", "truncated", "deadlocked"}):
        raise ValueError("Complete episode receipt differs from pinned arrays.")
    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        rows, reset = payload["observations"], payload["reset_masks"]
        if (rows.ndim != 3 or rows.shape[0] != 5 or rows.shape[2] != 209
                or not 1 <= rows.shape[1] <= 800
                or rows.dtype != np.float32 or not np.isfinite(rows).all()
                or reset.dtype != np.bool_ or reset.shape != rows.shape[:2]
                or not reset[:, 0].all() or reset[:, 1:].any()
                or receipt.get("steps") != rows.shape[1]
                or receipt.get("episode_id") != f"collection-seed-{seed}-episode-0"):
            raise ValueError("Episode receipt shape or reset alignment differs.")


def inspect_source(snapshot, source_config, source_run, expected_features):  # noqa: C901
    """Only called in a subprocess whose import root is the pinned snapshot."""
    import numpy as np
    import numpy._core.multiarray as multiarray
    import torch
    from llm_mappo.mappo_baseline.runner import checkpoint_schema_for_experiment
    from scripts.run_medium5_hard_charge_formal_batch import load_experiment
    from llm_mappo.mappo_baseline.networks import MAPPOActor
    from llm_mappo.mappo_baseline.structured_observation import (
        DIRECT_GOAL_OBSERVATION_V2, StructuredObservationSchema,
    )

    for name in ("llm_mappo.mappo_baseline.runner",
                 "llm_mappo.mappo_baseline.structured_observation",
                 "llm_mappo.mappo_baseline.networks",
                 "scripts.run_medium5_hard_charge_formal_batch"):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(snapshot):
            raise ValueError("Worker imported code outside immutable runtime snapshot.")

    native_config = (snapshot / "configs/hard_charge_v2/formal_multibudget"
                     / "seed_0007/medium-5ag-baseline.yaml")
    if digest(native_config) != digest(source_config):
        raise ValueError("Original config bytes differ from pinned code configuration.")
    experiment = load_experiment(native_config, "baseline", 7, root=snapshot)
    manifest = read(source_run / "run_manifest.json")
    checkpoint = source_run / "checkpoint_final.pt"
    state = read(source_run / "state.json")
    if (digest(source_config) != CONFIG_SHA
            or experiment.schema != "three-scene-hard-charge-v2"
            or experiment.method != "MAPPO-FP-DG"
            or experiment.approval_status != "approved"
            or manifest["environment"] != dict(experiment.environment)
            or manifest["training"] != dict(experiment.training)
            or state.get("status") != "complete"
            or experiment.training["seed"] != 7
            or experiment.training["environment_steps"] != 900000):
        raise ValueError("New source is not the pinned completed hard-charge run.")
    values = experiment.environment
    if (values["n_agents"] != 5 or values["max_steps"] != 800
            or values["profile_id"] != "medium-5ag-hard-charge-v2"
            or values.get("hard_charge_until_full") is not True
            or values["charging_rate"] != 0.1
            or values["charge_release_threshold"] != 1.0):
        raise ValueError("Native hard-charge environment contract differs.")
    features = list(StructuredObservationSchema(
        observation_schema=DIRECT_GOAL_OBSERVATION_V2).feature_names)
    if features != expected_features:
        raise ValueError("Snapshot 209D feature order differs from v3 serializer.")
    allowed = {
        "numpy._core.multiarray._reconstruct", "numpy.core.multiarray._reconstruct",
        "numpy._core.multiarray.scalar", "numpy.core.multiarray.scalar",
        "numpy.dtype", "numpy.ndarray", "builtins.bytes",
    }
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))
    if not unsafe.issubset(allowed):
        raise ValueError("Checkpoint contains unapproved pickle globals.")
    safe = [(multiarray._reconstruct, "numpy._core.multiarray._reconstruct"),
            (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            (multiarray.scalar, "numpy._core.multiarray.scalar"),
            (multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype, np.ndarray, bytes,
            *[type(np.dtype(name)) for name in (
                "float32", "float64", "int32", "int64", "uint32", "uint64",
                "bool", "str", "object")]]
    with torch.serialization.safe_globals(safe):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = payload.get("identity", {})
    if (payload.get("schema") != checkpoint_schema_for_experiment(experiment)
            or identity != manifest["identity"]
            or identity.get("code_commit") != SOURCE_COMMIT
            or identity.get("config_sha256") != CONFIG_SHA
            or identity.get("schema") != experiment.schema
            or identity.get("method") != experiment.method
            or identity.get("seed") != 7
            or payload.get("global_environment_steps") != 900000):
        raise ValueError("Final checkpoint identity differs from source manifest.")
    config = payload["trainer"]["config"]
    if {k: config.get(k) for k in (
            "observation_dim", "action_dim", "hidden_dim")} != {
            "observation_dim": 209, "action_dim": 5, "hidden_dim": 64}:
        raise ValueError("Checkpoint Actor dimensions differ.")
    actor = MAPPOActor(209, 5, 64)
    actor.load_state_dict(payload["trainer"]["actor"], strict=True)
    if not all(torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("Actor contains non-finite parameters.")
    actor.eval()
    source = dict(
        verified_source=True, source_config_path=str(source_config),
        source_config_sha256=CONFIG_SHA,
        source_environment_sha256=canonical(dict(values)),
        source_checkpoint_path=str(checkpoint),
        source_checkpoint_sha256=digest(checkpoint),
        source_manifest_path=str(source_run / "run_manifest.json"),
        source_manifest_sha256=digest(source_run / "run_manifest.json"),
        source_state_sha256=digest(source_run / "state.json"),
        source_code_commit=SOURCE_COMMIT, actor_identity=identity,
    )
    return experiment, actor, source


def check_charging_holds(before, after, actions, masks):
    """Inspect natural transitions; never place an agent or change its actions."""
    violations, full_releases = [], 0
    for agent_id, prior in before.items():
        agent = after.env.agents[agent_id - 1]
        position, direction, battery = prior
        released = agent_id not in after._charging_holds
        expected = min(1.0, battery + 0.1)
        if ((agent.x, agent.y) != position or agent.dir != direction
                or int(actions[agent_id - 1]) != 0
                or masks[agent_id - 1].tolist() != [True, False, False, False, False]
                or abs(float(agent.battery) - expected) > 1e-9
                or released != (float(agent.battery) >= 1.0)):
            violations.append(dict(
                agent_id=int(agent_id), before_position=list(position),
                after_position=[int(agent.x), int(agent.y)],
                before_battery=battery, after_battery=float(agent.battery),
                requested_action=int(actions[agent_id - 1]), released=released))
        elif released:
            full_releases += 1
    return violations, full_releases


def collect_native(directory, experiment, actor, episode_count):  # noqa: C901
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
    schema = StructuredObservationSchema(observation_schema=DIRECT_GOAL_OBSERVATION_V2)
    policy = FrozenMAPPOPolicy(actor, identity["source_actor_contract"]
                               ["source_checkpoint_sha256"], "cpu")
    torch.set_num_threads(1)
    for number, seed in enumerate(range(10000, 10000 + episode_count), 1):
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
                verify_receipt(candidate, completion, identity_sha, seed)
                if candidate == temporary:
                    os.replace(temporary, path)
                atomic(receipt_path, completion)
            if receipt_path.exists():
                verify_receipt(path, read(receipt_path), identity_sha, seed)
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


def run(args):
    sys.path.insert(0, str(ROOT))
    from llm_mappo.mappo_llm_kd.goal_full_dataset_run import (
        COLLECTION_V3, MEDIUM_BUDGETS, SALT, SCHEMA, _implementation_hashes,
    )

    if not 1 <= args.episode_count <= 120:
        raise ValueError("Only the fixed 120-episode prefix is authorized.")
    runtime_name = SOURCE_COMMIT + "-collection-lf"
    snapshot = (ROOT / "artifacts/llm_kd/runtime_snapshots" / runtime_name).resolve()
    stamp = snapshot.parent / f"{runtime_name}.json"
    if stamp.exists():
        if read(stamp) != snapshot_identity(snapshot):
            raise ValueError("Immutable runtime snapshot has changed.")
    else:
        if snapshot.exists():
            raise ValueError("Partial runtime snapshot exists without receipt.")
        content = subprocess.check_output(
            ["git", "-c", "core.autocrlf=false", "archive", SOURCE_COMMIT,
             "llm_mappo", "rware", "scripts",
             "configs/hard_charge_v2"], cwd=ROOT)
        snapshot.mkdir(parents=True)
        extract_snapshot(content, snapshot)
        atomic(stamp, snapshot_identity(snapshot))
    request = dict(snapshot=str(snapshot),
                   source_config=str(Path(args.source_config).resolve()),
                   source_run=str(Path(args.source_run).resolve()),
                   feature_names=list(SCHEMA.feature_names))
    input_path = snapshot.parent / f"{SOURCE_COMMIT}-inspection-input.json"
    atomic(input_path, request)
    command = [sys.executable, str(Path(__file__).resolve()), "_worker",
               "--input", str(input_path)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    inspected = json.loads(subprocess.check_output(
        command, cwd=snapshot, env=env, text=True))
    source = inspected["source"]
    identity = dict(
        schema=COLLECTION_V3, source_actor_contract=source,
        rollout_environment_contract=inspected["rollout"],
        collection_contract=dict(
            seed_start=10000, initial_episode_target=120, maximum_episodes=120,
            max_steps=800, n_agents=5, history_length=8,
            split_ratios=dict(train=.70, validation=.15, test=.15),
            split_salt=SALT, window_budgets=MEDIUM_BUDGETS,
            behavior_policy_id=f"MAPPO-FP-DG:{source['source_checkpoint_sha256']}",
            sampling="seeded_stochastic_original_frozen_policy",
            learner_inference_threads=1),
        implementation_sha256=_implementation_hashes(), synthetic_execution=False,
        native_runtime=dict(commit=SOURCE_COMMIT, snapshot_sha256=canonical(read(stamp)),
                            collector_sha256=digest(__file__),
                            feature_names_sha256=canonical(request["feature_names"])),
    )
    directory = (Path(args.output_root).resolve()
                 / f"hard-charge-v3-{canonical(identity)[:16]}")
    if args.inspect_only:
        print(json.dumps(dict(run_directory=str(directory), source=source,
                              rollout=inspected["rollout"], created=False)), flush=True)
        return
    if args.resume:
        if read(directory / "collection_identity.json") != identity:
            raise ValueError("Native collection resume identity differs.")
        if ((directory / "selection_frozen.json").exists()
                or ((directory / "labels").exists()
                    and any((directory / "labels").iterdir()))):
            raise ValueError("Frozen or labeled collection cannot extend.")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "episodes").mkdir()
        atomic(directory / "collection_identity.json", identity)
        atomic(directory / "launch_metadata.json", dict(
            native_runtime=str(snapshot), source_commit=SOURCE_COMMIT,
            source_config=request["source_config"],
            current_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    atomic(directory / "worker_input.json", dict(
        **request, directory=str(directory), episode_count=args.episode_count,
        identity_sha256=canonical(identity)))
    print(json.dumps(dict(run_directory=str(directory))), flush=True)
    subprocess.run(command[:-1] + [str(directory / "worker_input.json")],
                   cwd=snapshot, env=env, check=True)


def worker(input_path):
    request = read(input_path)
    snapshot = Path(request["snapshot"])
    sys.path.insert(0, str(snapshot))
    experiment, actor, source = inspect_source(
        snapshot, Path(request["source_config"]), Path(request["source_run"]),
        request["feature_names"])
    if "directory" in request:
        directory = Path(request["directory"])
        identity = read(directory / "collection_identity.json")
        if (canonical(identity) != request["identity_sha256"]
                or source != identity["source_actor_contract"]):
            raise ValueError("Worker source identity differs from collection launch.")
        collect_native(directory, experiment, actor, request["episode_count"])
    else:
        reward_files = {name: digest(snapshot / name) for name in (
            "llm_mappo/reward_v2.py", "llm_mappo/warehouse_environment.py",
            "llm_mappo/environment.py", "rware/warehouse.py")}
        print(json.dumps(dict(source=source, rollout=dict(
            schema=experiment.schema, config_sha256=CONFIG_SHA,
            environment=dict(experiment.environment),
            environment_sha256=canonical(dict(experiment.environment)),
            reward_definition_sha256=canonical(dict(
                native_reward_code=reward_files,
                environment=dict(experiment.environment))),
            reward_definition_sha256_source="native-code-and-environment-derived",
            native_reward_code_sha256=reward_files))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        parser.add_argument("_worker")
        parser.add_argument("--input", required=True)
        worker(parser.parse_args().input)
    else:
        parser.add_argument("--source-run", required=True)
        parser.add_argument("--source-config", required=True)
        parser.add_argument("--output-root",
                            default="artifacts/llm_kd/phase_20261007/dataset")
        parser.add_argument("--episode-count", type=int, default=120)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--inspect-only", action="store_true")
        run(parser.parse_args())
