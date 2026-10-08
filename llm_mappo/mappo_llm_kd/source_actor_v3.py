"""Strict historical Actor provenance, independent of the rollout contract."""

from hashlib import sha256
import json
from pathlib import Path

from llm_mappo.mappo_baseline.runner import load_baseline_experiment
from llm_mappo.medium5_contract import CORE_MEDIUM5_C1_SCHEMA
from .collector import load_medium5_collection_policy


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                  .encode("utf-8")).hexdigest()


def load_pinned_medium_source(
    *, source_config, source_checkpoint, actor_artifact, actor_sha256,
    source_manifest=None,
):
    """Load only a completed final seed-7 old C1 source, retaining original hashes."""
    config_path, checkpoint = Path(source_config), Path(source_checkpoint)
    manifest_path = Path(source_manifest or checkpoint.parent / "run_manifest.json")
    state_path = manifest_path.parent / "state.json"
    experiment = load_baseline_experiment(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        experiment.schema != CORE_MEDIUM5_C1_SCHEMA
        or experiment.method != "MAPPO-FP-DG"
        or experiment.approval_status != "approved"
        or experiment.training["seed"] != 7
        or checkpoint.name != "checkpoint_final.pt"
        or state.get("status") != "complete"
        or manifest.get("environment") != dict(experiment.environment)
        or manifest.get("training") != dict(experiment.training)
    ):
        raise ValueError("Source requires a completed final medium C1 seed-7 run.")
    identity = manifest.get("identity", {})
    expected = {
        "schema": experiment.schema, "method": experiment.method,
        "config_sha256": experiment.config_sha256, "seed": 7,
        "reward_definition_sha256": experiment.reward_definition_sha256,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Source manifest identity differs from source config.")
    commit = identity.get("code_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("Source manifest has no recorded code identity.")
    expected.update(
        code_commit=commit,
        environment_sha256=canonical_sha256(dict(experiment.environment)),
        source_checkpoint_sha256=file_sha256(checkpoint),
    )
    policy = load_medium5_collection_policy(
        actor_artifact, device="cpu", expected_artifact_sha256=actor_sha256,
        expected_identity=expected,
    )
    return policy, {
        "verified_source": True,
        "source_config_path": str(config_path.resolve()),
        "source_config_sha256": experiment.config_sha256,
        "source_environment_sha256": expected["environment_sha256"],
        "source_checkpoint_path": str(checkpoint.resolve()),
        "source_checkpoint_sha256": expected["source_checkpoint_sha256"],
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(manifest_path),
        "source_state_sha256": file_sha256(state_path),
        "actor_path": str(Path(actor_artifact).resolve()),
        "actor_sha256": actor_sha256, "actor_identity": expected,
        "source_code_commit": commit,
    }
