"""Strict native hard-charge lineage admission shared by labeling and pretraining."""

import json
from pathlib import Path

from .medium_extension_dataset import (
    LINEAGE_SCHEMA, verify_medium_extension_dataset,
)
from .source_actor_v3 import canonical_sha256


SOURCE_CHECKPOINT_SHA256 = (
    "f112cb62dc77d4b7d6ccf66b0b90b2d088e8f6a191bcccd905b2dcb0a825ed91"
)
SOURCE_COMMIT = "3798ec6b593f4c92f982685ba133e7acdb184a5f"


def verify_native_derived_dataset(dataset, manifest, *, artifact_root=None):
    """Verify only the approved multi-pool medium selection and its raw source."""
    dataset = Path(dataset)
    if manifest.get("lineage_schema") != LINEAGE_SCHEMA:
        raise ValueError("The native derived lineage schema is incompatible.")
    identity_path = dataset / "collection_identity.json"
    if not identity_path.is_file():
        raise ValueError("The native derived collection identity is missing.")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    source = identity.get("source_actor_contract", {})
    actor = source.get("actor_identity", {})
    rollout = identity.get("rollout_environment_contract", {})
    environment = rollout.get("environment", {})
    if (identity.get("schema") != "medium-train-extension-dataset-v1"
            or identity.get("synthetic_execution") is not False
            or source.get("verified_source") is not True
            or source.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or source.get("source_code_commit") != SOURCE_COMMIT
            or source.get("source_environment_sha256")
            != rollout.get("environment_sha256")
            or actor.get("schema") != "three-scene-hard-charge-v2"
            or actor.get("method") != "MAPPO-FP-DG"
            or actor.get("seed") != 7
            or actor.get("n_agents") != 5
            or actor.get("profile_id") != "medium-5ag-hard-charge-v2"
            or actor.get("charging_queue_policy") != "battery-priority-yield-v1"
            or rollout.get("schema") != "three-scene-hard-charge-v2"
            or environment.get("profile_id") != "medium-5ag-hard-charge-v2"
            or environment.get("scene_id") != "medium-5ag"
            or environment.get("n_agents") != 5
            or environment.get("max_steps") != 800
            or environment.get("queue_size") != 5
            or environment.get("charging_rate") != 0.1
            or environment.get("charge_threshold") != 0.3
            or environment.get("charge_release_threshold") != 1.0
            or environment.get("hard_charge_until_full") is not True
            or manifest.get("collection_identity_sha256") != canonical_sha256(identity)
            or manifest.get("source_actor_contract") != source
            or manifest.get("rollout_environment_contract") != rollout
            or manifest.get("environment_sha256") != rollout.get("environment_sha256")):
        raise ValueError("Native hard-charge source or derived identity differs.")
    frozen = verify_medium_extension_dataset(
        dataset, require_frozen=True, artifact_root=artifact_root,
    )
    if (not isinstance(frozen, dict)
            or frozen.get("lineage_sha256") != canonical_sha256(identity)
            or frozen.get("environment") != "medium-5ag"
            or frozen.get("total_samples") != 3000):
        raise ValueError("Native derived freeze differs from source identity.")
    return frozen
