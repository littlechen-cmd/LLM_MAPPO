"""Narrow representation initialization for final MAPPO-family Actors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from llm_mappo.mappo_baseline.networks import MAPPOActor

from .pretraining import normalize_pretraining_identity
from .representation_export import import_actor_representation


REPRESENTATION_INITIALIZATION_V1 = "mappo-representation-initialization-v1"


@dataclass(frozen=True)
class RepresentationInitialization:
    """Public identity and authenticated path for one Encoder+GRU artifact."""

    artifact_path: str
    artifact_sha256: str
    identity: Mapping[str, object]


def load_representation_initialization(
    values: Mapping[str, object],
) -> RepresentationInitialization:
    """Reject partial, legacy, or unauthenticated initialization settings."""

    expected = {"schema", "artifact_path", "artifact_sha256", "identity"}
    if not isinstance(values, Mapping) or set(values) != expected:
        raise ValueError("Representation initialization fields must be exact.")
    if values["schema"] != REPRESENTATION_INITIALIZATION_V1:
        raise ValueError("Representation initialization schema is incompatible.")
    artifact_path = values["artifact_path"]
    artifact_sha256 = values["artifact_sha256"]
    if not isinstance(artifact_path, str) or not artifact_path:
        raise ValueError("Representation initialization path is invalid.")
    if not _is_sha256(artifact_sha256):
        raise ValueError("Representation initialization SHA-256 is invalid.")
    identity = (
        dict(values["identity"])
        if isinstance(values["identity"], Mapping)
        else None
    )
    if identity is None:
        raise ValueError("Representation initialization identity is invalid.")
    identity = normalize_pretraining_identity(identity)
    return RepresentationInitialization(artifact_path, artifact_sha256, identity)


def initialize_final_actor(
    actor: MAPPOActor, initialization: RepresentationInitialization
) -> dict[str, object]:
    """Apply only authenticated Encoder+GRU weights to an already-built Actor."""

    if not isinstance(initialization, RepresentationInitialization):
        raise ValueError("Representation initialization contract is invalid.")
    return import_actor_representation(
        Path(initialization.artifact_path),
        actor,
        expected_identity=initialization.identity,
        expected_artifact_sha256=initialization.artifact_sha256,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
