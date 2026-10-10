"""Strict Encoder + GRU-only representation artifacts for Phase 3."""

from __future__ import annotations

import copy
import hmac
import io
import json
import os
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor

from llm_mappo.mappo_baseline.networks import MAPPOActor

from .pretraining import (
    HIDDEN_DIM,
    OBSERVATION_DIM,
    PRETRAINING_EXPORT_V1,
    validate_pretraining_identity,
    normalize_pretraining_identity,
)


_TOP_LEVEL_KEYS = (
    "export_schema",
    "identity",
    "payload_content_sha256",
    "actor_representation",
)
_REPRESENTATION_PREFIXES = ("encoder.", "gru.")


def export_actor_representation(
    path: str | Path,
    actor: MAPPOActor,
    *,
    identity: Mapping[str, object],
) -> dict:
    """Atomically export only the exact Actor Encoder and GRU tensors."""

    _validate_actor(actor)
    identity_copy = normalize_pretraining_identity(copy.deepcopy(dict(identity)))
    representation = _representation_state(actor)
    payload = {
        "export_schema": PRETRAINING_EXPORT_V1,
        "identity": identity_copy,
        "payload_content_sha256": _payload_content_digest(
            identity_copy, representation
        ),
        "actor_representation": representation,
    }
    _validate_payload(payload, actor, expected_identity=identity_copy)
    artifact_buffer = io.BytesIO()
    torch.save(payload, artifact_buffer)
    artifact_bytes = artifact_buffer.getvalue()
    file_sha256 = sha256(artifact_bytes).hexdigest()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(artifact_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "identity": copy.deepcopy(identity_copy),
        "payload_content_sha256": payload["payload_content_sha256"],
        "file_sha256": file_sha256,
    }


def import_actor_representation(  # noqa: C901
    path: str | Path,
    actor: MAPPOActor,
    *,
    expected_identity: Mapping[str, object],
    expected_artifact_sha256: str,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Authenticate file bytes, validate, dry-load, then apply representation."""

    _validate_actor(actor)
    expected_copy = normalize_pretraining_identity(
        copy.deepcopy(dict(expected_identity))
    )
    if not _is_sha256(expected_artifact_sha256):
        raise ValueError(
            "expected_artifact_sha256 must be a lowercase SHA-256 digest."
        )
    artifact_path = Path(path)
    try:
        artifact_bytes = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError("Representation artifact bytes could not be read.") from error
    file_sha256 = sha256(artifact_bytes).hexdigest()
    if not hmac.compare_digest(file_sha256, expected_artifact_sha256):
        raise ValueError("Representation artifact file digest does not match.")
    try:
        payload = torch.load(
            io.BytesIO(artifact_bytes),
            map_location=map_location,
            weights_only=True,
        )
    except Exception as error:
        raise ValueError(
            "Representation artifact could not be safely loaded."
        ) from error
    _validate_payload(payload, actor, expected_identity=expected_copy)

    candidate = _merged_actor_state(actor, payload["actor_representation"])
    dry_actor = copy.deepcopy(actor)
    try:
        dry_actor.load_state_dict(candidate, strict=True)
    except Exception as error:
        raise ValueError("Representation artifact failed dry-load.") from error

    before = copy.deepcopy(actor.state_dict())
    try:
        actor.load_state_dict(candidate, strict=True)
    except Exception as error:
        actor.load_state_dict(before, strict=True)
        raise ValueError(
            "Representation artifact could not be atomically applied."
        ) from error
    return {
        "identity": copy.deepcopy(payload["identity"]),
        "payload_content_sha256": payload["payload_content_sha256"],
        "file_sha256": file_sha256,
    }


def _validate_actor(actor: object) -> None:
    if not isinstance(actor, MAPPOActor):
        raise ValueError("Representation import/export requires a MAPPOActor.")
    if actor.observation_dim != OBSERVATION_DIM or actor.hidden_dim != HIDDEN_DIM:
        raise ValueError("Representation import/export requires the 209D/64D Actor.")


def _representation_state(actor: MAPPOActor) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in actor.state_dict().items()
        if key.startswith(_REPRESENTATION_PREFIXES)
    }


def _validate_payload(payload, actor, *, expected_identity):  # noqa: C901
    if not isinstance(payload, Mapping) or tuple(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("Representation artifact top-level keys must be exact.")
    if payload["export_schema"] != PRETRAINING_EXPORT_V1:
        raise ValueError("Representation export schema is incompatible.")
    validate_pretraining_identity(payload["identity"])
    if not _strict_equal(
        normalize_pretraining_identity(payload["identity"]), expected_identity
    ):
        raise ValueError("Representation identity does not match expected_identity.")

    expected = _representation_state(actor)
    supplied = payload["actor_representation"]
    if not isinstance(supplied, Mapping) or tuple(supplied) != tuple(expected):
        raise ValueError("Actor representation tensor keys and order must be exact.")
    for key, tensor in supplied.items():
        reference = expected[key]
        if not isinstance(tensor, Tensor):
            raise ValueError(f"actor_representation.{key} must be a tensor.")
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(
                f"actor_representation.{key} shape and dtype must be exact."
            )
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(
                f"actor_representation.{key} must contain only finite values."
            )

    digest = payload["payload_content_sha256"]
    if not _is_sha256(digest):
        raise ValueError(
            "payload_content_sha256 must be a lowercase SHA-256 digest."
        )
    if digest != _payload_content_digest(payload["identity"], supplied):
        raise ValueError("Representation payload digest does not match its content.")


def _payload_content_digest(identity, representation) -> str:
    digest = sha256()
    encoded_identity = json.dumps(
        identity,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(encoded_identity).to_bytes(8, "big"))
    digest.update(encoded_identity)
    for key, tensor in representation.items():
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("ascii")
        shape_bytes = json.dumps(list(tensor.shape), separators=(",", ":")).encode(
            "ascii"
        )
        value_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
        for part in (key_bytes, dtype_bytes, shape_bytes, value_bytes):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _merged_actor_state(actor, representation):
    state = copy.deepcopy(actor.state_dict())
    state.update({key: value.detach().clone() for key, value in representation.items()})
    return state


def _strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return tuple(left) == tuple(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
