"""Immutable O1 evidence shards, failure classification, and O2 receipts."""

import errno
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Mapping, Optional

from llm_mappo.linux_server_runtime import write_new_atomic_json


@dataclass(frozen=True)
class RunIdentity:
    """The immutable fields that must agree before any shard is reused."""

    code_commit: str
    config_sha256: str
    immutable_machine_sha256: str
    environment_sha256: str

    def to_dict(self) -> Mapping[str, str]:
        return asdict(self)


class ExternalGpuInterferenceError(RuntimeError):
    """A monitored external PID appeared after the owner began an O1 run."""


def _identity_payload(identity: RunIdentity) -> Mapping[str, str]:
    return identity.to_dict()


def write_shard(
    path: Path, payload: Mapping[str, object], identity: RunIdentity
) -> None:
    """Create one immutable O1 shard with its complete identity."""
    if "schema" not in payload:
        raise ValueError("shard payload must declare a schema")
    write_new_atomic_json(
        path,
        {**payload, "run_identity": _identity_payload(identity)},
    )


def write_new_atomic_text(path: Path, text: str) -> None:
    """Atomically create an immutable text artifact without overwriting evidence."""
    if path.exists():
        raise FileExistsError("immutable artifact already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError("immutable artifact already exists: " + str(path))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_valid_shard(
    path: Path, identity: RunIdentity, schema: str
) -> Optional[Mapping[str, object]]:
    """Return a matching completed shard; reject any existing mismatch or corruption."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("existing shard is corrupt") from error
    if payload.get("run_identity") != _identity_payload(identity):
        raise ValueError("existing shard identity does not match")
    if payload.get("schema") != schema:
        raise ValueError("existing shard schema does not match")
    return payload


def classify_failure(error: BaseException) -> Literal["infrastructure", "algorithm"]:
    """Classify only the frozen recovery allowlist as infrastructure failures."""
    if isinstance(error, (KeyboardInterrupt, ExternalGpuInterferenceError)):
        return "infrastructure"
    if isinstance(error, SystemExit) and error.code in {130, 143}:
        return "infrastructure"
    if isinstance(error, OSError) and error.errno in {
        errno.EIO,
        errno.ENOSPC,
        errno.EDQUOT,
        errno.ESTALE,
    }:
        return "infrastructure"
    return "algorithm"


def write_o1_gate_receipt(
    summary: Mapping[str, object], identity: RunIdentity, path: Path
) -> None:
    """Write the O2-only receipt after both O1 normal-Gate decisions are true."""
    if summary.get("gate_pass") is not True:
        raise ValueError("Gate did not pass; an O2 receipt cannot be written")
    if summary.get("runtime_gate_pass") is not True:
        raise ValueError("Gate did not pass the runtime decision")
    if summary.get("memory_gate_pass") is not True:
        raise ValueError("Gate did not pass the memory decision")
    summary_json = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    write_new_atomic_json(
        path,
        {
            "schema": "o1-gate-receipt-v1",
            "run_identity": _identity_payload(identity),
            "summary_sha256": sha256(summary_json.encode("utf-8")).hexdigest(),
            "next_required_phase": "O2",
        },
    )


def verify_o1_gate_receipt(
    path: Path, expected_identity: RunIdentity
) -> Mapping[str, object]:
    """Verify an immutable O1 Go receipt before a future O2 entry point consumes it."""
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("O1 receipt is unreadable") from error
    if receipt.get("schema") != "o1-gate-receipt-v1":
        raise ValueError("O1 receipt schema does not match")
    if receipt.get("run_identity") != _identity_payload(expected_identity):
        raise ValueError("O1 receipt identity does not match")
    if receipt.get("next_required_phase") != "O2":
        raise ValueError("O1 receipt does not authorize the required next phase")
    return receipt
