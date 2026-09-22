"""Versioned identities and one-shot held-out diagnostic access."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping, TypeVar
import weakref

from .config import ResolvedConfig, load_approved_m1_config


DIAGNOSTIC_RUN_V1 = "representation-diagnostic-run-v1"
TEST_ACCESS_LEDGER_V1 = "representation-test-access-ledger-v1"
TEST_ACCESS_RECEIPT_V1 = "representation-test-access-receipt-v1"
FRESH_PROBE_TRAINING_RECEIPT_V1 = "fresh-probe-training-receipt-v1"
REPRESENTATION_VARIANTS = (
    "normal_history",
    "shuffled_label",
    "shuffled_history",
    "single_frame",
    "random_init",
)

_RUN_FIELDS = {
    "schema_version", "metric_definition_version", "m1_config_schema_version",
    "m1_config_sha256", "dataset_sha256", "representation_variants",
    "representation_seeds", "episode_partitions", "empty_ledger_anchor_sha256",
}
_PARTITIONS = ("train", "validation", "test")
_LEDGER_FIELDS = {"schema_version", "run_identity", "receipts"}
_RECEIPT_FIELDS = {
    "schema_version", "representation_variant", "representation_seed",
    "model_selection_frozen", "thresholds_frozen", "sequence",
    "previous_entry_hash", "entry_hash",
}
_FRESH_PROBE_TRAINING_RECEIPT_FIELDS = {
    "schema_version", "diagnostic_run_sha256", "m1_config_sha256",
    "dataset_sha256", "representation_variant", "representation_seed",
    "receipt_sha256",
}
_FRESH_PROBE_RECEIPT_FACTORY_TOKEN = object()
_ISSUED_FRESH_PROBE_RECEIPTS = weakref.WeakSet()
_Result = TypeVar("_Result")


class FreshProbeTrainingReceipt(Mapping[str, object]):
    """Immutable runtime capability issued after full run-identity validation."""

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(self, payload: Mapping[str, object], *, _factory_token) -> None:
        if _factory_token is not _FRESH_PROBE_RECEIPT_FACTORY_TOKEN:
            raise TypeError(
                "Fresh probe training receipts must be issued by their factory."
            )
        self._payload = copy.deepcopy(dict(payload))
        self._factory_token = _factory_token

    def __getitem__(self, key: str) -> object:
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)


def create_diagnostic_run_identity(
    *, m1_config_path: str | Path, m1_approval_path: str | Path,
    dataset_sha256: str,
    train_episode_ids: tuple[str, ...],
    validation_episode_ids: tuple[str, ...],
    test_episode_ids: tuple[str, ...],
    representation_seeds: tuple[int, ...],
) -> dict[str, object]:
    """Create the complete immutable input identity for one diagnostic run."""
    m1_config = _load_approved_m1(m1_config_path, m1_approval_path)
    identity: dict[str, object] = {
        "schema_version": DIAGNOSTIC_RUN_V1,
        "metric_definition_version": m1_config.values.get(
            "metric_definition_version"
        ),
        "m1_config_schema_version": m1_config.values.get("schema_version"),
        "m1_config_sha256": m1_config.sha256,
        "dataset_sha256": dataset_sha256,
        "representation_variants": list(REPRESENTATION_VARIANTS),
        "representation_seeds": list(representation_seeds),
        "episode_partitions": {
            "train": list(train_episode_ids),
            "validation": list(validation_episode_ids),
            "test": list(test_episode_ids),
        },
    }
    identity["empty_ledger_anchor_sha256"] = _empty_ledger_anchor(identity)
    _validate_diagnostic_run_identity(identity, m1_config)
    return identity


def validate_diagnostic_run_identity(
    identity: Mapping[str, object], m1_config_path: str | Path,
    m1_approval_path: str | Path,
) -> None:
    """Reject incomplete or altered diagnostic run identities."""
    m1_config = _load_approved_m1(m1_config_path, m1_approval_path)
    _validate_diagnostic_run_identity(identity, m1_config)


def create_fresh_probe_training_receipt(
    *, diagnostic_identity: Mapping[str, object],
    representation_variant: str,
    representation_seed: int,
    m1_config_path: str | Path,
    m1_approval_path: str | Path,
) -> FreshProbeTrainingReceipt:
    """Derive a test-metadata-free receipt for fixture probe training."""

    m1_config = _load_approved_m1(m1_config_path, m1_approval_path)
    identity = copy.deepcopy(dict(diagnostic_identity))
    _validate_diagnostic_run_identity(identity, m1_config)
    if representation_variant not in REPRESENTATION_VARIANTS:
        raise ValueError("Fresh probe receipt has an unknown representation variant.")
    if (
        type(representation_seed) is not int
        or representation_seed not in identity["representation_seeds"]
    ):
        raise ValueError("Fresh probe receipt has an unknown representation seed.")
    receipt: dict[str, object] = {
        "schema_version": FRESH_PROBE_TRAINING_RECEIPT_V1,
        "diagnostic_run_sha256": _diagnostic_run_identity_sha256(identity),
        "m1_config_sha256": m1_config.sha256,
        "dataset_sha256": identity["dataset_sha256"],
        "representation_variant": representation_variant,
        "representation_seed": representation_seed,
    }
    receipt["receipt_sha256"] = _fresh_probe_receipt_sha256(receipt)
    issued = FreshProbeTrainingReceipt(
        receipt, _factory_token=_FRESH_PROBE_RECEIPT_FACTORY_TOKEN
    )
    _ISSUED_FRESH_PROBE_RECEIPTS.add(issued)
    return issued


def validate_fresh_probe_training_receipt(
    receipt: FreshProbeTrainingReceipt,
    m1_config_path: str | Path,
    m1_approval_path: str | Path,
) -> dict[str, object]:
    """Reject a changed or M1-incompatible compact probe-training receipt."""

    m1_config = _load_approved_m1(m1_config_path, m1_approval_path)
    if (
        not isinstance(receipt, FreshProbeTrainingReceipt)
        or receipt._factory_token is not _FRESH_PROBE_RECEIPT_FACTORY_TOKEN
        or receipt not in _ISSUED_FRESH_PROBE_RECEIPTS
    ):
        raise ValueError("Fresh probe receipt was not issued by its factory.")
    result = copy.deepcopy(dict(receipt))
    if set(result) != _FRESH_PROBE_TRAINING_RECEIPT_FIELDS:
        raise ValueError("Fresh probe receipt fields must be exact.")
    if result["schema_version"] != FRESH_PROBE_TRAINING_RECEIPT_V1:
        raise ValueError("Fresh probe receipt schema is unknown.")
    _validate_sha256(result["diagnostic_run_sha256"], "diagnostic_run_sha256")
    _validate_sha256(result["m1_config_sha256"], "m1_config_sha256")
    _validate_sha256(result["dataset_sha256"], "dataset_sha256")
    _validate_sha256(result["receipt_sha256"], "receipt_sha256")
    if result["m1_config_sha256"] != m1_config.sha256:
        raise ValueError("Fresh probe receipt does not match the approved M1.")
    if (
        type(result["representation_variant"]) is not str
        or result["representation_variant"] not in REPRESENTATION_VARIANTS
    ):
        raise ValueError("Fresh probe receipt has an unknown representation variant.")
    probe = m1_config.values["probe"]
    if (
        type(result["representation_seed"]) is not int
        or result["representation_seed"] not in probe["seeds"]
    ):
        raise ValueError("Fresh probe receipt has an unapproved representation seed.")
    if result["receipt_sha256"] != _fresh_probe_receipt_sha256(result):
        raise ValueError("Fresh probe receipt hash is invalid.")
    return result


def _validate_diagnostic_run_identity(
    identity: Mapping[str, object], m1_config: ResolvedConfig
) -> None:
    _validate_resolved_m1_hash(m1_config)
    if not isinstance(identity, Mapping) or set(identity) != _RUN_FIELDS:
        raise ValueError("Diagnostic run identity fields must be exact.")
    expected_m1 = {
        "schema_version": DIAGNOSTIC_RUN_V1,
        "metric_definition_version": m1_config.values.get(
            "metric_definition_version"
        ),
        "m1_config_schema_version": m1_config.values.get("schema_version"),
        "m1_config_sha256": m1_config.sha256,
    }
    if any(identity[name] != value for name, value in expected_m1.items()):
        raise ValueError("Diagnostic run identity does not match the M1 config.")
    _validate_sha256(identity["dataset_sha256"], "dataset_sha256")
    supplied_anchor = identity["empty_ledger_anchor_sha256"]
    _validate_sha256(supplied_anchor, "empty_ledger_anchor_sha256")
    without_anchor = dict(identity)
    del without_anchor["empty_ledger_anchor_sha256"]
    if not hmac.compare_digest(supplied_anchor, _empty_ledger_anchor(without_anchor)):
        raise ValueError("Diagnostic run empty-ledger anchor was altered.")
    if identity["representation_variants"] != list(REPRESENTATION_VARIANTS):
        raise ValueError("Diagnostic run requires the exact representation variants.")
    probe = m1_config.values.get("probe")
    expected_seeds = probe.get("seeds") if isinstance(probe, Mapping) else None
    seeds = identity["representation_seeds"]
    if seeds != expected_seeds:
        raise ValueError("Representation seeds must match the exact M1 probe seeds.")
    if not isinstance(seeds, list) or not all(type(seed) is int for seed in seeds):
        raise ValueError("Representation seeds must be an integer list.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Representation seeds must be unique integers.")
    _validate_episode_partitions(identity["episode_partitions"])


def _validate_episode_partitions(partitions: object) -> None:
    if not isinstance(partitions, Mapping) or set(partitions) != set(_PARTITIONS):
        raise ValueError("Episode partition identity fields must be exact.")
    seen: set[str] = set()
    for name in _PARTITIONS:
        episode_ids = partitions[name]
        if not isinstance(episode_ids, list) or not episode_ids:
            raise ValueError("Every episode partition must be a non-empty list.")
        if any(type(item) is not str or not item for item in episode_ids):
            raise ValueError("Episode identities must be non-empty strings.")
        duplicate = len(set(episode_ids)) != len(episode_ids)
        if duplicate or seen.intersection(episode_ids):
            raise ValueError(
                "Episode partitions contain duplicate or overlap identities."
            )
        seen.update(episode_ids)


@dataclass(frozen=True)
class TestAccessGrant:
    """A receipt plus the new full-file digest the caller must pin."""
    ledger_sha256: str
    receipt: dict[str, object]


@dataclass(frozen=True)
class TestEvaluationResult:
    """The metric result and the access grant that preceded computation."""
    metrics: object
    ledger_sha256: str
    receipt: dict[str, object]


class TestAccessLedger:
    """One-shot access for cooperating processes using an external SHA anchor.

    The contract detects history rewrites only while the caller retains the
    previously returned digest. The exclusive lock coordinates cooperating
    writers; it is not a defense against an arbitrary filesystem attacker.
    File fsync is used everywhere, and directory fsync is best-effort on
    platforms that support opening directories (not Windows).
    """
    __test__ = False

    def __init__(
        self, path: str | Path, run_identity: Mapping[str, object],
        m1_config_path: str | Path, m1_approval_path: str | Path,
    ) -> None:
        m1_config = _load_approved_m1(m1_config_path, m1_approval_path)
        _validate_diagnostic_run_identity(run_identity, m1_config)
        self.path = Path(path)
        self._identity = copy.deepcopy(dict(run_identity))
        self._m1_config = ResolvedConfig(
            values=copy.deepcopy(m1_config.values), sha256=m1_config.sha256
        )

    def request_test_access(
        self, *, representation_variant: str, representation_seed: int,
        model_selection_frozen: bool, thresholds_frozen: bool,
        expected_ledger_sha256: str,
    ) -> TestAccessGrant:
        """Atomically append one receipt after verifying the caller's anchor."""
        self._validate_request(
            representation_variant, representation_seed,
            model_selection_frozen, thresholds_frozen,
        )
        _validate_sha256(expected_ledger_sha256, "expected_ledger_sha256")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ValueError("Test-access ledger is already being updated.") from exc
        os.close(descriptor)
        try:
            payload, initial = self._load_snapshot(expected_ledger_sha256)
            pair = (representation_variant, representation_seed)
            if any(_receipt_pair(item) == pair for item in payload["receipts"]):
                raise ValueError(
                    "This representation seed has already been accessed on test."
                )
            receipt = self._new_receipt(
                payload["receipts"], representation_variant, representation_seed
            )
            payload["receipts"].append(receipt)
            encoded = _canonical_json_bytes(payload)
            self._publish(encoded, initial=initial)
            return TestAccessGrant(
                ledger_sha256=sha256(encoded).hexdigest(),
                receipt=copy.deepcopy(receipt),
            )
        finally:
            lock_path.unlink(missing_ok=True)

    def evaluate_once(
        self, *, representation_variant: str, representation_seed: int,
        model_selection_frozen: bool, thresholds_frozen: bool,
        expected_ledger_sha256: str,
        metrics_callback: Callable[[], _Result],
    ) -> TestEvaluationResult:
        """Reserve test access before invoking metric computation."""
        if not callable(metrics_callback):
            raise TypeError("metrics_callback must be callable.")
        grant = self.request_test_access(
            representation_variant=representation_variant,
            representation_seed=representation_seed,
            model_selection_frozen=model_selection_frozen,
            thresholds_frozen=thresholds_frozen,
            expected_ledger_sha256=expected_ledger_sha256,
        )
        metrics = metrics_callback()
        return TestEvaluationResult(
            metrics=metrics,
            ledger_sha256=grant.ledger_sha256,
            receipt=grant.receipt,
        )

    def _validate_request(self, variant, seed, model_frozen, thresholds_frozen):
        if model_frozen is not True:
            raise ValueError("Diagnostic model selection must be frozen.")
        if thresholds_frozen is not True:
            raise ValueError("Diagnostic thresholds must be frozen.")
        if variant not in REPRESENTATION_VARIANTS:
            raise ValueError("Unknown representation variant.")
        if seed not in self._identity["representation_seeds"]:
            raise ValueError("Unknown representation seed.")

    def _load_snapshot(self, expected_sha: str) -> tuple[dict[str, object], bool]:
        if not self.path.exists():
            anchor = self._identity["empty_ledger_anchor_sha256"]
            if not hmac.compare_digest(expected_sha, anchor):
                raise ValueError(
                    "Expected pinned ledger SHA does not match empty anchor."
                )
            return self._empty_payload(), True
        try:
            encoded = self.path.read_bytes()
        except OSError as exc:
            raise ValueError("Test-access ledger could not be read.") from exc
        actual_sha = sha256(encoded).hexdigest()
        if not hmac.compare_digest(expected_sha, actual_sha):
            raise ValueError("Expected pinned ledger SHA does not match ledger bytes.")
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("Test-access ledger is not valid JSON.") from exc
        self._validate_payload(payload)
        return payload, False

    def _empty_payload(self) -> dict[str, object]:
        return {
            "schema_version": TEST_ACCESS_LEDGER_V1,
            "run_identity": copy.deepcopy(self._identity),
            "receipts": [],
        }

    def _validate_payload(self, payload: object) -> None:
        if not isinstance(payload, dict) or set(payload) != _LEDGER_FIELDS:
            raise ValueError("Test-access ledger fields must be exact.")
        if payload["schema_version"] != TEST_ACCESS_LEDGER_V1:
            raise ValueError("Unknown test-access ledger schema.")
        _validate_diagnostic_run_identity(payload["run_identity"], self._m1_config)
        if payload["run_identity"] != self._identity:
            raise ValueError("Test-access ledger run identity was altered.")
        receipts = payload["receipts"]
        if not isinstance(receipts, list):
            raise ValueError("Test-access receipts must be a list.")
        previous = self._identity["empty_ledger_anchor_sha256"]
        seen = set()
        for sequence, receipt in enumerate(receipts, start=1):
            self._validate_receipt(receipt, sequence, previous)
            pair = _receipt_pair(receipt)
            if pair in seen:
                raise ValueError("Test-access ledger contains duplicate receipts.")
            seen.add(pair)
            previous = receipt["entry_hash"]

    def _new_receipt(self, receipts, variant, seed) -> dict[str, object]:
        previous = (
            receipts[-1]["entry_hash"] if receipts
            else self._identity["empty_ledger_anchor_sha256"]
        )
        receipt = {
            "schema_version": TEST_ACCESS_RECEIPT_V1,
            "sequence": len(receipts) + 1,
            "representation_variant": variant,
            "representation_seed": seed,
            "model_selection_frozen": True,
            "thresholds_frozen": True,
            "previous_entry_hash": previous,
        }
        receipt["entry_hash"] = _receipt_hash(receipt)
        return receipt

    def _validate_receipt(self, receipt, sequence, previous) -> None:
        if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
            raise ValueError("Test-access receipt fields must be exact.")
        if receipt["schema_version"] != TEST_ACCESS_RECEIPT_V1:
            raise ValueError("Unknown test-access receipt schema.")
        if receipt["sequence"] != sequence:
            raise ValueError("Test-access receipt sequence is not contiguous.")
        if receipt["previous_entry_hash"] != previous:
            raise ValueError("Test-access receipt hash chain is broken.")
        if receipt["entry_hash"] != _receipt_hash(receipt):
            raise ValueError("Test-access receipt entry hash is invalid.")
        if (receipt["model_selection_frozen"] is not True
                or receipt["thresholds_frozen"] is not True):
            raise ValueError("Test-access receipt does not prove both freezes.")
        if receipt["representation_variant"] not in REPRESENTATION_VARIANTS:
            raise ValueError("Test-access receipt has an unknown variant.")
        if receipt["representation_seed"] not in self._identity[
            "representation_seeds"
        ]:
            raise ValueError("Test-access receipt has an unknown seed.")

    def _publish(self, encoded: bytes, *, initial: bool) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if initial:
                try:
                    os.link(temporary, self.path)
                except FileExistsError as exc:
                    raise ValueError(
                        "Initial ledger destination appeared during publication."
                    ) from exc
            else:
                os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _empty_ledger_anchor(identity_without_anchor: Mapping[str, object]) -> str:
    payload = {
        "schema_version": TEST_ACCESS_LEDGER_V1,
        "run_identity_without_empty_anchor": identity_without_anchor,
        "receipts": [],
    }
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _diagnostic_run_identity_sha256(identity: Mapping[str, object]) -> str:
    return sha256(_canonical_json_bytes(identity)).hexdigest()


def _fresh_probe_receipt_sha256(receipt: Mapping[str, object]) -> str:
    hashed = {
        name: value for name, value in receipt.items() if name != "receipt_sha256"
    }
    return sha256(_canonical_json_bytes(hashed)).hexdigest()


def _receipt_hash(receipt: Mapping[str, object]) -> str:
    hashed = {name: value for name, value in receipt.items() if name != "entry_hash"}
    return sha256(_canonical_json_bytes(hashed)).hexdigest()


def _receipt_pair(receipt: Mapping[str, object]) -> tuple[object, object]:
    return receipt["representation_variant"], receipt["representation_seed"]


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_sha256(value: object, name: str) -> None:
    if (type(value) is not str or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name} must be a lowercase SHA-256 identity.")


def _validate_resolved_m1_hash(m1_config: ResolvedConfig) -> None:
    if not isinstance(m1_config, ResolvedConfig):
        raise ValueError("M1 config must be a resolved configuration.")
    encoded = json.dumps(
        m1_config.values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if sha256(encoded).hexdigest() != m1_config.sha256:
        raise ValueError("M1 config hash does not match its resolved values.")


def _load_approved_m1(
    config_path: str | Path, approval_path: str | Path
) -> ResolvedConfig:
    """Return only an M1 mapping verified against its owner approval receipt."""
    return load_approved_m1_config(config_path, approval_path)
