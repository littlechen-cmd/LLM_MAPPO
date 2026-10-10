"""Strict response parsing, atomic cache, resumability, and failure receipts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping

from .prompt import build_label_prompt
from .provider import (
    OpenAICompatibleProvider,
    ProviderCallError,
    contains_credential,
)


LABEL_RECORD_V1 = "llm-semantic-label-record-v1"
LABEL_FAILURE_V1 = "llm-semantic-label-failure-v1"
_DIMENSIONS = (
    "mission_urgency",
    "concession_appropriateness",
    "interaction_risk",
    "deadlock_severity",
)


class LabelingFailure(RuntimeError):
    """A bounded, redacted labeling failure."""


@dataclass(frozen=True)
class ParsedLabel:
    scores: dict[str, float]
    evidence: dict[str, tuple[str, ...]]


def parse_label_response(
    content: str, available_evidence_paths: Iterable[str]
) -> ParsedLabel:
    """Parse only the exact four-score/evidence JSON response."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty.")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("LLM response is not strict JSON.") from error
    if not isinstance(payload, dict) or set(payload) != {"scores", "evidence"}:
        raise ValueError("LLM response must contain exactly scores and evidence.")
    scores = payload["scores"]
    evidence = payload["evidence"]
    if not isinstance(scores, dict) or set(scores) != set(_DIMENSIONS):
        raise ValueError("scores must contain exactly the four semantic dimensions.")
    if not isinstance(evidence, dict) or set(evidence) != set(_DIMENSIONS):
        raise ValueError("evidence must contain exactly the four semantic dimensions.")
    return ParsedLabel(
        scores=_parse_scores(scores),
        evidence=_parse_evidence(evidence, set(available_evidence_paths)),
    )


def _parse_scores(scores: dict[str, object]) -> dict[str, float]:
    parsed = {}
    for dimension in _DIMENSIONS:
        value = scores[dimension]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{dimension} must be a finite numeric score.")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{dimension} must be within [0, 1].")
        parsed[dimension] = numeric
    return parsed


def _parse_evidence(
    evidence: dict[str, object], allowed: set[str]
) -> dict[str, tuple[str, ...]]:
    parsed = {}
    for dimension in _DIMENSIONS:
        paths = evidence[dimension]
        if not isinstance(paths, list) or not 1 <= len(paths) <= 4:
            raise ValueError("Each evidence value must contain one to four paths.")
        if any(not isinstance(path, str) or path not in allowed for path in paths):
            raise ValueError("Response contains an unavailable evidence path.")
        if len(set(paths)) != len(paths):
            raise ValueError("Evidence paths must be unique per dimension.")
        parsed[dimension] = tuple(paths)
    return parsed


class AtomicLabelStore:
    """Durable append-only JSONL store with linear total write cost."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cached_records: list[dict[str, object]] | None = None
        self._cached_stat: tuple[int, int] | None = None
        self._success_index: dict[str, dict[str, object]] = {}
        self._lock = RLock()

    def records(self) -> list[dict[str, object]]:
        with self._lock:
            return self._records_unlocked()

    def _records_unlocked(self) -> list[dict[str, object]]:
        if not self.path.exists():
            self._cached_records = []
            self._cached_stat = None
            self._success_index = {}
            return []
        stat = self.path.stat()
        marker = (stat.st_size, stat.st_mtime_ns)
        if marker == self._cached_stat and self._cached_records is not None:
            return list(self._cached_records)
        records = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid label store JSON at line {line_number}."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Invalid label store record at line {line_number}.")
            records.append(record)
        self._cached_records = records
        self._cached_stat = marker
        self._success_index = {
            str(item["cache_key"]): item for item in records
            if item.get("status") == "ok" and isinstance(item.get("cache_key"), str)
        }
        return list(records)

    def successful(
        self, cache_key: str, available_evidence_paths: Iterable[str]
    ) -> dict[str, object] | None:
        """Return only one complete, current, public cached success record."""

        with self._lock:
            self._refresh_if_changed()
            record = self._success_index.get(cache_key)
            if record is not None:
                _validate_cached_success_record(record, available_evidence_paths)
            return record

    def append(self, record: Mapping[str, object]) -> None:
        with self._lock:
            self._append_unlocked(record)

    def _append_unlocked(self, record: Mapping[str, object]) -> None:
        self._refresh_if_changed()
        records = self._cached_records
        assert records is not None
        public_record = dict(record)
        if contains_credential(public_record):
            raise ValueError("Label record contains credential material.")
        encoded = (
            json.dumps(public_record, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        records.append(public_record)
        if public_record.get("status") == "ok" and isinstance(
            public_record.get("cache_key"), str
        ):
            self._success_index[public_record["cache_key"]] = public_record
        stat = self.path.stat()
        self._cached_records = records
        self._cached_stat = (stat.st_size, stat.st_mtime_ns)

    def _refresh_if_changed(self) -> None:
        if not self.path.exists():
            if self._cached_records is None or self._cached_stat is not None:
                self._records_unlocked()
            return
        stat = self.path.stat()
        if self._cached_records is None or self._cached_stat != (
            stat.st_size, stat.st_mtime_ns
        ):
            self._records_unlocked()


def label_view(
    view_id: str,
    semantic_view: Mapping[str, object],
    provider: OpenAICompatibleProvider,
    store: AtomicLabelStore,
    *,
    cache_namespace: str = "primary",
) -> dict[str, object]:
    """Label one view with bounded retries, cache, and redacted receipts."""

    if not isinstance(view_id, str) or not view_id:
        raise ValueError("view_id must be a non-empty string.")
    if not isinstance(cache_namespace, str) or not cache_namespace:
        raise ValueError("cache_namespace must be a non-empty string.")
    prompt = build_label_prompt(semantic_view)
    cache_key = _cache_key(
        prompt.semantic_view_sha256,
        prompt.prompt_version,
        provider,
        cache_namespace,
    )
    cached = store.successful(cache_key, prompt.available_evidence_paths)
    if cached is not None:
        return cached
    retry_history = []
    last_code = "labeling_failed"
    for _ in range(provider.config.retry_limit + 1):
        try:
            completion = provider.complete(prompt)
            parsed = parse_label_response(
                completion.content, prompt.available_evidence_paths
            )
            receipt = {**completion.receipt, "retry_history": retry_history}
            record = {
                "schema_version": LABEL_RECORD_V1,
                "status": "ok",
                "view_id": view_id,
                "cache_key": cache_key,
                "cache_namespace": cache_namespace,
                "scores": parsed.scores,
                "evidence": {
                    name: list(paths) for name, paths in parsed.evidence.items()
                },
                "receipt": receipt,
            }
            store.append(record)
            return record
        except ProviderCallError as error:
            last_code = error.code
        except ValueError:
            last_code = "response_validation_failed"
        retry_history.append(last_code)
    failure = {
        "schema_version": LABEL_FAILURE_V1,
        "status": "failed",
        "view_id": view_id,
        "cache_key": cache_key,
        "cache_namespace": cache_namespace,
        "provider_id": provider.config.provider_id,
        "requested_model": provider.config.requested_model,
        "semantic_view_sha256": prompt.semantic_view_sha256,
        "error_code": last_code,
        "retry_history": retry_history,
    }
    store.append(failure)
    raise LabelingFailure(last_code) from None


def _cache_key(
    semantic_view_sha256: str,
    prompt_version: str,
    provider: OpenAICompatibleProvider,
    cache_namespace: str,
) -> str:
    identity = {
        "semantic_view_sha256": semantic_view_sha256,
        "prompt_version": prompt_version,
        "provider": provider.config.public_identity(),
        "cache_namespace": cache_namespace,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _validate_cached_success_record(
    record: Mapping[str, object], available_evidence_paths: Iterable[str]
) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "view_id",
        "cache_key",
        "cache_namespace",
        "scores",
        "evidence",
        "receipt",
    }
    if set(record) != expected_fields:
        raise ValueError("cached label record has an invalid field set.")
    if record["schema_version"] != LABEL_RECORD_V1 or record["status"] != "ok":
        raise ValueError("cached label record has an unsupported schema or status.")
    for field in ("view_id", "cache_key", "cache_namespace"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"cached label record has an invalid {field}.")
    _validate_cached_label_values(record, available_evidence_paths)
    _validate_cached_receipt(record["receipt"])


def _validate_cached_label_values(
    record: Mapping[str, object], available_evidence_paths: Iterable[str]
) -> None:
    scores = record["scores"]
    evidence = record["evidence"]
    if not isinstance(scores, dict) or set(scores) != set(_DIMENSIONS):
        raise ValueError("cached label record has invalid scores.")
    if not isinstance(evidence, dict) or set(evidence) != set(_DIMENSIONS):
        raise ValueError("cached label record has invalid evidence.")
    try:
        _parse_scores(scores)
        _parse_evidence(evidence, set(available_evidence_paths))
    except ValueError as error:
        raise ValueError("cached label record has invalid label values.") from error


def _validate_cached_receipt(receipt: object) -> None:
    if not isinstance(receipt, dict) or not receipt:
        raise ValueError("cached label record has an invalid receipt.")
    identity_fields = (
        "provider_id",
        "requested_model",
        "returned_model",
        "prompt_version",
        "semantic_view_sha256",
    )
    if any(not isinstance(receipt.get(field), str) or not receipt[field]
           for field in identity_fields):
        raise ValueError("cached label record has an incomplete receipt.")
    if contains_credential(receipt):
        raise ValueError("cached label record contains credential material.")
