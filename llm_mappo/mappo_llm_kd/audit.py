"""Deterministic shared audit subset and model-comparison receipts."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from llm_mappo.mappo_baseline.structured_observation import (
    DIRECT_GOAL_OBSERVATION_V2,
    StructuredObservationSchema,
)

from .prompt import build_label_prompt
from .config import ResolvedConfig
from .labeling import AtomicLabelStore, LabelingFailure, label_view
from .provider import OpenAICompatibleProvider, ProviderConfig
from .temporal_view import serialize_temporal_semantic_view


SHARED_AUDIT_SUBSET_V1 = "shared-label-audit-subset-v1"
_SCHEMA = StructuredObservationSchema(
    observation_schema=DIRECT_GOAL_OBSERVATION_V2
)
_INDEX = {name: index for index, name in enumerate(_SCHEMA.feature_names)}


def run_label_audit(
    resolved: ResolvedConfig,
    *,
    mode: str,
    provider_factory=OpenAICompatibleProvider,
    provider_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run or resume the approved smoke/audit matrix and write receipts."""

    if mode not in {"smoke", "audit"}:
        raise ValueError("Label audit mode must be smoke or audit.")
    values = resolved.values
    output = Path(values["output_directory"]) / resolved.sha256[:12]
    models = _select_models(values["models"], provider_ids)
    samples = build_shared_audit_samples()
    write_shared_audit_subset(output / "shared-audit-subset.jsonl", samples)
    selected = samples[:1] if mode == "smoke" else samples
    repeat_count = 0 if mode == "smoke" else values["audit_subset"][
        "repeat_sample_count"
    ]
    stores = []
    for model_values in models:
        provider = provider_factory(ProviderConfig(**model_values))
        store = AtomicLabelStore(output / f"{provider.config.provider_id}.jsonl")
        stores.append(store)
        for sample in selected:
            _attempt_label(sample, provider, store, "primary")
        for sample in selected[:repeat_count]:
            _attempt_label(sample, provider, store, "repeat-1")
    records = [record for store in stores for record in store.records()]
    successful = [record for record in records if record.get("status") == "ok"]
    failed = [record for record in records if record.get("status") == "failed"]
    try:
        comparison = build_model_comparison_receipt(
            successful,
            required_provider_ids=[
                model["provider_id"] for model in values["models"]
            ],
            required_primary_ids=[sample["sample_id"] for sample in samples],
            required_repeat_ids=[
                sample["sample_id"]
                for sample in samples[:values["audit_subset"]["repeat_sample_count"]]
            ],
        )
    except ValueError:
        comparison = {
            "schema_version": "llm-model-comparison-receipt-v1",
            "status": "incomplete",
            "reason": "insufficient_successful_provider_coverage",
            "successful_providers": sorted(
                {
                    record["receipt"]["provider_id"]
                    for record in successful
                }
            ),
            "teacher_selected": False,
            "selected_teacher": None,
        }
    _write_json_atomic(output / "model-comparison.json", comparison)
    receipt = {
        "schema_version": "llm-label-audit-run-receipt-v1",
        "l1_config_sha256": resolved.sha256,
        "mode": mode,
        "audit_subset_sha256": _canonical_sha256(samples),
        "successful_records": len(successful),
        "failed_records": len(failed),
        "comparison_status": comparison["status"],
        "requested_provider_ids": [
            model["provider_id"] for model in models
        ],
        "teacher_selected": False,
        "selected_teacher": None,
    }
    _write_json_atomic(output / "run-receipt.json", receipt)
    return receipt


def _select_models(models, provider_ids):
    if provider_ids is None:
        return models
    requested = tuple(provider_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Provider subset must be non-empty and unique.")
    by_id = {model["provider_id"]: model for model in models}
    if any(provider_id not in by_id for provider_id in requested):
        raise ValueError("Provider subset contains an unconfigured provider.")
    return [by_id[provider_id] for provider_id in requested]


def _attempt_label(sample, provider, store, namespace):
    try:
        label_view(
            sample["sample_id"],
            sample["temporal_semantic_view"],
            provider,
            store,
            cache_namespace=namespace,
        )
    except LabelingFailure:
        return


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def build_shared_audit_samples() -> list[dict[str, object]]:
    """Build eight synthetic observable histories with no expected labels."""

    definitions = _audit_definitions()
    samples = []
    for ordinal, (stratum, rows) in enumerate(definitions):
        view = serialize_temporal_semantic_view(
            np.stack(rows),
            np.asarray([True] + [False] * (len(rows) - 1)),
            _SCHEMA.feature_names,
            8,
        )
        prompt = build_label_prompt(view)
        samples.append(
            {
                "schema_version": SHARED_AUDIT_SUBSET_V1,
                "sample_id": f"shared-{ordinal:03d}",
                "stratum": stratum,
                "semantic_view_sha256": prompt.semantic_view_sha256,
                "temporal_semantic_view": view,
            }
        )
    return samples


def write_shared_audit_subset(
    path: str | Path, samples: Sequence[Mapping[str, object]]
) -> None:
    """Atomically persist the deterministic pre-dataset audit subset."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            stream.write(
                json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def build_model_comparison_receipt(
    records: Sequence[Mapping[str, object]],
    *,
    required_provider_ids: Sequence[str] = ("deepseek-official", "gpt-terra"),
    required_primary_ids: Sequence[str] = tuple(
        f"shared-{index:03d}" for index in range(8)
    ),
    required_repeat_ids: Sequence[str] = tuple(
        f"shared-{index:03d}" for index in range(4)
    ),
) -> dict[str, object]:
    """Compare matched scores while deliberately making no teacher selection."""

    successful = [record for record in records if record.get("status") == "ok"]
    coverage = _audit_coverage(
        successful, required_provider_ids, required_primary_ids, required_repeat_ids
    )
    providers = sorted(
        {record["receipt"]["provider_id"] for record in successful}
    )
    primary = {
        (record["receipt"]["provider_id"], record["view_id"]): record
        for record in successful
        if record.get("cache_namespace") == "primary"
    }
    repeat_disagreement = {
        provider: _repeat_disagreement(successful, primary, provider)
        for provider in providers
    }
    if len(providers) != 2 or not coverage["complete"]:
        return {
            "schema_version": "llm-model-comparison-receipt-v1",
            "status": "incomplete",
            "reason": "insufficient_successful_provider_coverage",
            "coverage": coverage,
            "providers": providers,
            "requested_models": sorted(
                {
                    record["receipt"]["requested_model"]
                    for record in successful
                }
            ),
            "returned_models": sorted(
                {
                    record["receipt"]["returned_model"]
                    for record in successful
                }
            ),
            "cross_model_mean_absolute_disagreement": None,
            "repeated_call_mean_absolute_disagreement": repeat_disagreement,
            "teacher_selected": False,
            "selected_teacher": None,
        }
    shared_ids = sorted(
        set(view for provider, view in primary if provider == providers[0])
        & set(view for provider, view in primary if provider == providers[1])
    )
    if not shared_ids:
        raise ValueError("Model comparison has no matched primary samples.")
    cross_differences = []
    for view_id in shared_ids:
        first = primary[(providers[0], view_id)]["scores"]
        second = primary[(providers[1], view_id)]["scores"]
        cross_differences.extend(
            abs(float(first[name]) - float(second[name]))
            for name in first
        )
    return {
        "schema_version": "llm-model-comparison-receipt-v1",
        "status": "complete",
        "coverage": coverage,
        "providers": providers,
        "requested_models": sorted(
            {record["receipt"]["requested_model"] for record in successful}
        ),
        "returned_models": sorted(
            {record["receipt"]["returned_model"] for record in successful}
        ),
        "matched_primary_samples": len(shared_ids),
        "cross_model_mean_absolute_disagreement": float(
            np.mean(cross_differences)
        ),
        "repeated_call_mean_absolute_disagreement": repeat_disagreement,
        "teacher_selected": False,
        "selected_teacher": None,
    }


def _audit_coverage(records, providers, primary_ids, repeat_ids):
    expected = {
        provider: {
            "primary": set(primary_ids),
            "repeat-1": set(repeat_ids),
        }
        for provider in providers
    }
    seen = {}
    for record in records:
        key = (
            record["receipt"]["provider_id"],
            record.get("cache_namespace"),
            record.get("view_id"),
        )
        seen[key] = seen.get(key, 0) + 1
    missing = []
    duplicates = []
    for provider, namespaces in expected.items():
        for namespace, view_ids in namespaces.items():
            for view_id in sorted(view_ids):
                count = seen.get((provider, namespace, view_id), 0)
                if count == 0:
                    missing.append(f"{provider}:{namespace}:{view_id}")
                elif count != 1:
                    duplicates.append(f"{provider}:{namespace}:{view_id}")
    unexpected = [
        f"{provider}:{namespace}:{view_id}"
        for (provider, namespace, view_id), count in sorted(seen.items())
        if provider not in expected
        or namespace not in expected[provider]
        or view_id not in expected[provider][namespace]
    ]
    return {
        "complete": not missing and not duplicates and not unexpected,
        "missing": missing,
        "duplicates": duplicates,
        "unexpected": unexpected,
    }


def _repeat_disagreement(records, primary, provider):
    differences = []
    for record in records:
        if record["receipt"]["provider_id"] != provider:
            continue
        if record.get("cache_namespace") == "primary":
            continue
        original = primary.get((provider, record["view_id"]))
        if original is None:
            continue
        differences.extend(
            abs(float(record["scores"][name]) - float(original["scores"][name]))
            for name in record["scores"]
        )
    return float(np.mean(differences)) if differences else None


def _frame(
    *,
    orientation="up",
    phase="task",
    action=0,
    blocked=False,
    distance=0.5,
    priority=0.5,
    battery=0.8,
    carrying=False,
):
    row = np.zeros(_SCHEMA.dimension, dtype=np.float32)
    row[_INDEX[f"self_orientation_{orientation}"]] = 1.0
    row[_INDEX[f"self_phase_{phase}"]] = 1.0
    row[_INDEX[f"self_previous_action_{action}"]] = 1.0
    row[_INDEX[f"goal_type_{phase}"]] = 1.0
    row[_INDEX["self_previous_blocked"]] = float(blocked)
    row[_INDEX["self_battery"]] = battery
    row[_INDEX["self_priority_weight"]] = priority
    row[_INDEX["self_carrying"]] = float(carrying)
    row[_INDEX["goal_normalized_distance"]] = distance
    return row


def _peer(row, *, slot=0, forward=0.1, right=0.0, priority=0.5, blocked=False):
    prefix = f"peer_{slot}_"
    row[_INDEX[prefix + "relative_forward"]] = forward
    row[_INDEX[prefix + "relative_right"]] = right
    row[_INDEX[prefix + "orientation_down"]] = 1.0
    row[_INDEX[prefix + "battery"]] = 0.8
    row[_INDEX[prefix + "phase_task"]] = 1.0
    row[_INDEX[prefix + "priority_weight"]] = priority
    row[_INDEX[prefix + "previous_action_1"]] = 1.0
    row[_INDEX[prefix + "previous_blocked"]] = float(blocked)
    row[_INDEX[prefix + "valid"]] = 1.0


def _audit_definitions():
    idle = _frame(phase="idle", priority=0.0, distance=0.0)
    urgent = [
        _frame(priority=1.0, distance=distance, action=1)
        for distance in (0.9, 0.7, 0.55)
    ]
    loaded = [
        _frame(phase="delivery", carrying=True, priority=1.0, distance=distance)
        for distance in (0.5, 0.3)
    ]
    charging = _frame(phase="charging", battery=0.1, distance=0.4)
    close_peer = _frame()
    _peer(close_peer, forward=0.08)
    close_peer[_INDEX["map_f1_r0_other_agv"]] = 1.0
    persistent = [_frame(action=1, blocked=True, distance=0.5) for _ in range(4)]
    for row in persistent:
        _peer(row, forward=0.08, blocked=True)
    oscillation = [
        _frame(orientation=orientation, action=action)
        for orientation, action in (("up", 3), ("right", 3), ("down", 3), ("left", 0))
    ]
    concession = _frame(priority=0.5)
    _peer(concession, right=0.08, forward=0.0, priority=1.0)
    concession[_INDEX["map_f0_r1_other_agv"]] = 1.0
    concession[_INDEX["map_f1_r0_shelf"]] = 1.0
    concession[_INDEX["map_f-1_r0_shelf"]] = 1.0
    return [
        ("idle_clear", [idle]),
        ("urgent_task", urgent),
        ("loaded_delivery", loaded),
        ("low_battery_charging", [charging]),
        ("close_peer", [close_peer]),
        ("persistent_blocking", persistent),
        ("orientation_oscillation", oscillation),
        ("lower_priority_concession", [concession]),
    ]
