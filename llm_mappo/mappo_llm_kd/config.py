"""Strict, hashable configuration boundaries for Phase 3 LLM KD runs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping

import yaml


CONFIG_SCHEMAS: dict[str, dict[str, type]] = {
    "label_audit": {
        "schema_version": str,
        "diagnostic_config_sha256": str,
        "prompt_version": str,
        "temporal_view_schema": str,
        "models": list,
        "audit_subset": dict,
        "cache_policy": str,
        "concurrency": int,
        "logical_completion_max": int,
        "http_attempt_max": int,
        "output_token_max": int,
        "prompt_character_max": int,
        "output_directory": str,
    },
    "dataset_build": {
        "schema_version": str,
        "diagnostic_config_sha256": str,
        "environment_id": str,
        "n_agents": int,
        "behavior_policy_checkpoint": str,
        "behavior_policy_sha256": str,
        "collection_seeds": list,
        "episode_budget": int,
        "history_length": int,
        "summary_horizon": int,
        "split_ratios": dict,
        "sampling_strata": list,
        "output_directory": str,
    },
    "pretraining": {
        "schema_version": str,
        "diagnostic_config_sha256": str,
        "dataset_sha256": str,
        "representation_seeds": list,
        "device": str,
        "batch_size": int,
        "sequence_length": int,
        "max_epochs": int,
        "optimizer": str,
        "learning_rate": float,
        "loss": str,
        "dimension_weights": list,
        "reliability_rule": str,
        "early_stopping_patience": int,
        "early_stopping_min_delta": float,
        "output_directory": str,
    },
    "representation_diagnostics": {
        "schema_version": str,
        "diagnostic_config_sha256": str,
        "dataset_sha256": str,
        "representation_artifacts": dict,
        "controls": list,
        "probe_seeds": list,
        "device": str,
        "batch_size": int,
        "max_epochs": int,
        "optimizer": str,
        "learning_rate": float,
        "loss": str,
        "early_stopping_patience": int,
        "early_stopping_min_delta": float,
        "output_directory": str,
    },
    "downstream_diagnostics": {
        "schema_version": str,
        "diagnostic_config_sha256": str,
        "representation_sha256": str,
        "arms": list,
        "environment_id": str,
        "n_agents": int,
        "seed": int,
        "environment_steps": int,
        "workers": int,
        "rollout_length": int,
        "optimization": dict,
        "evaluation_seeds": list,
        "evaluation_episodes": int,
        "checkpoint_interval": int,
        "devices": dict,
        "output_directory": str,
    },
}

_L1_MODEL_FIELDS = {
    "provider_id": str,
    "requested_model": str,
    "base_url": str,
    "endpoint_path": str,
    "api_key_environment_variable": str,
    "temperature": float,
    "seed": (int, type(None)),
    "timeout_seconds": float,
    "retry_limit": int,
    "max_completion_tokens": int,
    "max_prompt_characters": int,
    "response_format": str,
    "extra_body": dict,
}
_L1_AUDIT_FIELDS = {
    "schema_version": str,
    "sample_count": int,
    "repeat_sample_count": int,
    "repeat_selection": str,
}

_M1_TOP_LEVEL = {
    "schema_version",
    "metric_definition_version",
    "semantic_dimensions",
    "test_access",
    "label_metrics",
    "prediction_metrics",
    "probe",
    "controls",
    "representation_health_metrics",
    "downstream_report",
    "go_no_go_thresholds",
}
_M1_SECTION_FIELDS = {
    "test_access": {
        "split_unit",
        "selection_partition",
        "test_evaluations_per_representation_seed",
        "test_access_after_model_and_threshold_freeze",
    },
    "label_metrics": {
        "parse_rate_denominator",
        "retry_rate_denominator",
        "forbidden_content_rate_denominator",
        "evidence_path_valid_rate_denominator",
        "standard_deviation",
        "boundary_epsilon",
        "disagreement",
    },
    "prediction_metrics": {
        "losses",
        "constant_predictor",
        "delta_direction",
        "relative_reduction",
        "zero_constant_error_rule",
    },
    "probe": {
        "architecture",
        "initialization",
        "representation_state",
        "training_labels",
        "selection_partition",
        "test_evaluations",
        "seeds",
        "optimizer",
        "learning_rate",
        "batch_size",
        "max_epochs",
        "loss",
        "early_stopping_patience",
        "early_stopping_min_delta",
    },
    "representation_health_metrics": {
        "embedding_variance",
        "effective_rank",
        "temporal_sensitivity",
        "gradient_norm",
        "import_export_error",
        "non_finite_count",
    },
    "downstream_report": {"arms", "factor_design", "seed_claim"},
    "go_no_go_thresholds": {
        "label",
        "end_to_end",
        "probe",
        "representation_health",
    },
}
_M1_THRESHOLD_FIELDS = {
    "label": {
        "parse_rate_min",
        "retry_rate_max",
        "forbidden_content_rate_max",
        "evidence_path_valid_rate_min",
        "per_dimension_standard_deviation_min",
        "per_dimension_boundary_saturation_rate_max",
        "repeated_call_mean_absolute_disagreement_max",
        "cross_model_mean_absolute_disagreement_max",
    },
    "end_to_end": {
        "macro_mae_relative_reduction_min",
        "per_dimension_mae_relative_reduction_min",
        "dimensions_meeting_minimum",
        "maximum_dimension_absolute_mae_regression",
        "joint_rmse_relative_reduction_min",
    },
    "probe": {
        "normal_vs_constant_macro_mae_relative_reduction_min",
        "normal_vs_random_init_macro_mae_relative_reduction_min",
        "normal_vs_shuffled_label_macro_mae_relative_reduction_min",
        "temporal_dimension_names",
        "normal_vs_shuffled_history_temporal_mae_relative_reduction_min",
        "normal_vs_single_frame_temporal_mae_relative_reduction_min",
        "representation_seed_count",
        "maximum_macro_mae_coefficient_of_variation",
    },
    "representation_health": {
        "non_finite_count_max",
        "mean_feature_variance_min",
        "effective_rank_min",
        "temporal_sensitivity_mean_l2_min",
        "encoder_gradient_norm_min",
        "gru_gradient_norm_min",
        "import_export_max_absolute_error_max",
    },
}


@dataclass(frozen=True)
class ResolvedConfig:
    """A fully explicit mapping and its canonical content identity."""

    values: dict[str, object]
    sha256: str


def validate_explicit_config(kind: str, values: Mapping[str, object]) -> None:
    """Reject unknown, absent, or incorrectly typed run-affecting fields."""

    if kind not in CONFIG_SCHEMAS:
        raise ValueError(f"Unknown Phase 3 config kind: {kind}")
    schema = CONFIG_SCHEMAS[kind]
    supplied = set(values)
    missing = set(schema) - supplied
    extra = supplied - set(schema)
    if missing:
        raise ValueError(f"Config has missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"Config has extra fields: {sorted(extra)}")
    for name, expected_type in schema.items():
        expected_types = (
            expected_type if isinstance(expected_type, tuple) else (expected_type,)
        )
        if type(values[name]) not in expected_types:
            names = "/".join(item.__name__ for item in expected_types)
            raise ValueError(
                f"Config field {name} must have exact type {names}."
            )


def load_m1_config(path: str | Path) -> ResolvedConfig:
    """Load and strictly identify the owner-reviewable M1 contract."""

    return _resolve_m1_bytes(Path(path).read_bytes())


def _resolve_m1_bytes(encoded_config: bytes) -> ResolvedConfig:
    """Resolve one immutable M1 byte snapshot."""

    values = yaml.safe_load(encoded_config.decode("utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Resolved diagnostic config must be a mapping.")
    missing = _M1_TOP_LEVEL - set(values)
    extra = set(values) - _M1_TOP_LEVEL
    if missing:
        raise ValueError(f"M1 config has missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"M1 config has extra fields: {sorted(extra)}")
    _validate_m1_identities(values)
    _validate_m1_sections(values)
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ResolvedConfig(values=values, sha256=sha256(encoded).hexdigest())


def load_approved_m1_config(
    config_path: str | Path, approval_path: str | Path
) -> ResolvedConfig:
    """Load each M1 input once and return only the approved config snapshot."""

    resolved, _ = _load_approved_m1_snapshot(config_path, approval_path)
    return resolved


def verify_m1_approval(
    config_path: str | Path, approval_path: str | Path
) -> dict[str, object]:
    """Verify that owner approval names the exact canonical M1 config."""

    _, approval = _load_approved_m1_snapshot(config_path, approval_path)
    return approval


def _load_approved_m1_snapshot(
    config_path: str | Path, approval_path: str | Path
) -> tuple[ResolvedConfig, dict[str, object]]:
    config_bytes = Path(config_path).read_bytes()
    approval_bytes = Path(approval_path).read_bytes()
    resolved = _resolve_m1_bytes(config_bytes)
    approval = json.loads(approval_bytes)
    _validate_m1_approval_record(approval, resolved)
    return resolved, approval


def _validate_m1_approval_record(
    approval: object, resolved: ResolvedConfig
) -> None:
    expected_fields = {
        "schema_version",
        "gate",
        "status",
        "approved_on",
        "approval_source",
        "approved_config_sha256",
    }
    _validate_mapping_keys(approval, expected_fields, "M1 approval")
    if approval["schema_version"] != "phase3-owner-gate-approval-v1":
        raise ValueError("Unknown M1 approval schema version.")
    if approval["gate"] != "M1" or approval["status"] != "approved":
        raise ValueError("M1 approval record is not approved.")
    if approval["approved_config_sha256"] != resolved.sha256:
        raise ValueError("M1 approval hash does not match the resolved config.")


def load_l1_config(
    path: str | Path,
    m1_config_path: str | Path,
    m1_approval_path: str | Path,
) -> ResolvedConfig:
    """Load the strict two-provider L1 proposal bound to approved M1."""

    m1 = load_approved_m1_config(m1_config_path, m1_approval_path)
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("L1 config must be a mapping.")
    validate_explicit_config("label_audit", values)
    if values["schema_version"] != "phase3-label-audit-config-v1":
        raise ValueError("Unknown L1 label-audit schema version.")
    if values["diagnostic_config_sha256"] != m1.sha256:
        raise ValueError("L1 config does not reference the approved M1 hash.")
    _validate_l1_models(values["models"])
    _validate_typed_mapping(
        values["audit_subset"], _L1_AUDIT_FIELDS, "audit_subset"
    )
    _validate_l1_runtime_contract(values)
    _validate_l1_budgets(values)
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ResolvedConfig(values=values, sha256=sha256(encoded).hexdigest())


def verify_l1_approval(
    config_path: str | Path,
    m1_config_path: str | Path,
    m1_approval_path: str | Path,
    approval_path: str | Path,
) -> dict[str, object]:
    """Verify that owner approval names the exact canonical L1 config."""

    resolved = load_l1_config(
        config_path, m1_config_path, m1_approval_path
    )
    approval = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "gate",
        "status",
        "approved_on",
        "approval_source",
        "approved_config_sha256",
    }
    _validate_mapping_keys(approval, expected_fields, "L1 approval")
    if approval["schema_version"] != "phase3-owner-gate-approval-v1":
        raise ValueError("Unknown L1 approval schema version.")
    if approval["gate"] != "L1" or approval["status"] != "approved":
        raise ValueError("L1 approval record is not approved.")
    if approval["approved_config_sha256"] != resolved.sha256:
        raise ValueError("L1 approval hash does not match the resolved config.")
    return approval


def _validate_l1_models(models: object) -> None:
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("L1 config requires exactly two provider models.")
    for index, model in enumerate(models):
        _validate_typed_mapping(model, _L1_MODEL_FIELDS, f"models[{index}]")
        if not model["base_url"].startswith("https://"):
            raise ValueError("Every L1 provider base URL must use HTTPS.")
    if len({model["provider_id"] for model in models}) != 2:
        raise ValueError("L1 provider IDs must be distinct.")
    if len({model["api_key_environment_variable"] for model in models}) != 2:
        raise ValueError("Each L1 provider needs a distinct key environment variable.")


def _validate_l1_runtime_contract(values: dict[str, object]) -> None:
    expected = {
        "prompt_version": "llm-label-prompt-v3",
        "temporal_view_schema": "temporal-semantic-view-v1",
        "cache_policy": "read_write_by_prompt_view_provider_and_namespace",
        "concurrency": 1,
    }
    subset_expected = {
        "schema_version": "phase3-shared-label-audit-subset-v1",
        "sample_count": 8,
        "repeat_sample_count": 4,
        "repeat_selection": "first_four_by_sample_id",
    }
    if any(
        values[name] != expected_value
        for name, expected_value in expected.items()
    ):
        raise ValueError("L1 config contains an unsupported runtime contract.")
    if values["audit_subset"] != subset_expected:
        raise ValueError("L1 config contains an unsupported runtime contract.")


def _validate_l1_budgets(values: dict[str, object]) -> None:
    subset = values["audit_subset"]
    per_model = subset["sample_count"] + subset["repeat_sample_count"]
    models = values["models"]
    expected_logical = per_model * len(models)
    expected_attempts = sum(
        per_model * (model["retry_limit"] + 1) for model in models
    )
    expected_outputs = sum(
        per_model * model["max_completion_tokens"] for model in models
    )
    expected_characters = sum(
        per_model * model["max_prompt_characters"] for model in models
    )
    expected = {
        "logical_completion_max": expected_logical,
        "http_attempt_max": expected_attempts,
        "output_token_max": expected_outputs,
        "prompt_character_max": expected_characters,
    }
    for name, expected_value in expected.items():
        if values[name] != expected_value:
            raise ValueError(f"L1 budget field {name} is inconsistent.")


def _validate_typed_mapping(
    value: object, schema: dict[str, object], section: str
) -> None:
    _validate_mapping_keys(value, set(schema), section)
    for name, expected in schema.items():
        expected_types = expected if isinstance(expected, tuple) else (expected,)
        if type(value[name]) not in expected_types:
            raise ValueError(f"L1 field {section}.{name} has the wrong type.")


def _validate_m1_identities(values: dict[str, object]) -> None:
    if values["schema_version"] != "phase3-m1-diagnostics-v1":
        raise ValueError("Unknown M1 diagnostic schema version.")
    if values["metric_definition_version"] != "llm-kd-diagnostic-metrics-v1":
        raise ValueError("Unknown diagnostic metric definition version.")


def _validate_m1_sections(values: dict[str, object]) -> None:
    for section, expected_fields in _M1_SECTION_FIELDS.items():
        _validate_mapping_keys(values[section], expected_fields, section)
    thresholds = values["go_no_go_thresholds"]
    for section, expected_fields in _M1_THRESHOLD_FIELDS.items():
        _validate_mapping_keys(
            thresholds[section], expected_fields, f"go_no_go_thresholds.{section}"
        )
        _validate_threshold_values(section, thresholds[section])
    expected_dimensions = [
        "mission_urgency",
        "concession_appropriateness",
        "interaction_risk",
        "deadlock_severity",
    ]
    if values["semantic_dimensions"] != expected_dimensions:
        raise ValueError("M1 config must use the exact four semantic dimensions.")
    expected_controls = [
        "constant_predictor",
        "random_init",
        "shuffled_label",
        "shuffled_history",
        "single_frame",
        "normal_history",
    ]
    if values["controls"] != expected_controls:
        raise ValueError("M1 config must use the exact diagnostic controls.")


def _validate_threshold_values(section: str, values: dict[str, object]) -> None:
    for name, threshold in values.items():
        if name == "temporal_dimension_names":
            if not isinstance(threshold, list):
                raise ValueError("Temporal dimension threshold must be a list.")
            continue
        if type(threshold) not in (int, float):
            raise ValueError(f"M1 threshold {section}.{name} must be numeric.")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError(
                f"M1 threshold {section}.{name} must be finite and non-negative."
            )


def _validate_mapping_keys(
    value: object, expected: set[str], section: str
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"M1 section {section} must be a mapping.")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing:
        raise ValueError(f"M1 section {section} has missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"M1 section {section} has extra fields: {sorted(extra)}")
