"""Versioned label, prediction, and representation diagnostics."""

from __future__ import annotations

import math

import numpy as np


SEMANTIC_DIMENSIONS = (
    "mission_urgency",
    "concession_appropriateness",
    "interaction_risk",
    "deadlock_severity",
)
DIAGNOSTIC_METRICS_V1 = "llm-kd-diagnostic-metrics-v1"
DOWNSTREAM_ARMS = (
    "baseline",
    "llm_kd",
    "astar_kd_no_validity",
    "astar_kd_no_validity_plus_llm_kd",
)


def semantic_prediction_metrics(
    truth: np.ndarray, prediction: np.ndarray, train_labels: np.ndarray
) -> dict[str, object]:
    """Compute MAE/MSE jointly and against train-only constant means."""

    expected = _score_matrix(truth, "truth")
    predicted = _score_matrix(prediction, "prediction")
    training = _score_matrix(train_labels, "train_labels")
    if expected.shape != predicted.shape:
        raise ValueError("truth and prediction shapes must match.")
    constant = np.mean(training, axis=0)
    absolute = np.abs(predicted - expected)
    squared = np.square(predicted - expected)
    constant_absolute = np.abs(expected - constant)
    constant_squared = np.square(expected - constant)
    per_dimension = {}
    for index, name in enumerate(SEMANTIC_DIMENSIONS):
        mae = float(np.mean(absolute[:, index]))
        mse = float(np.mean(squared[:, index]))
        constant_mae = float(np.mean(constant_absolute[:, index]))
        constant_mse = float(np.mean(constant_squared[:, index]))
        per_dimension[name] = {
            "mae": mae,
            "mse": mse,
            "constant_mae": constant_mae,
            "constant_mse": constant_mse,
            "mae_delta": constant_mae - mae,
            "mse_delta": constant_mse - mse,
            "mae_relative_reduction": _relative_reduction(mae, constant_mae),
            "mse_relative_reduction": _relative_reduction(mse, constant_mse),
        }
    macro_mae = float(np.mean(absolute))
    macro_mse = float(np.mean(squared))
    constant_macro_mae = float(np.mean(constant_absolute))
    constant_macro_mse = float(np.mean(constant_squared))
    joint_rmse = float(np.mean(np.sqrt(np.mean(squared, axis=1))))
    constant_joint_rmse = float(
        np.mean(np.sqrt(np.mean(constant_squared, axis=1)))
    )
    return {
        "metric_definition_version": DIAGNOSTIC_METRICS_V1,
        "per_dimension": per_dimension,
        "macro": {
            "mae": macro_mae,
            "mse": macro_mse,
            "constant_mae": constant_macro_mae,
            "constant_mse": constant_macro_mse,
            "mae_relative_reduction": _relative_reduction(
                macro_mae, constant_macro_mae
            ),
            "mse_relative_reduction": _relative_reduction(
                macro_mse, constant_macro_mse
            ),
        },
        "joint": {
            "sample_root_mean_squared_error": joint_rmse,
            "constant_sample_root_mean_squared_error": constant_joint_rmse,
            "relative_reduction": _relative_reduction(
                joint_rmse,
                constant_joint_rmse,
            ),
        },
    }


def label_diagnostic_metrics(
    *,
    scores: np.ndarray,
    requested_samples: int,
    parsed_samples: int,
    retry_count: int,
    forbidden_response_count: int,
    valid_evidence_path_count: int,
    evidence_path_count: int,
    repeated_call_pairs: tuple[np.ndarray, np.ndarray] | None,
    cross_model_pairs: tuple[np.ndarray, np.ndarray] | None,
    boundary_epsilon: float,
) -> dict[str, object]:
    """Compute label reliability and non-collapse statistics."""

    values = _label_score_matrix(scores)
    if requested_samples < 1 or not 0 <= parsed_samples <= requested_samples:
        raise ValueError("Sample counts are inconsistent.")
    if values.shape[0] != parsed_samples:
        raise ValueError("parsed_samples must equal the number of score rows.")
    if retry_count < 0 or forbidden_response_count < 0:
        raise ValueError("Failure counts cannot be negative.")
    if not 0 <= valid_evidence_path_count <= evidence_path_count:
        raise ValueError("Evidence-path counts are inconsistent.")
    if not 0.0 <= boundary_epsilon < 0.5:
        raise ValueError("boundary_epsilon must be in [0, 0.5).")
    per_dimension = {}
    if values.shape[0]:
        saturated = (values <= boundary_epsilon) | (
            values >= 1.0 - boundary_epsilon
        )
        for index, name in enumerate(SEMANTIC_DIMENSIONS):
            per_dimension[name] = {
                "mean": float(np.mean(values[:, index])),
                "standard_deviation": float(np.std(values[:, index])),
                "boundary_saturation_rate": float(np.mean(saturated[:, index])),
            }
    else:
        for name in SEMANTIC_DIMENSIONS:
            per_dimension[name] = {
                "mean": None,
                "standard_deviation": None,
                "boundary_saturation_rate": None,
            }
    return {
        "metric_definition_version": DIAGNOSTIC_METRICS_V1,
        "parse_rate": parsed_samples / requested_samples,
        "retry_rate": retry_count / requested_samples,
        "forbidden_content_rate": (
            forbidden_response_count / requested_samples
        ),
        "evidence_path_valid_rate": (
            valid_evidence_path_count / evidence_path_count
            if evidence_path_count
            else None
        ),
        "per_dimension": per_dimension,
        "repeated_call_mean_absolute_disagreement": _pair_disagreement(
            repeated_call_pairs, "repeated_call_pairs"
        ),
        "cross_model_mean_absolute_disagreement": _pair_disagreement(
            cross_model_pairs, "cross_model_pairs"
        ),
    }


def representation_health_metrics(
    *,
    embeddings: np.ndarray,
    temporal_control_embeddings: np.ndarray,
    imported_embeddings: np.ndarray,
    encoder_gradient_norm: float,
    gru_gradient_norm: float,
) -> dict[str, object]:
    """Measure collapse, temporal sensitivity, gradient flow, and round-trip."""

    values = _embedding_matrix(embeddings, "embeddings")
    temporal = _embedding_matrix(
        temporal_control_embeddings, "temporal_control_embeddings"
    )
    imported = _embedding_matrix(imported_embeddings, "imported_embeddings")
    if values.shape != temporal.shape or values.shape != imported.shape:
        raise ValueError("All embedding matrices must have identical shapes.")
    non_finite_count = int(
        np.count_nonzero(~np.isfinite(values))
        + np.count_nonzero(~np.isfinite(temporal))
        + np.count_nonzero(~np.isfinite(imported))
    )
    if non_finite_count:
        mean_variance = effective_rank = sensitivity = round_trip = None
    else:
        mean_variance = float(np.mean(np.var(values, axis=0)))
        effective_rank = _effective_rank(values)
        sensitivity = float(np.mean(np.linalg.norm(values - temporal, axis=1)))
        round_trip = float(np.max(np.abs(values - imported)))
    return {
        "metric_definition_version": DIAGNOSTIC_METRICS_V1,
        "non_finite_count": non_finite_count,
        "mean_feature_variance": mean_variance,
        "effective_rank": effective_rank,
        "temporal_sensitivity_mean_l2": sensitivity,
        "import_export_max_absolute_error": round_trip,
        "encoder_gradient_norm": _finite_nonnegative(
            encoder_gradient_norm, "encoder_gradient_norm"
        ),
        "gru_gradient_norm": _finite_nonnegative(
            gru_gradient_norm, "gru_gradient_norm"
        ),
    }


def build_downstream_diagnostic_report(
    arms: dict[str, dict[str, object]]
) -> dict[str, object]:
    """Build the fixed four-arm diagnostic-only report envelope."""

    if set(arms) != set(DOWNSTREAM_ARMS):
        raise ValueError("Downstream report requires the exact four arms.")
    return {
        "metric_definition_version": DIAGNOSTIC_METRICS_V1,
        "design": "2x2-llm-initialization-by-astar-kd",
        "claim_scope": "diagnostic-only-not-formal-performance-proof",
        "arms": {name: arms[name] for name in DOWNSTREAM_ARMS},
    }


def _score_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] != 4:
        raise ValueError(f"{name} must have shape [sample, 4].")
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must contain finite [0, 1] scores.")
    return result


def _label_score_matrix(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError("scores must have shape [sample, 4].")
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("scores must contain finite [0, 1] values.")
    return result


def _embedding_matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] != 64:
        raise ValueError(f"{name} must have shape [sample>=2, 64].")
    return result


def _relative_reduction(value: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return 0.0 if value == 0.0 else None
    return (baseline - value) / baseline


def _pair_disagreement(
    pair: tuple[np.ndarray, np.ndarray] | None, name: str
) -> float | None:
    if pair is None:
        return None
    first = _score_matrix(pair[0], name)
    second = _score_matrix(pair[1], name)
    if first.shape != second.shape:
        raise ValueError(f"{name} matrices must have identical shapes.")
    return float(np.mean(np.abs(first - second)))


def _effective_rank(values: np.ndarray) -> float:
    centered = values - np.mean(values, axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular_values)
    positive = energy[energy > np.finfo(np.float64).eps]
    if positive.size == 0:
        return 0.0
    probabilities = positive / np.sum(positive)
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result
