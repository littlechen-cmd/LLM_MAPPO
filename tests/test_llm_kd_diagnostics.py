"""Tests for predeclared Phase 3 diagnostic contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from llm_mappo.mappo_llm_kd.config import (
    CONFIG_SCHEMAS,
    load_m1_config,
    validate_explicit_config,
    verify_m1_approval,
)
from llm_mappo.mappo_llm_kd.controls import (
    build_fresh_probe,
    shuffled_history_control,
    shuffled_label_indices,
    single_frame_control,
)
from llm_mappo.mappo_llm_kd.diagnostics import (
    build_downstream_diagnostic_report,
    label_diagnostic_metrics,
    representation_health_metrics,
    semantic_prediction_metrics,
)


M1_CONFIG = "configs/llm_kd/representation-diagnostics.yaml"
M1_APPROVAL = "configs/llm_kd/representation-diagnostics-approval.json"
DIMENSIONS = (
    "mission_urgency",
    "concession_appropriateness",
    "interaction_risk",
    "deadlock_severity",
)


def test_prediction_metrics_compare_each_dimension_and_joint_to_train_mean():
    train = np.asarray([[0.0] * 4, [1.0] * 4])
    truth = np.asarray([[0.0, 1.0, 0.25, 0.75]])

    metrics = semantic_prediction_metrics(truth, truth.copy(), train)

    assert tuple(metrics["per_dimension"]) == DIMENSIONS
    for values in metrics["per_dimension"].values():
        assert values["mae"] == 0.0
        assert values["mse"] == 0.0
        assert values["constant_mae"] > 0.0
        assert values["mae_relative_reduction"] == 1.0
    assert metrics["macro"]["mae"] == 0.0
    assert metrics["joint"]["sample_root_mean_squared_error"] == 0.0


def test_joint_rmse_is_distinct_from_macro_mae_for_concentrated_error():
    train = np.asarray([[0.0] * 4, [1.0] * 4])
    truth = np.asarray([[0.0] * 4])
    prediction = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    metrics = semantic_prediction_metrics(truth, prediction, train)
    assert metrics["macro"]["mae"] == 0.25
    assert metrics["joint"]["sample_root_mean_squared_error"] == 0.5
    assert metrics["joint"]["constant_sample_root_mean_squared_error"] == 0.5
    assert metrics["joint"]["relative_reduction"] == 0.0


def test_prediction_metrics_reject_test_derived_or_invalid_constant_labels():
    valid = np.full((2, 4), 0.5)
    with pytest.raises(ValueError, match="train_labels"):
        semantic_prediction_metrics(valid, valid, np.empty((0, 4)))
    invalid = valid.copy()
    invalid[0, 0] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        semantic_prediction_metrics(invalid, valid, valid)


def test_label_metrics_freeze_rate_distribution_and_disagreement_math():
    scores = np.asarray([[0.0] * 4, [0.5] * 4, [1.0] * 4])
    metrics = label_diagnostic_metrics(
        scores=scores,
        requested_samples=4,
        parsed_samples=3,
        retry_count=1,
        forbidden_response_count=0,
        valid_evidence_path_count=11,
        evidence_path_count=12,
        repeated_call_pairs=(scores[:2], scores[1:]),
        cross_model_pairs=(scores[:1], scores[2:]),
        boundary_epsilon=0.05,
    )

    assert metrics["parse_rate"] == 0.75
    assert metrics["retry_rate"] == 0.25
    assert metrics["forbidden_content_rate"] == 0.0
    assert metrics["evidence_path_valid_rate"] == pytest.approx(11 / 12)
    assert metrics["repeated_call_mean_absolute_disagreement"] == 0.5
    assert metrics["cross_model_mean_absolute_disagreement"] == 1.0
    assert metrics["per_dimension"]["mission_urgency"] == {
        "mean": 0.5,
        "standard_deviation": pytest.approx(np.std([0.0, 0.5, 1.0])),
        "boundary_saturation_rate": pytest.approx(2 / 3),
    }


def test_label_metrics_report_missing_matched_pairs_as_null_not_zero():
    scores = np.asarray([[0.25] * 4])
    metrics = label_diagnostic_metrics(
        scores=scores,
        requested_samples=1,
        parsed_samples=1,
        retry_count=0,
        forbidden_response_count=0,
        valid_evidence_path_count=4,
        evidence_path_count=4,
        repeated_call_pairs=None,
        cross_model_pairs=None,
        boundary_epsilon=0.05,
    )
    assert metrics["repeated_call_mean_absolute_disagreement"] is None
    assert metrics["cross_model_mean_absolute_disagreement"] is None


def test_label_metrics_record_total_parse_failure_without_crashing():
    metrics = label_diagnostic_metrics(
        scores=np.empty((0, 4)),
        requested_samples=2,
        parsed_samples=0,
        retry_count=2,
        forbidden_response_count=0,
        valid_evidence_path_count=0,
        evidence_path_count=0,
        repeated_call_pairs=None,
        cross_model_pairs=None,
        boundary_epsilon=0.05,
    )
    assert metrics["parse_rate"] == 0.0
    assert metrics["evidence_path_valid_rate"] is None
    assert metrics["per_dimension"]["mission_urgency"] == {
        "mean": None,
        "standard_deviation": None,
        "boundary_saturation_rate": None,
    }


def test_label_metrics_reject_count_that_disagrees_with_score_rows():
    with pytest.raises(ValueError, match="parsed_samples"):
        label_diagnostic_metrics(
            scores=np.asarray([[0.5] * 4]),
            requested_samples=2,
            parsed_samples=2,
            retry_count=0,
            forbidden_response_count=0,
            valid_evidence_path_count=4,
            evidence_path_count=4,
            repeated_call_pairs=None,
            cross_model_pairs=None,
            boundary_epsilon=0.05,
        )


def test_seeded_controls_preserve_labels_padding_and_final_frames():
    indices = shuffled_label_indices(sample_count=8, seed=17)
    assert sorted(indices.tolist()) == list(range(8))
    assert np.all(indices != np.arange(8))
    assert np.array_equal(indices, shuffled_label_indices(8, 17))

    sequences = np.arange(2 * 5 * 3).reshape(2, 5, 3)
    valid = np.asarray(
        [[False, True, True, True, True], [False, False, True, True, True]]
    )
    shuffled = shuffled_history_control(sequences, valid, seed=19)
    assert np.array_equal(shuffled[:, -1], sequences[:, -1])
    assert np.array_equal(shuffled[~valid], sequences[~valid])
    assert np.array_equal(shuffled, shuffled_history_control(sequences, valid, 19))
    for sample in range(2):
        prefix = np.flatnonzero(valid[sample])[:-1]
        assert sorted(shuffled[sample, prefix, 0].tolist()) == sorted(
            sequences[sample, prefix, 0].tolist()
        )


def test_single_frame_control_removes_every_earlier_valid_frame():
    sequences = np.ones((2, 4, 3), dtype=np.float32)
    valid = np.asarray([[False, True, True, True], [True, True, True, True]])
    controlled, controlled_mask = single_frame_control(sequences, valid)
    assert np.count_nonzero(controlled[:, :-1]) == 0
    assert np.array_equal(controlled[:, -1], sequences[:, -1])
    assert np.array_equal(
        controlled_mask,
        np.asarray([[False, False, False, True], [False, False, False, True]]),
    )


def test_fresh_probe_is_new_linear_64_to_4_with_sigmoid_each_time():
    first = build_fresh_probe(seed=31)
    second = build_fresh_probe(seed=32)
    assert isinstance(first[0], torch.nn.Linear)
    assert (first[0].in_features, first[0].out_features) == (64, 4)
    assert isinstance(first[1], torch.nn.Sigmoid)
    assert first is not second
    assert not torch.equal(first[0].weight, second[0].weight)


def test_representation_health_exposes_collapse_gradients_and_round_trip():
    embeddings = np.eye(8, 64, dtype=np.float64)
    metrics = representation_health_metrics(
        embeddings=embeddings,
        temporal_control_embeddings=np.roll(embeddings, 1, axis=0),
        imported_embeddings=embeddings.copy(),
        encoder_gradient_norm=0.2,
        gru_gradient_norm=0.3,
    )

    assert metrics["non_finite_count"] == 0
    assert metrics["mean_feature_variance"] > 0
    assert metrics["effective_rank"] == pytest.approx(7.0)
    assert metrics["temporal_sensitivity_mean_l2"] > 0
    assert metrics["import_export_max_absolute_error"] == 0.0
    assert metrics["encoder_gradient_norm"] == 0.2
    assert metrics["gru_gradient_norm"] == 0.3


def test_non_finite_embeddings_are_counted_without_emitting_invalid_metrics():
    embeddings = np.zeros((2, 64))
    embeddings[0, 0] = np.nan
    metrics = representation_health_metrics(
        embeddings=embeddings,
        temporal_control_embeddings=np.zeros((2, 64)),
        imported_embeddings=np.zeros((2, 64)),
        encoder_gradient_norm=0.0,
        gru_gradient_norm=0.0,
    )
    assert metrics["non_finite_count"] == 1
    assert metrics["effective_rank"] is None
    assert metrics["mean_feature_variance"] is None


def test_downstream_report_requires_exact_four_arms_and_disclaims_proof():
    arms = {
        name: {"seed": 7, "metrics": {"tasks": index}}
        for index, name in enumerate(
            (
                "baseline",
                "llm_kd",
                "astar_kd_no_validity",
                "astar_kd_no_validity_plus_llm_kd",
            )
        )
    }
    report = build_downstream_diagnostic_report(arms)
    assert report["design"] == "2x2-llm-initialization-by-astar-kd"
    assert report["claim_scope"] == "diagnostic-only-not-formal-performance-proof"
    assert report["arms"] == arms
    with pytest.raises(ValueError, match="exact four arms"):
        build_downstream_diagnostic_report({"baseline": arms["baseline"]})


def test_all_five_run_config_types_reject_missing_extra_and_wrong_types():
    assert set(CONFIG_SCHEMAS) == {
        "label_audit",
        "dataset_build",
        "pretraining",
        "representation_diagnostics",
        "downstream_diagnostics",
    }
    for kind, schema in CONFIG_SCHEMAS.items():
        complete = {name: _value_for(expected) for name, expected in schema.items()}
        validate_explicit_config(kind, complete)
        missing = dict(complete)
        missing.pop(next(iter(schema)))
        with pytest.raises(ValueError, match="missing"):
            validate_explicit_config(kind, missing)
        with pytest.raises(ValueError, match="extra"):
            validate_explicit_config(kind, {**complete, "hidden_default": 1})
        wrong = dict(complete)
        integer_field = next(
            name for name, expected in schema.items() if expected is int
        )
        wrong[integer_field] = True
        with pytest.raises(ValueError, match="exact type"):
            validate_explicit_config(kind, wrong)


def test_downstream_diagnostics_rejects_collection_behavior_checkpoint():
    schema = CONFIG_SCHEMAS["downstream_diagnostics"]
    complete = {name: _value_for(expected) for name, expected in schema.items()}

    with pytest.raises(ValueError, match="extra"):
        validate_explicit_config(
            "downstream_diagnostics",
            {**complete, "behavior_policy_checkpoint": "collection.pt"},
        )


def test_m1_config_hash_is_stable_and_freezes_probe_and_test_access():
    first = load_m1_config(M1_CONFIG)
    second = load_m1_config(M1_CONFIG)
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64
    assert first.values["schema_version"] == "phase3-m1-diagnostics-v1"
    assert first.values["test_access"] == {
        "split_unit": "episode",
        "selection_partition": "validation",
        "test_evaluations_per_representation_seed": 1,
        "test_access_after_model_and_threshold_freeze": True,
    }
    assert first.values["probe"]["architecture"] == "Linear(64,4)+Sigmoid"
    assert first.values["controls"] == [
        "constant_predictor",
        "random_init",
        "shuffled_label",
        "shuffled_history",
        "single_frame",
        "normal_history",
    ]


def test_m1_approval_matches_the_exact_resolved_config_hash():
    approval = verify_m1_approval(M1_CONFIG, M1_APPROVAL)
    assert approval["gate"] == "M1"
    assert approval["status"] == "approved"
    assert approval["approved_config_sha256"] == load_m1_config(M1_CONFIG).sha256


def test_m1_loader_rejects_missing_or_extra_nested_contract_fields(tmp_path):
    values = yaml.safe_load(Path(M1_CONFIG).read_text(encoding="utf-8"))
    del values["probe"]["batch_size"]
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="probe.*missing"):
        load_m1_config(incomplete)

    values = yaml.safe_load(Path(M1_CONFIG).read_text(encoding="utf-8"))
    values["test_access"]["peek_at_test"] = True
    extra = tmp_path / "extra.yaml"
    extra.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="test_access.*extra"):
        load_m1_config(extra)

    values = yaml.safe_load(Path(M1_CONFIG).read_text(encoding="utf-8"))
    values["go_no_go_thresholds"]["label"]["parse_rate_min"] = "0.98"
    wrong_type = tmp_path / "wrong-type.yaml"
    wrong_type.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold.*numeric"):
        load_m1_config(wrong_type)


def _value_for(expected):
    return {
        str: "value",
        int: 1,
        float: 0.5,
        bool: True,
        list: ["value"],
        dict: {"value": 1},
    }[expected]
