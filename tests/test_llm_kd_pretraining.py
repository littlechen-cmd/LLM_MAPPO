import pytest
import torch

from llm_mappo.mappo_llm_kd.pretraining import (
    ACTOR_ARCHITECTURE_V1,
    HIDDEN_DIM,
    OBSERVATION_DIM,
    PRETRAINING_CHECKPOINT_V1,
    PRETRAINING_EXPORT_V1,
    PRETRAINING_RUN_V1,
    REQUIRED_PRETRAINING_IDENTITY_FIELDS,
    SEMANTIC_DIMENSIONS,
    SEMANTIC_DIM,
    SEMANTIC_ONTOLOGY_V1,
    PretrainingBatch,
    SemanticPretrainingModel,
    masked_semantic_loss,
    validate_pretraining_identity,
    validate_prediction_tensor,
)
from llm_mappo.mappo_baseline.networks import MAPPOActor


def _batch(**overrides):
    values = {
        "observations": torch.zeros(2, 3, 209),
        "valid_masks": torch.ones(2, 3, dtype=torch.bool),
        "reset_masks": torch.zeros(2, 3, dtype=torch.bool),
        "targets": torch.full((2, 3, 4), 0.5),
        "initial_hidden": torch.zeros(2, 64),
    }
    values.update(overrides)
    return PretrainingBatch(**values)


def test_pretraining_contract_versions_and_identity_are_explicit():
    assert PRETRAINING_RUN_V1 == "semantic-pretraining-run-v1"
    assert PRETRAINING_CHECKPOINT_V1 == "semantic-pretraining-checkpoint-v1"
    assert PRETRAINING_EXPORT_V1 == "actor-representation-export-v1"
    assert SEMANTIC_DIMENSIONS == (
        "mission_urgency",
        "concession_appropriateness",
        "interaction_risk",
        "deadlock_severity",
    )


def test_pretraining_identity_requires_exact_compatible_fields():
    identity = {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": 209,
        "ontology_schema": "agent-centric-four-score-ontology-v1",
        "semantic_dimensions": list(SEMANTIC_DIMENSIONS),
        "actor_architecture": "mappo-actor-encoder-gru-v1",
        "hidden_dim": 64,
        "dataset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }
    validate_pretraining_identity(identity)
    with pytest.raises(ValueError, match="fields and order"):
        validate_pretraining_identity({**identity, "extra": "forbidden"})
    incompatible = {**identity, "observation_dim": 208}
    with pytest.raises(ValueError, match="observation_dim"):
        validate_pretraining_identity(incompatible)
    malformed_hash = {**identity, "dataset_sha256": "A" * 64}
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_pretraining_identity(malformed_hash)
    assert REQUIRED_PRETRAINING_IDENTITY_FIELDS == (
        "observation_schema",
        "observation_dim",
        "ontology_schema",
        "semantic_dimensions",
        "actor_architecture",
        "hidden_dim",
        "dataset_sha256",
        "config_sha256",
    )


def test_pretraining_batch_accepts_only_the_frozen_tensor_contract():
    batch = _batch()
    assert batch.batch_size == 2
    assert batch.time_steps == 3
    assert batch.observations.shape == (2, 3, 209)
    assert batch.targets.shape == (2, 3, 4)
    assert batch.initial_hidden.shape == (2, 64)
    validate_prediction_tensor(torch.zeros(2, 3, 4), batch)


def test_pretraining_batch_keeps_recurrent_history_separate_from_final_targets():
    target_masks = torch.tensor([[False, False, True], [False, False, True]])
    batch = _batch(target_masks=target_masks)
    predictions = torch.tensor(
        [[[1.0] * 4, [1.0] * 4, [0.0] * 4]] * 2
    )
    targets = torch.tensor(
        [[[0.0] * 4, [0.0] * 4, [1.0] * 4]] * 2
    )

    loss = masked_semantic_loss(
        predictions, targets, batch.target_masks, torch.ones(4)
    )

    assert torch.equal(batch.valid_masks, torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(batch.target_masks, target_masks)
    assert loss.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observations", torch.zeros(2, 209), "rank 3"),
        ("observations", torch.zeros(2, 3, 208), "209"),
        ("valid_masks", torch.ones(2, 3), "boolean"),
        ("reset_masks", torch.zeros(2, 3), "boolean"),
        ("valid_masks", torch.ones(2, 2, dtype=torch.bool), "batch/time"),
        ("reset_masks", torch.zeros(2, 2, dtype=torch.bool), "batch/time"),
        ("targets", torch.zeros(2, 4), "rank 3"),
        ("targets", torch.zeros(2, 3, 3), "four"),
        ("targets", torch.zeros(2, 2, 4), "batch/time"),
        ("initial_hidden", torch.zeros(2, 1, 64), "rank 2"),
        ("initial_hidden", torch.zeros(2, 63), "64"),
        ("initial_hidden", torch.zeros(3, 64), "batch"),
    ],
)
def test_pretraining_batch_rejects_wrong_rank_dimension_or_axes(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        _batch(**{field: value})


def test_pretraining_batch_requires_boolean_masks_and_valid_final_frames():
    valid = torch.tensor([[True, True, False], [True, False, False]])
    with pytest.raises(ValueError, match="final frame"):
        _batch(valid_masks=valid)


def test_pretraining_batch_rejects_empty_or_noncontiguous_sequences():
    with pytest.raises(ValueError, match="non-empty"):
        _batch(
            observations=torch.zeros(0, 3, 209),
            valid_masks=torch.zeros(0, 3, dtype=torch.bool),
            reset_masks=torch.zeros(0, 3, dtype=torch.bool),
            targets=torch.zeros(0, 3, 4),
            initial_hidden=torch.zeros(0, 64),
        )
    with pytest.raises(ValueError, match="contiguous"):
        _batch(valid_masks=torch.tensor([[True, False, True], [True, True, True]]))


@pytest.mark.parametrize("field", ["observations", "targets", "initial_hidden"])
def test_pretraining_batch_rejects_non_finite_floating_tensors(field):
    value = getattr(_batch(), field).clone()
    value.reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _batch(**{field: value})


def test_pretraining_batch_rejects_targets_outside_unit_interval():
    targets = torch.zeros(2, 3, 4)
    targets[0, 0, 0] = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _batch(targets=targets)


@pytest.mark.parametrize(
    "prediction",
    [
        torch.zeros(2, 4),
        torch.zeros(2, 3, 3),
        torch.zeros(2, 2, 4),
        torch.full((2, 3, 4), float("inf")),
    ],
)
def test_prediction_contract_rejects_wrong_or_non_finite_output(prediction):
    with pytest.raises(ValueError):
        validate_prediction_tensor(prediction, _batch())


def test_public_package_exports_all_frozen_pretraining_constants():
    import llm_mappo.mappo_llm_kd as llm_kd

    expected = {
        "ACTOR_ARCHITECTURE_V1": ACTOR_ARCHITECTURE_V1,
        "HIDDEN_DIM": HIDDEN_DIM,
        "OBSERVATION_DIM": OBSERVATION_DIM,
        "SEMANTIC_DIM": SEMANTIC_DIM,
        "SEMANTIC_ONTOLOGY_V1": SEMANTIC_ONTOLOGY_V1,
    }
    for name, value in expected.items():
        assert getattr(llm_kd, name) == value
        assert name in llm_kd.__all__


def test_semantic_model_wraps_the_supplied_actor_and_only_adds_linear_head():
    actor = MAPPOActor(observation_dim=209, action_dim=5, hidden_dim=64)
    model = SemanticPretrainingModel(actor)

    assert model.actor is actor
    assert isinstance(model.semantic_head, torch.nn.Linear)
    assert model.semantic_head.in_features == 64
    assert model.semantic_head.out_features == 4
    assert not hasattr(model, "critic")


def test_semantic_model_adapts_batch_first_and_preserves_episode_resets():
    torch.manual_seed(101)
    actor = MAPPOActor(observation_dim=209, action_dim=5, hidden_dim=64)
    model = SemanticPretrainingModel(actor)
    observations = torch.randn(2, 3, 209)
    resets = torch.tensor([[False, True, False], [False, False, False]])
    batch = _batch(
        observations=observations,
        reset_masks=resets,
        initial_hidden=torch.randn(2, 64),
    )

    predictions = model(batch)
    encoded, _ = actor._sequence(
        observations.transpose(0, 1),
        batch.initial_hidden,
        (~resets).to(observations.dtype).transpose(0, 1),
    )
    expected = torch.sigmoid(model.semantic_head(encoded)).transpose(0, 1)

    assert predictions.shape == (2, 3, 4)
    torch.testing.assert_close(predictions, expected)


def test_left_padding_never_enters_encoder_or_gru():
    torch.manual_seed(103)
    actor = MAPPOActor(observation_dim=209, action_dim=5, hidden_dim=64)
    model = SemanticPretrainingModel(actor)
    valid = torch.tensor([[False, False, True, True], [False, True, True, True]])
    observations = torch.randn(2, 4, 209)
    altered = observations.clone()
    altered[~valid] = 1.0e6
    reset_masks = torch.tensor(
        [[False, False, True, False], [False, True, False, False]]
    )
    values = {
        "valid_masks": valid,
        "reset_masks": reset_masks,
        "targets": torch.full((2, 4, 4), 0.5),
        "initial_hidden": torch.randn(2, 64),
    }

    original_predictions = model(_batch(observations=observations, **values))
    altered_predictions = model(_batch(observations=altered, **values))

    torch.testing.assert_close(
        original_predictions[valid], altered_predictions[valid]
    )
    assert torch.equal(original_predictions[~valid], torch.zeros(3, 4))


def test_masked_semantic_loss_ignores_padding_and_applies_dimension_weights():
    predictions = torch.tensor(
        [[[99.0, 99.0, 99.0, 99.0], [0.0, 0.5, 1.0, 0.5]]]
    )
    targets = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.0, 1.0]]]
    )
    valid = torch.tensor([[False, True]])
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])

    loss = masked_semantic_loss(predictions, targets, valid, weights)

    assert loss.item() == pytest.approx((1.0 + 0.0 + 3.0 + 1.0) / 10.0)


def test_masked_semantic_loss_does_not_compute_through_padded_values():
    predictions = torch.tensor(
        [[[float("nan")] * 4, [0.2, 0.4, 0.6, 0.8]]],
        requires_grad=True,
    )
    targets = torch.zeros(1, 2, 4)
    valid = torch.tensor([[False, True]])

    loss = masked_semantic_loss(predictions, targets, valid, torch.ones(4))
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(predictions.grad).all()
    assert torch.equal(predictions.grad[~valid], torch.zeros(1, 4))


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (torch.ones(3), "four"),
        (torch.tensor([1.0, -1.0, 1.0, 1.0]), "non-negative"),
        (torch.zeros(4), "positive total"),
        (torch.tensor([1.0, 1.0, float("nan"), 1.0]), "finite"),
    ],
)
def test_masked_semantic_loss_rejects_invalid_dimension_weights(weights, message):
    with pytest.raises(ValueError, match=message):
        masked_semantic_loss(
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 4),
            torch.ones(1, 1, dtype=torch.bool),
            weights,
        )


def test_tiny_cpu_step_updates_head_gru_and_encoder_with_finite_gradients():
    torch.manual_seed(107)
    actor = MAPPOActor(observation_dim=209, action_dim=5, hidden_dim=64)
    model = SemanticPretrainingModel(actor)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _batch(
        observations=torch.randn(2, 3, 209),
        valid_masks=torch.tensor([[False, True, True], [True, True, True]]),
        reset_masks=torch.tensor([[False, True, False], [True, False, False]]),
        targets=torch.tensor(
            [
                [[0.0] * 4, [0.1, 0.3, 0.7, 0.9], [0.9, 0.7, 0.3, 0.1]],
                [[0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2], [0.3] * 4],
            ]
        ),
    )
    watched = {
        "head": model.semantic_head.weight,
        "gru": actor.gru.weight_ih_l0,
        "encoder": actor.encoder[1].weight,
    }
    before = {name: parameter.detach().clone() for name, parameter in watched.items()}

    predictions = model(batch)
    loss = masked_semantic_loss(
        predictions, batch.targets, batch.valid_masks, torch.ones(4)
    )
    optimizer.zero_grad()
    loss.backward()

    for parameter in watched.values():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    optimizer.step()
    for name, parameter in watched.items():
        assert not torch.equal(before[name], parameter.detach())
