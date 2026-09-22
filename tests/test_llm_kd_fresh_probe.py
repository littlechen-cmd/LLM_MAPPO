"""Fixture-only tests for the frozen-representation fresh probe."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from hashlib import sha256
import json

import pytest
import torch

from llm_mappo.mappo_llm_kd import fresh_probe as fresh_probe_module
from llm_mappo.mappo_llm_kd.fresh_probe import (
    train_validation_selected_fresh_probe,
)
from llm_mappo.mappo_llm_kd import representation_diagnostics
from llm_mappo.mappo_llm_kd.representation_diagnostics import (
    create_diagnostic_run_identity,
    create_fresh_probe_training_receipt,
)


M1_CONFIG = "configs/llm_kd/representation-diagnostics.yaml"
M1_APPROVAL = "configs/llm_kd/representation-diagnostics-approval.json"
SHA = "a" * 64


def _fixture_partition():
    """Build non-constant, hand-checkable 64D embeddings and four scores."""

    scores = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    embeddings = torch.zeros((scores.shape[0], 64), dtype=torch.float32)
    embeddings[:, :4] = scores.mul(16.0).sub(8.0)
    return embeddings, scores


def _identity():
    return create_diagnostic_run_identity(
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        dataset_sha256=SHA,
        train_episode_ids=("train-001",),
        validation_episode_ids=("validation-001",),
        test_episode_ids=("test-001",),
        representation_seeds=(7301, 7302, 7303),
    )


def _training_receipt():
    creator = getattr(
        representation_diagnostics,
        "create_fresh_probe_training_receipt",
        None,
    )
    assert callable(creator), "The compact fresh-probe receipt boundary must exist."
    return creator(
        diagnostic_identity=_identity(),
        representation_variant="normal_history",
        representation_seed=7301,
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
    )


def test_fresh_probe_freezes_embeddings_and_selects_using_validation_loss():
    """Catch a probe that backpropagates into representations or skips validation."""

    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()
    train_embeddings.requires_grad_(True)
    validation_embeddings.requires_grad_(True)
    original_train = train_embeddings.detach().clone()
    original_validation = validation_embeddings.detach().clone()

    result = train_validation_selected_fresh_probe(
        representation_variant="normal_history",
        representation_seed=7301,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        validation_embeddings=validation_embeddings,
        validation_labels=validation_labels,
        training_receipt=_training_receipt(),
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        device="cpu",
    )

    constant = train_labels.mean(dim=0)
    constant_validation_mse = torch.mean(
        torch.square(validation_labels - constant)
    ).item()
    assert result.validation_predictions.shape == (12, 4)
    assert result.validation_mse < constant_validation_mse
    assert result.selected_epoch < result.epochs_ran <= 100
    assert result.dataset_sha256 == SHA
    assert result.probe.training is False
    assert all(not parameter.requires_grad for parameter in result.probe.parameters())
    assert torch.equal(train_embeddings.detach(), original_train)
    assert torch.equal(validation_embeddings.detach(), original_validation)
    assert train_embeddings.grad is None
    assert validation_embeddings.grad is None


def test_training_receipt_excludes_test_partition_metadata():
    """Catch a probe receipt that carries test episode metadata downstream."""

    receipt = _training_receipt()

    assert set(receipt) == {
        "schema_version",
        "diagnostic_run_sha256",
        "m1_config_sha256",
        "dataset_sha256",
        "representation_variant",
        "representation_seed",
        "receipt_sha256",
    }
    assert "episode_partitions" not in receipt
    assert "test" not in receipt


def test_fresh_probe_rejects_a_tampered_training_receipt():
    """Catch probe use that accepts an altered compact training receipt."""

    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()

    receipt = dict(_training_receipt())
    receipt["representation_variant"] = "random_init"

    with pytest.raises(ValueError, match="receipt"):
        train_validation_selected_fresh_probe(
            representation_variant="normal_history",
            representation_seed=7301,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            validation_embeddings=validation_embeddings,
            validation_labels=validation_labels,
            training_receipt=receipt,
            m1_config_path=M1_CONFIG,
            m1_approval_path=M1_APPROVAL,
            device="cpu",
        )


def test_fresh_probe_rejects_a_self_consistent_forged_training_receipt():
    """Catch a plain mapping that recomputes a hash without full-ID validation."""

    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()
    receipt = dict(_training_receipt())
    receipt["dataset_sha256"] = "b" * 64
    hashed = {
        name: value for name, value in receipt.items() if name != "receipt_sha256"
    }
    receipt["receipt_sha256"] = sha256(
        (
            json.dumps(hashed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="issued"):
        train_validation_selected_fresh_probe(
            representation_variant="normal_history",
            representation_seed=7301,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            validation_embeddings=validation_embeddings,
            validation_labels=validation_labels,
            training_receipt=receipt,
            m1_config_path=M1_CONFIG,
            m1_approval_path=M1_APPROVAL,
            device="cpu",
        )


def test_training_receipt_factory_rejects_a_non_integer_seed():
    """Catch Python numeric equality admitting a float as an approved seed."""

    with pytest.raises(ValueError, match="seed"):
        create_fresh_probe_training_receipt(
            diagnostic_identity=_identity(),
            representation_variant="normal_history",
            representation_seed=7301.0,
            m1_config_path=M1_CONFIG,
            m1_approval_path=M1_APPROVAL,
        )


def test_fresh_probe_rejects_an_unissued_copy_of_a_training_receipt():
    """Catch an object copy that inherits fields without factory issuance."""

    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()
    copied_receipt = copy.copy(_training_receipt())

    with pytest.raises(ValueError, match="issued"):
        train_validation_selected_fresh_probe(
            representation_variant="normal_history",
            representation_seed=7301,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            validation_embeddings=validation_embeddings,
            validation_labels=validation_labels,
            training_receipt=copied_receipt,
            m1_config_path=M1_CONFIG,
            m1_approval_path=M1_APPROVAL,
            device="cpu",
        )


def test_fresh_probe_rejects_an_unissued_deepcopy_of_a_training_receipt():
    """Catch a deep-copied capability that was never factory-issued."""

    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()
    copied_receipt = copy.deepcopy(_training_receipt())

    with pytest.raises(ValueError, match="issued"):
        train_validation_selected_fresh_probe(
            representation_variant="normal_history",
            representation_seed=7301,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            validation_embeddings=validation_embeddings,
            validation_labels=validation_labels,
            training_receipt=copied_receipt,
            m1_config_path=M1_CONFIG,
            m1_approval_path=M1_APPROVAL,
            device="cpu",
        )


def test_fresh_probe_protects_all_visible_cuda_rng_states_on_cpu(
    monkeypatch,
):
    """Catch CPU probe initialization that still mutates another GPU's RNG."""

    forked_devices = []

    @contextmanager
    def recording_fork_rng(*, devices):
        forked_devices.append(list(devices))
        yield

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.random, "fork_rng", recording_fork_rng)
    monkeypatch.setattr(fresh_probe_module.torch, "manual_seed", lambda seed: None)
    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()

    train_validation_selected_fresh_probe(
        representation_variant="normal_history",
        representation_seed=7301,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        validation_embeddings=validation_embeddings,
        validation_labels=validation_labels,
        training_receipt=_training_receipt(),
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        device="cpu",
    )

    assert forked_devices == [[0, 1]]


def test_fresh_probe_preserves_real_cpu_rng_state():
    """Catch training that changes the caller's actual CPU RNG stream."""

    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()
    torch.manual_seed(481516)
    expected_rng_state = torch.random.get_rng_state().clone()

    train_validation_selected_fresh_probe(
        representation_variant="normal_history",
        representation_seed=7301,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        validation_embeddings=validation_embeddings,
        validation_labels=validation_labels,
        training_receipt=_training_receipt(),
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        device="cpu",
    )

    assert torch.equal(torch.random.get_rng_state(), expected_rng_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_fresh_probe_preserves_real_visible_cuda_rng_states():
    """Catch CPU probe initialization that changes a real CUDA RNG stream."""

    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()
    expected_rng_states = [
        state.clone() for state in torch.cuda.get_rng_state_all()
    ]

    train_validation_selected_fresh_probe(
        representation_variant="normal_history",
        representation_seed=7301,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        validation_embeddings=validation_embeddings,
        validation_labels=validation_labels,
        training_receipt=_training_receipt(),
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        device="cpu",
    )

    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(torch.cuda.get_rng_state_all(), expected_rng_states)
    )


def test_fresh_probe_rejects_non_float32_embeddings_before_training():
    """Catch implicit dtype mismatches between diagnostic embeddings and probe."""

    train_embeddings, train_labels = _fixture_partition()
    validation_embeddings, validation_labels = _fixture_partition()

    with pytest.raises(ValueError, match="float32"):
        train_validation_selected_fresh_probe(
            representation_variant="normal_history",
            representation_seed=7301,
            train_embeddings=train_embeddings.double(),
            train_labels=train_labels.double(),
            validation_embeddings=validation_embeddings.double(),
            validation_labels=validation_labels.double(),
            training_receipt=_training_receipt(),
            m1_config_path=M1_CONFIG,
            m1_approval_path=M1_APPROVAL,
            device="cpu",
        )


def test_fresh_probe_keeps_the_validation_selected_state_when_losses_conflict():
    """Catch accidental selection of the ever-improving training-loss state."""

    train_embeddings = torch.zeros((12, 64), dtype=torch.float32)
    validation_embeddings = torch.zeros((12, 64), dtype=torch.float32)
    train_labels = torch.zeros((12, 4), dtype=torch.float32)
    validation_labels = torch.ones((12, 4), dtype=torch.float32)

    result = train_validation_selected_fresh_probe(
        representation_variant="normal_history",
        representation_seed=7301,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        validation_embeddings=validation_embeddings,
        validation_labels=validation_labels,
        training_receipt=_training_receipt(),
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        device="cpu",
    )

    returned_mse = torch.mean(
        torch.square(result.validation_predictions - validation_labels)
    ).item()
    assert result.selected_epoch == 0
    assert result.epochs_ran == 11
    assert result.validation_mse == pytest.approx(returned_mse)
