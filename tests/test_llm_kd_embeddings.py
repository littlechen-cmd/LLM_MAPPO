from __future__ import annotations

import copy
from hashlib import sha256
import random

import numpy as np
import pytest
import torch
from torch import nn

from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_llm_kd.representation_diagnostics import (
    create_diagnostic_run_identity,
)
from llm_mappo.mappo_llm_kd.representation_embeddings import (
    FrozenRecurrentEmbeddingExtractor,
    embedding_input_variant,
)
from llm_mappo.mappo_llm_kd.representation_export import (
    export_actor_representation,
)


M1_CONFIG = "configs/llm_kd/representation-diagnostics.yaml"
M1_APPROVAL = "configs/llm_kd/representation-diagnostics-approval.json"
SHA = "a" * 64


def _artifact_identity():
    return {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": 209,
        "ontology_schema": "agent-centric-four-score-ontology-v1",
        "semantic_dimensions": [
            "mission_urgency",
            "concession_appropriateness",
            "interaction_risk",
            "deadlock_severity",
        ],
        "actor_architecture": "mappo-actor-encoder-gru-v1",
        "hidden_dim": 64,
        "dataset_sha256": SHA,
        "config_sha256": "b" * 64,
    }


def _diagnostic_identity():
    return create_diagnostic_run_identity(
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        dataset_sha256=SHA,
        train_episode_ids=("train",),
        validation_episode_ids=("validation",),
        test_episode_ids=("test",),
        representation_seeds=(7301, 7302, 7303),
    )


def _actor(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return MAPPOActor(209, 5, 64)


def _batch():
    observations = torch.arange(3 * 5 * 209, dtype=torch.float32).reshape(3, 5, 209)
    observations /= observations.max()
    valid = torch.tensor(
        [[False, False, True, True, True],
         [False, True, True, True, True],
         [True, True, True, True, True]]
    )
    resets = torch.tensor(
        [[False, False, True, False, False],
         [False, True, False, True, False],
         [True, False, False, False, False]]
    )
    hidden = torch.linspace(-0.2, 0.2, 3 * 64).reshape(3, 64)
    return observations, valid, resets, hidden


def _snapshot(actor):
    return {
        key: value.detach().clone()
        for key, value in actor.state_dict().items()
    }


def _assert_snapshot(actor, expected):
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def _assert_numpy_rng_equal(left, right):
    assert left[0] == right[0]
    assert np.array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _side_effect_snapshot(actor, inputs):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
        "state": _snapshot(actor),
        "training": {name: module.training for name, module in actor.named_modules()},
        "parameters": {
            name: (
                parameter.requires_grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for name, parameter in actor.named_parameters()
        },
        "inputs": [value.detach().clone() for value in inputs],
        "input_meta": [
            (
                tuple(value.shape),
                value.dtype,
                value.device,
                value.requires_grad,
                None if value.grad is None else value.grad.detach().clone(),
            )
            for value in inputs
        ],
    }


def _assert_side_effect_snapshot(actor, inputs, expected):
    assert random.getstate() == expected["python"]
    _assert_numpy_rng_equal(np.random.get_state(), expected["numpy"])
    assert torch.equal(torch.random.get_rng_state(), expected["torch"])
    for actual, wanted in zip(torch.cuda.get_rng_state_all(), expected["cuda"]):
        assert torch.equal(actual, wanted)
    _assert_snapshot(actor, expected["state"])
    assert {
        name: module.training for name, module in actor.named_modules()
    } == expected["training"]
    for name, parameter in actor.named_parameters():
        requires_grad, grad = expected["parameters"][name]
        assert parameter.requires_grad is requires_grad
        if grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, grad, rtol=0, atol=0)
    for actual, wanted, metadata in zip(
        inputs, expected["inputs"], expected["input_meta"]
    ):
        shape, dtype, device, requires_grad, grad = metadata
        assert (tuple(actual.shape), actual.dtype, actual.device) == (
            shape, dtype, device
        )
        assert actual.requires_grad is requires_grad
        if grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, grad, rtol=0, atol=0)
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


def _from_artifact(tmp_path, seed=7301):
    source = _actor(11)
    path = tmp_path / "fixture-representation.pt"
    receipt = export_actor_representation(path, source, identity=_artifact_identity())
    target = _actor(12)
    extractor = FrozenRecurrentEmbeddingExtractor.from_artifact(
        path,
        target,
        expected_artifact_identity=_artifact_identity(),
        expected_artifact_sha256=receipt["file_sha256"],
        diagnostic_identity=_diagnostic_identity(),
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        representation_seed=seed,
    )
    return extractor, source, receipt


def test_shuffled_label_is_explicitly_label_only_and_never_an_embedding_transform():
    assert embedding_input_variant("shuffled_label") == "normal_history"
    assert embedding_input_variant("normal_history") == "normal_history"
    with pytest.raises(ValueError, match="embedding-input variant"):
        embedding_input_variant("constant_predictor")


def test_authenticated_artifact_is_bound_to_diagnostic_identity(tmp_path):
    extractor, _, receipt = _from_artifact(tmp_path)

    assert extractor.identity["artifact_file_sha256"] == receipt["file_sha256"]
    assert extractor.identity["diagnostic_run_sha256"] == sha256(
        __import__("json").dumps(
            _diagnostic_identity(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()

    wrong = copy.deepcopy(_artifact_identity())
    wrong["dataset_sha256"] = "c" * 64
    source = _actor(13)
    path = tmp_path / "wrong.pt"
    receipt = export_actor_representation(path, source, identity=wrong)
    with pytest.raises(ValueError, match="dataset identity"):
        FrozenRecurrentEmbeddingExtractor.from_artifact(
            path,
            _actor(14),
            expected_artifact_identity=wrong,
            expected_artifact_sha256=receipt["file_sha256"],
            diagnostic_identity=_diagnostic_identity(),
            m1_config_path=M1_CONFIG,
            m1_approval_path=M1_APPROVAL,
            representation_seed=7301,
        )


def test_direct_construction_cannot_bypass_artifact_or_random_init_factory():
    with pytest.raises(ValueError, match="factory"):
        FrozenRecurrentEmbeddingExtractor(_actor(15), {"source": "forged"})


def test_normal_history_uses_exact_actor_sequence_and_correct_padding_resets(tmp_path):
    extractor, source, _ = _from_artifact(tmp_path)
    observations, valid, resets, hidden = _batch()
    result = extractor.extract(
        observations, valid, resets, hidden,
        variant="normal_history", include_all_valid=True,
    )

    expected = torch.zeros(3, 5, 64)
    with torch.no_grad():
        for row in range(3):
            first = int(torch.nonzero(valid[row])[0])
            encoded, _ = source._sequence(
                observations[row, first:].unsqueeze(1),
                hidden[row].unsqueeze(0),
                (~resets[row, first:]).float().unsqueeze(1),
            )
            expected[row, first:] = encoded.squeeze(1)
    torch.testing.assert_close(result.final_embeddings, expected[:, -1])
    torch.testing.assert_close(result.all_embeddings, expected)
    assert torch.equal(result.valid_masks, valid)
    assert result.final_embeddings.shape == (3, 64)
    assert result.final_embeddings.grad_fn is None


@pytest.mark.parametrize("variant", ["normal_history", "shuffled_history", "single_frame"])
def test_controls_preserve_shape_masks_seed_and_never_mutate_state_or_rng(
    tmp_path, variant
):
    extractor, _, _ = _from_artifact(tmp_path)
    observations, valid, resets, hidden = _batch()
    before_state = _snapshot(extractor.actor)
    torch.manual_seed(991)
    before_rng = torch.random.get_rng_state().clone()

    first = extractor.extract(
        observations, valid, resets, hidden,
        variant=variant, include_all_valid=True,
    )
    second = extractor.extract(
        observations, valid, resets, hidden,
        variant=variant, include_all_valid=True,
    )

    torch.testing.assert_close(first.final_embeddings, second.final_embeddings)
    assert first.all_embeddings.shape == (3, 5, 64)
    assert torch.equal(first.valid_masks, valid)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    _assert_snapshot(extractor.actor, before_state)
    assert not extractor.actor.training
    assert all(not parameter.requires_grad for parameter in extractor.representation_parameters())
    assert all(parameter.grad is None for parameter in extractor.representation_parameters())


def test_final_only_omits_all_embeddings_without_changing_final_values(tmp_path):
    extractor, _, _ = _from_artifact(tmp_path)
    args = _batch()
    all_result = extractor.extract(
        *args, variant="single_frame", include_all_valid=True
    )
    final_result = extractor.extract(
        *args, variant="single_frame", include_all_valid=False
    )
    assert final_result.all_embeddings is None
    torch.testing.assert_close(final_result.final_embeddings, all_result.final_embeddings)


def test_random_init_is_explicit_seeded_frozen_and_does_not_pollute_global_rng():
    diagnostic = _diagnostic_identity()
    torch.manual_seed(1234)
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before = torch.random.get_rng_state().clone()
    before_cuda = [value.clone() for value in torch.cuda.get_rng_state_all()]
    first = FrozenRecurrentEmbeddingExtractor.random_init(
        diagnostic_identity=diagnostic,
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        representation_seed=7302,
        actor_architecture="mappo-actor-encoder-gru-v1",
        observation_dim=209,
        action_dim=5,
        hidden_dim=64,
    )
    assert random.getstate() == before_python
    _assert_numpy_rng_equal(np.random.get_state(), before_numpy)
    assert torch.equal(torch.random.get_rng_state(), before)
    for actual, wanted in zip(torch.cuda.get_rng_state_all(), before_cuda):
        assert torch.equal(actual, wanted)
    second = FrozenRecurrentEmbeddingExtractor.random_init(
        diagnostic_identity=diagnostic,
        m1_config_path=M1_CONFIG,
        m1_approval_path=M1_APPROVAL,
        representation_seed=7302,
        actor_architecture="mappo-actor-encoder-gru-v1",
        observation_dim=209,
        action_dim=5,
        hidden_dim=64,
    )
    for left, right in zip(
        first.representation_parameters(), second.representation_parameters()
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert first.identity["source"] == "random_init"
    assert first.identity["representation_seed"] == 7302
    observations, valid, resets, hidden = _batch()
    first_result = first.extract(
        observations, valid, resets, hidden,
        variant="random_init", include_all_valid=True,
    )
    second_result = second.extract(
        observations, valid, resets, hidden,
        variant="random_init", include_all_valid=True,
    )
    torch.testing.assert_close(
        first_result.all_embeddings, second_result.all_embeddings, rtol=0, atol=0
    )


def test_unknown_seed_and_invalid_inputs_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="representation seed"):
        _from_artifact(tmp_path, seed=99)
    extractor, _, _ = _from_artifact(tmp_path)
    observations, valid, resets, hidden = _batch()
    with pytest.raises(ValueError, match="contiguous valid suffix"):
        extractor.extract(
            observations,
            torch.tensor([[True, False, True, True, True], *valid[1:]]),
            resets,
            hidden,
            variant="normal_history",
        )
    with pytest.raises(ValueError, match="shuffled_label"):
        extractor.extract(
            observations, valid, resets, hidden,
            variant="shuffled_label",
        )


def test_extraction_rejects_reenabled_training_or_grad_state(tmp_path):
    extractor, _, _ = _from_artifact(tmp_path)
    extractor.actor.train()
    with pytest.raises(ValueError, match="eval mode"):
        extractor.extract(*_batch(), variant="normal_history")
    extractor.actor.eval()
    next(extractor.representation_parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        extractor.extract(*_batch(), variant="normal_history")


@pytest.mark.parametrize(
    "effect",
    [
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "caller_input",
        "working_input",
        "working_requires_grad",
        "working_gradient",
        "training_mode",
        "requires_grad",
        "gradient",
        "buffer",
        "parameter",
        "delete_parameter",
        "add_parameter",
        "replace_parameter",
        "delete_buffer",
        "add_buffer",
        "replace_buffer",
        "delete_submodule",
        "add_submodule",
        "replace_submodule",
    ],
)
def test_sequence_side_effects_are_restored_before_fail_closed(
    tmp_path, monkeypatch, effect
):
    extractor, _, _ = _from_artifact(tmp_path)
    actor = extractor.actor
    actor.register_buffer("diagnostic_test_buffer", torch.tensor([3.0]))
    inputs = _batch()
    original_sequence = MAPPOActor._sequence
    parameter = next(actor.encoder.parameters())
    if effect == "gradient":
        parameter.grad = torch.full_like(parameter, 0.25)
    expected = _side_effect_snapshot(actor, inputs)

    def side_effecting_sequence(working_actor, observations, hidden, masks):
        parameter = next(working_actor.encoder.parameters())
        if effect == "python_rng":
            random.random()
        elif effect == "numpy_rng":
            np.random.random()
        elif effect == "torch_rng":
            torch.rand(1)
        elif effect == "caller_input":
            inputs[0].resize_(1).add_(7.0)
        elif effect == "working_input":
            observations.add_(1.0)
        elif effect in ("working_requires_grad", "working_gradient"):
            base = observations
            while base._base is not None:
                base = base._base
            if effect == "working_requires_grad":
                base.requires_grad_(True)
            else:
                base.grad = torch.ones_like(base)
        elif effect == "training_mode":
            working_actor.train()
        elif effect == "requires_grad":
            parameter.requires_grad_(True)
        elif effect == "gradient":
            parameter.grad = torch.ones_like(parameter)
        elif effect == "buffer":
            working_actor.diagnostic_test_buffer.add_(1.0)
        elif effect == "parameter":
            parameter.add_(1.0)
        elif effect == "delete_parameter":
            del working_actor.encoder[1]._parameters["weight"]
        elif effect == "add_parameter":
            working_actor.register_parameter(
                "injected_parameter", nn.Parameter(torch.zeros(1))
            )
        elif effect == "replace_parameter":
            working_actor.encoder[1].weight = nn.Parameter(
                torch.zeros_like(working_actor.encoder[1].weight)
            )
        elif effect == "delete_buffer":
            del working_actor._buffers["diagnostic_test_buffer"]
        elif effect == "add_buffer":
            working_actor.register_buffer("injected_buffer", torch.zeros(1))
        elif effect == "replace_buffer":
            working_actor.diagnostic_test_buffer = torch.zeros(2)
        elif effect == "delete_submodule":
            del working_actor._modules["encoder"]
        elif effect == "add_submodule":
            working_actor.add_module("injected_module", nn.Identity())
        elif effect == "replace_submodule":
            working_actor.encoder = nn.Identity()
        return original_sequence(working_actor, observations, hidden, masks)

    monkeypatch.setattr(MAPPOActor, "_sequence", side_effecting_sequence)
    with pytest.raises(RuntimeError, match="side effect"):
        extractor.extract(*inputs, variant="normal_history")
    _assert_side_effect_snapshot(actor, inputs, expected)


def test_exception_path_restores_every_mutated_external_state(tmp_path, monkeypatch):
    extractor, _, _ = _from_artifact(tmp_path)
    actor = extractor.actor
    actor.register_buffer("diagnostic_test_buffer", torch.tensor([4.0]))
    inputs = _batch()
    expected = _side_effect_snapshot(actor, inputs)
    parameter = next(actor.encoder.parameters())

    def mutating_failure(working_actor, observations, hidden, masks):
        parameter = next(working_actor.encoder.parameters())
        random.random()
        np.random.random()
        torch.rand(1)
        observations.resize_(1).add_(9.0)
        working_actor.train()
        parameter.requires_grad_(True)
        parameter.grad = torch.ones_like(parameter)
        working_actor.diagnostic_test_buffer.add_(2.0)
        parameter.add_(2.0)
        raise ValueError("injected sequence failure")

    monkeypatch.setattr(MAPPOActor, "_sequence", mutating_failure)
    with pytest.raises(RuntimeError, match="side effect"):
        extractor.extract(*inputs, variant="normal_history")
    _assert_side_effect_snapshot(actor, inputs, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_rng_side_effect_is_restored_before_failure(tmp_path, monkeypatch):
    extractor, _, _ = _from_artifact(tmp_path)
    actor = extractor.actor
    inputs = _batch()
    expected = _side_effect_snapshot(actor, inputs)
    original_sequence = MAPPOActor._sequence

    def cuda_rng_sequence(working_actor, observations, hidden, masks):
        torch.rand(1, device="cuda")
        return original_sequence(working_actor, observations, hidden, masks)

    monkeypatch.setattr(MAPPOActor, "_sequence", cuda_rng_sequence)
    with pytest.raises(RuntimeError, match="side effect"):
        extractor.extract(*inputs, variant="normal_history")
    _assert_side_effect_snapshot(actor, inputs, expected)


def test_caller_autograd_metadata_is_restored_after_mutation(tmp_path, monkeypatch):
    extractor, _, _ = _from_artifact(tmp_path)
    inputs = list(_batch())
    inputs[0].requires_grad_(True)
    inputs[0].grad = torch.full_like(inputs[0], 0.25)
    inputs[3].requires_grad_(True)
    inputs[3].grad = torch.full_like(inputs[3], 0.5)
    inputs = tuple(inputs)
    expected = _side_effect_snapshot(extractor.actor, inputs)
    original_sequence = MAPPOActor._sequence

    def mutate_caller_metadata(working_actor, observations, hidden, masks):
        inputs[0].requires_grad_(False)
        inputs[0].grad = torch.zeros_like(inputs[0])
        inputs[3].requires_grad_(False)
        inputs[3].grad = None
        return original_sequence(working_actor, observations, hidden, masks)

    monkeypatch.setattr(MAPPOActor, "_sequence", mutate_caller_metadata)
    with pytest.raises(RuntimeError, match="side effect"):
        extractor.extract(*inputs, variant="normal_history")
    _assert_side_effect_snapshot(extractor.actor, inputs, expected)
