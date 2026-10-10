"""Frozen recurrent embedding extraction for representation diagnostics."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from llm_mappo.mappo_baseline.networks import MAPPOActor

from .pretraining import ACTOR_ARCHITECTURE_V1, HIDDEN_DIM, OBSERVATION_DIM
from .representation_diagnostics import validate_diagnostic_run_identity
from .representation_export import import_actor_representation


EMBEDDING_INPUT_VARIANTS = (
    "normal_history",
    "shuffled_history",
    "single_frame",
    "random_init",
)
_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class RecurrentEmbeddings:
    """Detached final embeddings and, optionally, their padded time series."""

    final_embeddings: Tensor
    all_embeddings: Tensor | None
    valid_masks: Tensor
    identity: Mapping[str, object]


def embedding_input_variant(diagnostic_variant: str) -> str:
    """Resolve label-only shuffled-label to its unchanged embedding input."""

    if diagnostic_variant == "shuffled_label":
        return "normal_history"
    if diagnostic_variant in EMBEDDING_INPUT_VARIANTS:
        return diagnostic_variant
    raise ValueError("Unknown embedding-input variant.")


class FrozenRecurrentEmbeddingExtractor:
    """Use one existing MAPPO Actor Encoder + GRU as a frozen representation."""

    def __init__(
        self,
        actor: MAPPOActor,
        identity: Mapping[str, object],
        *,
        _factory_token=None,
    ) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise ValueError(
                "Use the authenticated-artifact or random-init factory."
            )
        _validate_actor(actor)
        self.actor = actor
        self.identity = copy.deepcopy(dict(identity))
        for parameter in self.representation_parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.actor.eval()

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        actor: MAPPOActor,
        *,
        expected_artifact_identity: Mapping[str, object],
        expected_artifact_sha256: str,
        diagnostic_identity: Mapping[str, object],
        m1_config_path: str | Path,
        m1_approval_path: str | Path,
        representation_seed: int,
    ) -> "FrozenRecurrentEmbeddingExtractor":
        """Authenticate and import a Group 6 representation into one Actor."""

        diagnostic = _validated_diagnostic_identity(
            diagnostic_identity, m1_config_path, m1_approval_path,
            representation_seed,
        )
        if expected_artifact_identity.get("dataset_sha256") != diagnostic[
            "dataset_sha256"
        ]:
            raise ValueError(
                "Representation artifact and diagnostic dataset identity differ."
            )
        receipt = import_actor_representation(
            path,
            actor,
            expected_identity=expected_artifact_identity,
            expected_artifact_sha256=expected_artifact_sha256,
        )
        identity = _extraction_identity(
            diagnostic,
            representation_seed=representation_seed,
            source="authenticated_artifact",
            artifact_file_sha256=receipt["file_sha256"],
            artifact_payload_sha256=receipt["payload_content_sha256"],
        )
        return cls(actor, identity, _factory_token=_FACTORY_TOKEN)

    @classmethod
    def random_init(
        cls,
        *,
        diagnostic_identity: Mapping[str, object],
        m1_config_path: str | Path,
        m1_approval_path: str | Path,
        representation_seed: int,
        actor_architecture: str,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int,
    ) -> "FrozenRecurrentEmbeddingExtractor":
        """Build the exact random-init control without advancing global RNG."""

        diagnostic = _validated_diagnostic_identity(
            diagnostic_identity, m1_config_path, m1_approval_path,
            representation_seed,
        )
        if (
            actor_architecture != ACTOR_ARCHITECTURE_V1
            or observation_dim != OBSERVATION_DIM
            or hidden_dim != HIDDEN_DIM
            or type(action_dim) is not int
            or action_dim < 1
        ):
            raise ValueError(
                "Random-init Actor architecture must be explicit and exact."
            )
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        try:
            with torch.random.fork_rng(devices=_cuda_device_indices()):
                torch.manual_seed(representation_seed)
                actor = MAPPOActor(observation_dim, action_dim, hidden_dim)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        identity = _extraction_identity(
            diagnostic,
            representation_seed=representation_seed,
            source="random_init",
            artifact_file_sha256=None,
            artifact_payload_sha256=None,
        )
        return cls(actor, identity, _factory_token=_FACTORY_TOKEN)

    def representation_parameters(self):
        """Iterate only the shared Actor Encoder + GRU parameters."""

        yield from self.actor.encoder.parameters()
        yield from self.actor.gru.parameters()

    def extract(  # noqa: C901
        self,
        observations: Tensor,
        valid_masks: Tensor,
        reset_masks: Tensor,
        initial_hidden: Tensor,
        *,
        variant: str,
        include_all_valid: bool = False,
    ) -> RecurrentEmbeddings:
        """Extract detached final 64D states with exact left-padding resets."""

        if self.actor.training:
            raise ValueError("Frozen representation extraction requires eval mode.")
        if any(
            parameter.requires_grad
            for parameter in self.representation_parameters()
        ):
            raise ValueError(
                "Encoder and GRU representation parameters must stay frozen."
            )
        if variant == "shuffled_label":
            raise ValueError(
                "shuffled_label changes later label assignment, not embeddings; "
                "use normal_history for embedding extraction."
            )
        if variant not in EMBEDDING_INPUT_VARIANTS:
            raise ValueError("Unknown embedding-input variant.")
        if variant == "random_init" and self.identity["source"] != "random_init":
            raise ValueError("random_init requires a random-init extractor.")
        if variant != "random_init" and self.identity["source"] == "random_init":
            raise ValueError("A random-init extractor accepts only random_init.")
        _validate_inputs(observations, valid_masks, reset_masks, initial_hidden)

        external_inputs = (
            observations, valid_masks, reset_masks, initial_hidden,
        )
        call_state = _capture_call_state(self.actor, external_inputs)
        working_actor = None
        working_inputs = None
        working_state = None
        result = None
        error = None
        try:
            working_actor = copy.deepcopy(self.actor)
            working_inputs = _prepare_working_inputs(
                external_inputs,
                variant=variant,
                seed=self.identity["representation_seed"],
            )
            working_state = _capture_actor_state(working_actor)
            working_input_state = [
                _capture_tensor(value) for value in working_inputs
            ]
            with torch.no_grad():
                result = _extract_all(
                    working_actor, *working_inputs
                )
        except BaseException as caught:
            error = caught
        side_effects = _find_call_side_effects(
            self.actor, external_inputs, call_state
        )
        if working_actor is not None and working_state is not None:
            side_effects.extend(
                _find_actor_side_effects(working_actor, working_state, "working")
            )
            if any(
                not _tensor_equal(value, expected)
                for value, expected in zip(
                    working_inputs, working_input_state
                )
            ):
                side_effects.append("working_input")
        _restore_call_state(self.actor, external_inputs, call_state)
        if side_effects:
            names = ", ".join(side_effects)
            raise RuntimeError(
                f"Embedding extraction side effect detected and restored: {names}."
            ) from error
        if error is not None:
            raise error
        all_embeddings = result

        final = all_embeddings[:, -1].detach()
        complete = all_embeddings.detach() if include_all_valid else None
        return RecurrentEmbeddings(
            final_embeddings=final,
            all_embeddings=complete,
            valid_masks=valid_masks.detach().clone(),
            identity=copy.deepcopy(self.identity),
        )


def _validated_diagnostic_identity(
    identity, m1_config_path, m1_approval_path, representation_seed
):
    diagnostic = copy.deepcopy(dict(identity))
    validate_diagnostic_run_identity(
        diagnostic, m1_config_path, m1_approval_path
    )
    if (
        type(representation_seed) is not int
        or representation_seed not in diagnostic["representation_seeds"]
    ):
        raise ValueError("Unknown representation seed for diagnostic identity.")
    return diagnostic


def _extraction_identity(
    diagnostic,
    *,
    representation_seed,
    source,
    artifact_file_sha256,
    artifact_payload_sha256,
):
    encoded = json.dumps(
        diagnostic, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "diagnostic_run_sha256": sha256(encoded).hexdigest(),
        "dataset_sha256": diagnostic["dataset_sha256"],
        "representation_seed": representation_seed,
        "actor_architecture": ACTOR_ARCHITECTURE_V1,
        "observation_dim": OBSERVATION_DIM,
        "hidden_dim": HIDDEN_DIM,
        "source": source,
        "artifact_file_sha256": artifact_file_sha256,
        "artifact_payload_sha256": artifact_payload_sha256,
    }


def _validate_actor(actor):
    if (
        not isinstance(actor, MAPPOActor)
        or actor.observation_dim != OBSERVATION_DIM
        or actor.hidden_dim != HIDDEN_DIM
    ):
        raise ValueError("Embedding extraction requires the 209D/64D MAPPOActor.")


def _validate_inputs(  # noqa: C901
    observations, valid_masks, reset_masks, initial_hidden
):
    if not all(
        isinstance(value, Tensor)
        for value in (observations, valid_masks, reset_masks, initial_hidden)
    ):
        raise ValueError("Embedding inputs must be torch.Tensor values.")
    if observations.ndim != 3 or observations.shape[-1] != OBSERVATION_DIM:
        raise ValueError("observations must have shape [batch,time,209].")
    axes = tuple(observations.shape[:2])
    if any(size < 1 for size in axes):
        raise ValueError("Embedding batch and time axes must be non-empty.")
    if (
        valid_masks.dtype is not torch.bool
        or reset_masks.dtype is not torch.bool
        or tuple(valid_masks.shape) != axes
        or tuple(reset_masks.shape) != axes
    ):
        raise ValueError("Boolean valid/reset masks must match batch/time axes.")
    if tuple(initial_hidden.shape) != (axes[0], HIDDEN_DIM):
        raise ValueError("initial_hidden must have shape [batch,64].")
    if not observations.is_floating_point() or not initial_hidden.is_floating_point():
        raise ValueError("Observations and hidden states must be floating point.")
    if observations.device != initial_hidden.device:
        raise ValueError("Observations and hidden states must share a device.")
    if (
        valid_masks.device != observations.device
        or reset_masks.device != observations.device
    ):
        raise ValueError("Observations and masks must share a device.")
    if (
        not torch.isfinite(observations).all()
        or not torch.isfinite(initial_hidden).all()
    ):
        raise ValueError("Observations and hidden states must be finite.")
    if not valid_masks[:, -1].all():
        raise ValueError("Every sequence must have a valid final frame.")
    if torch.any(reset_masks & ~valid_masks):
        raise ValueError("reset_masks cannot mark padded frames.")
    for row in valid_masks:
        first = int(torch.nonzero(row, as_tuple=False)[0])
        if not row[first:].all():
            raise ValueError("valid_masks must describe one contiguous valid suffix.")


def _shuffle_valid_history(observations, valid_masks, seed):
    result = _clone_tensor(observations)
    generator = np.random.default_rng(seed)
    with torch.no_grad():
        for row in range(observations.shape[0]):
            indices = torch.nonzero(valid_masks[row], as_tuple=False).flatten()
            prefix = indices[:-1]
            if prefix.numel() > 1:
                shift = int(generator.integers(1, prefix.numel()))
                permutation = torch.roll(prefix, shifts=shift)
                result[row, prefix] = observations[row, permutation]
    return result


def _prepare_working_inputs(external_inputs, *, variant, seed):
    observations, valid_masks, reset_masks, initial_hidden = (
        _clone_tensor(value) for value in external_inputs
    )
    if variant == "shuffled_history":
        observations = _shuffle_valid_history(observations, valid_masks, seed)
    elif variant == "single_frame":
        source = observations
        observations = torch.zeros_like(source)
        with torch.no_grad():
            observations[:, -1] = source[:, -1]
        observations.requires_grad_(source.requires_grad)
        valid_masks.zero_()
        valid_masks[:, -1] = True
        reset_masks.zero_()
        reset_masks[:, -1] = True
    return observations, valid_masks, reset_masks, initial_hidden


def _extract_all(actor, observations, valid_masks, reset_masks, initial_hidden):
    output = torch.zeros(
        (*observations.shape[:2], HIDDEN_DIM),
        dtype=observations.dtype,
        device=observations.device,
    )
    for row in range(observations.shape[0]):
        first = int(torch.nonzero(valid_masks[row], as_tuple=False)[0])
        sequence = observations[row, first:].unsqueeze(1)
        continuation = (~reset_masks[row, first:]).to(sequence.dtype).unsqueeze(1)
        encoded, _ = actor._sequence(
            sequence, initial_hidden[row].unsqueeze(0), continuation
        )
        output[row, first:] = encoded.squeeze(1)
    return output


def _cuda_device_indices():
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def _capture_tensor(value):
    return {
        "value": value.detach().clone(),
        "shape": tuple(value.shape),
        "dtype": value.dtype,
        "device": value.device,
        "requires_grad": value.requires_grad,
        "grad": None if value.grad is None else _capture_tensor_value(value.grad),
    }


def _capture_tensor_value(value):
    return {
        "value": value.detach().clone(),
        "shape": tuple(value.shape),
        "dtype": value.dtype,
        "device": value.device,
    }


def _clone_tensor(value):
    clone = value.detach().clone()
    clone.requires_grad_(value.requires_grad)
    clone.grad = None if value.grad is None else value.grad.detach().clone()
    return clone


def _capture_actor_state(actor):
    modules = dict(actor.named_modules())
    return {
        "parameters": {
            name: {**_capture_tensor(parameter), "object": parameter}
            for name, parameter in actor.named_parameters()
        },
        "buffers": {
            name: {**_capture_tensor(buffer), "object": buffer}
            for name, buffer in actor.named_buffers()
        },
        "modules": modules,
        "registries": {
            name: {
                "modules": tuple(module._modules.items()),
                "parameters": tuple(module._parameters.items()),
                "buffers": tuple(module._buffers.items()),
            }
            for name, module in modules.items()
        },
        "training": {
            name: module.training for name, module in modules.items()
        },
        "state_dict_keys": tuple(actor.state_dict()),
    }


def _capture_call_state(actor, external_inputs):
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state().clone(),
        "cuda_rng": [state.clone() for state in torch.cuda.get_rng_state_all()],
        "actor": _capture_actor_state(actor),
        "external_inputs": [_capture_tensor(value) for value in external_inputs],
    }


def _find_call_side_effects(actor, external_inputs, state):
    effects = []
    if random.getstate() != state["python_rng"]:
        effects.append("python_rng")
    if not _numpy_rng_equal(np.random.get_state(), state["numpy_rng"]):
        effects.append("numpy_rng")
    if not torch.equal(torch.random.get_rng_state(), state["torch_rng"]):
        effects.append("torch_cpu_rng")
    current_cuda = torch.cuda.get_rng_state_all()
    if len(current_cuda) != len(state["cuda_rng"]) or any(
        not torch.equal(actual, expected)
        for actual, expected in zip(current_cuda, state["cuda_rng"])
    ):
        effects.append("torch_cuda_rng")
    effects.extend(_find_actor_side_effects(actor, state["actor"], "source"))
    if any(
        not _tensor_equal(value, expected)
        for value, expected in zip(external_inputs, state["external_inputs"])
    ):
        effects.append("caller_input")
    return effects


def _find_actor_side_effects(actor, state, label):  # noqa: C901
    effects = []
    try:
        current_modules = dict(actor.named_modules())
        current_parameters = dict(actor.named_parameters())
        current_buffers = dict(actor.named_buffers())
        current_state_keys = tuple(actor.state_dict())
    except Exception:
        return [f"{label}_actor_registry"]
    if current_state_keys != state["state_dict_keys"]:
        effects.append(f"{label}_state_structure")
    if tuple(current_parameters) != tuple(state["parameters"]):
        effects.append(f"{label}_parameter_structure")
    else:
        for name, expected in state["parameters"].items():
            parameter = current_parameters[name]
            if parameter is not expected["object"]:
                effects.append(f"{label}_parameter_replaced:{name}")
            elif not _tensor_equal(parameter, expected):
                effects.append(f"{label}_parameter:{name}")
            if parameter.requires_grad is not expected["requires_grad"]:
                effects.append(f"{label}_requires_grad:{name}")
            if not _optional_snapshot_equal(parameter.grad, expected["grad"]):
                effects.append(f"{label}_gradient:{name}")
    if tuple(current_buffers) != tuple(state["buffers"]):
        effects.append(f"{label}_buffer_structure")
    else:
        for name, expected in state["buffers"].items():
            if current_buffers[name] is not expected["object"]:
                effects.append(f"{label}_buffer_replaced:{name}")
            elif not _tensor_equal(current_buffers[name], expected):
                effects.append(f"{label}_buffer:{name}")
    if tuple(current_modules) != tuple(state["modules"]) or any(
        current_modules[name] is not module
        for name, module in state["modules"].items()
        if name in current_modules
    ):
        effects.append(f"{label}_module_structure")
    for name, expected in state["registries"].items():
        if name not in current_modules:
            continue
        module = current_modules[name]
        for registry in ("modules", "parameters", "buffers"):
            actual = tuple(getattr(module, f"_{registry}").items())
            if len(actual) != len(expected[registry]) or any(
                key != old_key or value is not old_value
                for (key, value), (old_key, old_value)
                in zip(actual, expected[registry])
            ):
                effects.append(f"{label}_{registry}_registry:{name}")
    current_training = {
        name: module.training for name, module in current_modules.items()
    }
    if current_training != state["training"]:
        effects.append(f"{label}_module_training_mode")
    return effects


def _restore_call_state(actor, external_inputs, state):
    actor_state = state["actor"]
    parameters = dict(actor.named_parameters())
    for name, expected in actor_state["parameters"].items():
        parameter = parameters[name]
        parameter.data = expected["value"].detach().clone()
        parameter.requires_grad_(expected["requires_grad"])
        parameter.grad = (
            None
            if expected["grad"] is None
            else expected["grad"]["value"].detach().clone()
        )
    buffers = dict(actor.named_buffers())
    for name, expected in actor_state["buffers"].items():
        buffers[name].data = expected["value"].detach().clone()
    modules = dict(actor.named_modules())
    for name, training in actor_state["training"].items():
        modules[name].training = training
    for value, expected in zip(external_inputs, state["external_inputs"]):
        value.data = expected["value"].detach().clone()
        value.requires_grad_(expected["requires_grad"])
        value.grad = (
            None
            if expected["grad"] is None
            else expected["grad"]["value"].detach().clone()
        )
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.random.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def _tensor_equal(value, expected):
    return (
        tuple(value.shape) == expected["shape"]
        and value.dtype == expected["dtype"]
        and value.device == expected["device"]
        and torch.equal(value.detach(), expected["value"])
        and value.requires_grad is expected["requires_grad"]
        and _optional_snapshot_equal(value.grad, expected["grad"])
    )


def _optional_snapshot_equal(value, expected):
    if value is None or expected is None:
        return value is None and expected is None
    return (
        tuple(value.shape) == expected["shape"]
        and value.dtype == expected["dtype"]
        and value.device == expected["device"]
        and torch.equal(value.detach(), expected["value"])
    )


def _numpy_rng_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )
