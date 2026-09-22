"""Strict, resumable checkpoints for Phase 3 semantic pretraining."""

from __future__ import annotations

import copy
import os
import random
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from .pretraining import (
    PRETRAINING_CHECKPOINT_V1,
    SemanticPretrainingModel,
    validate_pretraining_identity,
)


_TOP_LEVEL_KEYS = (
    "checkpoint_schema",
    "identity",
    "actor_representation",
    "semantic_head",
    "optimizer",
    "rng",
    "progress",
    "early_stopping",
)
_RNG_KEYS = ("python", "numpy", "torch_cpu", "torch_cuda")
_PROGRESS_KEYS = ("global_step", "epoch", "batch_in_epoch")
_EARLY_STOPPING_KEYS = (
    "best_validation_loss",
    "bad_epochs",
    "patience",
    "minimum_improvement",
)


def save_pretraining_checkpoint(
    path: str | Path,
    model: SemanticPretrainingModel,
    optimizer: torch.optim.Optimizer,
    *,
    identity: Mapping[str, object],
    progress: Mapping[str, object],
    early_stopping: Mapping[str, object],
) -> None:
    """Validate and atomically save all state needed for exact continuation."""

    payload = _build_payload(model, optimizer, identity, progress, early_stopping)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pretraining_checkpoint(
    path: str | Path,
    model: SemanticPretrainingModel,
    optimizer: torch.optim.Optimizer,
    *,
    expected_identity: Mapping[str, object],
    map_location: str | torch.device = "cpu",
) -> dict:
    """Validate a safe payload completely, then atomically restore its state."""

    expected_identity_copy = copy.deepcopy(dict(expected_identity))
    validate_pretraining_identity(expected_identity_copy)
    try:
        payload = torch.load(
            Path(path), map_location=map_location, weights_only=True
        )
    except Exception as error:
        raise ValueError("Pretraining checkpoint could not be safely loaded.") from error
    validated = _validate_payload(
        payload,
        model,
        optimizer,
        expected_identity=expected_identity_copy,
    )

    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng_before = _capture_rng_state()
    try:
        _apply_model_state(model, validated)
        optimizer.load_state_dict(validated["optimizer"]["state_dict"])
        _restore_rng_state(validated["rng"])
    except Exception as error:
        model.load_state_dict(model_before, strict=True)
        optimizer.load_state_dict(optimizer_before)
        _restore_rng_state(rng_before)
        raise ValueError(
            "Pretraining checkpoint could not be atomically applied."
        ) from error
    return {
        "identity": copy.deepcopy(validated["identity"]),
        "progress": copy.deepcopy(validated["progress"]),
        "early_stopping": copy.deepcopy(validated["early_stopping"]),
    }


def _build_payload(model, optimizer, identity, progress, early_stopping):
    _require_model_optimizer(model, optimizer)
    identity_copy = copy.deepcopy(dict(identity))
    validate_pretraining_identity(identity_copy)
    progress_copy = _validate_progress(progress)
    early_copy = _validate_early_stopping(early_stopping)
    actor_state = {
        key: value.detach().clone()
        for key, value in model.actor.state_dict().items()
        if key.startswith(("encoder.", "gru."))
    }
    head_state = {
        key: value.detach().clone()
        for key, value in model.semantic_head.state_dict().items()
    }
    optimizer_payload = {
        "class": type(optimizer).__name__,
        "state_dict": copy.deepcopy(optimizer.state_dict()),
    }
    payload = {
        "checkpoint_schema": PRETRAINING_CHECKPOINT_V1,
        "identity": identity_copy,
        "actor_representation": actor_state,
        "semantic_head": head_state,
        "optimizer": optimizer_payload,
        "rng": _capture_rng_state(),
        "progress": progress_copy,
        "early_stopping": early_copy,
    }
    _validate_payload(payload, model, optimizer)
    return payload


def _validate_payload(
    payload, model, optimizer, *, expected_identity=None
):  # noqa: C901
    _require_model_optimizer(model, optimizer)
    if not isinstance(payload, Mapping) or tuple(payload.keys()) != _TOP_LEVEL_KEYS:
        raise ValueError("Checkpoint top-level keys and order must be exact.")
    if payload["checkpoint_schema"] != PRETRAINING_CHECKPOINT_V1:
        raise ValueError("Checkpoint schema is incompatible.")
    validate_pretraining_identity(payload["identity"])
    if (
        expected_identity is not None
        and not _strict_equal(payload["identity"], expected_identity)
    ):
        raise ValueError("Checkpoint identity does not match expected_identity.")
    progress = _validate_progress(payload["progress"])
    _validate_early_stopping(payload["early_stopping"])
    _validate_rng_state(payload["rng"])

    expected_actor = {
        key: value
        for key, value in model.actor.state_dict().items()
        if key.startswith(("encoder.", "gru."))
    }
    _validate_tensor_state(
        "actor_representation", payload["actor_representation"], expected_actor
    )
    _validate_tensor_state(
        "semantic_head", payload["semantic_head"], model.semantic_head.state_dict()
    )
    _validate_optimizer(payload["optimizer"], optimizer, model, progress)

    dry_model = copy.deepcopy(model)
    dry_optimizer = copy.deepcopy(optimizer)
    _apply_model_state(dry_model, payload)
    try:
        dry_optimizer.load_state_dict(
            copy.deepcopy(payload["optimizer"]["state_dict"])
        )
    except Exception as error:
        raise ValueError("Optimizer state is incompatible.") from error
    return payload


def _require_model_optimizer(model, optimizer):
    if not isinstance(model, SemanticPretrainingModel):
        raise ValueError("Checkpoint requires a SemanticPretrainingModel.")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("Checkpoint requires a torch optimizer.")


def _validate_tensor_state(name, supplied, expected):
    if not isinstance(supplied, Mapping) or tuple(supplied.keys()) != tuple(expected):
        raise ValueError(f"{name} state keys and order must be exact.")
    for key, tensor in supplied.items():
        reference = expected[key]
        if not isinstance(tensor, Tensor):
            raise ValueError(f"{name}.{key} must be a tensor.")
        if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
            raise ValueError(f"{name}.{key} shape and dtype must be exact.")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name}.{key} must contain only finite values.")


def _validate_optimizer(supplied, optimizer, model, progress):  # noqa: C901
    if not isinstance(supplied, Mapping) or tuple(supplied.keys()) != (
        "class", "state_dict"
    ):
        raise ValueError("Optimizer keys and order must be exact.")
    if supplied["class"] != type(optimizer).__name__:
        raise ValueError("Optimizer class is incompatible.")
    state_dict = supplied["state_dict"]
    if not isinstance(state_dict, Mapping) or tuple(state_dict.keys()) != (
        "state", "param_groups"
    ):
        raise ValueError("Optimizer state top-level keys and order must be exact.")
    current = optimizer.state_dict()
    expected_parameters = (
        list(model.actor.encoder.parameters())
        + list(model.actor.gru.parameters())
        + list(model.semantic_head.parameters())
    )
    live_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len(live_parameters) != len(expected_parameters) or any(
        supplied_parameter is not expected_parameter
        for supplied_parameter, expected_parameter in zip(
            live_parameters, expected_parameters
        )
    ):
        raise ValueError(
            "Optimizer must track exactly the Actor representation and SemanticHead."
        )
    groups = state_dict["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(current["param_groups"]):
        raise ValueError("Optimizer parameter groups are incompatible.")
    saved_ids = []
    for supplied_group, current_group, live_group in zip(
        groups, current["param_groups"], optimizer.param_groups
    ):
        if not isinstance(supplied_group, Mapping):
            raise ValueError("Optimizer parameter group must be a mapping.")
        if tuple(supplied_group.keys()) != tuple(current_group.keys()):
            raise ValueError("Optimizer parameter-group keys must be exact.")
        for key in supplied_group:
            if key == "params":
                continue
            _validate_optimizer_hyperparameter(key, supplied_group[key])
            _validate_optimizer_hyperparameter(key, current_group[key])
            if not _strict_equal(supplied_group[key], current_group[key]):
                raise ValueError(
                    f"Optimizer hyperparameter {key} is incompatible."
                )
        ids = supplied_group["params"]
        if not isinstance(ids, list) or len(ids) != len(live_group["params"]):
            raise ValueError("Optimizer parameter-group structure is incompatible.")
        if any(type(parameter_id) is not int for parameter_id in ids):
            raise ValueError("Optimizer parameter identifiers must be integers.")
        if ids != current_group["params"]:
            raise ValueError(
                "Optimizer parameter identifier sequence is incompatible."
            )
        saved_ids.extend(ids)
    if len(set(saved_ids)) != len(saved_ids):
        raise ValueError("Optimizer parameter identifiers must be unique.")
    current_ids = [
        parameter_id
        for group in current["param_groups"]
        for parameter_id in group["params"]
    ]
    if saved_ids != current_ids or set(saved_ids) != set(current_ids):
        raise ValueError(
            "Optimizer parameter identifiers must exactly cover the expected set."
        )
    state = state_dict["state"]
    if not isinstance(state, Mapping):
        raise ValueError("Optimizer per-parameter state must be a mapping.")
    if any(type(parameter_id) is not int for parameter_id in state):
        raise ValueError("Optimizer state identifiers must be built-in integers.")
    if not set(state).issubset(saved_ids):
        raise ValueError("Optimizer per-parameter state keys are incompatible.")
    if progress["global_step"] == 0 and state:
        raise ValueError("A zero-step checkpoint must have empty optimizer state.")
    if progress["global_step"] > 0 and set(state) != set(saved_ids):
        raise ValueError(
            "A stepped checkpoint must contain complete optimizer state."
        )
    expected_state = _expected_optimizer_state_contract(optimizer)
    for parameter_id, entry in state.items():
        if not isinstance(entry, Mapping):
            raise ValueError("Optimizer per-parameter state must be a mapping.")
        expected_entry = expected_state[parameter_id]
        if tuple(entry.keys()) != tuple(expected_entry.keys()):
            raise ValueError("Optimizer per-parameter state keys must be exact.")
        for key, value in entry.items():
            if not isinstance(value, Tensor):
                raise ValueError("Optimizer state values must be tensors.")
            expected_value = expected_entry[key]
            if (
                value.shape != expected_value.shape
                or value.dtype != expected_value.dtype
            ):
                label = "step" if key == "step" else "tensor"
                raise ValueError(
                    f"Optimizer {label} shape and dtype must be exact."
                )
            if not torch.isfinite(value).all().item():
                label = "step" if key == "step" else "state"
                raise ValueError(
                    f"Optimizer {label} must contain only finite values."
                )
            if key == "step" and value.item() < 0:
                raise ValueError("Optimizer step state must be non-negative.")


def _expected_optimizer_state_contract(optimizer):
    probe = copy.deepcopy(optimizer)
    for group in probe.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.zeros_like(parameter)
    try:
        probe.step()
    except Exception as error:
        raise ValueError("Optimizer state contract could not be derived.") from error
    state = probe.state_dict()["state"]
    if not isinstance(state, Mapping):
        raise ValueError("Optimizer state contract is malformed.")
    return state


def _validate_optimizer_hyperparameter(name, value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            raise ValueError(f"Optimizer hyperparameter {name} must be finite.")
        if name in {
            "lr", "eps", "weight_decay", "momentum", "dampening"
        } and value < 0:
            raise ValueError(
                f"Optimizer hyperparameter {name} must be non-negative."
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_optimizer_hyperparameter(name, item)
        if name == "betas" and any(item < 0 or item >= 1 for item in value):
            raise ValueError("Optimizer hyperparameter betas must be within [0, 1).")
        return
    raise ValueError(f"Optimizer hyperparameter {name} has an unsupported type.")


def _strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return tuple(left.keys()) == tuple(right.keys()) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _validate_progress(progress):
    if not isinstance(progress, Mapping) or tuple(progress.keys()) != _PROGRESS_KEYS:
        raise ValueError("Progress keys and order must be exact.")
    result = dict(progress)
    for key, value in result.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"Progress {key} must be a non-negative integer.")
    return result


def _validate_early_stopping(state):
    if not isinstance(state, Mapping) or tuple(state.keys()) != _EARLY_STOPPING_KEYS:
        raise ValueError("Early-stopping keys and order must be exact.")
    result = dict(state)
    for key in ("bad_epochs", "patience"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"Early-stopping {key} must be non-negative.")
    for key in ("best_validation_loss", "minimum_improvement"):
        value = result[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Early-stopping {key} must be numeric.")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Early-stopping {key} must be finite and non-negative.")
        result[key] = float(value)
    return result


def _capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def _validate_rng_state(state):  # noqa: C901
    if not isinstance(state, Mapping) or tuple(state.keys()) != _RNG_KEYS:
        raise ValueError("RNG state keys and order must be exact.")
    try:
        probe = random.Random()
        probe.setstate(state["python"])
    except Exception as error:
        raise ValueError("Python RNG state is malformed.") from error
    numpy_state = state["numpy"]
    expected_numpy = (
        "bit_generator", "keys", "position", "has_gauss", "cached_gaussian"
    )
    if (
        not isinstance(numpy_state, Mapping)
        or tuple(numpy_state.keys()) != expected_numpy
    ):
        raise ValueError("NumPy RNG state keys and order must be exact.")
    try:
        probe_numpy = np.random.RandomState()
        probe_numpy.set_state(_numpy_state_tuple(numpy_state))
    except Exception as error:
        raise ValueError("NumPy RNG state is malformed.") from error
    cpu = state["torch_cpu"]
    if not isinstance(cpu, Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError("Torch CPU RNG state is malformed.")
    try:
        generator = torch.Generator(device="cpu")
        generator.set_state(cpu.cpu())
    except Exception as error:
        raise ValueError("Torch CPU RNG state is malformed.") from error
    cuda = state["torch_cuda"]
    if cuda is not None:
        if not torch.cuda.is_available() or not isinstance(cuda, list):
            raise ValueError("Torch CUDA RNG state is incompatible.")
        if len(cuda) != torch.cuda.device_count():
            raise ValueError("Torch CUDA RNG device count is incompatible.")
        if any(
            not isinstance(item, Tensor) or item.dtype != torch.uint8 or item.ndim != 1
            for item in cuda
        ):
            raise ValueError("Torch CUDA RNG state is malformed.")


def _numpy_state_tuple(state):
    return (
        state["bit_generator"],
        np.asarray(state["keys"], dtype=np.uint32),
        state["position"],
        state["has_gauss"],
        state["cached_gaussian"],
    )


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(_numpy_state_tuple(state["numpy"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _apply_model_state(model, payload):
    actor_state = model.actor.state_dict()
    actor_state.update(payload["actor_representation"])
    model.actor.load_state_dict(actor_state, strict=True)
    model.semantic_head.load_state_dict(payload["semantic_head"], strict=True)
