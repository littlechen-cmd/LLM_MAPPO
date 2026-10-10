"""Fresh linear probes for fixture-only representation diagnostics."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .config import load_approved_m1_config
from .representation_diagnostics import (
    FreshProbeTrainingReceipt,
    REPRESENTATION_VARIANTS,
    validate_fresh_probe_training_receipt,
)


_EMBEDDING_DIMENSION = 64
_SEMANTIC_DIMENSION = 4


class FreshSemanticProbe(nn.Module):
    """The M1-prescribed ``Linear(64, 4) + Sigmoid`` probe."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(_EMBEDDING_DIMENSION, _SEMANTIC_DIMENSION)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings: Tensor) -> Tensor:
        """Predict the four bounded semantic scores from one final embedding."""

        return self.sigmoid(self.linear(embeddings))


@dataclass(frozen=True)
class FreshProbeResult:
    """Validation-selected probe and its fixture-only audit values."""

    probe: FreshSemanticProbe
    representation_variant: str
    representation_seed: int
    diagnostic_run_sha256: str
    dataset_sha256: str
    selected_epoch: int
    epochs_ran: int
    validation_mse: float
    validation_predictions: Tensor


def train_validation_selected_fresh_probe(
    *,
    representation_variant: str,
    representation_seed: int,
    train_embeddings: Tensor,
    train_labels: Tensor,
    validation_embeddings: Tensor,
    validation_labels: Tensor,
    training_receipt: FreshProbeTrainingReceipt,
    m1_config_path: str | Path,
    m1_approval_path: str | Path,
    device: str,
) -> FreshProbeResult:
    """Fit a fresh probe on train data and choose it solely by validation MSE.

    This boundary intentionally has no test-partition argument.  Detaching the
    provided embeddings prevents probe loss from updating an upstream
    representation, even if a caller accidentally passes tensors that require
    gradients.
    """

    probe_config, receipt = _load_probe_inputs(
        training_receipt, m1_config_path, m1_approval_path
    )
    _validate_identity(
        representation_variant, representation_seed, probe_config, receipt
    )
    _validate_partition(train_embeddings, train_labels, "train")
    _validate_partition(validation_embeddings, validation_labels, "validation")
    target_device = _resolve_device(device)
    train_x = train_embeddings.detach().to(target_device)
    train_y = train_labels.detach().to(target_device)
    validation_x = validation_embeddings.detach().to(target_device)
    validation_y = validation_labels.detach().to(target_device)

    with torch.random.fork_rng(devices=_all_cuda_device_indices()):
        torch.manual_seed(representation_seed)
        probe = FreshSemanticProbe().to(target_device)
        optimizer = torch.optim.Adam(
            probe.parameters(), lr=probe_config["learning_rate"]
        )
        loss = nn.MSELoss()
        best_state = None
        best_loss = float("inf")
        selected_epoch = -1
        stale_epochs = 0
        epochs_ran = 0
        for epoch in range(probe_config["max_epochs"]):
            _train_one_epoch(
                probe, optimizer, loss, train_x, train_y,
                batch_size=probe_config["batch_size"],
            )
            validation_loss = _validation_loss(
                probe, loss, validation_x, validation_y
            )
            epochs_ran = epoch + 1
            if validation_loss < (
                best_loss - probe_config["early_stopping_min_delta"]
            ):
                best_state = copy.deepcopy(probe.state_dict())
                best_loss = validation_loss
                selected_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= probe_config["early_stopping_patience"]:
                    break

        if best_state is None:
            raise RuntimeError("Fresh probe did not produce a validation state.")
        probe.load_state_dict(best_state, strict=True)
        predictions = _predict(probe, validation_x)

    probe.eval()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return FreshProbeResult(
        probe=probe,
        representation_variant=representation_variant,
        representation_seed=representation_seed,
        diagnostic_run_sha256=receipt["diagnostic_run_sha256"],
        dataset_sha256=receipt["dataset_sha256"],
        selected_epoch=selected_epoch,
        epochs_ran=epochs_ran,
        validation_mse=best_loss,
        validation_predictions=predictions.detach().cpu(),
    )


def _load_probe_inputs(training_receipt, m1_config_path, m1_approval_path):
    resolved = load_approved_m1_config(m1_config_path, m1_approval_path)
    probe = resolved.values["probe"]
    expected = {
        "architecture": "Linear(64,4)+Sigmoid",
        "initialization": "fresh_per_representation_and_seed",
        "representation_state": "frozen",
        "training_labels": "true_train_partition_labels",
        "selection_partition": "validation",
        "test_evaluations": 1,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 256,
        "max_epochs": 100,
        "loss": "MSE",
        "early_stopping_patience": 10,
        "early_stopping_min_delta": 0.0001,
    }
    if not isinstance(probe, dict) or any(
        probe.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("Approved M1 probe configuration is incompatible.")
    if not isinstance(probe.get("seeds"), list):
        raise ValueError("Approved M1 probe seeds must be an explicit list.")
    receipt = validate_fresh_probe_training_receipt(
        training_receipt, m1_config_path, m1_approval_path
    )
    if receipt["m1_config_sha256"] != resolved.sha256:
        raise ValueError("Fresh probe receipt does not match M1 snapshot.")
    return probe, receipt


def _validate_identity(variant, seed, probe_config, receipt):
    if variant not in REPRESENTATION_VARIANTS:
        raise ValueError("Fresh probe has an unknown representation variant.")
    if type(seed) is not int or seed not in probe_config["seeds"]:
        raise ValueError("Fresh probe seed is not approved by M1.")
    if (
        receipt["representation_variant"] != variant
        or receipt["representation_seed"] != seed
    ):
        raise ValueError("Fresh probe receipt does not match the requested probe.")


def _validate_partition(embeddings, labels, partition):
    if not isinstance(embeddings, Tensor) or not isinstance(labels, Tensor):
        raise ValueError(f"{partition} embeddings and labels must be tensors.")
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] < 1
        or embeddings.shape[1] != _EMBEDDING_DIMENSION
    ):
        raise ValueError(f"{partition} embeddings must have shape [sample, 64].")
    if (
        labels.ndim != 2
        or tuple(labels.shape) != (embeddings.shape[0], _SEMANTIC_DIMENSION)
    ):
        raise ValueError(f"{partition} labels must have shape [sample, 4].")
    if embeddings.dtype is not torch.float32 or labels.dtype is not torch.float32:
        raise ValueError(f"{partition} embeddings and labels must be float32.")
    if not torch.isfinite(embeddings).all() or not torch.isfinite(labels).all():
        raise ValueError(f"{partition} embeddings and labels must be finite.")
    if torch.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError(f"{partition} labels must be within [0, 1].")


def _resolve_device(value):
    if type(value) is not str:
        raise ValueError("Fresh probe device must be a string.")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Fresh probe CUDA device is unavailable.")
    return device


def _all_cuda_device_indices():
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def _train_one_epoch(probe, optimizer, loss, embeddings, labels, *, batch_size):
    probe.train()
    for start in range(0, embeddings.shape[0], batch_size):
        stop = min(start + batch_size, embeddings.shape[0])
        optimizer.zero_grad(set_to_none=True)
        batch_loss = loss(probe(embeddings[start:stop]), labels[start:stop])
        batch_loss.backward()
        optimizer.step()


def _validation_loss(probe, loss, embeddings, labels):
    with torch.no_grad():
        probe.eval()
        return float(loss(probe(embeddings), labels).item())


def _predict(probe, embeddings):
    with torch.no_grad():
        probe.eval()
        return probe(embeddings)
