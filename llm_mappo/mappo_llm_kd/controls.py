"""Exact seeded controls and fresh probes for representation diagnostics."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def shuffled_label_indices(sample_count: int, seed: int) -> np.ndarray:
    """Return a deterministic derangement preserving complete label vectors."""

    if sample_count < 2:
        raise ValueError("Shuffled-label control requires at least two samples.")
    generator = np.random.default_rng(seed)
    shift = int(generator.integers(1, sample_count))
    return np.roll(np.arange(sample_count), shift)


def shuffled_history_control(
    sequences: np.ndarray, valid_mask: np.ndarray, seed: int
) -> np.ndarray:
    """Shuffle each valid history prefix while retaining its final frame."""

    values = np.asarray(sequences)
    mask = np.asarray(valid_mask)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("Expected [sample,time,feature] sequences and mask.")
    if mask.dtype != np.bool_:
        raise ValueError("The valid history mask must be boolean.")
    result = values.copy()
    generator = np.random.default_rng(seed)
    for sample in range(values.shape[0]):
        valid_indices = np.flatnonzero(mask[sample])
        if valid_indices.size == 0 or valid_indices[-1] != values.shape[1] - 1:
            raise ValueError("Every sequence must have a valid final frame.")
        expected = np.arange(valid_indices[0], values.shape[1])
        if not np.array_equal(valid_indices, expected):
            raise ValueError("Valid history frames must form a contiguous suffix.")
        prefix = valid_indices[:-1]
        if prefix.size > 1:
            shift = int(generator.integers(1, prefix.size))
            result[sample, prefix] = values[sample, np.roll(prefix, shift)]
    return result


def build_fresh_probe(seed: int) -> nn.Sequential:
    """Build the fixed 64-to-4 sigmoid probe without changing global RNG."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(64, 4), nn.Sigmoid())


def single_frame_control(
    sequences: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Retain only each sequence's valid final frame and clear its history."""

    values = np.asarray(sequences)
    mask = np.asarray(valid_mask)
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("Expected [sample,time,feature] sequences and mask.")
    if mask.dtype != np.bool_ or not np.all(mask[:, -1]):
        raise ValueError("Every sequence must have a boolean valid final frame.")
    controlled = np.zeros_like(values)
    controlled[:, -1] = values[:, -1]
    controlled_mask = np.zeros_like(mask)
    controlled_mask[:, -1] = True
    return controlled, controlled_mask
