"""Observable selection predicate and immutable derivative regression checks."""

from __future__ import annotations

import numpy as np
import pytest

from llm_mappo.mappo_llm_kd.coordination_dataset import (
    _LEDGER, classify_trajectory,
)
from llm_mappo.mappo_llm_kd.medium_prompt import observable_coordination_evidence
from llm_mappo.mappo_llm_kd.temporal_view import (
    serialize_temporal_semantic_view,
)


def _frames() -> np.ndarray:
    rows = np.zeros((8, 209), dtype=np.float32)
    rows[:, 0] = 1.0
    rows[:, 5] = 0.8
    rows[:, 7] = 1.0
    rows[:, 21] = 1.0
    rows[:, 11] = 1.0
    rows[:, 19] = 0.5
    return rows


def _action(rows: np.ndarray, index: int, action: int,
            blocked: bool = False) -> None:
    rows[index, 11:16] = 0
    rows[index, 11 + action] = 1
    rows[index, 16] = float(blocked)


def _pure(rows: np.ndarray) -> dict:
    masks = np.zeros(8, dtype=np.bool_)
    masks[0] = True
    view = serialize_temporal_semantic_view(rows, masks, _LEDGER, 8)
    return observable_coordination_evidence(view)


@pytest.mark.parametrize("kind", ["blocked", "oscillation", "goal_change",
                                  "phase_change", "unmarked_forward"])
def test_vectorized_witness_matches_pure_view(kind: str) -> None:
    rows = _frames()
    if kind in ("blocked", "goal_change", "phase_change"):
        for index in (1, 2, 7):
            _action(rows, index, 1, blocked=True)
    else:
        for index, action in zip((1, 2, 3, 4), (2, 3, 2, 3)):
            _action(rows, index, action)
    if kind == "goal_change":
        rows[6, 21] = 0
        rows[6, 22] = 1
    if kind == "phase_change":
        rows[6, 7] = 0
        rows[6, 8] = 1
    if kind == "unmarked_forward":
        _action(rows, 6, 1, blocked=False)
    flags = classify_trajectory(rows)
    pure = _pure(rows)
    assert bool(flags["blocked_forward"][0]) == pure[
        "blocked_forward_supported"
    ]
    assert bool(flags["oscillation"][0]) == pure["oscillation_supported"]
    assert bool(flags["blocked_forward"][0] or flags["oscillation"][0]) == (
        kind in {"blocked", "oscillation"}
    )


def test_no_goal_or_battery_cannot_support_positive_witness() -> None:
    rows = _frames()
    for index in (1, 2, 7):
        _action(rows, index, 1, blocked=True)
    rows[7, 5] = 0
    assert not classify_trajectory(rows)["blocked_forward"][0]
    rows[7, 5] = 0.8
    rows[7, 21] = 0
    rows[7, 20] = 1
    assert not classify_trajectory(rows)["blocked_forward"][0]
