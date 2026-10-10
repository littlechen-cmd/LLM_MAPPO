"""Cached candidate decoding must preserve exact original window semantics."""

import json

import numpy as np

from llm_mappo.mappo_llm_kd import goal_full_dataset_run as run


def test_cached_views_preserve_reset_censor_charge_actions_and_candidate_flags():
    ledger = {name: index for index, name in enumerate(run.SCHEMA.feature_names)}
    rows = np.zeros((15, 209), dtype=np.float32)
    for index, row in enumerate(rows):
        phase = "task" if index < 8 else "charging"
        row[ledger["self_orientation_up"]] = 1
        row[ledger[f"self_phase_{phase}"]] = 1
        row[ledger[f"goal_type_{phase}"]] = 1
        row[ledger[f"self_previous_action_{index % 4}"]] = 1
        row[ledger["self_previous_blocked"]] = float(index % 4 == 1)
        row[ledger["self_priority_weight"]] = .75 if phase == "task" else 0
        row[ledger["self_battery"]] = .8 if phase == "task" else .2 + (index - 8) * .1
        if index < 10:
            row[ledger["goal_relative_forward"]] = .1
            row[ledger["goal_normalized_distance"]] = .1
        else:
            row[ledger["map_f0_r0_charger"]] = 1
    resets = np.zeros(15, dtype=bool)
    resets[0] = True
    expected = [(end, run._window_view(rows[end - 7:end + 1], resets[end - 7:end + 1]))
                for end in range(7, 15)]
    actual = list(run._iter_window_views(rows, resets))
    assert [end for end, _ in actual] == list(range(7, 15))
    assert actual[0][1]["summary"]["history_left_censored"] is False
    assert all(view["summary"]["history_left_censored"] for _, view in actual[1:])
    for (_, old), (_, cached) in zip(expected, actual):
        assert json.dumps(cached, separators=(",", ":")) == json.dumps(
            old, separators=(",", ":"))
        assert run.canonical_sha256(cached) == run.canonical_sha256(old)
        assert run._observable_flags(cached) == run._observable_flags(old)
