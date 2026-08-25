import json

import numpy as np
import pytest

from llm_mappo.semantic_v3 import (
    SemanticDatasetV3,
    SemanticRecordV3,
    SemanticViewV3,
    TruncatedExponentialOOD,
)


def test_semantic_view_v3_encodes_the_frozen_61_dimensions_and_anonymous_neighbors():
    focal = {
        "position": (2, 2), "orientation": "right", "battery_ratio": 0.5,
        "loaded": False, "priority_present": True, "priority_rank": 0.0,
        "target_kind": "task", "on_highway": True,
        "at_charging_station": False, "at_picking_station": False,
        "adjacent_highway": {"forward": True, "right": False, "backward": True, "left": False},
    }
    peers = [
        {"position": (3, 2), "loaded": True, "battery_ratio": 0.8, "dead": False,
         "priority_present": True, "priority_rank": 0.1, "target_kind": "delivery",
         "at_charging_station": False},
    ]
    view = SemanticViewV3.from_state("layout", 5, 5, focal, peers)

    assert view.vector.shape == (61,)
    assert view.json_view["semantic_view_version"] == "semantic-view-v3"
    assert view.json_view["neighbors"][0]["mask"] is True
    assert [item["mask"] for item in view.json_view["neighbors"][1:]] == [False, False]
    assert np.array_equal(view.vector[47:], np.zeros(14))


def test_semantic_dataset_retrieval_uses_only_valid_61d_records_and_fails_closed():
    vector = np.zeros(61, dtype=np.float64)
    dataset = SemanticDatasetV3.from_records([
        {"vector": vector.tolist(), "scores": [0.2, 0.4, 0.6], "validity": 1},
        {"vector": (vector + 0.1).tolist(), "scores": [0.4, 0.5, 0.6], "validity": 1},
        {"vector": (vector + 0.2).tolist(), "scores": [0.6, 0.7, 0.8], "validity": 1},
    ])
    target, validity, reliability = dataset.retrieve(vector)

    assert validity == 1.0
    assert reliability >= 0.0
    assert target.shape == (3,)
    empty = SemanticDatasetV3.from_records([])
    target, validity, reliability = empty.retrieve(vector)
    assert np.array_equal(target, np.zeros(3))
    assert (validity, reliability) == (0.0, 0.0)


def test_semantic_record_requires_exact_three_score_json_schema():
    payload = {
        "task_persistence": 0.2,
        "task_persistence_reason": "An active task remains.",
        "yielding_preference": 0.4,
        "yielding_preference_reason": "A nearby peer makes yielding plausible.",
        "coordination_risk": 0.6,
        "coordination_risk_reason": "The local area is constrained.",
    }
    record = SemanticRecordV3.parse_response(json.dumps(payload))
    assert record.validity == 1.0
    assert np.allclose(record.scores, np.asarray([0.2, 0.4, 0.6]))

    payload["unexpected"] = True
    with pytest.raises(ValueError, match="exactly"):
        SemanticRecordV3.parse_response(json.dumps(payload))


def test_truncated_exponential_ood_uses_standardized_61d_distances():
    references = np.vstack([np.zeros(61), np.ones(61), np.full(61, 2.0)])
    ood = TruncatedExponentialOOD.fit(references)

    assert ood.reliability(np.zeros(61)) > 0.0
    assert ood.reliability(np.full(61, 100.0)) == 0.0
    with pytest.raises(ValueError, match="61"):
        ood.reliability(np.zeros(615))
