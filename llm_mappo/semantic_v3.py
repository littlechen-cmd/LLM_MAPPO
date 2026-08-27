"""Offline semantic-view-v3 encoding and fail-closed three-score retrieval."""

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

import numpy as np


_KINDS = ("task", "delivery", "charging", "idle")
_ORIENTATIONS = ("up", "down", "left", "right")
_SCORE_KEYS = (
    "task_persistence",
    "yielding_preference",
    "coordination_risk",
)
_RESPONSE_KEYS = {
    key
    for score in _SCORE_KEYS
    for key in (score, f"{score}_reason")
}


@dataclass(frozen=True)
class SemanticRecordV3:
    """Strict parser result; audit reasons never enter retrieval features."""

    scores: np.ndarray
    reasons: tuple[str, str, str]
    validity: float

    @classmethod
    def parse_response(cls, content: str) -> "SemanticRecordV3":
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Semantic response must be one JSON object.") from error
        if not isinstance(parsed, dict) or set(parsed) != _RESPONSE_KEYS:
            raise ValueError("Semantic response must contain exactly six keys.")
        scores, reasons = [], []
        for key in _SCORE_KEYS:
            value = parsed[key]
            reason = parsed[f"{key}_reason"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Semantic score must be a finite number.")
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Semantic score must be in [0, 1].")
            if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 1000:
                raise ValueError("Semantic reason must be a non-empty string.")
            scores.append(float(value))
            reasons.append(reason.strip())
        return cls(np.asarray(scores, dtype=np.float32), tuple(reasons), 1.0)


@dataclass(frozen=True)
class SemanticViewV3:
    json_view: dict
    vector: np.ndarray

    @classmethod
    def from_state(cls, layout_hash, width, height, focal, peers):
        orientation = focal["orientation"]
        if orientation not in _ORIENTATIONS:
            raise ValueError("Invalid focal orientation.")
        f = focal["position"]
        scale = max(width - 1, height - 1, 1)
        distance_scale = max((width - 1) + (height - 1), 1)
        rows = []
        for peer in peers:
            dx, dy = peer["position"][0] - f[0], peer["position"][1] - f[1]
            forward, right = _relative(orientation, dx, dy)
            rows.append((
                abs(dx) + abs(dy), forward, right, -int(peer["loaded"]),
                int(peer["dead"]), -int(peer["priority_present"]),  # noqa: E501
                peer["priority_rank"],
                _KINDS.index(peer["target_kind"]), peer["battery_ratio"],
                -int(peer["at_charging_station"]), peer, forward / scale, right / scale,
                (abs(dx) + abs(dy)) / distance_scale,
            ))
        rows.sort(key=lambda row: row[:10])
        neighbors, vector = [], []
        focal_values = [focal["battery_ratio"], float(focal["loaded"]),
                        float(focal["priority_present"]), focal["priority_rank"]]
        focal_values += _one_hot(focal["target_kind"], _KINDS)
        focal_values += _one_hot(orientation, _ORIENTATIONS)
        focal_values += [
            float(focal["on_highway"]), float(focal["at_charging_station"]),
            float(focal["at_picking_station"]),
        ]
        focal_values += [
            float(focal["adjacent_highway"][key])
            for key in ("forward", "right", "backward", "left")
        ]
        vector.extend(focal_values)
        for row in rows[:3]:
            peer, forward, right, distance = row[10:]
            record = {
                "mask": True, "relative_forward": forward, "relative_right": right,
                "normalized_manhattan_distance": distance,
                "loaded": bool(peer["loaded"]), "battery_ratio": peer["battery_ratio"],
                "dead": bool(peer["dead"]),
                "priority_present": bool(peer["priority_present"]),
                "priority_rank": peer["priority_rank"],
                "target_kind": peer["target_kind"],
                "at_charging_station": bool(peer["at_charging_station"]),
            }
            neighbors.append(record)
            vector.extend([
                1.0, forward, right, distance, float(peer["loaded"]),
                peer["battery_ratio"], float(peer["dead"]),
                float(peer["priority_present"]), peer["priority_rank"],
            ])
            vector.extend(_one_hot(peer["target_kind"], _KINDS))
            vector.append(float(peer["at_charging_station"]))
        while len(neighbors) < 3:
            neighbors.append({
                "mask": False, "relative_forward": 0.0, "relative_right": 0.0,
                "normalized_manhattan_distance": 0.0, "loaded": False,
                "battery_ratio": 0.0, "dead": False, "priority_present": False,
                "priority_rank": 0.0, "target_kind": "idle",
                "at_charging_station": False,
            })
            vector.extend([0.0] * 14)
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (61,) or not np.all(np.isfinite(values)):
            raise ValueError("semantic-view-v3 must encode to 61 finite values.")
        focal_json = {key: value for key, value in focal.items() if key != "position"}
        return cls(
            {"semantic_view_version": "semantic-view-v3", "layout_hash": layout_hash,
             "focal": focal_json, "neighbors": neighbors}, values
        )


@dataclass
class SemanticDatasetV3:
    vectors: np.ndarray
    scores: np.ndarray
    ood: "TruncatedExponentialOOD | None" = None

    @classmethod
    def from_records(cls, records: Sequence[Mapping]):
        from llm_mappo.o3_guard import reject_o3_provenance

        reject_o3_provenance(records, context="semantic/OOD dataset")
        valid = [record for record in records if record.get("validity") == 1]
        if not valid:
            return cls(
                np.empty((0, 61), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                None,
            )
        vectors = np.asarray([record["vector"] for record in valid], dtype=np.float64)
        scores = np.asarray([record["scores"] for record in valid], dtype=np.float64)
        if (
            vectors.ndim != 2
            or vectors.shape[1] != 61
            or scores.shape != (len(valid), 3)
        ):
            raise ValueError("Semantic records require 61D vectors and three scores.")
        if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(scores)):
            raise ValueError("Semantic records must be finite.")
        return cls(vectors, scores, TruncatedExponentialOOD.fit(vectors))

    def retrieve(self, query):
        query = np.asarray(query, dtype=np.float64)
        if (
            query.shape != (61,)
            or not np.all(np.isfinite(query))
            or len(self.vectors) < 3
        ):
            return np.zeros(3, dtype=np.float32), 0.0, 0.0
        distances = np.sqrt(np.mean((self.vectors - query) ** 2, axis=1))
        indices = np.argsort(distances, kind="stable")[:3]
        selected = distances[indices]
        if np.any(selected == 0):
            target = self.scores[indices[selected == 0]].mean(axis=0)
        else:
            weights = (selected + 1e-6) ** -2
            target = (
                self.scores[indices] * (weights / weights.sum())[:, None]
            ).sum(axis=0)
        reliability = self.ood.reliability(query) if self.ood is not None else 0.0
        return target.astype(np.float32), 1.0, reliability


@dataclass(frozen=True)
class TruncatedExponentialOOD:
    """Frozen validity-independent OOD reliability in semantic-view-v3 space."""

    mean: np.ndarray
    scale: np.ndarray
    q95: float
    q99: float
    normalized_references: np.ndarray

    @classmethod
    def fit(cls, references: np.ndarray) -> "TruncatedExponentialOOD":
        values = np.asarray(references, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 61 or len(values) < 3:
            raise ValueError("OOD references require at least three 61D records.")
        if not np.all(np.isfinite(values)):
            raise ValueError("OOD references must be finite.")
        mean = values.mean(axis=0)
        scale = np.maximum(values.std(axis=0), 1e-3)
        normalized = (values - mean) / scale
        distances = []
        for index, value in enumerate(normalized):
            others = np.delete(normalized, index, axis=0)
            nearest = np.sort(
                np.sqrt(np.mean((others - value) ** 2, axis=1))
            )[:3]
            distances.append(nearest.mean())
        q95, q99 = np.quantile(distances, [0.95, 0.99], method="linear")
        if not np.isfinite(q95) or not np.isfinite(q99) or q95 <= 1e-6 or q99 < q95:
            raise ValueError("OOD quantiles are invalid.")
        return cls(mean, scale, float(q95), float(q99), normalized)

    def reliability(self, query: np.ndarray) -> float:
        value = np.asarray(query, dtype=np.float64)
        if value.shape != (61,):
            raise ValueError("OOD query must contain 61 values.")
        if not np.all(np.isfinite(value)):
            return 0.0
        normalized = (value - self.mean) / self.scale
        distances = np.sqrt(
            np.mean((self.normalized_references - normalized) ** 2, axis=1)
        )
        distance = float(np.sort(distances)[:3].mean())
        return 0.0 if distance > self.q99 else float(np.exp(-distance / self.q95))


def _one_hot(value, choices):
    if value not in choices:
        raise ValueError("Invalid semantic category.")
    return [float(value == item) for item in choices]


def _relative(orientation, dx, dy):
    coordinates = {
        "up": (-dy, dx), "down": (dy, -dx),
        "left": (-dx, -dy), "right": (dx, dy),
    }
    return coordinates[orientation]
