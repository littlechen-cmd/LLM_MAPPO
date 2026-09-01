"""Frozen three-dimensional E1 semantic controls; never modify raw LLM records."""

from collections import defaultdict
import hashlib
from typing import Mapping, Sequence

import numpy as np

from llm_mappo.semantic_v3 import SemanticDatasetV3


def rule_kd_v3(records: Sequence[Mapping]) -> SemanticDatasetV3:
    """Derive the preregistered v3 rule teacher from deidentified semantic views."""
    derived = []
    for record in records:
        if record.get("validity") != 1: continue
        focal = record["semantic_view"]["focal"]; neighbors = record["semantic_view"]["neighbors"]
        persistence = 0.1 if focal["target_kind"] == "charging" or focal["battery_ratio"] <= .30 else (.9 if focal["loaded"] else (.8 if focal["priority_present"] else .5))
        close = any(item["mask"] and item["normalized_manhattan_distance"] <= .15 for item in neighbors)
        yielding = .8 if focal["priority_present"] and focal["priority_rank"] == 0 else (.2 if close else .6)
        risk = .8 if close else (.5 if sum(focal["adjacent_highway"].values()) <= 2 else .2)
        derived.append({"vector": record["vector"], "scores": [persistence, yielding, risk], "validity": 1})
    return SemanticDatasetV3.from_records(derived)


def shuffle_kd_v3(records: Sequence[Mapping]) -> SemanticDatasetV3:
    """Jointly derange valid 3D tuples within each pre-registered stratum."""
    groups = defaultdict(list)
    for record in records:
        if record.get("validity") == 1: groups[record["stratum"]].append(record)
    derived = []
    for stratum, rows in sorted(groups.items()):
        if len(rows) < 2: raise ValueError("ShuffleKD-v3 requires at least two valid records per stratum.")
        shift = int.from_bytes(hashlib.sha256(("shuffle-kd-v3|" + stratum).encode()).digest()[:8], "big") % (len(rows) - 1) + 1
        for index, row in enumerate(rows):
            donor = rows[(index + shift) % len(rows)]
            derived.append({"vector": row["vector"], "scores": donor["scores"], "validity": 1})
    return SemanticDatasetV3.from_records(derived)
