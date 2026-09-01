"""Fail-closed evidence aggregation for the owner-run E1 CUDA functional smoke."""

import json
from pathlib import Path


_EXPECTED = {
    ("MAPPO-DG", 9001), ("Fixed-AStarKD+LLMKD", 9002),
    ("RuleKD-v3", 9003), ("NoOOD-v1", 9004),
    ("RC-AStarKD+LLMKD", 9001), ("QMIX-DG", 9002),
    ("ShuffleKD-v3", 9003), ("NoGoalHint-v1", 9004),
}


def frozen_smoke_waves() -> tuple[tuple[tuple[str, int, int], ...], ...]:
    """Return the only two owner-approved four-slot smoke waves."""
    return (
        (("MAPPO-DG", 9001, 0), ("Fixed-AStarKD+LLMKD", 9002, 0),
         ("RuleKD-v3", 9003, 1), ("NoOOD-v1", 9004, 1)),
        (("RC-AStarKD+LLMKD", 9001, 0), ("QMIX-DG", 9002, 0),
         ("ShuffleKD-v3", 9003, 1), ("NoGoalHint-v1", 9004, 1)),
    )


def aggregate_cuda_smoke(root: str | Path) -> dict:
    """Check identity, 128→256 resume, device binding and non-performance safety."""
    records = []
    for path in Path(root).rglob("smoke_receipt.json"):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    identities = {(item.get("group"), item.get("seed")) for item in records}
    if identities != _EXPECTED:
        raise ValueError("E1 CUDA smoke does not contain the exact eight frozen runs.")
    for item in records:
        if item.get("steps_before_resume") != 128 or item.get("steps_after_resume") != 256:
            raise ValueError("E1 CUDA smoke did not prove 128-to-256 resume.")
        if item.get("planner_query_count") != 0 or item.get("online_llm_calls") != 0:
            raise ValueError("E1 CUDA smoke violates the zero-call contract.")
        if item.get("finite") is not True or item.get("device") not in {"cuda:0", "cuda:1"}:
            raise ValueError("E1 CUDA smoke device or numerical evidence is incompatible.")
    gpu_counts = {gpu: sum(item["physical_gpu"] == gpu for item in records) for gpu in (0, 1)}
    if gpu_counts != {0: 4, 1: 4}:
        raise ValueError("E1 CUDA smoke must contain four runs per physical GPU.")
    return {"schema": "e1-cuda-smoke-aggregate-v1", "pass": True,
            "run_count": 8, "total_environment_steps": 2048, "gpu_run_counts": gpu_counts}
