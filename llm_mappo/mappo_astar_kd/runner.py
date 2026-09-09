"""Phase 2 entrypoints, kept separate from the Phase 1 CLI surface."""

from __future__ import annotations

from pathlib import Path

from llm_mappo.mappo_baseline.runner import BaselineRun, load_baseline_experiment


def load_astar_kd_experiment(path: str | Path):
    """Load one Phase 2 A*KD training or checkpoint-evaluation profile."""

    experiment = load_baseline_experiment(path)
    if experiment.schema != "phase2-mappo-astar-kd-v1":
        raise ValueError("Phase 2 A*KD runner requires an A*KD experiment schema.")
    return experiment


class AStarKDRun(BaselineRun):
    """Named Phase 2 run type sharing the frozen recurrent MAPPO mechanics."""
