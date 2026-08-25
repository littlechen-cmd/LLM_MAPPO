"""Compact O0 evidence logs; full per-state teacher arrays are prohibited."""

import json
from pathlib import Path
from typing import Mapping

import numpy as np


_REQUIRED = {
    "event",
    "real_env_steps",
    "loss_total",
    "planner_query_count",
    "teacher_valid_count",
    "shadow_attempted_count",
    "pollution_counters",
}
_FORBIDDEN = {"teacher_preferences", "teacher_arrays", "full_state_arrays"}


def validate_o0_log_record(record: Mapping) -> None:
    """Fail closed on incomplete, non-finite, or oversized evidence records."""
    missing = _REQUIRED - set(record)
    if missing:
        raise ValueError(f"O0 log record is missing {sorted(missing)[0]}.")
    forbidden = _FORBIDDEN & set(record)
    if forbidden:
        raise ValueError(f"O0 log record contains forbidden {sorted(forbidden)[0]}.")
    for name, value in record.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise ValueError(f"O0 log field {name} must be finite.")
    counters = record["pollution_counters"]
    if not isinstance(counters, Mapping) or any(
        value != 0 for value in counters.values()
    ):
        raise ValueError("O0 pollution counters must all be zero.")


class O0RunLogger:
    """Append compact validated JSONL records to an optimization artifact directory."""

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.path = self.output_directory / "o0_evidence.jsonl"

    def write(self, record: Mapping) -> None:
        validate_o0_log_record(record)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
