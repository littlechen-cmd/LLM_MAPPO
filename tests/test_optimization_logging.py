"""O1 compact evidence logging contracts."""

import pytest

from llm_mappo.optimization_logging import validate_o0_log_record


def test_log_record_rejects_nonfinite_or_full_teacher_arrays():
    record = {
        "event": "update",
        "real_env_steps": 32,
        "loss_total": 1.0,
        "planner_query_count": 0,
        "teacher_valid_count": 3,
        "shadow_attempted_count": 1,
        "pollution_counters": {"astar": 0, "llm": 0, "planner": 0},
    }
    validate_o0_log_record(record)
    with pytest.raises(ValueError, match="finite"):
        validate_o0_log_record({**record, "loss_total": float("nan")})
    with pytest.raises(ValueError, match="forbidden"):
        validate_o0_log_record({**record, "teacher_preferences": [[1.0, 0.0, 0.0]]})
