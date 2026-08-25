import numpy as np

from llm_mappo.pure_motion_teacher import PureMotionQuery, PureMotionTeacher


def _query(**changes):
    values = {
        "layout_hash": "layout-v1",
        "width": 5,
        "height": 5,
        "blocked_coordinates": (),
        "own_pose": (1, 2),
        "orientation": "RIGHT",
        "goal": (4, 2),
        "occupied_coordinates": (),
        "pure_motion_mask": (False, True, True, True, False),
    }
    values.update(changes)
    return PureMotionQuery(**values)


def test_pure_motion_teacher_returns_a_finite_motion_only_preference():
    result = PureMotionTeacher().query(_query())

    assert result.valid
    assert result.failure_reason == "ok"
    assert result.motion_preferences.shape == (5,)
    assert result.motion_preferences.dtype == np.float32
    assert result.motion_preferences[0] == 0.0
    assert result.motion_preferences[4] == 0.0
    assert np.isclose(result.motion_preferences.sum(), 1.0)
    assert result.motion_preferences[1] > result.motion_preferences[2]
    assert result.motion_preferences[1] > result.motion_preferences[3]
    assert result.diagnostics["expanded_nodes"] <= 512


def test_pure_motion_teacher_is_deterministic_and_cache_keys_ignore_high_level_data():
    teacher = PureMotionTeacher()
    first = teacher.query(_query())
    second = teacher.query(_query())

    assert np.array_equal(first.motion_preferences, second.motion_preferences)
    assert second.diagnostics["cache_hit"] is True


def test_pure_motion_teacher_fail_closes_for_mandatory_toggle_and_no_root_action():
    mandatory = PureMotionTeacher().query(_query(mandatory_toggle_load=True))
    no_root = PureMotionTeacher().query(
        _query(pure_motion_mask=(False, False, False, False, False))
    )

    assert not mandatory.valid
    assert mandatory.failure_reason == "mandatory_toggle_load"
    assert np.array_equal(mandatory.motion_preferences, np.zeros(5, dtype=np.float32))
    assert not no_root.valid
    assert no_root.failure_reason == "no_physical_root_action"


def test_pure_motion_teacher_treats_current_anonymous_occupancy_as_a_blocker():
    unoccupied = PureMotionTeacher().query(_query())
    occupied = PureMotionTeacher().query(_query(occupied_coordinates=((2, 2),)))

    assert unoccupied.valid
    assert occupied.valid
    assert occupied.motion_preferences[1] == 0.0
    assert not np.array_equal(unoccupied.motion_preferences, occupied.motion_preferences)


def test_pure_motion_teacher_marks_budget_exhaustion_when_no_root_is_certified():
    result = PureMotionTeacher(expansion_budget=1).query(_query())

    assert not result.valid
    assert result.failure_reason == "budget_exceeded"
    assert np.array_equal(result.motion_preferences, np.zeros(5, dtype=np.float32))
