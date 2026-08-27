"""O3-B frozen ASCII-map and loaded-highway graph contracts."""

from collections import deque
from pathlib import Path

import pytest


LAYOUT_DIRECTORY = Path("rware/layouts/o3")
NARROW_PATH = LAYOUT_DIRECTORY / "unseen_narrow_passage_v1.txt"
CENTRAL_PATH = LAYOUT_DIRECTORY / "unseen_central_cross_v1.txt"

NARROW_CORRIDOR = tuple((x, 9) for x in range(7, 16))
NARROW_CHARGING_STATIONS = (
    (2, 0), (5, 3), (2, 13), (5, 19),
    (11, 0), (18, 3), (18, 13), (21, 19),
)
CENTRAL_CENTER = (11, 9)
CENTRAL_ARMS = {
    "north": ((11, 6), (11, 7), (11, 8)),
    "south": ((11, 10), (11, 11), (11, 12)),
    "west": ((8, 9), (9, 9), (10, 9)),
    "east": ((12, 9), (13, 9), (14, 9)),
}
CENTRAL_CHARGING_STATIONS = (
    (0, 0), (23, 0), (0, 19), (23, 19),
    (4, 0), (19, 0), (4, 19), (19, 19),
)


def _read_layout(path):
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    rows = raw.decode("utf-8").splitlines()
    assert len(rows) == 20
    assert {len(row) for row in rows} == {24}
    assert set("".join(rows)) <= {"X", ".", "G"}
    assert "".join(rows).count("X") == 144
    assert "".join(rows).count(".") == 334
    assert "".join(rows).count("G") == 2
    return rows


def _highways(rows):
    return {
        (x, y)
        for y, row in enumerate(rows)
        for x, cell in enumerate(row)
        if cell in ".G"
    }


def _neighbors(point):
    x, y = point
    return {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)}


def _components(nodes):
    remaining = set(nodes)
    components = []
    while remaining:
        start = min(remaining)
        component = {start}
        queue = deque([start])
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            for neighbor in _neighbors(current) & remaining:
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def _assert_common_contract(rows, charging_stations):
    highways = _highways(rows)
    assert len(_components(highways)) == 1
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell == "X":
                assert _neighbors((x, y)) & highways
    assert len(charging_stations) == len(set(charging_stations)) == 8
    for x, y in charging_stations:
        assert rows[y][x] == "."
        assert _neighbors((x, y)) & highways


@pytest.mark.parametrize(
    ("path", "stations"),
    [
        (NARROW_PATH, NARROW_CHARGING_STATIONS),
        (CENTRAL_PATH, CENTRAL_CHARGING_STATIONS),
    ],
)
def test_o3_maps_freeze_canonical_size_density_and_accessibility(path, stations):
    """Catch malformed O3 files or topology-only controls drifting."""
    _assert_common_contract(_read_layout(path), stations)


def test_narrow_map_has_one_eight_cell_loaded_transport_corridor():
    """Catch alternate cross-zone routes weakening the narrow-passage certificate."""
    rows = _read_layout(NARROW_PATH)
    highways = _highways(rows)
    goals = {
        (x, y)
        for y, row in enumerate(rows)
        for x, cell in enumerate(row)
        if cell == "G"
    }
    assert goals == {(0, 5), (0, 14)}
    assert set(NARROW_CORRIDOR) <= highways
    assert not (set(NARROW_CORRIDOR) & set(NARROW_CHARGING_STATIONS))
    assert any(
        rows[y][x] == "X" for y in range(20) for x in range(0, 7)
    )
    assert any(
        rows[y][x] == "X" for y in range(20) for x in range(16, 24)
    )
    for articulation in NARROW_CORRIDOR[1:-1]:
        components = _components(highways - {articulation})
        assert len(components) == 2
        assert any((0, 0) in item for item in components)
        assert any((23, 19) in item for item in components)


def test_central_map_has_one_four_way_loaded_transport_articulation():
    """Catch quadrant bypasses or an underspecified central crossing."""
    rows = _read_layout(CENTRAL_PATH)
    highways = _highways(rows)
    goals = {
        (x, y)
        for y, row in enumerate(rows)
        for x, cell in enumerate(row)
        if cell == "G"
    }
    assert goals == {(3, 3), (20, 16)}
    assert _neighbors(CENTRAL_CENTER) & highways == _neighbors(CENTRAL_CENTER)
    assert all(set(arm) <= highways for arm in CENTRAL_ARMS.values())
    critical = {CENTRAL_CENTER}.union(*map(set, CENTRAL_ARMS.values()))
    assert not (critical & set(CENTRAL_CHARGING_STATIONS))
    components = _components(highways - {CENTRAL_CENTER})
    assert len(components) == 4
    endpoints = {(11, 6), (11, 12), (8, 9), (14, 9)}
    assert {
        next(index for index, component in enumerate(components) if point in component)
        for point in endpoints
    } == {0, 1, 2, 3}
