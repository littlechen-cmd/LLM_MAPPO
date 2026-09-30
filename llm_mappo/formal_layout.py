"""Validated conversion of the approved static warehouse JSON to RWARE layout."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


FORMAL_MAP_PATH = Path(__file__).resolve().parent / "data" / "formal_environment.json"
FORMAL_MAP_SHA256 = (
    "2a03a4abfe281de6865f39399d870f5ba9931d2520a6829bdb01788d50ca22fc"
)


def load_formal_layout() -> tuple[str, tuple[tuple[int, int], ...],  # noqa: C901
                                  tuple[tuple[int, int], ...]]:
    """Return RWARE layout and stations, rejecting an altered map."""

    raw = FORMAL_MAP_PATH.read_bytes()
    if sha256(raw).hexdigest() != FORMAL_MAP_SHA256:
        raise ValueError("Formal warehouse map hash differs from approved JSON.")
    data = json.loads(raw)
    width = int(data["map"]["width"])
    height = int(data["map"]["height"])
    if (width, height) != (50, 28):
        raise ValueError("Formal warehouse map dimensions are incompatible.")
    grid = [["."] * width for _ in range(height)]
    for x, y, block_width, block_height in data["shelves"]:
        for shelf_y in range(y, y + block_height):
            for shelf_x in range(x, x + block_width):
                if not (0 <= shelf_x < width and 0 <= shelf_y < height):
                    raise ValueError("Formal warehouse shelf lies outside the map.")
                if grid[shelf_y][shelf_x] != ".":
                    raise ValueError("Formal warehouse shelves overlap.")
                grid[shelf_y][shelf_x] = "X"
    stations = []
    for kind in ("picking_stations", "charging_stations"):
        coordinates = tuple(
            tuple(point)
            for side in ("left", "right")
            for point in data[kind][side]
        )
        for x, y in coordinates:
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError("Formal warehouse station lies outside the map.")
            if grid[y][x] != ".":
                raise ValueError("Formal warehouse stations overlap shelves or goals.")
            if kind == "picking_stations":
                grid[y][x] = "g"
        stations.append(coordinates)
    picking, charging = stations
    if (sum(row.count("X") for row in grid), len(picking), len(charging)) != (
        504, 10, 8
    ):
        raise ValueError("Formal warehouse cell and station counts differ.")
    return "\n".join("".join(row) for row in grid), picking, charging
