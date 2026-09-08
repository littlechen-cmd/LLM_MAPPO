"""Render owner-review previews for the Phase 1 RWARE-derived layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from llm_mappo.environment import DynamicWarehouse
from rware.warehouse import RewardType


PRIORITY_COLORS = {
    "A": "#d55e00",
    "B": "#e69f00",
    "C": "#0072b2",
    "D": "#6a3d9a",
}
CHARGER_COLOR = "#009e73"
PICKING_COLOR = "#cc79a7"
SHELF_COLOR = "#4c566a"
HIGHWAY_COLOR = "#f5f5f5"


@dataclass(frozen=True)
class LayoutProfile:
    """Static owner-review proposal; not a training configuration."""

    name: str
    shelf_rows: int
    n_agents: int
    workload: str
    request_queue_size: int
    batch_interval: int | None
    batch_size_range: tuple[int, int] | None
    task_completion_target: int | None
    charging_stations: tuple[tuple[int, int], ...]
    picking_stations: tuple[tuple[int, int], ...]
    priority_schedule: tuple[str, ...] = ("A", "B", "C", "D")
    max_steps: int = 500
    battery_cost_scale: float = 1.10
    charge_threshold: float = 0.30
    charge_release_threshold: float = 0.80


PROFILES = (
    LayoutProfile(
        name="medium-3ag-low-workload",
        shelf_rows=2,
        n_agents=3,
        workload="low",
        request_queue_size=3,
        batch_interval=120,
        batch_size_range=(1, 1),
        task_completion_target=6,
        charging_stations=((0, 0), (19, 0), (0, 23)),
        picking_stations=((9, 21),),
    ),
    LayoutProfile(
        name="medium-5ag-low-workload",
        shelf_rows=2,
        n_agents=5,
        workload="low",
        request_queue_size=5,
        batch_interval=120,
        batch_size_range=(1, 1),
        task_completion_target=8,
        charging_stations=((0, 0), (19, 0), (0, 23), (19, 23), (9, 0)),
        picking_stations=((9, 21), (10, 21)),
    ),
    LayoutProfile(
        name="large-5ag-high-workload",
        shelf_rows=3,
        n_agents=5,
        workload="to-be-determined",
        request_queue_size=5,
        batch_interval=None,
        batch_size_range=None,
        task_completion_target=None,
        charging_stations=((0, 0), (19, 0), (0, 32), (19, 32), (9, 0)),
        picking_stations=((9, 30), (10, 30)),
    ),
)


def _make_environment(profile: LayoutProfile) -> DynamicWarehouse:
    environment = DynamicWarehouse(
        column_height=8,
        shelf_rows=profile.shelf_rows,
        shelf_columns=5,
        n_agents=profile.n_agents,
        msg_bits=0,
        sensor_range=4,
        request_queue_size=profile.request_queue_size,
        max_inactivity_steps=None,
        max_steps=profile.max_steps,
        reward_type=RewardType.INDIVIDUAL,
        # Defaults only instantiate the layout for a static preview; they are not
        # a proposed workload for the large layout.
        batch_interval=profile.batch_interval or 120,
        batch_size_range=profile.batch_size_range or (1, 1),
        charging_stations=profile.charging_stations,
        battery_cost_scale=profile.battery_cost_scale,
        outer_aisle_width=2,
        initial_priority_label="A",
        priority_schedule=profile.priority_schedule,
        task_completion_target=profile.task_completion_target or 2,
    )
    environment.reset(seed=7)
    return environment


def _reachable_cells(environment: DynamicWarehouse) -> set[tuple[int, int]]:
    walkable = {
        (x, y)
        for y in range(environment.grid_size[0])
        for x in range(environment.grid_size[1])
        if environment._is_highway(x, y)
    }
    pending = [next(iter(walkable))]
    reached = set(pending)
    while pending:
        x, y = pending.pop()
        for neighbour in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if neighbour in walkable and neighbour not in reached:
                reached.add(neighbour)
                pending.append(neighbour)
    return reached


def _validate_profile(
    environment: DynamicWarehouse, profile: LayoutProfile
) -> dict[str, object]:
    reachable = _reachable_cells(environment)
    station_cells = set(profile.charging_stations) | set(profile.picking_stations)
    if len(profile.charging_stations) != profile.n_agents:
        raise ValueError("Phase 1 requires exactly one charger per AGV.")
    if len(profile.picking_stations) != profile.n_agents // 2:
        raise ValueError("Phase 1 picking-station count must equal floor(AGV / 2).")
    if len(station_cells) != len(profile.charging_stations) + len(profile.picking_stations):
        raise ValueError("Charging and picking stations must not overlap.")
    if not set(profile.picking_stations).issubset(set(environment.goals)):
        raise ValueError("Picking stations must be a subset of RWARE delivery goals.")
    if not station_cells.issubset(reachable):
        raise ValueError("Every proposed station must be reachable.")
    return {
        "grid_width": environment.grid_size[1],
        "grid_height": environment.grid_size[0],
        "walkable_cells": len(reachable),
        "all_walkable_cells_connected": len(reachable)
        == sum(int(environment._is_highway(x, y))
               for y in range(environment.grid_size[0])
               for x in range(environment.grid_size[1])),
        "rware_goals": [list(item) for item in environment.goals],
    }


def _priority_shelves(
    environment: DynamicWarehouse
) -> Iterable[tuple[tuple[int, int], str]]:
    shelves = sorted((shelf.x, shelf.y) for shelf in environment.shelfs)
    indices = (0, len(shelves) // 3, 2 * len(shelves) // 3, len(shelves) - 1)
    for index, priority in zip(indices, ("A", "B", "C", "D")):
        yield shelves[index], priority


def _draw_preview(
    environment: DynamicWarehouse, profile: LayoutProfile, output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10, 12))
    width = environment.grid_size[1]
    height = environment.grid_size[0]
    axis.set_facecolor("#ffffff")

    for y in range(height):
        for x in range(width):
            color = HIGHWAY_COLOR if environment._is_highway(x, y) else SHELF_COLOR
            axis.add_patch(Rectangle((x, y), 1, 1, facecolor=color, edgecolor="#dddddd"))

    for (x, y), priority in _priority_shelves(environment):
        axis.add_patch(
            Rectangle((x, y), 1, 1, facecolor=PRIORITY_COLORS[priority], edgecolor="#333333")
        )
    for x, y in profile.charging_stations:
        axis.add_patch(Rectangle((x, y), 1, 1, facecolor=CHARGER_COLOR, edgecolor="#222222"))
    for x, y in profile.picking_stations:
        axis.add_patch(Rectangle((x, y), 1, 1, facecolor=PICKING_COLOR, edgecolor="#222222"))

    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_xticks(range(width + 1))
    axis.set_yticks(range(height + 1))
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.grid(False)
    axis.set_title(profile.name, pad=18)
    axis.legend(
        handles=[
            Patch(facecolor=PRIORITY_COLORS["A"], label="Priority A"),
            Patch(facecolor=PRIORITY_COLORS["B"], label="Priority B"),
            Patch(facecolor=PRIORITY_COLORS["C"], label="Priority C"),
            Patch(facecolor=PRIORITY_COLORS["D"], label="Priority D"),
            Patch(facecolor=CHARGER_COLOR, label="Charging station"),
            Patch(facecolor=PICKING_COLOR, label="Picking station"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict[str, object]] = {}
    for profile in PROFILES:
        environment = _make_environment(profile)
        try:
            report[profile.name] = {
                "profile": asdict(profile),
                "audit": _validate_profile(environment, profile),
            }
            _draw_preview(environment, profile, arguments.output / f"{profile.name}.png")
        finally:
            environment.close()
    (arguments.output / "profile_proposal.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
