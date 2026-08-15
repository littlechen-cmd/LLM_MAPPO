"""Deterministic software frames for headless Phase 2 visualizations."""

from __future__ import annotations

import numpy as np

from rware.warehouse import Direction


def render_warehouse_frame(warehouse, cell_size: int = 32) -> np.ndarray:
    """Draw a warehouse snapshot without relying on an OpenGL frame buffer."""
    if cell_size < 8:
        raise ValueError("cell_size must be at least 8 pixels.")
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Software rendering requires Pillow.") from error

    rows, columns = warehouse.grid_size
    image = Image.new("RGB", (columns * cell_size + 1, rows * cell_size + 1), "white")
    draw = ImageDraw.Draw(image)
    for row in range(rows + 1):
        y = row * cell_size
        draw.line((0, y, columns * cell_size, y), fill="black", width=1)
    for column in range(columns + 1):
        x = column * cell_size
        draw.line((x, 0, x, rows * cell_size), fill="black", width=1)

    for x, y in warehouse.goals:
        _fill_cell(draw, x, y, cell_size, "#3c3c3c")
        _label_cell(draw, x, y, cell_size, "G", "white")
    _draw_charging_stations(draw, warehouse, cell_size)

    requested = set(warehouse.request_queue)
    task_labels = {
        task.shelf_id: task.label for task in warehouse.task_queue.active_tasks
    }
    for shelf in warehouse.shelfs:
        color = "#008080" if shelf in requested else "#483d8b"
        _fill_cell(draw, shelf.x, shelf.y, cell_size, color, padding=3)
        label = task_labels.get(shelf.id)
        if label is not None:
            _label_cell(draw, shelf.x, shelf.y, cell_size, label, "#ffffff")

    for agent in warehouse.agents:
        color = "#ff0000" if agent.carrying_shelf is not None else "#ff8c00"
        left = agent.x * cell_size + cell_size // 4
        top = agent.y * cell_size + cell_size // 4
        right = (agent.x + 1) * cell_size - cell_size // 4
        bottom = (agent.y + 1) * cell_size - cell_size // 4
        draw.ellipse((left, top, right, bottom), fill=color, outline="black")
        center_x = (agent.x + 0.5) * cell_size
        center_y = (agent.y + 0.5) * cell_size
        dx, dy = _direction_vector(agent.dir)
        draw.line(
            (
                center_x,
                center_y,
                center_x + dx * cell_size / 4,
                center_y + dy * cell_size / 4,
            ),
            fill="black",
            width=2,
        )
        _label_cell(draw, agent.x, agent.y, cell_size, str(agent.id), "black")
    return np.asarray(image)


def _draw_charging_stations(draw, warehouse, cell_size: int) -> None:
    """Show station availability and low-battery reservations in each frame."""
    reservations = getattr(warehouse, "charging_reservations", {})
    occupied_stations = {
        (agent.x, agent.y): agent.id
        for agent in warehouse.agents
        if (agent.x, agent.y) in getattr(warehouse, "charging_stations", ())
    }
    for x, y in getattr(warehouse, "charging_stations", ()):
        station = x, y
        color = "#b22222" if station in occupied_stations else "#008000"
        label = str(reservations[station]) if station in reservations else "C"
        _label_cell(draw, x, y, cell_size, label, color)


def _fill_cell(draw, x, y, cell_size, color, padding: int = 0) -> None:
    draw.rectangle(
        (
            x * cell_size + 1 + padding,
            y * cell_size + 1 + padding,
            (x + 1) * cell_size - 1 - padding,
            (y + 1) * cell_size - 1 - padding,
        ),
        fill=color,
    )


def _label_cell(draw, x, y, cell_size, text: str, color: str) -> None:
    draw.text(
        ((x + 0.5) * cell_size, (y + 0.5) * cell_size),
        text,
        fill=color,
        anchor="mm",
    )


def _direction_vector(direction: Direction) -> tuple[int, int]:
    return {
        Direction.UP: (0, -1),
        Direction.DOWN: (0, 1),
        Direction.LEFT: (-1, 0),
        Direction.RIGHT: (1, 0),
    }[direction]
