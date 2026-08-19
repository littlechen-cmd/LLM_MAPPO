import numpy as np

from figures.core.plot_core_comparison import (
    GROUP_ORDER,
    METRICS,
    plot as plot_comparison,
)
from figures.core.plot_learning_curves import (
    PANELS,
    plot as plot_learning_curves,
)


def test_core_comparison_writes_png_and_svg(tmp_path):
    rows = []
    for group_index, group in enumerate(GROUP_ORDER):
        for metric, _ in METRICS:
            mean = float(group_index + 1)
            rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": mean - 0.1,
                    "ci95_high": mean + 0.1,
                }
            )
    output = tmp_path / "comparison"

    plot_comparison(rows, output)

    assert output.with_suffix(".png").stat().st_size > 0
    assert output.with_suffix(".svg").stat().st_size > 0


def test_learning_curves_write_png_and_svg(tmp_path):
    steps = np.linspace(1000.0, 10000.0, 20)
    curves = {}
    for group_index, group in enumerate(GROUP_ORDER):
        runs = []
        for seed_index in range(3):
            metrics = {
                metric: np.linspace(
                    group_index + seed_index / 10,
                    group_index + seed_index / 10 + 1,
                    len(steps),
                )
                for metric, _ in PANELS
            }
            runs.append((steps, metrics))
        curves[group] = runs
    output = tmp_path / "learning"

    plot_learning_curves(curves, output)

    assert output.with_suffix(".png").stat().st_size > 0
    assert output.with_suffix(".svg").stat().st_size > 0
