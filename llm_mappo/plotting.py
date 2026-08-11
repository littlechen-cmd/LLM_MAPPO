"""Phase 3 plotting utilities.

Reads training artifacts (``episodes.csv`` and ``updates.csv``) produced by
:mod:`llm_mappo.phase3_training` and renders publication-quality figures to a
local directory.  All figures are also written as TensorBoard scalar curves by
the training loop itself, so this module focuses on static PNG/PDF outputs for
reports and offline analysis.

The module avoids matplotlib GUI back-ends on import, so it can run headless on
Windows servers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Force a non-interactive backend before importing pyplot so the module works
# on headless Windows sessions.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Shared style configuration
# ---------------------------------------------------------------------------

FIGURE_DPI = 150
DEFAULT_FIGSIZE = (8, 5)

# Stable colours for the priority labels used by Phase 3.
PRIORITY_COLORS = {
    "A": "#d62728",  # red – highest priority
    "B": "#1f77b4",  # blue
    "C": "#2ca02c",  # green
    "none": "#7f7f7f",  # grey
}

PHASE2_BASELINE_COLLISIONS = 9.585
PHASE2_BASELINE_COMPLETION = 0.973
GATE_COMPLETION = 0.95
GATE_COLLISIONS = 2.0
GATE_DEADLOCK = 0.05


def _configure_style() -> None:
    """Apply a clean default style for all figures."""
    plt.rcParams.update(
        {
            "figure.figsize": DEFAULT_FIGSIZE,
            "figure.dpi": FIGURE_DPI,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "savefig.bbox": "tight",
        }
    )


_configure_style()


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


@dataclass
class TrainingData:
    """Container for the rows written by :func:`train_phase3`."""

    episodes: List[Dict[str, object]]
    updates: List[Dict[str, object]]
    summary: Optional[Dict[str, object]]
    run_dir: Path


def load_training_data(run_dir: str | Path) -> TrainingData:
    """Load ``episodes.csv``, ``updates.csv`` and ``summary.json`` for a run."""
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")
    episodes = _read_csv(run_path / "episodes.csv")
    updates = _read_csv(run_path / "updates.csv")
    summary = _read_json(run_path / "summary.json")
    return TrainingData(
        episodes=episodes,
        updates=updates,
        summary=summary,
        run_dir=run_path,
    )


def _read_csv(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [_coerce_row(row) for row in reader]


def _read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_row(row: Dict[str, str]) -> Dict[str, object]:
    """Convert CSV string values to int/float/bool where possible."""
    converted: Dict[str, object] = {}
    for key, value in row.items():
        if value is None or value == "":
            converted[key] = None
            continue
        try:
            converted[key] = int(value)
        except ValueError:
            try:
                converted[key] = float(value)
            except ValueError:
                if value.lower() in {"true", "false"}:
                    converted[key] = value.lower() == "true"
                else:
                    converted[key] = value
    return converted


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------


def moving_average(values: Sequence[float], window: int = 20) -> np.ndarray:
    """Return a centred moving average; preserves length via edge clamping."""
    if not values:
        return np.asarray([], dtype=float)
    array = np.asarray(values, dtype=float)
    if window <= 1:
        return array
    kernel = np.ones(window) / window
    padded = np.pad(array, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _episodes_axis(episodes: Sequence[Dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [int(record.get("episode", index + 1)) for index, record in enumerate(episodes)],
        dtype=int,
    )


def _updates_axis(updates: Sequence[Dict[str, object]]) -> np.ndarray:
    return np.asarray(
        [int(record.get("update", index + 1)) for index, record in enumerate(updates)],
        dtype=int,
    )


# ---------------------------------------------------------------------------
# Figure writers – training curves
# ---------------------------------------------------------------------------


def plot_episode_metric(
    data: TrainingData,
    field: str,
    output_path: Path,
    *,
    title: str,
    ylabel: str,
    gate: Optional[float] = None,
    gate_label: Optional[str] = None,
    smooth_window: int = 20,
    baseline: Optional[float] = None,
    baseline_label: Optional[str] = None,
) -> Path:
    """Render a single per-episode metric with optional smoothing and gates."""
    episodes = _episodes_axis(data.episodes)
    values = [float(record.get(field, np.nan)) for record in data.episodes]
    if not values or np.all(np.isnan(values)):
        return output_path
    fig, ax = plt.subplots()
    ax.plot(episodes, values, alpha=0.35, color="#1f77b4", label="raw")
    smoothed = moving_average(values, smooth_window)
    ax.plot(
        episodes, smoothed, color="#1f77b4", linewidth=2,
        label=f"smooth({smooth_window})",
    )
    if gate is not None:
        ax.axhline(
            gate, color="#d62728", linestyle="--", linewidth=1.2,
            label=gate_label or "gate",
        )
    if baseline is not None:
        ax.axhline(
            baseline,
            color="#9467bd",
            linestyle=":",
            linewidth=1.2,
            label=baseline_label or "baseline",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    _save(fig, output_path)
    return output_path


def plot_priority_completion_steps(data: TrainingData, output_path: Path) -> Path:
    """Per-priority mean completion steps — Phase 3 specific."""
    series: Dict[str, List[Tuple[int, float]]] = {}
    for record in data.episodes:
        episode = int(record.get("episode", 0))
        for key, value in record.items():
            if key.startswith("priority_") and key.endswith(
                "_mean_completion_steps"
            ):
                label = key[len("priority_"):-len("_mean_completion_steps")]
                if value is not None:
                    series.setdefault(label, []).append((episode, float(value)))
    if not series:
        return output_path
    fig, ax = plt.subplots()
    for label in sorted(series):
        points = series[label]
        x = np.asarray([point[0] for point in points], dtype=int)
        values = [point[1] for point in points]
        colour = PRIORITY_COLORS.get(label, "#7f7f7f")
        ax.plot(
            x, moving_average(values, 20), color=colour, linewidth=2,
            label=f"Priority {label}",
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean completion steps")
    ax.set_title("Per-priority task completion latency")
    ax.legend(loc="best")
    _save(fig, output_path)
    return output_path


def plot_loss_curves(data: TrainingData, output_path: Path) -> Path:
    """Policy / value / entropy loss curves from ``updates.csv``."""
    if not data.updates:
        return output_path
    updates = _updates_axis(data.updates)
    fields = ["policy_loss", "value_loss", "entropy"]
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 4))
    for ax, field in zip(axes, fields):
        values = [float(record.get(field, np.nan)) for record in data.updates]
        ax.plot(updates, values, color="#ff7f0e", linewidth=1.5)
        ax.set_xlabel("Update")
        ax.set_ylabel(field)
        ax.set_title(field.replace("_", " ").title())
    fig.suptitle("Training loss / entropy")
    _save(fig, output_path)
    return output_path


def plot_engagement_loss(data: TrainingData, output_path: Path) -> Path:
    """Engagement-distillation MSE curve — Phase 3 specific."""
    if not data.updates:
        return output_path
    updates = _updates_axis(data.updates)
    values = [float(record.get("engagement_loss", np.nan)) for record in data.updates]
    if np.all(np.isnan(values)):
        return output_path
    fig, ax = plt.subplots()
    ax.plot(updates, values, color="#2ca02c", linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Engagement MSE")
    ax.set_title("Phase 3 engagement-head distillation loss")
    _save(fig, output_path)
    return output_path


def plot_reservation_kl(data: TrainingData, output_path: Path) -> Path:
    """Reservation-KL curve — only populated in Phase 3b."""
    if not data.updates:
        return output_path
    updates = _updates_axis(data.updates)
    values = [float(record.get("reservation_kl", np.nan)) for record in data.updates]
    if np.all(np.isnan(values)):
        return output_path
    fig, ax = plt.subplots()
    ax.plot(updates, values, color="#9467bd", linewidth=2)
    ax.set_xlabel("Update")
    ax.set_ylabel("Reservation KL")
    ax.set_title("A* reservation-KL distillation (Phase 3b)")
    _save(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Evaluation comparison
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Aggregated evaluation metrics for a single configuration."""

    label: str
    completion: float
    collisions: float
    deadlock: float
    completion_std: float = 0.0
    collisions_std: float = 0.0
    deadlock_std: float = 0.0


def plot_eval_comparison(results: Sequence[EvalResult], output_path: Path) -> Path:
    """Bar chart comparing completion / collisions / deadlock across configs."""
    if not results:
        return output_path
    labels = [r.label for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("task_completion_rate", "completion", GATE_COMPLETION),
        ("mean_collisions_per_episode", "collisions", GATE_COLLISIONS),
        ("deadlock_rate", "deadlock", GATE_DEADLOCK),
    ]
    colors = ["#1f77b4", "#d62728", "#9467bd"]
    for ax, (title, attr, gate) in zip(axes, metrics):
        means = [getattr(r, attr) for r in results]
        stds = [getattr(r, f"{attr}_std") for r in results]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
        ax.axhline(gate, color="#d62728", linestyle="--", linewidth=1.0, label="gate")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(title)
        ax.legend(loc="best")
    fig.suptitle("Phase 3 evaluation comparison")
    _save(fig, output_path)
    return output_path


def plot_phase2_vs_phase3(phase3_result: EvalResult, output_path: Path) -> Path:
    """Side-by-side comparison against the Phase 2 No-Go baseline."""
    results = [
        EvalResult(
            label="Phase 2 (No-Go)",
            completion=PHASE2_BASELINE_COMPLETION,
            collisions=PHASE2_BASELINE_COLLISIONS,
            deadlock=0.05,
        ),
        phase3_result,
    ]
    return plot_eval_comparison(results, output_path)


# ---------------------------------------------------------------------------
# Engagement diagnostic (requires sampled engagement data)
# ---------------------------------------------------------------------------


def plot_engagement_by_priority(
    samples: Sequence[Tuple[str, float]],
    output_path: Path,
) -> Path:
    """Box/violin plot of engagement outputs grouped by priority label.

    ``samples`` is a sequence of ``(label, engagement_value)`` tuples collected
    during evaluation.  ``label`` is one of ``"A"``, ``"B"``, ``"C"`` or
    ``"none"``.
    """
    if not samples:
        return output_path
    grouped: Dict[str, List[float]] = {}
    for label, value in samples:
        grouped.setdefault(label, []).append(float(value))
    order = [label for label in ("A", "B", "C", "none") if label in grouped]
    data = [grouped[label] for label in order]
    colors = [PRIORITY_COLORS.get(label, "#7f7f7f") for label in order]
    fig, ax = plt.subplots()
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body, colour in zip(parts["bodies"], colors):
        body.set_facecolor(colour)
        body.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_xlabel("Task priority label")
    ax.set_ylabel("Engagement head output")
    ax.set_title("Engagement head response by priority (Phase 3 diagnostic)")
    _save(fig, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def render_all_figures(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    phase2_comparison: Optional[EvalResult] = None,
    engagement_samples: Optional[Sequence[Tuple[str, float]]] = None,
) -> List[Path]:
    """Render every available figure for a Phase 3 run.

    Figures that lack source data are skipped silently so the function can be
    called on partial / in-progress runs.
    """
    data = load_training_data(run_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    # P0 training curves
    written.append(
        plot_episode_metric(
            data,
            "task_completion_rate",
            out / "episode_completion.png",
            title="Task completion rate",
            ylabel="Completion rate",
            gate=GATE_COMPLETION,
            gate_label="gate 0.95",
        )
    )
    written.append(
        plot_episode_metric(
            data,
            "collisions",
            out / "episode_collisions.png",
            title="Collisions per episode",
            ylabel="Collisions",
            gate=GATE_COLLISIONS,
            gate_label="gate 2.0",
            baseline=PHASE2_BASELINE_COLLISIONS,
            baseline_label="Phase 2 baseline",
        )
    )
    written.append(
        plot_episode_metric(
            data,
            "deadlocked",
            out / "episode_deadlock.png",
            title="Deadlock flag (moving average)",
            ylabel="Deadlock",
            gate=GATE_DEADLOCK,
            gate_label="gate 0.05",
        )
    )
    written.append(
        plot_episode_metric(
            data,
            "reward",
            out / "episode_reward.png",
            title="Episode reward",
            ylabel="Reward",
        )
    )
    written.append(
        plot_episode_metric(
            data,
            "agent_deaths",
            out / "episode_deaths.png",
            title="AGV deaths per episode",
            ylabel="Deaths",
        )
    )
    written.append(
        plot_episode_metric(
            data,
            "blocked_forwards",
            out / "episode_blocked.png",
            title="Blocked forwards per episode",
            ylabel="Blocked",
        )
    )
    # Phase 3 specific
    written.append(
        plot_priority_completion_steps(data, out / "priority_completion_steps.png")
    )
    written.append(plot_loss_curves(data, out / "training_losses.png"))
    written.append(plot_engagement_loss(data, out / "engagement_loss.png"))
    written.append(plot_reservation_kl(data, out / "reservation_kl.png"))
    # Phase 2 comparison
    if phase2_comparison is not None:
        written.append(
            plot_phase2_vs_phase3(phase2_comparison, out / "phase2_vs_phase3.png")
        )
    # Engagement diagnostic
    if engagement_samples:
        written.append(
            plot_engagement_by_priority(
                engagement_samples, out / "engagement_by_priority.png"
            )
        )
    return [path for path in written if path.exists()]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _save(fig, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return output_path
