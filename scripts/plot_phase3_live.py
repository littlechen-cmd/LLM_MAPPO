"""Live Phase 3 figure refresher.

Watches a Phase 3 run directory and re-renders the figure set whenever
``episodes.csv`` or ``updates.csv`` changes.  Use alongside
``tensorboard --logdir ...`` for the scalar view; this script keeps the PNG
files in ``figures/`` up to date for quick inspection without a browser.

Usage::

    python scripts/plot_phase3_live.py --run-dir artifacts/phase3a_dual_head/seed_007
    python scripts/plot_phase3_live.py --run-dir ... \
        --interval 30 --output-dir figures/phase3a

The script polls the CSV mtimes instead of relying on watchdog so it works on
any platform without extra dependencies.  Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from llm_mappo.plotting import render_all_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Seconds between refreshes (default: 15)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Render once and exit instead of polling",
    )
    return parser.parse_args()


def _mtimes(run_dir: Path) -> tuple[float, float]:
    ep_path = run_dir / "episodes.csv"
    up_path = run_dir / "updates.csv"
    episodes = ep_path.stat().st_mtime if ep_path.exists() else 0.0
    updates = up_path.stat().st_mtime if up_path.exists() else 0.0
    return episodes, updates


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    last_ep, last_up = 0.0, 0.0
    iteration = 0
    try:
        while True:
            iteration += 1
            ep_mtime, up_mtime = _mtimes(run_dir)
            changed = (ep_mtime != last_ep) or (up_mtime != last_up)
            if changed or iteration == 1:
                try:
                    written = render_all_figures(run_dir, output_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{time.strftime('%H:%M:%S')}] error: {exc}")
                else:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] rendered {len(written)} figures "
                        f"-> {output_dir.resolve()}"
                    )
                last_ep, last_up = ep_mtime, up_mtime
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
