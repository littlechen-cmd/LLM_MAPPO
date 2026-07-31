"""Render a trained Phase 2 MAPPO checkpoint as a paced window or GIF."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
from time import sleep
from typing import Sequence

import numpy as np

from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase2_training import load_policy
from llm_mappo.visualization import render_warehouse_frame


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="artifacts/phase2_small_astar_bc/seed_007/checkpoint_final.pt",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", choices=("rgb_array", "human"), default="rgb_array")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--output",
        help="GIF output path. Defaults beside the checkpoint for rgb_array mode.",
    )
    return parser.parse_args()


def run_visualization(
    checkpoint_path: str | Path,
    seed: int,
    episodes: int,
    render_mode: str,
    fps: float,
    output_path: str | Path | None = None,
) -> dict:
    """Run deterministic checkpoint actions and optionally record RGB frames."""
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    policy, config, _ = load_policy(checkpoint_path)
    env = Phase2Warehouse(
        n_agents=config["n_agents"],
        max_steps=config["max_steps"],
        env_id=config.get("env_id", "llm-mappo-medium-3ag-v1"),
        waypoint_reward=config.get("waypoint_reward", 1.0),
        oracle_interaction_mask=config.get("oracle_interaction_mask", True),
        deadlock_steps=config.get("deadlock_steps", 120),
        render_mode="human" if render_mode == "human" else None,
    )
    frames: list[np.ndarray] = []
    episode_metrics = []
    try:
        for episode in range(episodes):
            observations = env.reset(seed=seed + episode)
            _capture_frame(env, render_mode, frames)
            while True:
                masks = env.action_masks()
                actions, _, _ = policy.act(
                    observations, masks, deterministic=True
                )
                transition = env.step(actions)
                observations = transition.observations
                _capture_frame(env, render_mode, frames)
                if render_mode == "human":
                    sleep(1.0 / fps)
                if (
                    transition.terminated
                    or transition.truncated
                    or transition.metrics.deadlocked
                ):
                    episode_metrics.append(transition.metrics.as_dict())
                    break
    finally:
        env.close()

    saved_output = None
    if render_mode == "rgb_array":
        destination = _gif_destination(checkpoint_path, output_path)
        _write_gif(frames, destination, fps)
        saved_output = str(destination)
    return {"episodes": episode_metrics, "frames": len(frames), "output": saved_output}


def _capture_frame(
    env: Phase2Warehouse, render_mode: str, frames: list[np.ndarray]
) -> None:
    if render_mode == "rgb_array":
        frame = render_warehouse_frame(env.env)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError("rgb_array rendering did not produce an RGB frame.")
        frames.append(frame.copy())
    else:
        env.render()


def _gif_destination(checkpoint_path, output_path) -> Path:
    if output_path is not None:
        return Path(output_path)
    checkpoint = Path(checkpoint_path)
    return checkpoint.parent / "visualization.gif"


def _write_gif(frames: Sequence[np.ndarray], destination: Path, fps: float) -> None:
    if not frames:
        raise RuntimeError("Cannot write a GIF without rendered frames.")
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "GIF output requires Pillow. Install the dev extras."
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=round(1000.0 / fps),
        loop=0,
    )


def main():
    args = parse_args()
    summary = run_visualization(
        args.checkpoint,
        seed=args.seed,
        episodes=args.episodes,
        render_mode=args.render,
        fps=args.fps,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
