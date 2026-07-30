"""Repeatable Phase 1 Go/No-Go validation for dynamics and rendering."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from time import monotonic, sleep

import gymnasium as gym
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import rware  # noqa: E402,F401 - registers the custom Gymnasium environment.


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase1_medium_3ag.yaml",
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--render-frames", type=int)
    return parser.parse_args()


def load_config(path):
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def make_environment(environment_id, max_steps, render_mode=None):
    kwargs = {
        "max_steps": max_steps,
        "disable_env_checker": True,
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gym.make(environment_id, **kwargs)


def run_headless(environment_id, episodes, episode_steps, seed):
    env = make_environment(environment_id, episode_steps)
    executed_steps = 0
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            for _ in range(episode_steps):
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                executed_steps += 1
                if terminated or truncated:
                    break
    finally:
        env.close()
    return {"episodes": episodes, "executed_steps": executed_steps}


def run_render_benchmark(
    environment_id,
    max_steps,
    render_warmup_frames,
    render_frames,
    render_fps,
    max_frame_seconds,
    seed,
):
    env = make_environment(environment_id, max_steps, render_mode="rgb_array")
    frame_times = []
    nonblank_frames = 0
    frame_interval = 1 / render_fps
    try:
        env.reset(seed=seed)
        for _ in range(render_warmup_frames):
            _validate_rendered_frame(env.render())
        next_frame = monotonic()
        for _ in range(render_frames):
            now = monotonic()
            if now < next_frame:
                sleep(next_frame - now)
            started = monotonic()
            frame = env.render()
            elapsed = monotonic() - started
            frame_times.append(elapsed)
            _validate_rendered_frame(frame)
            nonblank_frames += 1
            next_frame += frame_interval
    finally:
        env.close()

    max_frame = max(frame_times)
    if max_frame > max_frame_seconds:
        raise AssertionError(
            f"Maximum frame time {max_frame:.4f}s exceeds {max_frame_seconds:.4f}s."
        )
    return {
        "warmup_frames": render_warmup_frames,
        "frames": render_frames,
        "nonblank_frames": nonblank_frames,
        "average_frame_seconds": round(float(np.mean(frame_times)), 6),
        "max_frame_seconds": round(float(max_frame), 6),
    }


def _validate_rendered_frame(frame):
    if frame is None or frame.ndim != 3 or frame.shape[-1] != 3:
        raise AssertionError("rgb_array rendering did not return an RGB image.")
    if np.ptp(frame) == 0:
        raise AssertionError("Rendered frame is blank or a uniform color.")


def main():
    args = parse_args()
    config = load_config(args.config)
    environment = config["environment"]
    validation = config["validation"]
    episodes = args.episodes or validation["headless_episodes"]
    episode_steps = args.episode_steps or validation["episode_steps"]
    render_frames = args.render_frames or validation["render_frames"]
    result = {
        "headless": run_headless(
            environment["id"],
            episodes,
            episode_steps,
            environment["seed"],
        ),
        "render": run_render_benchmark(
            environment["id"],
            environment["max_steps"],
            validation["render_warmup_frames"],
            render_frames,
            validation["render_fps"],
            validation["max_frame_seconds"],
            environment["seed"],
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
