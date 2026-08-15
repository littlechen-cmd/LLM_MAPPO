"""Replay Phase 2 MAPPO or A* policies in one stable real-time viewer."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import yaml

from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase2_expert import AStarExpert
from llm_mappo.phase2_training import load_policy
from llm_mappo.phase3_training import load_phase3_policy
from llm_mappo.visualization import render_warehouse_frame
from rware.warehouse import Action


DEFAULT_CHECKPOINT = (
    "artifacts/phase2_medium_3ag_astar_bc_kl/seed_007/checkpoint_final.pt"
)
DEFAULT_CONFIG = "configs/phase2_medium_3ag_astar_bc.yaml"


@dataclass(frozen=True)
class EnvironmentSettings:
    """Parameters needed to recreate the environment used by a controller."""

    env_id: str
    n_agents: int
    max_steps: int
    waypoint_reward: float
    oracle_interaction_mask: bool
    deadlock_steps: int
    charge_threshold: float = 0.2
    priority_schedule: Optional[Sequence[str]] = None
    batch_interval: Optional[int] = None
    batch_size_range: Optional[Sequence[int]] = None
    initial_priority_label: str = "B"
    request_queue_size: Optional[int] = None
    task_completion_target: Optional[int] = None
    include_priority_features: bool = False


class LiveViewer:
    """Keep one Tk window alive while its image is updated frame by frame."""

    def __init__(self, maximum_width: int, maximum_height: int):
        if maximum_width < 1 or maximum_height < 1:
            raise ValueError("Live-view dimensions must be positive.")
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ImportError as error:
            raise RuntimeError(
                "Live rendering requires Tkinter and Pillow. Use --no-live for "
                "GIF or trace output."
            ) from error
        self._tk = tk
        self._image = Image
        self._image_tk = ImageTk
        self._maximum_width = maximum_width
        self._maximum_height = maximum_height
        self._closed = False
        self._root = tk.Tk()
        self._root.title("LLM-MAPPO warehouse visualization")
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._label = tk.Label(self._root, borderwidth=0)
        self._label.pack()

    def show(self, frame: np.ndarray, title: str) -> bool:
        """Draw a frame in the existing window and process close events."""
        if self._closed:
            return False
        try:
            image = self._image.fromarray(frame)
            image.thumbnail((self._maximum_width, self._maximum_height))
            photo = self._image_tk.PhotoImage(image=image)
            self._root.title(title)
            self._label.configure(image=photo)
            self._label.image = photo
            self._root.update_idletasks()
            self._root.update()
            return not self._closed
        except self._tk.TclError:
            self._closed = True
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass


def parse_args():
    parser = ArgumentParser(
        description=(
            "Replay MAPPO checkpoints or the reservation-aware A* expert. "
            "Multiple seeds are displayed sequentially in one window."
        )
    )
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--controller", choices=("policy", "expert"), default="policy")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument(
        "--held-out-seed-protocol",
        action="store_true",
        help=(
            "Reproduce evaluation seeds as seed * 10000 + episode offset, "
            "as used by the Phase 3 gates."
        ),
    )
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--cell-size", type=int, default=28)
    parser.add_argument("--record-gif", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/visualizations")
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--window-width", type=int, default=1400)
    parser.add_argument("--window-height", type=int, default=900)
    return parser.parse_args()


def run_visualization(
    controller_kind: str,
    checkpoint_path: str | Path,
    config_path: str | Path,
    seeds: Sequence[int],
    episodes_per_seed: int,
    fps: float,
    cell_size: int,
    live: bool,
    record_gif: bool,
    record_trace: bool,
    output_dir: str | Path,
    held_out_seed_protocol: bool = False,
    window_width: int = 1400,
    window_height: int = 900,
) -> dict:
    """Replay each requested seed sequentially and return saved replay metadata."""
    if not seeds:
        raise ValueError("Provide at least one seed.")
    if episodes_per_seed < 1:
        raise ValueError("episodes_per_seed must be positive.")
    if fps <= 0.0:
        raise ValueError("fps must be positive.")
    if cell_size < 8:
        raise ValueError("cell_size must be at least 8 pixels.")
    settings, controller = _load_controller(
        controller_kind, checkpoint_path, config_path
    )
    env = Phase2Warehouse(**settings.__dict__)
    destination = Path(output_dir)
    viewer = LiveViewer(window_width, window_height) if live else None
    results = []
    try:
        for seed in seeds:
            for episode in range(episodes_per_seed):
                environment_seed = _episode_seed(
                    seed, episode, held_out_seed_protocol
                )
                result, keep_showing = _replay_episode(
                    env=env,
                    controller=controller,
                    seed=seed,
                    environment_seed=environment_seed,
                    episode_index=episode + 1,
                    fps=fps,
                    cell_size=cell_size,
                    viewer=viewer,
                    record_gif=record_gif,
                    record_trace=record_trace,
                    output_dir=destination,
                )
                results.append(result)
                if not keep_showing:
                    return _write_summary(destination, controller_kind, results)
    finally:
        env.close()
        if viewer is not None:
            viewer.close()
    return _write_summary(destination, controller_kind, results)


def _load_controller(
    controller_kind: str, checkpoint_path: str | Path, config_path: str | Path
) -> tuple[EnvironmentSettings, Callable[[Phase2Warehouse, np.ndarray], np.ndarray]]:
    if controller_kind == "policy":
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        if _is_phase3_checkpoint(checkpoint):
            policy, config, _ = load_phase3_policy(checkpoint_path)
            settings = _settings_from_phase3_config(config)
        else:
            policy, config, _ = load_policy(checkpoint_path)
            settings = _settings_from_mapping(config)

        def choose_actions(env: Phase2Warehouse, masks: np.ndarray) -> np.ndarray:
            observations = env._observations()
            outputs = policy.act(observations, masks, deterministic=True)
            return outputs[0]

        return settings, choose_actions
    if controller_kind == "expert":
        with Path(config_path).open("r", encoding="utf-8") as stream:
            source = yaml.safe_load(stream) or {}
        expert = AStarExpert()

        def choose_actions(env: Phase2Warehouse, masks: np.ndarray) -> np.ndarray:
            actions, _ = expert.act(env, masks)
            return actions

        return _settings_from_mapping(source.get("environment", {})), choose_actions
    raise ValueError(f"Unsupported controller: {controller_kind}")


def _is_phase3_checkpoint(checkpoint: Mapping[str, object]) -> bool:
    """Return True when the checkpoint stores a Phase 3 dual-head policy."""
    phase = checkpoint.get("phase")
    if phase is not None:
        return str(phase).startswith("3") or str(phase) == "4"
    model_state = checkpoint.get("model_state", {})
    return any(str(key).startswith("actor.motion_encoder") for key in model_state)


def _settings_from_mapping(source: Mapping[str, object]) -> EnvironmentSettings:
    priority_schedule = source.get("priority_schedule")
    batch_size_range = source.get("batch_size_range")
    return EnvironmentSettings(
        env_id=str(source.get("env_id", source.get("id", "llm-mappo-medium-3ag-v1"))),
        n_agents=int(source.get("n_agents", 3)),
        max_steps=int(source.get("max_steps", 400)),
        waypoint_reward=float(source.get("waypoint_reward", 0.01)),
        oracle_interaction_mask=bool(source.get("oracle_interaction_mask", True)),
        deadlock_steps=int(source.get("deadlock_steps", 180)),
        charge_threshold=float(source.get("charge_threshold", 0.2)),
        priority_schedule=tuple(priority_schedule) if priority_schedule else None,
        batch_interval=source.get("batch_interval"),
        batch_size_range=(
            tuple(batch_size_range) if batch_size_range is not None else None
        ),
        initial_priority_label=str(source.get("initial_priority_label", "B")),
        request_queue_size=source.get("request_queue_size"),
        task_completion_target=source.get("task_completion_target"),
        include_priority_features=bool(
            source.get("include_priority_features", False)
        ),
    )


def _settings_from_phase3_config(source: Mapping[str, object]) -> EnvironmentSettings:
    settings = _settings_from_mapping(source)
    return EnvironmentSettings(
        **{
            **settings.__dict__,
            # Phase 3 always reserves the priority observation dimensions.
            "include_priority_features": True,
        }
    )


def _replay_episode(
    env: Phase2Warehouse,
    controller: Callable[[Phase2Warehouse, np.ndarray], np.ndarray],
    seed: int,
    environment_seed: int,
    episode_index: int,
    fps: float,
    cell_size: int,
    viewer: LiveViewer | None,
    record_gif: bool,
    record_trace: bool,
    output_dir: Path,
) -> tuple[dict, bool]:
    env.reset(seed=environment_seed)
    frames: list[np.ndarray] = []
    trace = []
    started = monotonic()
    keep_showing = _display_frame(
        env, seed, episode_index, "initial", cell_size, viewer, frames, record_gif
    )
    transition = None
    while keep_showing:
        masks = env.action_masks()
        actions = controller(env, masks)
        transition = env.step(actions)
        trace.append(_trace_entry(env, transition.info, actions))
        keep_showing = _display_frame(
            env,
            seed,
            episode_index,
            f"step {transition.info['step']}",
            cell_size,
            viewer,
            frames,
            record_gif,
        )
        _pace(started, transition.info["step"], fps)
        if (
            transition.terminated
            or transition.truncated
            or transition.metrics.deadlocked
        ):
            break
    metrics = (
        transition.metrics.as_dict()
        if transition is not None
        else env.metrics.as_dict()
    )
    result = {
        "seed": seed,
        "environment_seed": environment_seed,
        "episode": episode_index,
        "metrics": metrics,
        "frames": len(frames) if record_gif else len(trace) + 1,
    }
    if record_gif:
        gif_path = output_dir / f"seed_{seed:03d}_episode_{episode_index:02d}.gif"
        _write_gif(frames, gif_path, fps)
        result["gif"] = str(gif_path)
    if record_trace:
        trace_path = output_dir / (
            f"seed_{seed:03d}_episode_{episode_index:02d}_trace.json"
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(
                {"seed": seed, "metrics": metrics, "trace": trace},
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        result["trace"] = str(trace_path)
    return result, keep_showing


def _display_frame(
    env: Phase2Warehouse,
    seed: int,
    episode: int,
    status: str,
    cell_size: int,
    viewer: LiveViewer | None,
    frames: list[np.ndarray],
    record_gif: bool,
) -> bool:
    if viewer is None and not record_gif:
        return True
    frame = render_warehouse_frame(env.env, cell_size=cell_size)
    if record_gif:
        frames.append(frame.copy())
    if viewer is None:
        return True
    return viewer.show(frame, f"LLM-MAPPO | seed {seed} | episode {episode} | {status}")


def _pace(started: float, step: int, fps: float) -> None:
    remaining = step / fps - (monotonic() - started)
    if remaining > 0.0:
        sleep(remaining)


def _trace_entry(
    env: Phase2Warehouse, info: dict, actions: Sequence[int]
) -> dict:
    """Persist the state needed to diagnose target progress and livelock."""
    return {
        "step": info["step"],
        "actions": [int(action) for action in actions],
        "action_names": [Action(int(action)).name for action in actions],
        "agents": info["agents"],
        "tasks": info["tasks"],
        "events": info["events"],
        "collisions": info["collisions"],
        "blocked_forwards": info["blocked_forwards"],
        "diagnostics": [_agent_diagnostics(env, agent.id) for agent in env.env.agents],
    }


def _agent_diagnostics(env: Phase2Warehouse, agent_id: int) -> dict:
    agent = env.env.agents[agent_id - 1]
    target, target_kind = env._target_for_agent(agent_id)
    task = env.env.task_queue.task_for_agent(agent_id)
    return {
        "agent_id": agent_id,
        "position": [agent.x, agent.y],
        "direction": agent.dir.name,
        "battery": round(float(agent.battery), 6),
        "carrying_shelf_id": (
            agent.carrying_shelf.id if agent.carrying_shelf is not None else None
        ),
        "task_id": task.task_id if task is not None else None,
        "task_label": task.label if task is not None else None,
        "picking_lock_steps": agent.picking_lock_steps,
        "target": list(target),
        "target_kind": target_kind,
        "target_distance": abs(agent.x - target[0]) + abs(agent.y - target[1]),
    }


def _episode_seed(seed: int, episode_offset: int, held_out_protocol: bool) -> int:
    if held_out_protocol:
        return seed * 10_000 + episode_offset
    return seed + episode_offset


def _write_gif(frames: Sequence[np.ndarray], destination: Path, fps: float) -> None:
    if not frames:
        raise RuntimeError("Cannot write a GIF without rendered frames.")
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("GIF output requires Pillow.") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        destination,
        save_all=True,
        append_images=images[1:],
        duration=round(1000.0 / fps),
        loop=0,
    )


def _write_summary(output_dir: Path, controller: str, results: Sequence[dict]) -> dict:
    summary = {"controller": controller, "results": list(results)}
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    summary["summary"] = str(path)
    return summary


def _json_default(value):
    """Convert NumPy metric and task identifiers for persisted replay data."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported trace value: {type(value).__name__}")


def main():
    args = parse_args()
    summary = run_visualization(
        controller_kind=args.controller,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        seeds=args.seeds,
        episodes_per_seed=args.episodes_per_seed,
        fps=args.fps,
        cell_size=args.cell_size,
        live=not args.no_live,
        record_gif=args.record_gif,
        record_trace=args.trace,
        output_dir=args.output_dir,
        held_out_seed_protocol=args.held_out_seed_protocol,
        window_width=args.window_width,
        window_height=args.window_height,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
