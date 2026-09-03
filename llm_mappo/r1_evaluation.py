"""Deterministic, headless R1-C checkpoint evaluation and replay evidence."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from llm_mappo.e1_vector_env import _new_environment, _semantic_batch
from llm_mappo.optimization_student import O0StudentActor
from llm_mappo.visualization import render_warehouse_frame


def evaluate_r1c_checkpoint(
    *,
    directory: str | Path,
    environment: Mapping[str, Any],
    run,
    dataset,
    identity: Mapping[str, Any],
    seeds: tuple[int, ...],
    device: str | torch.device,
) -> dict[str, Any]:
    """Evaluate one exact R1-C final checkpoint with deterministic argmax.

    The checkpoint identity is compared to the stored run manifest before model
    weights are used.  Replays are sampled, headless GIFs rather than GUI runs.
    """
    root = Path(directory)
    checkpoint_path = root / "checkpoint_final.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("identity") != dict(identity):
        raise ValueError("R1-C evaluation checkpoint identity is incompatible.")
    actor = O0StudentActor().to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    results = [
        _episode(actor, environment, run, dataset, int(seed), device)
        for seed in seeds
    ]
    if not results:
        raise ValueError("R1-C evaluation requires at least one frozen seed.")
    lowest = min(
        results,
        key=lambda item: (item["metrics"]["task_completion_rate"], item["seed"]),
    )
    successful = next(
        (item for item in results if item["metrics"]["success"]), None
    )
    replay_root = root / "evaluation" / "replays"
    replay_root.mkdir(parents=True, exist_ok=True)
    fixed = _episode(
        actor, environment, run, dataset, int(seeds[0]), device,
        replay_path=replay_root / f"fixed_seed_{int(seeds[0]):04d}.gif",
    )
    lowest_replay = _episode(
        actor, environment, run, dataset, int(lowest["seed"]), device,
        replay_path=replay_root / f"lowest_seed_{int(lowest['seed']):04d}.gif",
    )
    successful_replay: str | None = None
    if successful is not None:
        success_path = replay_root / f"success_seed_{int(successful['seed']):04d}.gif"
        _episode(
            actor, environment, run, dataset, int(successful["seed"]), device,
            replay_path=success_path,
        )
        successful_replay = str(success_path.relative_to(root))
    result = {
        "schema": "r1c-deterministic-evaluation-v1",
        "checkpoint_sha256": _sha256(checkpoint_path),
        "identity": dict(identity),
        "policy": "deterministic_argmax",
        "episodes": results,
        "mean_task_completion_rate": float(np.mean([
            item["metrics"]["task_completion_rate"] for item in results
        ])),
        "mean_completed_tasks": float(np.mean([
            item["metrics"]["completed_tasks"] for item in results
        ])),
        "fixed_replay": str(Path(fixed["replay_path"]).relative_to(root)),
        "lowest_completion_replay": str(
            Path(lowest_replay["replay_path"]).relative_to(root)
        ),
        "successful_replay": successful_replay or "unavailable",
    }
    output = root / "evaluation" / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_evaluation_plot(output.parent / "evaluation_metrics.png", results)
    result["metrics_plot"] = str(
        (output.parent / "evaluation_metrics.png").relative_to(root)
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _episode(actor, values, run, dataset, seed, device, replay_path=None):
    environment = _new_environment(values, run)
    frames = []
    try:
        observations = environment.reset(seed=seed)
        for step in range(int(values["max_steps"])):
            if replay_path is not None and step % 10 == 0:
                frames.append(render_warehouse_frame(environment.env, cell_size=16))
            semantic, _, _, _ = _semantic_batch(environment, dataset, "none")
            masks = environment.action_masks()
            with torch.no_grad():
                logits = actor(
                    torch.as_tensor(observations, dtype=torch.float32, device=device),
                    torch.as_tensor(semantic, dtype=torch.float32, device=device),
                ).action_logits
                logits = logits.masked_fill(
                    ~torch.as_tensor(masks, dtype=torch.bool, device=device), -1e9
                )
                actions = logits.argmax(dim=-1).cpu().numpy().astype(np.int64)
            transition = environment.step(actions)
            observations = transition.observations
            if (
                transition.terminated
                or transition.truncated
                or transition.metrics.deadlocked
            ):
                break
        if replay_path is not None:
            frames.append(render_warehouse_frame(environment.env, cell_size=16))
            _write_gif(Path(replay_path), frames)
        item = {"seed": int(seed), "metrics": transition.metrics.as_dict()}
        if replay_path is not None:
            item["replay_path"] = str(replay_path)
        return item
    finally:
        environment.close()


def _write_gif(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        raise ValueError("R1-C replay has no renderable frames.")
    from PIL import Image
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=100,
        loop=0, optimize=False,
    )


def _write_evaluation_plot(path: Path, episodes: list[Mapping[str, Any]]) -> None:
    """Write a compact, dependency-free visual summary of fixed evaluations."""
    from PIL import Image, ImageDraw

    width, height = 760, 360
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    names = ("completion", "tasks", "collisions")
    colors = ("#1f77b4", "#2ca02c", "#d62728")
    values = (
        [float(item["metrics"]["task_completion_rate"]) for item in episodes],
        [float(item["metrics"]["completed_tasks"]) for item in episodes],
        [float(item["metrics"]["collisions"]) for item in episodes],
    )
    for row, (name, color, series) in enumerate(zip(names, colors, values)):
        top = 30 + row * 105
        maximum = max(max(series), 1.0)
        draw.text((12, top), name, fill="black")
        for index, value in enumerate(series):
            left = 135 + index * 110
            bar_height = int(65 * value / maximum)
            draw.rectangle((left, top + 78 - bar_height, left + 72, top + 78),
                           fill=color)
            draw.text((left, top + 82), str(episodes[index]["seed"]), fill="black")
            draw.text((left, top + 62 - bar_height), f"{value:.2f}", fill="black")
    image.save(path)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
