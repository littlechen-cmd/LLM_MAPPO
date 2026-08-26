"""Owner-only 12-worker runtime and memory gate for O1 calibration."""

import argparse
import csv
import gc
from dataclasses import dataclass
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import psutil
import torch

from llm_mappo.optimization_training import (
    OptimizationTrainer,
    OptimizationTrainingConfig,
)
from llm_mappo.optimization_buffer import OptimizationRolloutBuffer
from llm_mappo.reward_calibration import (
    CalibrationResult,
    deterministic_masked_argmax,
)
from llm_mappo.pure_motion_teacher import PureMotionTeacher
from llm_mappo.shadow_state import ShadowStateAdapter


REQUIRED_ARTIFACTS = {
    "manifest.json", "runtime.csv", "memory.csv",
    "branch_objects.csv", "summary.json",
}
_MIN_GROWTH = 64 * 1024 * 1024
_MIN_FRACTION = 0.05


@dataclass(frozen=True)
class BenchmarkConfig:
    condition: str

    def __post_init__(self):
        if self.condition not in {"baseline", "h4", "h12"}:
            raise ValueError("Unsupported benchmark condition.")

    @property
    def horizon(self):
        return 4 if self.condition == "h4" else 12


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description="Run the frozen owner-only O1 calibration runtime/memory gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--modes", choices=("baseline", "h4", "h12"), nargs="+", required=True
    )
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--warmup-vector-steps", type=int, required=True)
    parser.add_argument("--measure-vector-steps", type=int, required=True)
    parser.add_argument("--memory-warmup-windows", type=int, required=True)
    parser.add_argument("--memory-measure-windows", type=int, required=True)
    parser.add_argument("--output", required=True)
    parsed = parser.parse_args(arguments)
    if parsed.modes != ["baseline", "h4", "h12"]:
        parser.error("--modes must be exactly: baseline h4 h12")
    frozen = {
        "workers": 12, "repeats": 5, "warmup_vector_steps": 16,
        "measure_vector_steps": 128, "memory_warmup_windows": 2,
        "memory_measure_windows": 10,
    }
    for name, value in frozen.items():
        if getattr(parsed, name) != value:
            parser.error(f"--{name.replace('_', '-')} is frozen at {value}")
    return parsed


class _Worker:
    def __init__(self, config, seed, output, environment_index=0):
        self.trainer = OptimizationTrainer(config, output)
        self.seed = seed
        self.environment_index = environment_index
        self.episode = 0
        self.global_step = 0
        self.buffer = OptimizationRolloutBuffer(config.n_agents)
        self.rewards = []
        self.update_count = 0
        self.observations = self._reset()

    def _reset(self):
        episode_seed = self.seed + self.episode
        observations = self.trainer.environment.reset(seed=episode_seed)
        self.trainer.student_shadow.reset(seed=episode_seed)
        self.trainer.teacher_shadow.reset(seed=episode_seed)
        self.episode_step = 0
        return observations

    def step(self, mode):
        trainer = self.trainer
        semantic, targets, semantic_valid, reliability = trainer._semantic_batch()
        masks = trainer.environment.action_masks()
        with torch.inference_mode():
            output = trainer.actor(
                torch.as_tensor(
                    self.observations, dtype=torch.float32, device=trainer.device
                ),
                torch.as_tensor(semantic, dtype=torch.float32, device=trainer.device),
            )
        actions = deterministic_masked_argmax(output.action_logits.cpu().numpy(), masks)
        log_probs = trainer._selected_log_probs(output.action_logits, actions, masks)
        preferences, valid = trainer._teacher_batch(trainer.environment)
        address = {
            "run_seed": self.seed, "episode_index": self.episode,
            "episode_seed": self.seed + self.episode,
            "environment_index": self.environment_index,
            "real_global_step": self.global_step,
            "episode_step": self.episode_step,
        }
        selected = trainer.calibrator.select(**address)
        result = CalibrationResult(
            selected=selected,
            attempted=False,
            success=False,
            confidence=0.0,
            delta_g=None,
            failure_reason="selected_no_valid" if selected and not valid.any() else None,
        )
        if mode != "baseline" and selected and valid.any():
            snapshot = trainer.real_adapter.capture(**address)
            result = trainer.calibrator.run_paired_shadows(
                snapshot=snapshot, real_adapter=trainer.real_adapter,
                student_adapter=trainer.student_adapter,
                teacher_adapter=trainer.teacher_adapter,
                student_logits=trainer._student_logits,
                teacher_preferences=trainer._shadow_teacher_batch,
                initial_valid_mask=valid, critic_value=trainer._critic_value,
                gamma=trainer.config.gamma, address=address,
                horizon=BenchmarkConfig(mode).horizon, diagnostic=mode == "h4",
            )
        transition = trainer.environment.step(actions)
        self.buffer.add(
            physical_observations=self.observations,
            semantic_observations=semantic,
            actions=actions,
            log_probs=log_probs,
            action_masks=masks,
            astar_preferences=preferences,
            astar_valid=valid,
            calibration_selected=selected,
            reward_confidence=(
                1.0 if trainer.config.method == "fixed-kd" else result.confidence
            ),
            semantic_targets=targets,
            semantic_validity=semantic_valid,
            ood_reliability=reliability,
        )
        self.rewards.append(transition.team_reward)
        trainer.schedule.advance_real_env_steps(1)
        self.observations = transition.observations
        self.global_step += 1
        self.episode_step += 1
        if len(self.rewards) == trainer.config.rollout_length:
            trainer._update(self.buffer, self.rewards)
            self.buffer = OptimizationRolloutBuffer(trainer.config.n_agents)
            self.rewards = []
            self.update_count += 1
        if transition.terminated or transition.truncated:
            self.episode += 1
            self.observations = self._reset()

    @property
    def cache_entries(self):
        return len(self.trainer.teacher._cache)


def _workers(config, count, output):
    return [
        _Worker(
            config,
            config.seed + index,
            output / f"worker_{index:02d}",
            environment_index=index,
        )
        for index in range(count)
    ]


def _advance(workers, mode, steps):
    for _ in range(steps):
        for worker in workers:
            worker.step(mode)


def _live_object_counts():
    gc.collect()
    objects = gc.get_objects()
    return {
        "branch_objects": sum(
            isinstance(item, ShadowStateAdapter) for item in objects
        ),
        "teacher_objects": sum(
            isinstance(item, PureMotionTeacher) for item in objects
        ),
    }


def _runtime_child(payload, queue):
    try:
        config = OptimizationTrainingConfig.from_yaml(payload["config"])
        workers = _workers(config, payload["workers"], Path(payload["scratch"]))
        _advance(workers, payload["mode"], payload["warmup"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        _advance(workers, payload["mode"], payload["measure"])
        torch.cuda.synchronize()
        object_counts = _live_object_counts()
        queue.put({
            "ok": True, "seconds": time.perf_counter() - started,
            **object_counts,
            "teacher_cache_entries": sum(item.cache_entries for item in workers),
        })
    except Exception:
        queue.put({"ok": False, "traceback": traceback.format_exc()})


def _memory_child(payload, queue):
    try:
        config = OptimizationTrainingConfig.from_yaml(payload["config"])
        workers = _workers(config, payload["workers"], Path(payload["scratch"]))
        process = psutil.Process(os.getpid())
        rows = []
        total = payload["warmup_windows"] + payload["measure_windows"]
        for window in range(total):
            _advance(workers, "h12", payload["window_steps"])
            torch.cuda.synchronize()
            if window >= payload["warmup_windows"]:
                object_counts = _live_object_counts()
                rows.append({
                    "window": window - payload["warmup_windows"],
                    "cpu_rss_bytes": process.memory_info().rss,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(),
                    **object_counts,
                    "teacher_cache_entries": sum(
                        item.cache_entries for item in workers
                    ),
                })
        queue.put({"ok": True, "rows": rows})
    except Exception:
        queue.put({"ok": False, "traceback": traceback.format_exc()})


def _spawn(target, payload):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(payload, queue))
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(f"Benchmark child exited with code {process.exitcode}.")
    result = queue.get()
    if not result["ok"]:
        raise RuntimeError(result["traceback"])
    return result


def _rho(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or np.all(values == values[0]):
        return 0.0

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return float(np.corrcoef(np.arange(len(values)), ranks)[0, 1])


def analyze_memory_rows(rows):
    if len(rows) != 10:
        raise ValueError("Memory gate requires exactly 10 measured windows.")

    def growth(name):
        values = [int(row[name]) for row in rows]
        delta = values[-1] - values[0]
        threshold = max(_MIN_GROWTH, int(values[0] * _MIN_FRACTION))
        rho = _rho(values)
        return values[0], delta, threshold, rho, delta > threshold and rho >= 0.80

    rss_start, rss_delta, rss_threshold, rss_rho, rss_growth = growth(
        "cpu_rss_bytes"
    )
    cuda_start, cuda_delta, cuda_threshold, cuda_rho, cuda_growth = growth(
        "cuda_reserved_bytes"
    )
    branches = [int(row["branch_objects"]) for row in rows]
    teachers = [int(row.get("teacher_objects", 0)) for row in rows]
    caches = [int(row["teacher_cache_entries"]) for row in rows]
    branch_growth = branches[-1] > branches[0] and _rho(branches) >= 0.80
    teacher_growth = teachers[-1] > teachers[0] and _rho(teachers) >= 0.80
    return {
        "rss_start_bytes": rss_start,
        "rss_delta_bytes": rss_delta,
        "rss_growth_threshold_bytes": rss_threshold,
        "rss_rho": rss_rho,
        "cuda_reserved_start_bytes": cuda_start,
        "cuda_reserved_delta_bytes": cuda_delta,
        "cuda_reserved_growth_threshold_bytes": cuda_threshold,
        "cuda_reserved_rho": cuda_rho,
        "branch_object_growth": bool(branch_growth),
        "teacher_object_growth": bool(teacher_growth),
        "teacher_cache_growth": caches[-1] - caches[0],
        "persistent_growth": bool(
            rss_growth or cuda_growth or branch_growth or teacher_growth
        ),
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_gate(arguments):
    if not torch.cuda.is_available():
        raise RuntimeError("The O1 runtime gate requires a CUDA-capable owner machine.")
    config_path = Path(arguments.config)
    OptimizationTrainingConfig.from_yaml(config_path)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    common = {
        "config_hash": sha256(config_path.read_bytes()).hexdigest(),
        "code_commit": commit, "device": torch.cuda.get_device_name(),
        "cuda": str(torch.version.cuda), "pytorch": torch.__version__,
        "platform": platform.platform(),
        "workers": arguments.workers, "exit_status": 0,
    }
    runtime_rows, branch_rows = [], []
    for mode in arguments.modes:
        for repeat in range(arguments.repeats):
            result = _spawn(_runtime_child, {
                "config": str(config_path), "mode": mode,
                "workers": arguments.workers,
                "warmup": arguments.warmup_vector_steps,
                "measure": arguments.measure_vector_steps,
                "scratch": str(output / "scratch" / f"{mode}_{repeat}"),
            })
            runtime_rows.append({
                **common, "mode": mode,
                "horizon": 0 if mode == "baseline" else BenchmarkConfig(mode).horizon,
                "repeat": repeat,
                "warmup_vector_steps": arguments.warmup_vector_steps,
                "measure_vector_steps": arguments.measure_vector_steps,
                "seconds": result["seconds"],
            })
            branch_rows.append({
                **common, "mode": mode,
                "horizon": 0 if mode == "baseline" else BenchmarkConfig(mode).horizon,
                "repeat": repeat,
                "warmup_vector_steps": arguments.warmup_vector_steps,
                "measure_vector_steps": arguments.measure_vector_steps,
                "branch_objects": result["branch_objects"],
                "teacher_objects": result["teacher_objects"],
                "teacher_cache_entries": result["teacher_cache_entries"],
            })
    memory = _spawn(_memory_child, {
        "config": str(config_path), "workers": arguments.workers,
        "warmup_windows": arguments.memory_warmup_windows,
        "measure_windows": arguments.memory_measure_windows,
        "window_steps": arguments.measure_vector_steps,
        "scratch": str(output / "scratch" / "memory_h12"),
    })
    memory_rows = [
        {
            **common,
            "mode": "h12",
            "horizon": 12,
            "memory_warmup_windows": arguments.memory_warmup_windows,
            "memory_measure_windows": arguments.memory_measure_windows,
            "window_vector_steps": arguments.measure_vector_steps,
            **row,
        }
        for row in memory["rows"]
    ]
    medians = {
        mode: float(np.median([
            row["seconds"] for row in runtime_rows if row["mode"] == mode
        ])) for mode in arguments.modes
    }
    memory_analysis = analyze_memory_rows(memory_rows)
    ratio = medians["h12"] / medians["baseline"]
    summary = {
        **common, "runtime_medians_seconds": medians,
        "h12_baseline_runtime_ratio": ratio,
        "runtime_gate_pass": ratio <= 3.0, "memory": memory_analysis,
        "memory_gate_pass": not memory_analysis["persistent_growth"],
        "gate_pass": ratio <= 3.0 and not memory_analysis["persistent_growth"],
    }
    manifest = {
        **common, "config": vars(arguments),
        "required_artifacts": sorted(REQUIRED_ARTIFACTS),
    }
    _write_csv(output / "runtime.csv", runtime_rows)
    _write_csv(output / "memory.csv", memory_rows)
    _write_csv(output / "branch_objects.csv", branch_rows)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def main():
    print(json.dumps(run_gate(parse_arguments()), indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
