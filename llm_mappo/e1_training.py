"""E1 formal MAPPO core with raw three-dimensional noisy-teacher evidence.

This module is deliberately separate from the completed O2 calibration runner.
It never contacts an LLM: labels are loaded once from the immutable local JSONL
evidence and retrieved only in the frozen 61-dimensional semantic-view-v3 space.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import functional as functional

from llm_mappo.optimization_student import (
    ACTION_DIM,
    MOTION_ACTION_DIM,
    PHYSICAL_OBSERVATION_DIM,
    SEMANTIC_OBSERVATION_DIM,
    SEMANTIC_SCORE_DIM,
)
from llm_mappo.semantic_v3 import SemanticDatasetV3, SemanticViewV3
from llm_mappo.optimization_buffer import LinearEnvStepSchedule
from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.pure_motion_teacher import PureMotionQuery, PureMotionTeacher
from llm_mappo.reward_calibration import RewardCalibrator
from llm_mappo.shadow_state import ShadowStateAdapter
from llm_mappo.shadow_state import ShadowSnapshotV1, rebind_snapshot_rng_guard
from llm_mappo.e1_vector_env import E1VectorEnvironmentPool


_RAW_LABEL_PROMPT = "semantic-prompt-v5-state-contract"
_RAW_LABEL_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class RawSemanticEvidence:
    """The immutable E1 raw-label source and its training-safe projection."""

    records_path: Path
    manifest_path: Path
    records_sha256: str
    total_records: int
    valid_records: int
    invalid_records: int
    backend_fingerprint: str
    dataset: SemanticDatasetV3
    records: tuple[Mapping[str, Any], ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "evidence_kind": "exploratory_noisy_teacher",
            "records_sha256": self.records_sha256,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "request_model": _RAW_LABEL_MODEL,
            "prompt_version": _RAW_LABEL_PROMPT,
            "backend_fingerprint": self.backend_fingerprint,
        }


def load_e1_raw_semantic_evidence(records_path: str | Path) -> RawSemanticEvidence:
    """Load the frozen 800-attempt raw dataset without repairing any record.

    The one parser-invalid record remains excluded by the existing record-level
    validity rule.  No retry, completion request, deletion, or score mutation is
    performed here.
    """

    path = Path(records_path)
    manifest_path = path.with_name("manifest.json")
    try:
        raw = path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = [
            json.loads(line) for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("E1 raw semantic evidence is unreadable.") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("E1 raw semantic manifest must be a mapping.")
    expected_manifest = {
        "schema": "semantic-label-session-v1",
        "mode": "formal",
        "request_model": _RAW_LABEL_MODEL,
        "prompt_version": _RAW_LABEL_PROMPT,
    }
    for name, value in expected_manifest.items():
        if manifest.get(name) != value:
            raise ValueError(f"E1 raw semantic manifest {name} is incompatible.")
    if len(records) != 800:
        raise ValueError("E1 raw semantic evidence must contain exactly 800 records.")
    identities = [
        (record.get("scenario_id"), record.get("content_hash"))
        for record in records
    ]
    if any(not all(identity) for identity in identities) or len(set(identities)) != 800:
        raise ValueError("E1 raw semantic evidence has non-unique record identity.")
    if any(record.get("semantic_view_version") != "semantic-view-v3" for record in records):
        raise ValueError("E1 raw semantic evidence is not semantic-view-v3.")
    valid = [record for record in records if record.get("validity") == 1]
    invalid = [record for record in records if record.get("validity") == 0]
    if len(valid) != 799 or len(invalid) != 1:
        raise ValueError("E1 raw semantic evidence must preserve the 799/800 split.")
    fingerprints = {
        tuple(record.get("backend_tuple", ())) for record in valid
    }
    if len(fingerprints) != 1:
        raise ValueError("E1 raw semantic evidence mixes backend fingerprints.")
    backend = next(iter(fingerprints))
    if len(backend) != 3 or backend[0] != _RAW_LABEL_MODEL:
        raise ValueError("E1 raw semantic backend identity is incompatible.")
    frozen = tuple(manifest.get("frozen_backend_tuple", ()))
    if frozen != backend:
        raise ValueError("E1 raw semantic manifest fingerprint is incompatible.")
    dataset = SemanticDatasetV3.from_records(records)
    if len(dataset.scores) != 799:
        raise ValueError("E1 raw semantic dataset unexpectedly includes invalid labels.")
    return RawSemanticEvidence(
        records_path=path,
        manifest_path=manifest_path,
        records_sha256=sha256(raw).hexdigest(),
        total_records=len(records),
        valid_records=len(valid),
        invalid_records=len(invalid),
        backend_fingerprint=str(backend[2]),
        dataset=dataset,
        records=tuple(records),
    )


@dataclass(frozen=True)
class E1RolloutTensors:
    """Whole-team PPO tensors plus per-robot 3D semantic supervision."""

    physical_observations: Tensor
    semantic_observations: Tensor
    actions: Tensor
    old_log_probs: Tensor
    action_masks: Tensor
    astar_preferences: Tensor
    astar_valid: Tensor
    calibration_selected: Tensor
    reward_confidence: Tensor
    semantic_targets: Tensor
    semantic_validity: Tensor
    semantic_ood_reliability: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor

    def semantic_mse_loss(
        self, semantic_scores: Tensor, *, lambda_l: float
    ) -> tuple[Tensor, int]:
        """Apply one record-level validity×OOD weight to all three scores."""

        if semantic_scores.shape != self.semantic_targets.shape:
            raise ValueError("Semantic scores must match E1 semantic targets.")
        active = self.semantic_validity > 0.0
        denominator = int(active.sum().item())
        if not denominator:
            return semantic_scores.sum() * 0.0, 0
        per_record = (semantic_scores - self.semantic_targets).square().mean(dim=-1)
        weights = self.semantic_validity * self.semantic_ood_reliability
        return float(lambda_l) * (weights * per_record).sum() / denominator, denominator


class E1Rollout:
    """Update-boundary rollout with explicit per-robot semantic reliability."""

    def __init__(self, n_agents: int) -> None:
        if n_agents < 1:
            raise ValueError("E1 rollout requires at least one agent.")
        self.n_agents = int(n_agents)
        self._records: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, **values: Any) -> None:
        record = {
            "physical_observations": self._array(
                values["physical_observations"],
                (self.n_agents, PHYSICAL_OBSERVATION_DIM),
            ),
            "semantic_observations": self._array(
                values["semantic_observations"],
                (self.n_agents, SEMANTIC_OBSERVATION_DIM),
            ),
            "actions": self._array(values["actions"], (self.n_agents,), np.int64),
            "log_probs": self._array(values["log_probs"], (self.n_agents,)),
            "action_masks": self._array(
                values["action_masks"], (self.n_agents, ACTION_DIM), bool
            ),
            "astar_preferences": self._array(
                values["astar_preferences"], (self.n_agents, MOTION_ACTION_DIM)
            ),
            "astar_valid": self._array(values["astar_valid"], (self.n_agents,), bool),
            "semantic_targets": self._array(
                values["semantic_targets"],
                (self.n_agents, SEMANTIC_SCORE_DIM),
            ),
            "semantic_validity": self._array(
                values["semantic_validity"], (self.n_agents,)
            ),
            "semantic_ood_reliability": self._array(
                values["semantic_ood_reliability"], (self.n_agents,)
            ),
            "stream_id": int(values.get("stream_id", 0)),
            "reward": float(values["reward"]),
            "done": bool(values["done"]),
            "value": float(values["value"]),
            "calibration_selected": bool(values["calibration_selected"]),
            "reward_confidence": float(values["reward_confidence"]),
        }
        if not np.all(record["action_masks"].any(axis=-1)):
            raise ValueError("Every robot must retain a legal action.")
        if not np.isfinite(record["reward"]) or not np.isfinite(record["value"]):
            raise ValueError("E1 rollout reward and value must be finite.")
        for name in ("semantic_targets", "semantic_validity", "semantic_ood_reliability"):
            if not np.isfinite(record[name]).all() or not np.all(
                (record[name] >= 0.0) & (record[name] <= 1.0)
            ):
                raise ValueError(f"E1 {name} must be finite and in [0, 1].")
        if not 0.0 <= record["reward_confidence"] <= 1.0:
            raise ValueError("E1 reward confidence must be in [0, 1].")
        valid_preferences = record["astar_preferences"][record["astar_valid"]]
        if valid_preferences.size and not np.allclose(
            valid_preferences.sum(axis=-1), 1.0
        ):
            raise ValueError("Each valid E1 A* preference must sum to one.")
        self._records.append(record)

    def tensors(self, *, last_value: float | Mapping[int, float] | np.ndarray, device: str | torch.device) -> E1RolloutTensors:
        if not self._records:
            raise ValueError("Cannot update an empty E1 rollout.")
        rewards = np.asarray([record["reward"] for record in self._records], dtype=np.float32)
        dones = np.asarray([record["done"] for record in self._records], dtype=bool)
        values = np.asarray([record["value"] for record in self._records], dtype=np.float32)
        advantages = np.zeros(len(self._records), dtype=np.float32)
        stream_ids = {record["stream_id"] for record in self._records}
        if isinstance(last_value, Mapping):
            next_values = {stream: float(last_value.get(stream, 0.0)) for stream in stream_ids}
        elif np.isscalar(last_value):
            next_values = {stream: float(last_value) for stream in stream_ids}
        else:
            vector = np.asarray(last_value, dtype=np.float32).reshape(-1)
            next_values = {stream: float(vector[stream]) for stream in stream_ids}
        gae_by_stream = {stream: 0.0 for stream in stream_ids}
        for index in reversed(range(len(self._records))):
            stream = self._records[index]["stream_id"]
            keep_bootstrap = 1.0 - float(dones[index])
            delta = rewards[index] + 0.99 * next_values[stream] * keep_bootstrap - values[index]
            gae = delta + 0.99 * 0.95 * keep_bootstrap * gae_by_stream[stream]
            advantages[index] = gae
            gae_by_stream[stream] = gae
            next_values[stream] = float(values[index])

        def stack(name: str, dtype: torch.dtype) -> Tensor:
            return torch.as_tensor(
                np.stack([record[name] for record in self._records]),
                dtype=dtype,
                device=device,
            )

        return E1RolloutTensors(
            physical_observations=stack("physical_observations", torch.float32),
            semantic_observations=stack("semantic_observations", torch.float32),
            actions=stack("actions", torch.long),
            old_log_probs=stack("log_probs", torch.float32),
            action_masks=stack("action_masks", torch.bool),
            astar_preferences=stack("astar_preferences", torch.float32),
            astar_valid=stack("astar_valid", torch.bool),
            calibration_selected=torch.as_tensor(
                [record["calibration_selected"] for record in self._records],
                dtype=torch.bool,
                device=device,
            ),
            reward_confidence=torch.as_tensor(
                [record["reward_confidence"] for record in self._records],
                dtype=torch.float32,
                device=device,
            ),
            semantic_targets=stack("semantic_targets", torch.float32),
            semantic_validity=stack("semantic_validity", torch.float32),
            semantic_ood_reliability=stack("semantic_ood_reliability", torch.float32),
            rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
            dones=torch.as_tensor(dones, dtype=torch.bool, device=device),
            values=torch.as_tensor(values, dtype=torch.float32, device=device),
            advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
            returns=torch.as_tensor(
                advantages + values, dtype=torch.float32, device=device
            ),
        )

    @staticmethod
    def _array(values: Any, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
        array = np.asarray(values, dtype=dtype)
        if array.shape != shape:
            raise ValueError(f"Expected E1 shape {shape}, got {array.shape}.")
        return array.copy()


class E1PPOUpdater:
    """The shared MAPPO update used by E1 core groups.

    `method` selects only frozen teacher masks.  It never changes the policy,
    critic, PPO hyperparameters, data source, or schedule.
    """

    def __init__(
        self, *, actor, critic, method: str, device: str | torch.device,
        update_epochs: int = 4, minibatch_steps: int = 64,
    ) -> None:
        if method not in {"MAPPO-DG", "RC-AStarKD", "LLMKD", "RC-AStarKD+LLMKD",
                          "Fixed-AStarKD+LLMKD", "RuleKD-v3", "ShuffleKD-v3", "NoOOD-v1", "NoGoalHint-v1"}:
            raise ValueError("E1 does not support this formal method yet.")
        self.actor, self.critic = actor.to(device), critic.to(device)
        self.method, self.device = method, torch.device(device)
        self.update_epochs, self.minibatch_steps = int(update_epochs), int(minibatch_steps)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=3e-4
        )

    def update(self, rollout: E1Rollout, *, last_value: float | Mapping[int, float] | np.ndarray,
               lambda_a: float, lambda_l: float) -> dict[str, float | int]:
        data = rollout.tensors(last_value=last_value, device=self.device)
        advantages = (data.advantages - data.advantages.mean()) / (
            data.advantages.std(unbiased=False) + 1e-8
        )
        sums = {name: 0.0 for name in (
            "policy_loss", "value_loss", "entropy", "astar_loss", "semantic_loss",
            "total_loss",
        )}
        denominator, count = 0, 0
        for _ in range(self.update_epochs):
            order = torch.randperm(len(rollout), device=self.device)
            for start in range(0, len(rollout), self.minibatch_steps):
                index = order[start:start + self.minibatch_steps]
                output = self.actor(data.physical_observations[index], data.semantic_observations[index])
                logits = output.action_logits.masked_fill(~data.action_masks[index], -1e9)
                distribution = Categorical(logits=logits)
                ratio = torch.exp(distribution.log_prob(data.actions[index]) - data.old_log_probs[index])
                advantage = advantages[index].unsqueeze(-1)
                policy_loss = -torch.minimum(
                    ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage
                ).mean()
                value_loss = functional.mse_loss(
                    self.critic(data.physical_observations[index]), data.returns[index]
                )
                astar_loss = self._astar_loss(data, output.motion_logits, index, lambda_a)
                semantic_loss, active = self._semantic_loss(data, output.semantic_scores, index, lambda_l)
                total = policy_loss + 0.5 * value_loss - 0.01 * distribution.entropy().mean() + astar_loss + semantic_loss
                if not torch.isfinite(total):
                    raise RuntimeError("E1 MAPPO produced a non-finite loss.")
                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 0.5
                )
                self.optimizer.step()
                values = (policy_loss, value_loss, distribution.entropy().mean(), astar_loss, semantic_loss, total)
                for name, value in zip(sums, values):
                    sums[name] += float(value.detach().cpu())
                denominator += active
                count += 1
        return {**{name: value / count for name, value in sums.items()},
                "semantic_valid_denominator": denominator, "finite": 1}

    def _astar_loss(self, data, logits, index, lambda_a):
        if self.method not in {"RC-AStarKD", "RC-AStarKD+LLMKD", "Fixed-AStarKD+LLMKD", "RuleKD-v3", "ShuffleKD-v3", "NoOOD-v1", "NoGoalHint-v1"}:
            return logits.sum() * 0.0
        active = data.astar_valid[index] & data.calibration_selected[index].unsqueeze(-1)
        denominator = active.sum()
        if not bool(denominator):
            return logits.sum() * 0.0
        divergence = functional.kl_div(torch.log_softmax(logits, dim=-1),
            data.astar_preferences[index], reduction="none").sum(dim=-1)
        weights = active.to(logits.dtype) * data.reward_confidence[index].unsqueeze(-1)
        return float(lambda_a) * (weights * divergence).sum() / denominator

    def _semantic_loss(self, data, scores, index, lambda_l):
        if self.method not in {"LLMKD", "RC-AStarKD+LLMKD", "Fixed-AStarKD+LLMKD", "RuleKD-v3", "ShuffleKD-v3", "NoOOD-v1", "NoGoalHint-v1"}:
            return scores.sum() * 0.0, 0
        subset = E1RolloutTensors(**{
            name: (value[index] if isinstance(value, Tensor) and value.shape[:1] == data.actions.shape[:1] else value)
            for name, value in data.__dict__.items()
        })
        return subset.semantic_mse_loss(scores, lambda_l=lambda_l)


class E1Trainer:
    """Single formal MAPPO path for the E1 core 2×2 and Fixed control."""

    def __init__(self, *, run, environment: Mapping[str, Any], training: Mapping[str, Any],
                 labels: RawSemanticEvidence, device: str | torch.device) -> None:
        self.run, self.environment_values, self.training = run, dict(environment), dict(training)
        self.labels, self.device = labels, torch.device(device)
        if run.semantic_control not in {"none", "llm", "rule_v3", "shuffle_v3", "no_ood_v1"}:
            raise ValueError("E1 does not support this semantic control yet.")
        self.semantic_dataset = labels.dataset
        if run.semantic_control == "rule_v3":
            from llm_mappo.e1_semantic_controls import rule_kd_v3
            self.semantic_dataset = rule_kd_v3(labels.records)
        elif run.semantic_control == "shuffle_v3":
            from llm_mappo.e1_semantic_controls import shuffle_kd_v3
            self.semantic_dataset = shuffle_kd_v3(labels.records)
        self._seed(int(run.seed))
        self.actor, self.critic = O0StudentActor(), O0CentralizedCritic()
        self.updater = E1PPOUpdater(actor=self.actor, critic=self.critic,
            method=run.group, device=self.device,
            update_epochs=int(training["update_epochs"]),
            minibatch_steps=int(training["minibatch_steps"]))
        self.schedule = LinearEnvStepSchedule(int(run.real_environment_steps))
        self.teacher = None if run.astar_kd == "disabled" else PureMotionTeacher()
        self.calibrator = None if self.teacher is None else RewardCalibrator()
        self.num_env_workers = int(training.get("num_env_workers", 1))
        self.rollout_length = int(training.get("rollout_length", training["rollout_steps"]))
        if self.num_env_workers < 1 or self.rollout_length < 1:
            raise ValueError("E1 rollout worker count and length must be positive.")
        self.environment = self._new_environment()
        self.student_shadow = self._new_environment()
        self.teacher_shadow = self._new_environment()
        self.real_adapter = ShadowStateAdapter(self.environment, code_commit="e1-vector-v1")
        self.student_adapter = ShadowStateAdapter(self.student_shadow, code_commit="e1-vector-v1")
        self.teacher_adapter = ShadowStateAdapter(self.teacher_shadow, code_commit="e1-vector-v1")
        self.vector_pool = None
        self._runtime = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def close(self) -> None:
        if self.vector_pool is not None:
            self.vector_pool.close()
            self.vector_pool = None
        for environment in (self.environment, self.student_shadow, self.teacher_shadow):
            environment.close()

    def run_prefix(self, max_steps: int, *, on_update=None) -> dict[str, Any]:
        """Run a bounded real-environment prefix; owner runner owns checkpoints."""
        if not 1 <= max_steps <= self.run.real_environment_steps:
            raise ValueError("E1 prefix is outside the frozen real-step budget.")
        if self.num_env_workers > 1:
            return self._run_vector_prefix(max_steps, on_update=on_update)
        self.student_shadow.reset(seed=self.run.seed); self.teacher_shadow.reset(seed=self.run.seed)
        if self._runtime is None:
            observations = self.environment.reset(seed=self.run.seed)
            updates, episode_index, episode_step = 0, 0, 0
            counts = {"teacher_queries": 0, "shadow_calls": 0, "ema_updates": 0,
                      "semantic_valid_slots": 0, "semantic_total_slots": 0}; latest = {}
        else:
            self.environment.reset(seed=self.run.seed)
            self.real_adapter.restore_bytes(self._runtime["snapshot"])
            observations = self.environment._observations()
            updates, episode_index, episode_step = (self._runtime[name] for name in ("updates", "episode_index", "episode_step"))
            counts, latest = dict(self._runtime["counts"]), dict(self._runtime["latest_metrics"])
        rollout = E1Rollout(self.environment_values["n_agents"])
        metrics = {"semantic_loss": 0.0, "semantic_valid_denominator": 0}
        for step in range(self.schedule.global_env_steps, max_steps):
            semantic, targets, valid_semantic, ood = self._semantic_batch(self.environment)
            masks = self.environment.action_masks()
            actions, log_probs, value = self._actions(observations, semantic, masks)
            preferences, valid_astar = self._teacher_batch(self.environment)
            confidence, selected = self._calibration(observations, valid_astar, episode_index,
                episode_step, step, counts)
            transition = self.environment.step(actions)
            done = bool(transition.terminated or transition.truncated or transition.metrics.deadlocked)
            rollout.add(physical_observations=observations, semantic_observations=semantic,
                actions=actions, log_probs=log_probs, action_masks=masks,
                astar_preferences=preferences, astar_valid=valid_astar,
                calibration_selected=selected, reward_confidence=confidence,
                semantic_targets=targets, semantic_validity=valid_semantic,
                semantic_ood_reliability=ood, reward=transition.team_reward,
                done=done, value=value)
            counts["semantic_valid_slots"] += int((valid_semantic > 0).sum())
            counts["semantic_total_slots"] += len(valid_semantic)
            self.schedule.advance_real_env_steps(1)
            observations, latest = transition.observations, transition.metrics.as_dict()
            if len(rollout) == int(self.training["rollout_steps"]) or step + 1 == max_steps:
                last = 0.0 if done else self._value(observations)
                metrics = self.updater.update(rollout, last_value=last,
                    lambda_a=self.schedule.weights()[0], lambda_l=self.schedule.weights()[1])
                updates += 1
                rollout = E1Rollout(self.environment_values["n_agents"])
                self._runtime = self._capture_runtime(episode_index=episode_index,
                    episode_step=episode_step, updates=updates, counts=counts, latest=latest)
                if on_update is not None:
                    on_update(dict(metrics), self.runtime_state())
            if done:
                episode_index, episode_step = episode_index + 1, 0
                observations = self.environment.reset(seed=self.run.seed + episode_index)
            else:
                episode_step += 1
        self._runtime = self._capture_runtime(episode_index=episode_index,
            episode_step=episode_step, updates=updates, counts=counts, latest=latest)
        return {"group": self.run.group, "seed": self.run.seed,
                "real_env_steps": self.schedule.global_env_steps, "updates": updates,
                "latest_episode_metrics": latest, "planner_query_count": self.environment.planner_query_counter.count,
                "lambda_a": self.schedule.weights()[0], "lambda_l": self.schedule.weights()[1],
                "exploratory_noisy_teacher": self.run.semantic_control == "llm", **counts,
                **metrics}

    def _run_vector_prefix(self, max_steps: int, *, on_update=None) -> dict[str, Any]:  # noqa: C901
        """Collect real transitions in spawned CPU processes and learn centrally."""
        if max_steps % self.num_env_workers:
            raise ValueError("E1 vector prefix must end on a whole worker step.")
        self.student_shadow.reset(seed=self.run.seed)
        self.teacher_shadow.reset(seed=self.run.seed)
        self.vector_pool = E1VectorEnvironmentPool(self.environment_values, self.run,
            self.semantic_dataset, self.num_env_workers)
        if self._runtime is not None:
            state = self._runtime
            self.vector_pool.restore(state["worker_states"])
        updates = 0 if self._runtime is None else int(self._runtime["updates"])
        counts = ({"teacher_queries": 0, "shadow_calls": 0, "ema_updates": 0,
                   "semantic_valid_slots": 0, "semantic_total_slots": 0}
                  if self._runtime is None else dict(self._runtime["counts"]))
        latest = {} if self._runtime is None else dict(self._runtime["latest_metrics"])
        rollout = E1Rollout(self.environment_values["n_agents"])
        vector_ticks = 0
        metrics = {"semantic_loss": 0.0, "semantic_valid_denominator": 0}
        started = time.perf_counter()
        rollout_wall = policy_wall = update_wall = 0.0
        planner_queries = 0
        while self.schedule.global_env_steps < max_steps:
            payloads = self.vector_pool.payloads
            observations = np.stack([item["observations"] for item in payloads])
            semantic = np.stack([item["semantic"] for item in payloads])
            masks = np.stack([item["action_masks"] for item in payloads])
            policy_started = time.perf_counter()
            actions, log_probs, values = self._actions_batch(observations, semantic, masks)
            policy_wall += time.perf_counter() - policy_started
            base_step = self.schedule.global_env_steps
            addresses = [base_step + index for index in range(self.num_env_workers)]
            selected = np.zeros(self.num_env_workers, dtype=bool)
            if self.calibrator is not None:
                for index, item in enumerate(payloads):
                    selected[index] = self.calibrator.select(
                        run_seed=self.run.seed, episode_index=int(item["episode_index"]),
                        episode_seed=int(item["episode_seed"]), environment_index=index,
                        real_global_step=addresses[index], episode_step=int(item["episode_step"]))
                    counts["teacher_queries"] += len(item["astar_valid"])
            rollout_started = time.perf_counter()
            responses = self.vector_pool.step(actions, selected, addresses)
            rollout_wall += time.perf_counter() - rollout_started
            confidences = np.zeros(self.num_env_workers, dtype=np.float32)
            for index, response in enumerate(responses):
                item = payloads[index]
                valid = item["astar_valid"]
                if self.calibrator is not None and selected[index] and valid.any():
                    snapshot = self._restore_worker_snapshot(response["snapshot"])
                    result = self.calibrator.run_paired_shadows(
                        snapshot=snapshot, real_adapter=self.real_adapter,
                        student_adapter=self.student_adapter, teacher_adapter=self.teacher_adapter,
                        student_logits=lambda env, obs: self._shadow_logits(env, obs),
                        teacher_preferences=lambda env: self._teacher_batch(env),
                        initial_valid_mask=valid, critic_value=lambda obs: self._value(obs),
                        gamma=0.99, address=snapshot.payload["address"])
                    confidences[index] = 1.0 if self.run.astar_kd == "fixed" else float(result.confidence)
                    counts["shadow_calls"] += 1; counts["ema_updates"] += 1
                elif self.calibrator is not None:
                    result = self.calibrator.record_delta_g(selected=bool(selected[index]), any_valid=bool(valid.any()), delta_g=0.0)
                    confidences[index] = 1.0 if self.run.astar_kd == "fixed" and selected[index] else float(result.confidence)
                rollout.add(physical_observations=item["observations"], semantic_observations=item["semantic"],
                    actions=actions[index], log_probs=log_probs[index], action_masks=item["action_masks"],
                    astar_preferences=item["astar_preferences"], astar_valid=valid,
                    calibration_selected=bool(selected[index]), reward_confidence=float(confidences[index]),
                    semantic_targets=item["semantic_targets"], semantic_validity=item["semantic_validity"],
                    semantic_ood_reliability=item["semantic_ood_reliability"], reward=response["team_reward"],
                    done=response["done"], value=float(values[index]), stream_id=index)
                counts["semantic_valid_slots"] += int((item["semantic_validity"] > 0).sum())
                counts["semantic_total_slots"] += len(item["semantic_validity"])
                latest = response["latest_metrics"]
                planner_queries += int(response["planner_query_count"])
            self.schedule.advance_real_env_steps(self.num_env_workers)
            vector_ticks += 1
            if vector_ticks == self.rollout_length or self.schedule.global_env_steps == max_steps:
                bootstrap = self._values_batch(np.stack([item["observations"] for item in self.vector_pool.payloads]))
                update_started = time.perf_counter()
                metrics = self.updater.update(rollout, last_value=bootstrap,
                    lambda_a=self.schedule.weights()[0], lambda_l=self.schedule.weights()[1])
                update_wall += time.perf_counter() - update_started
                updates += 1; rollout = E1Rollout(self.environment_values["n_agents"]); vector_ticks = 0
                self._runtime = self._capture_vector_runtime(updates=updates, counts=counts, latest=latest)
                metrics.update(self._performance_metrics(started, rollout_wall, policy_wall, update_wall))
                if on_update is not None: on_update(dict(metrics), self.runtime_state())
        self._runtime = self._capture_vector_runtime(updates=updates, counts=counts, latest=latest)
        metrics.update(self._performance_metrics(started, rollout_wall, policy_wall, update_wall))
        return {"group": self.run.group, "seed": self.run.seed, "real_env_steps": self.schedule.global_env_steps,
                "updates": updates, "latest_episode_metrics": latest, "planner_query_count": planner_queries,
                "lambda_a": self.schedule.weights()[0], "lambda_l": self.schedule.weights()[1],
                "exploratory_noisy_teacher": self.run.semantic_control == "llm", **counts, **metrics}

    def _actions_batch(self, observations, semantic, masks):
        physical = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        semantic_tensor = torch.as_tensor(semantic, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            flat_output = self.updater.actor(physical.flatten(0, 1), semantic_tensor.flatten(0, 1))
            logits = flat_output.action_logits.reshape(*masks.shape[:2], -1).masked_fill(~mask_tensor, -1e9)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            values = self.updater.critic(physical)
        return (actions.cpu().numpy().astype(np.int64), log_probs.cpu().numpy().astype(np.float32),
                values.cpu().numpy().astype(np.float32))

    def _values_batch(self, observations):
        with torch.no_grad():
            values = self.updater.critic(torch.as_tensor(observations, dtype=torch.float32, device=self.device))
        return values.cpu().numpy().astype(np.float32)

    def _restore_worker_snapshot(self, raw_snapshot: bytes) -> ShadowSnapshotV1:
        """Prepare the learner-side mirror before importing a worker snapshot."""
        source = ShadowSnapshotV1.from_bytes(raw_snapshot)
        address = source.payload["address"]
        self.environment.reset(seed=int(address["episode_seed"]))
        snapshot = rebind_snapshot_rng_guard(source)
        self.real_adapter.restore(snapshot)
        return snapshot

    def _capture_vector_runtime(self, *, updates, counts, latest):
        addresses = []
        for index in range(self.num_env_workers):
            addresses.append({"run_seed": int(self.run.seed), "episode_index": 0, "episode_seed": 0,
                "environment_index": index, "real_global_step": self.schedule.global_env_steps, "episode_step": 0})
        snapshots = self.vector_pool.snapshot(addresses)
        states = [{"snapshot": item["snapshot"], "episode_index": item["episode_index"],
                   "episode_step": item["episode_step"]} for item in snapshots]
        return {"schema": "e1-runtime-v2", "worker_states": states,
                "schedule_state": self.schedule.state_dict(), "updates": int(updates),
                "counts": dict(counts), "latest_metrics": dict(latest)}

    def _performance_metrics(self, started, rollout_wall, policy_wall, update_wall):
        elapsed = max(time.perf_counter() - started, 1e-9)
        cuda_allocated = cuda_reserved = 0
        if self.device.type == "cuda":
            cuda_allocated = int(torch.cuda.max_memory_allocated(self.device))
            cuda_reserved = int(torch.cuda.max_memory_reserved(self.device))
        return {"num_env_workers": self.num_env_workers, "rollout_length": self.rollout_length,
                "global_environment_steps": self.schedule.global_env_steps,
                "environment_steps_per_second": self.schedule.global_env_steps / elapsed,
                "rollout_wall_time": rollout_wall, "policy_inference_time": policy_wall,
                "ppo_update_time": update_wall, "total_elapsed_time": elapsed,
                "peak_cuda_memory_allocated": cuda_allocated,
                "peak_cuda_memory_reserved": cuda_reserved}

    def runtime_state(self) -> dict[str, Any]:
        """Expose only an update-boundary snapshot for the checkpoint writer."""
        if self._runtime is None:
            raise RuntimeError("E1 runtime state is unavailable before an update.")
        return dict(self._runtime)

    def restore_runtime_state(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") == "e1-runtime-v2":
            required = {"schema", "worker_states", "schedule_state", "updates", "counts", "latest_metrics"}
            if set(state) != required or len(state["worker_states"]) != self.num_env_workers:
                raise ValueError("E1 vector runtime state is incompatible.")
            schedule = state["schedule_state"]
            if int(schedule.get("total_env_steps", -1)) != self.run.real_environment_steps:
                raise ValueError("E1 runtime budget is incompatible.")
            self.schedule.global_env_steps = int(schedule["global_env_steps"])
            self._runtime = dict(state)
            return
        required = {"schema", "snapshot", "schedule_state", "episode_index", "episode_step",
                    "updates", "counts", "latest_metrics"}
        if set(state) != required or state.get("schema") != "e1-runtime-v1":
            raise ValueError("E1 runtime state is incompatible.")
        schedule = state["schedule_state"]
        if int(schedule.get("total_env_steps", -1)) != self.run.real_environment_steps:
            raise ValueError("E1 runtime budget is incompatible.")
        self.schedule.global_env_steps = int(schedule["global_env_steps"])
        self._runtime = dict(state)

    def _capture_runtime(self, *, episode_index, episode_step, updates, counts, latest):
        address = {"run_seed": int(self.run.seed), "episode_index": int(episode_index),
                   "episode_seed": int(self.run.seed) + int(episode_index), "environment_index": 0,
                   "real_global_step": self.schedule.global_env_steps, "episode_step": int(episode_step)}
        return {"schema": "e1-runtime-v1", "snapshot": self.real_adapter.capture(**address).to_bytes(),
                "schedule_state": self.schedule.state_dict(), "episode_index": int(episode_index),
                "episode_step": int(episode_step), "updates": int(updates), "counts": dict(counts),
                "latest_metrics": dict(latest)}

    def _new_environment(self):
        values = self.environment_values
        return Phase2Warehouse(n_agents=int(values["n_agents"]), max_steps=int(values["max_steps"]),
            env_id=str(values["environment_id"]), charge_threshold=float(values["charge_threshold"]),
            charge_release_threshold=float(values["charge_release_threshold"]), battery_cost_scale=float(values["battery_cost_scale"]),
            deadlock_steps=int(values["deadlock_steps"]), batch_interval=int(values["dynamic_ingress_interval"]),
            batch_size_range=tuple(values["batch_size_range"]), initial_priority_label="A",
            request_queue_size=int(values["queue_size"]), task_completion_target=int(values["task_target"]),
            observation_schema=ObservationSchema(self.run.observation_schema))

    def _semantic_batch(self, environment):
        views = self._views(environment)
        values = [self.semantic_dataset.retrieve(view.vector) for view in views]
        targets, validity, ood = zip(*values)
        if self.run.semantic_control == "none":
            validity = [0.0] * len(views)
        if self.run.semantic_control == "no_ood_v1":
            ood = [1.0 if item else 0.0 for item in validity]
        return (np.stack([view.vector for view in views]).astype(np.float32),
                np.stack(targets).astype(np.float32), np.asarray(validity, dtype=np.float32),
                np.asarray(ood, dtype=np.float32))

    def _views(self, environment):
        warehouse = environment.env
        width, height = warehouse.grid_size[1], warehouse.grid_size[0]
        return [SemanticViewV3.from_state(
            warehouse.shadow_layout_hash(), width, height,
            self._semantic_agent_state(environment, agent, environment._target_for_agent(agent.id)[1]),
            [self._semantic_agent_state(environment, peer, environment._target_for_agent(peer.id)[1])
             for peer in warehouse.agents if peer.id != agent.id],
        ) for agent in warehouse.agents]

    def _semantic_agent_state(self, environment, agent, target_kind):
        warehouse = environment.env
        direction = agent.dir.name.lower()
        deltas = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
        forward = deltas[direction]
        right = deltas[{"up": "right", "right": "down", "down": "left", "left": "up"}[direction]]
        def highway(delta):
            x, y = agent.x + delta[0], agent.y + delta[1]
            return 0 <= x < warehouse.grid_size[1] and 0 <= y < warehouse.grid_size[0] and warehouse._is_highway(x, y)
        task = warehouse.task_queue.task_for_agent(agent.id)
        return {"position": (agent.x, agent.y), "orientation": direction, "battery_ratio": float(agent.battery),
                "loaded": agent.carrying_shelf is not None, "dead": bool(agent.dead), "priority_present": task is not None,
                "priority_rank": 0.0 if task is None else (ord(task.label[0]) - ord("A")) / 25.0, "target_kind": target_kind,
                "on_highway": warehouse._is_highway(agent.x, agent.y), "at_charging_station": (agent.x, agent.y) in warehouse.charging_stations,
                "at_picking_station": (agent.x, agent.y) in warehouse.picking_stations,
                "adjacent_highway": {"forward": highway(forward), "right": highway(right),
                                     "backward": highway((-forward[0], -forward[1])), "left": highway((-right[0], -right[1]))}}

    def _actions(self, observations, semantic, masks):
        physical = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self.updater.actor(physical, torch.as_tensor(semantic, dtype=torch.float32, device=self.device))
            distribution = Categorical(logits=output.action_logits.masked_fill(~torch.as_tensor(masks, dtype=torch.bool, device=self.device), -1e9))
            actions = distribution.sample()
            return actions.cpu().numpy().astype(np.int64), distribution.log_prob(actions).cpu().numpy().astype(np.float32), float(self.updater.critic(physical.unsqueeze(0)).item())

    def _value(self, observations):
        with torch.no_grad():
            return float(self.updater.critic(torch.as_tensor(observations, dtype=torch.float32, device=self.device).unsqueeze(0)).item())

    def _teacher_batch(self, environment):
        n = int(self.environment_values["n_agents"])
        if self.teacher is None:
            return np.zeros((n, 3), dtype=np.float32), np.zeros(n, dtype=bool)
        warehouse = environment.env
        shelves = tuple((x, y) for y in range(warehouse.grid_size[0]) for x in range(warehouse.grid_size[1]) if not warehouse._is_highway(x, y))
        results = []
        for agent in warehouse.agents:
            goal, _ = environment._target_for_agent(agent.id)
            results.append(self.teacher.query(PureMotionQuery(layout_hash=warehouse.shadow_layout_hash(), width=int(warehouse.grid_size[1]), height=int(warehouse.grid_size[0]),
                blocked_coordinates=shelves if agent.carrying_shelf is not None else (), own_pose=(int(agent.x), int(agent.y)), orientation=agent.dir.name,
                goal=(int(goal[0]), int(goal[1])), occupied_coordinates=tuple((int(peer.x), int(peer.y)) for peer in warehouse.agents if peer.id != agent.id),
                pure_motion_mask=(False, True, True, True, False), dead=bool(agent.dead), picking_lock=bool(agent.picking_lock_steps),
                mandatory_toggle_load=environment._requires_pickup(agent.id), footprint_class="loaded" if agent.carrying_shelf is not None else "unloaded")))
        return np.stack([item.motion_preferences[1:4] for item in results]), np.asarray([item.valid for item in results], dtype=bool)

    def _calibration(self, observations, valid, episode_index, episode_step, step, counts):
        if self.calibrator is None:
            return 0.0, False
        address = {"run_seed": self.run.seed, "episode_index": episode_index,
                   "episode_seed": self.run.seed + episode_index, "environment_index": 0,
                   "real_global_step": step, "episode_step": episode_step}
        selected = self.calibrator.select(**address)
        counts["teacher_queries"] += len(valid)
        if not selected or not valid.any():
            result = self.calibrator.record_delta_g(selected=selected, any_valid=bool(valid.any()), delta_g=0.0)
        else:
            result = self.calibrator.run_paired_shadows(snapshot=self.real_adapter.capture(**address), real_adapter=self.real_adapter,
                student_adapter=self.student_adapter, teacher_adapter=self.teacher_adapter,
                student_logits=lambda env, obs: self._shadow_logits(env, obs),
                teacher_preferences=lambda env: self._teacher_batch(env), initial_valid_mask=valid,
                critic_value=lambda obs: self._value(obs), gamma=0.99, address=address)
            counts["shadow_calls"] += 1; counts["ema_updates"] += 1
        return (1.0 if self.run.astar_kd == "fixed" and selected else float(result.confidence)), selected

    def _shadow_logits(self, environment, observations):
        semantic, _, _, _ = self._semantic_batch(environment)
        with torch.no_grad():
            return self.updater.actor(torch.as_tensor(observations, dtype=torch.float32, device=self.device), torch.as_tensor(semantic, dtype=torch.float32, device=self.device)).action_logits.cpu().numpy()

    @staticmethod
    def _seed(seed):
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
