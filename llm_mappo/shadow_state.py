"""Versioned, branch-local state snapshots for O0 reward calibration."""

import base64
import copy
from dataclasses import dataclass
from hashlib import sha256
import json
import random
from typing import Any, Dict, Iterable

import numpy as np


SHADOW_SCHEMA_VERSION = "o0-shadow-state-v1"
CRN_VERSION = "crn-v1"


def _json_value(value: Any) -> Any:  # noqa: C901
    """Return the frozen JSON representation used for hashes and snapshots."""
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        raw = contiguous.tobytes(order="C")
        return {
            "__ndarray__": True,
            "data": base64.b64encode(raw).decode("ascii"),
            "dtype": contiguous.dtype.str,
            "sha256": sha256(raw).hexdigest(),
            "shape": list(contiguous.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"__tuple__": [_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return {"__set__": sorted(_json_value(item) for item in value)}
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Snapshot values must be finite.")
        return value
    raise TypeError(f"Unsupported canonical snapshot value: {type(value)!r}")


def _from_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("__ndarray__") is True:
        raw = base64.b64decode(value["data"])
        if sha256(raw).hexdigest() != value["sha256"]:
            raise ValueError("Snapshot array digest mismatch.")
        array = np.frombuffer(raw, dtype=np.dtype(value["dtype"])).copy()
        return array.reshape(tuple(value["shape"]))
    if "__bytes__" in value:
        return base64.b64decode(value["__bytes__"])
    if "__tuple__" in value:
        return tuple(_from_json_value(item) for item in value["__tuple__"])
    if "__set__" in value:
        return set(_from_json_value(item) for item in value["__set__"])
    return {key: _from_json_value(item) for key, item in value.items()}


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        _json_value(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _global_rng_guard() -> str:
    """Hash global RNG state without importing it into shadow branches."""
    state: Dict[str, Any] = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    try:
        import torch

        state["torch_cpu"] = torch.get_rng_state().cpu().numpy()
        if torch.cuda.is_available():
            state["torch_cuda"] = [
                item.cpu().numpy() for item in torch.cuda.get_rng_state_all()
            ]
    except ImportError:
        pass
    return sha256(_canonical_bytes(state)).hexdigest()


@dataclass(frozen=True)
class ShadowSnapshotV1:
    """A JSON-serializable O0 branch snapshot with a canonical state digest."""

    payload: Dict[str, Any]
    state_hash: str

    @classmethod
    def create(cls, payload: Dict[str, Any]) -> "ShadowSnapshotV1":
        state_hash = sha256(_canonical_bytes(payload)).hexdigest()
        return cls(payload=copy.deepcopy(payload), state_hash=state_hash)

    def to_bytes(self) -> bytes:
        return _canonical_bytes(
            {
                "payload": self.payload,
                "schema_version": SHADOW_SCHEMA_VERSION,
                "state_hash": self.state_hash,
            }
        )

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ShadowSnapshotV1":
        try:
            encoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Snapshot bytes are not canonical UTF-8 JSON.") from error
        document = _from_json_value(encoded)
        if document.get("schema_version") != SHADOW_SCHEMA_VERSION:
            raise ValueError("Unsupported shadow snapshot schema.")
        snapshot = cls.create(document["payload"])
        if snapshot.state_hash != document.get("state_hash"):
            raise ValueError("Snapshot hash mismatch.")
        return snapshot


class EventAddressedRandomness:
    """Stateless `crn-v1` randomness keyed only by the frozen event address."""

    version = CRN_VERSION

    def _digest(
        self,
        *,
        episode_seed: int,
        real_global_step: int,
        shadow_offset: int,
        event_type: str,
        event_slot: int,
    ) -> bytes:
        key = [
            self.version,
            int(episode_seed),
            int(real_global_step),
            int(shadow_offset),
            str(event_type),
            int(event_slot),
        ]
        raw = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return sha256(raw).digest()

    def integer(self, *, low: int, high: int, **address: Any) -> int:
        if not low < high:
            raise ValueError("CRN integer range must satisfy low < high.")
        value = int.from_bytes(self._digest(**address)[:8], "big")
        return low + value % (high - low)

    def choose_without_replacement(
        self, candidates: Iterable[int], count: int, **address: Any
    ) -> list[int]:
        candidate_ids = sorted(int(candidate) for candidate in candidates)
        if count < 0 or count > len(candidate_ids):
            raise ValueError("CRN choice count is outside the candidate range.")
        ranked = []
        for candidate in candidate_ids:
            payload = self._digest(**address) + candidate.to_bytes(8, "big", signed=True)
            ranked.append((sha256(payload).digest(), candidate))
        return [candidate for _, candidate in sorted(ranked)[:count]]


class ShadowStateAdapter:
    """Capture/import only explicit mutable state into preconstructed branches."""

    def __init__(self, environment, *, code_commit: str) -> None:
        self.environment = environment
        self.code_commit = str(code_commit)
        self._active_address: Dict[str, int] = {}

    def _config_payload(self) -> Dict[str, Any]:
        environment = self.environment
        return {
            "adapter": {
                "n_agents": environment.n_agents,
                "max_steps": environment.max_steps,
                "env_id": environment.env_id,
                "charge_threshold": environment.charge_threshold,
                "charge_release_threshold": environment.charge_release_threshold,
                "battery_cost_scale": environment.battery_cost_scale,
                "deadlock_steps": environment.deadlock_steps,
                "waypoint_reward": environment.waypoint_reward,
                "observation_schema": environment.observation_schema.value,
                "oracle_interaction_mask": environment.oracle_interaction_mask,
                "batch_interval": environment.batch_interval,
                "batch_size_range": environment.batch_size_range,
                "request_queue_size": environment.request_queue_size,
                "task_completion_target": environment.task_completion_target,
            },
            "warehouse": environment.env.shadow_config_payload(),
            "wrapper_chain": self._wrapper_config(),
        }

    def _wrapper_config(self) -> list[str]:
        wrapper = self.environment._env
        names = []
        while True:
            names.append(f"{type(wrapper).__module__}.{type(wrapper).__qualname__}")
            if not hasattr(wrapper, "env"):
                return names
            wrapper = wrapper.env

    def config_hash(self) -> str:
        return sha256(_canonical_bytes(self._config_payload())).hexdigest()

    def capture(self, **address: int) -> ShadowSnapshotV1:
        required = {
            "run_seed",
            "episode_index",
            "episode_seed",
            "environment_index",
            "real_global_step",
            "episode_step",
        }
        if set(address) != required:
            raise ValueError("Snapshot address does not match the frozen schema.")
        if self.environment.env.renderer is not None:
            raise ValueError("Training shadow snapshots require a null renderer.")
        payload = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "code_commit": self.code_commit,
            "environment_config_hash": self.config_hash(),
            "layout_hash": self.environment.env.shadow_layout_hash(),
            "address": {name: int(address[name]) for name in sorted(address)},
            "wrapper": self._export_wrapper_state(),
            "space_rng": self._export_space_rng_state(),
            "warehouse": self.environment.env.export_shadow_state(),
            "adapter": self.environment.export_shadow_state(),
            "global_rng_guard": _global_rng_guard(),
        }
        self._active_address = copy.deepcopy(payload["address"])
        return ShadowSnapshotV1.create(payload)

    def restore(self, snapshot: ShadowSnapshotV1) -> None:
        payload = snapshot.payload
        if payload.get("schema_version") != SHADOW_SCHEMA_VERSION:
            raise ValueError("Unsupported shadow snapshot schema.")
        if payload.get("environment_config_hash") != self.config_hash():
            raise ValueError("Shadow snapshot config hash mismatch.")
        if payload.get("layout_hash") != self.environment.env.shadow_layout_hash():
            raise ValueError("Shadow snapshot layout hash mismatch.")
        if self.environment.env.renderer is not None:
            raise ValueError("Training shadow restore requires a null renderer.")
        self.environment.env.import_shadow_state(copy.deepcopy(payload["warehouse"]))
        self.environment.import_shadow_state(copy.deepcopy(payload["adapter"]))
        self._import_wrapper_state(copy.deepcopy(payload["wrapper"]))
        self._import_space_rng_state(copy.deepcopy(payload["space_rng"]))
        self._active_address = copy.deepcopy(payload["address"])
        if self.state_hash() != snapshot.state_hash:
            raise ValueError("Shadow snapshot restore hash mismatch.")

    def restore_bytes(self, raw: bytes) -> None:
        self.restore(ShadowSnapshotV1.from_bytes(raw))

    def state_hash(self) -> str:
        payload = {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "code_commit": self.code_commit,
            "environment_config_hash": self.config_hash(),
            "layout_hash": self.environment.env.shadow_layout_hash(),
            "address": copy.deepcopy(self._active_address),
            "wrapper": self._export_wrapper_state(),
            "space_rng": self._export_space_rng_state(),
            "warehouse": self.environment.env.export_shadow_state(),
            "adapter": self.environment.export_shadow_state(),
            "global_rng_guard": _global_rng_guard(),
        }
        return sha256(_canonical_bytes(payload)).hexdigest()

    def assert_global_rng_guard(self, snapshot: ShadowSnapshotV1) -> None:
        if _global_rng_guard() != snapshot.payload["global_rng_guard"]:
            raise ValueError("Global RNG changed during shadow calibration.")

    def _export_wrapper_state(self) -> list[Dict[str, Any]]:
        wrapper = self.environment._env
        states = []
        while True:
            mutable = {
                name: copy.deepcopy(wrapper.__dict__[name])
                for name in ("_has_reset", "_elapsed_steps")
                if name in wrapper.__dict__
            }
            states.append(
                {
                    "class": (
                        f"{type(wrapper).__module__}.{type(wrapper).__qualname__}"
                    ),
                    "state": mutable,
                }
            )
            if not hasattr(wrapper, "env"):
                return states
            wrapper = wrapper.env

    def _import_wrapper_state(self, states: list[Dict[str, Any]]) -> None:
        wrapper = self.environment._env
        for expected in states:
            actual = f"{type(wrapper).__module__}.{type(wrapper).__qualname__}"
            if actual != expected["class"]:
                raise ValueError("Shadow wrapper chain mismatch.")
            for name, value in expected["state"].items():
                wrapper.__dict__[name] = copy.deepcopy(value)
            if hasattr(wrapper, "env"):
                wrapper = wrapper.env

    def _export_space_rng_state(self) -> list[Dict[str, Any]]:
        states: list[Dict[str, Any]] = []
        self._collect_space_rng_state(
            self.environment._env.action_space, "action_space", states
        )
        self._collect_space_rng_state(
            self.environment._env.observation_space, "observation_space", states
        )
        return states

    def _collect_space_rng_state(self, space, path: str, states: list[Dict[str, Any]]):
        generator = space.__dict__.get("_np_random")
        states.append(
            {
                "path": path,
                "present": generator is not None,
                "state": (
                    copy.deepcopy(generator.bit_generator.state)
                    if generator is not None
                    else None
                ),
            }
        )
        spaces = getattr(space, "spaces", None)
        if isinstance(spaces, dict):
            for key in sorted(spaces):
                self._collect_space_rng_state(spaces[key], f"{path}.{key}", states)
        elif isinstance(spaces, (list, tuple)):
            for index, child in enumerate(spaces):
                self._collect_space_rng_state(child, f"{path}[{index}]", states)

    def _import_space_rng_state(self, states: list[Dict[str, Any]]) -> None:
        expected = {record["path"]: record for record in states}
        actual = self._space_paths()
        if set(expected) != set(actual):
            raise ValueError("Shadow space RNG path mismatch.")
        for path, target in actual.items():
            source = expected[path]
            if not source["present"]:
                target.__dict__["_np_random"] = None
                continue
            generator = target.__dict__.get("_np_random")
            if generator is None:
                generator = np.random.default_rng()
                target.__dict__["_np_random"] = generator
            generator.bit_generator.state = copy.deepcopy(source["state"])

    def _space_paths(self) -> Dict[str, Any]:
        paths: Dict[str, Any] = {}

        def visit(space, path: str) -> None:
            paths[path] = space
            spaces = getattr(space, "spaces", None)
            if isinstance(spaces, dict):
                for key in sorted(spaces):
                    visit(spaces[key], f"{path}.{key}")
            elif isinstance(spaces, (list, tuple)):
                for index, child in enumerate(spaces):
                    visit(child, f"{path}[{index}]")

        visit(self.environment._env.action_space, "action_space")
        visit(self.environment._env.observation_space, "observation_space")
        return paths
