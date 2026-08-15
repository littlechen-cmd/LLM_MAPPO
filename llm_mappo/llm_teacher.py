"""Offline LLM teacher contracts for Phase 4 semantic distillation.

The module deliberately separates untrusted model output from warehouse rules.
It never assigns AGVs or emits movement actions.  Training consumes cached labels
only; a provider is contacted exclusively by the explicit dataset CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence
from urllib import error, request

from llm_mappo.types import PriorityAdjustment, SemanticPreferenceLabel


OBSERVATION_VERSION = "phase4-semantic-v2"
_SEMANTIC_KEYS = frozenset(
    (
        "task_commitment",
        "task_reason",
        "local_assertiveness",
        "coordination_reason",
    )
)
_ADJUSTMENT_KEYS = frozenset(("adjustments",))


@dataclass(frozen=True)
class EngagementScenario:
    """A compact, auditable state description for one focal AGV."""

    scenario_id: str
    observation_version: str
    scenario_type: str
    observation: tuple[float, ...]
    agent_id: int
    battery: float
    loaded: bool
    priority_label: str | None
    target_kind: str
    nearby_agents: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LabelledScenario:
    """One cached teacher label paired with its exact actor observation."""

    scenario: EngagementScenario
    label: SemanticPreferenceLabel
    task_reason: str
    coordination_reason: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self.scenario)
        data.update(
            {
                "task_commitment": self.label.task_commitment,
                "local_assertiveness": self.label.local_assertiveness,
                "model": self.label.model,
                "created_at": self.label.created_at,
                "task_reason": self.task_reason,
                "coordination_reason": self.coordination_reason,
            }
        )
        return data


class TeacherProvider(Protocol):
    """Minimal boundary for an explicit offline labelling provider."""

    name: str

    def label_semantics(
        self, scenario: EngagementScenario
    ) -> tuple[float, str, float, str]:
        """Return task commitment and local assertiveness with rationales."""

    def parse_priority_instruction(
        self, instruction: str, active_tasks: Sequence[dict[str, Any]]
    ) -> Sequence[PriorityAdjustment]:
        """Translate a user instruction into label-only adjustments."""


class MockTeacher:
    """Deterministic local teacher for tests and CPU feasibility runs."""

    name = "mock-semantic-v2"

    def label_semantics(
        self, scenario: EngagementScenario
    ) -> tuple[float, str, float, str]:
        if scenario.scenario_type == "low_battery_diversion" or scenario.battery < 0.2:
            return (
                0.1,
                "battery safety overrides continued transport progress",
                0.8,
                "urgent charging passage is appropriate when locally safe",
            )
        if scenario.scenario_type == "narrow_corridor_yield" and not scenario.loaded:
            return (
                0.6,
                "the transport task remains active",
                0.2,
                "the empty AGV yields to the loaded peer in the narrow corridor",
            )
        if scenario.scenario_type == "station_exit_congestion":
            return (
                0.6,
                "the transport task remains active",
                0.25,
                "yield locally to clear the charging-station exit",
            )
        loaded_peers = sum(peer["loaded"] for peer in scenario.nearby_agents)
        higher_peer = any(
            peer["priority_label"] is not None
            and scenario.priority_label is not None
            and peer["priority_label"] < scenario.priority_label
            for peer in scenario.nearby_agents
        )
        if higher_peer:
            return (
                0.5,
                "continue the task after the higher-priority peer passes",
                0.2,
                "yield to the higher-priority nearby AGV",
            )
        if scenario.loaded and scenario.priority_label == "A" and loaded_peers:
            return (
                0.9,
                "loaded high-priority delivery remains strongly committed",
                0.85,
                "the high-priority loaded delivery has local precedence",
            )
        if scenario.loaded:
            return (
                0.7,
                "continue loaded delivery under normal conditions",
                0.7,
                "proceed normally when the local route is clear",
            )
        return (
            0.6,
            "continue the assigned transport task",
            0.7,
            "proceed normally when the local route is clear",
        )

    def parse_priority_instruction(
        self, instruction: str, active_tasks: Sequence[dict[str, Any]]
    ) -> Sequence[PriorityAdjustment]:
        # The mock only supports an explicit task label and target letter.  This
        # keeps local tests deterministic while exercising the same validator.
        import re

        match = re.search(r"\b([A-Z])(\d+)\b.*?\b([A-Z])\b", instruction)
        if match is None:
            raise ValueError("Mock teacher requires '<task> ... <target letter>'.")
        task = f"{match.group(1)}{match.group(2)}"
        new_label = f"{match.group(3)}{match.group(2)}"
        adjustments = [PriorityAdjustment(task, new_label, instruction)]
        conflict = next(
            (item for item in active_tasks if item.get("label") == new_label), None
        )
        if conflict is not None and conflict["label"] != task:
            adjustments.append(
                PriorityAdjustment(
                    conflict["label"],
                    task,
                    "swap to preserve unique active task labels",
                )
            )
        return tuple(adjustments)


class DeepSeekTeacher:
    """Explicit OpenAI-compatible DeepSeek client used only before training."""

    name = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-v4-flash",
        endpoint: str = "https://api.deepseek.com/chat/completions",
        timeout_seconds: float = 120.0,
        max_attempts: int = 3,
        max_tokens: int = 1024,
        thinking_enabled: bool = False,
        reasoning_effort: str | None = None,
        retry_backoff_seconds: Sequence[float] = (5.0, 15.0),
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for the DeepSeek teacher.")
        self.model = model
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, int(max_attempts))
        self.max_tokens = max(128, int(max_tokens))
        self.thinking_enabled = bool(thinking_enabled)
        if reasoning_effort not in (None, "high", "max"):
            raise ValueError("reasoning_effort must be 'high', 'max', or None.")
        if reasoning_effort is not None and not self.thinking_enabled:
            raise ValueError("reasoning_effort requires thinking_enabled=True.")
        self.reasoning_effort = reasoning_effort or "high"
        self.retry_backoff_seconds = tuple(
            float(value) for value in retry_backoff_seconds
        )
        suffix = f":thinking-{self.reasoning_effort}" if self.thinking_enabled else ""
        self.name = f"deepseek:{model}{suffix}"

    def label_semantics(
        self, scenario: EngagementScenario
    ) -> tuple[float, str, float, str]:
        content = self._complete(
            "Return JSON only with exactly these keys: task_commitment, task_reason, "
            "local_assertiveness, coordination_reason. Both scores must be numbers "
            "in [0, 1]. task_commitment means willingness to continue the original "
            "transport task; it never means willingness to reach a charging target. "
            "local_assertiveness means willingness to claim immediate local passage; "
            "low means yield and high means proceed only when hard rules permit. "
            "Priority order is A > B > C. Decision precedence is hard safety, battery "
            "safety, loaded-versus-empty yielding, task priority, then ordinary task "
            "progress. Never describe B as higher priority than A: when an empty A AGV "
            "yields to a loaded B AGV, attribute that yield to the load rule overriding "
            "priority. Treat every loaded field as authoritative and never infer that a "
            "peer is loaded when its loaded field is false. In particular, do not "
            "invent a loaded B peer in a low_battery_diversion scenario. Controlled "
            "reference "
            "directions: low_battery_diversion requires "
            "task_commitment <= 0.3; priority_conflict requires local_assertiveness >= "
            "0.7; narrow_corridor_yield requires local_assertiveness <= 0.3; and "
            "station_exit_congestion requires local_assertiveness <= 0.4 so the station "
            "occupant can clear the exit. Do not assign AGVs or propose actions, paths, "
            "station controls, or task-label changes.\n"
            + json.dumps(_teacher_scenario_view(scenario), ensure_ascii=False)
        )
        return parse_semantic_response(content)

    def parse_priority_instruction(
        self, instruction: str, active_tasks: Sequence[dict[str, Any]]
    ) -> Sequence[PriorityAdjustment]:
        content = self._complete(
            "Return JSON only: {\"adjustments\":[{\"task\":\"B4\","
            "\"new_label\":\"A4\",\"reason\":\"...\"}]}. Only change label "
            "letters, preserve numeric suffixes, do not assign AGVs, and only refer "
            "to active tasks.\nInstruction: "
            + instruction
            + "\nActive tasks: "
            + json.dumps(list(active_tasks), ensure_ascii=False)
        )
        return parse_priority_adjustments(content)

    def _complete(self, prompt: str) -> str:
        query = self._build_request(prompt)
        decoded = self._request_completion(query)
        return _completion_text(decoded)

    def _build_request(self, prompt: str) -> request.Request:
        body = {
            "model": self.model,
            "stream": False,
            "max_tokens": self.max_tokens,
            "thinking": {
                "type": "enabled" if self.thinking_enabled else "disabled"
            },
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a JSON-only warehouse semantic teacher.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if self.thinking_enabled:
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = 0.0
        payload = json.dumps(body).encode("utf-8")
        return request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    def _request_completion(self, query: request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with request.urlopen(query, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                message = (
                    f"DeepSeek teacher HTTP {exc.code} for model {self.model} "
                    f"(attempt {attempt}/{self.max_attempts}): {exc.reason}; {body}"
                )
                # Client errors usually indicate an invalid key, model, or payload.
                # Retrying those only repeats the same request and hides the cause.
                if exc.code < 500 and exc.code != 429:
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(message)
            except (error.URLError, TimeoutError) as exc:
                last_error = RuntimeError(
                    f"{type(exc).__name__} on attempt {attempt}/"
                    f"{self.max_attempts} for model {self.model} "
                    f"(timeout={self.timeout_seconds}s): {exc}"
                )
            if attempt < self.max_attempts:
                delay = self._retry_delay(attempt)
                if delay > 0:
                    time.sleep(delay)
        detail = str(last_error)
        raise RuntimeError(
            f"DeepSeek teacher request failed for model {self.model} after "
            f"{self.max_attempts} attempts "
            f"(timeout={self.timeout_seconds}s): {detail}"
        ) from last_error

    def _retry_delay(self, attempt: int) -> float:
        if not self.retry_backoff_seconds:
            return 0.0
        index = min(attempt - 1, len(self.retry_backoff_seconds) - 1)
        return self.retry_backoff_seconds[index]


def _teacher_scenario_view(scenario: EngagementScenario) -> dict[str, Any]:
    """Expose semantic state to the LLM while retaining raw observations offline."""
    return {
        "scenario_id": scenario.scenario_id,
        "observation_version": scenario.observation_version,
        "scenario_type": scenario.scenario_type,
        "focal_agent": {
            "agent_id": scenario.agent_id,
            "battery": scenario.battery,
            "loaded": scenario.loaded,
            "priority_label": scenario.priority_label,
            "target_kind": scenario.target_kind,
        },
        "nearby_agents": list(scenario.nearby_agents),
    }


def _completion_text(decoded: Any) -> str:
    message = _completion_message(decoded)
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    # Some newer reasoning-capable gateways put the answer in
    # reasoning_content when content is empty. Parsing remains bounded by the
    # same engagement/priority schema validators below.
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        bounded = _bounded_response_text(reasoning)
        if bounded is not None:
            return bounded
    diagnostics = _completion_diagnostics(decoded, message)
    raise ValueError(
        "DeepSeek response contained no bounded JSON answer; "
        f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
    )


def _completion_message(decoded: Any) -> dict[str, Any]:
    try:
        choices = decoded["choices"]
        message = choices[0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        diagnostics = _completion_diagnostics(decoded)
        raise ValueError(
            "DeepSeek response did not contain a chat completion; "
            f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
        ) from exc
    if not isinstance(message, dict):
        diagnostics = _completion_diagnostics(decoded)
        raise ValueError(
            "DeepSeek response message was not an object; "
            f"diagnostics={json.dumps(diagnostics, ensure_ascii=False)}"
        )
    return message


def _completion_diagnostics(
    decoded: Any, message: dict[str, Any] | None = None
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"response_type": type(decoded).__name__}
    if isinstance(decoded, dict):
        diagnostics["top_level_keys"] = sorted(decoded)
        choices = decoded.get("choices")
        diagnostics["choices_count"] = (
            len(choices) if isinstance(choices, list) else None
        )
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            diagnostics["finish_reason"] = choices[0].get("finish_reason")
        usage = decoded.get("usage")
        if isinstance(usage, dict):
            diagnostics["usage"] = {
                key: value
                for key, value in usage.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
    if message is not None:
        diagnostics["message_keys"] = sorted(message)
        diagnostics["content"] = _text_diagnostics(message.get("content"))
        diagnostics["reasoning_content"] = _text_diagnostics(
            message.get("reasoning_content")
        )
    return diagnostics


def _text_diagnostics(value: Any) -> dict[str, Any]:
    details: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, str):
        details["length"] = len(value)
        details["preview"] = value[:200].replace("\n", " ")
    return details


def build_engagement_scenarios(
    env, scenario_type: str = "normal_transport"
) -> list[EngagementScenario]:
    """Compress current adapter state without permitting teacher control output."""
    scenarios = []
    warehouse = env.env
    observations = env._observations()
    for index, agent in enumerate(warehouse.agents):
        task = warehouse.task_queue.task_for_agent(agent.id)
        _, target_kind = env._target_for_agent(agent.id)
        peers = []
        for peer in warehouse.agents:
            if peer.id == agent.id:
                continue
            peer_task = warehouse.task_queue.task_for_agent(peer.id)
            _, peer_target_kind = env._target_for_agent(peer.id)
            peers.append(
                {
                    "agent_id": int(peer.id),
                    "distance": int(abs(peer.x - agent.x) + abs(peer.y - agent.y)),
                    "loaded": bool(peer.carrying_shelf is not None),
                    "battery": round(float(peer.battery), 4),
                    "priority_label": peer_task.label[0] if peer_task else None,
                    "target_kind": peer_target_kind,
                    "at_charging_station": (peer.x, peer.y)
                    in warehouse.charging_stations,
                }
            )
        peers.sort(key=lambda item: (item["distance"], item["agent_id"]))
        observation = tuple(float(value) for value in observations[index])
        scenarios.append(
            EngagementScenario(
                scenario_id=_scenario_id(observation, agent.id, scenario_type),
                observation_version=OBSERVATION_VERSION,
                scenario_type=scenario_type,
                observation=observation,
                agent_id=int(agent.id),
                battery=round(float(agent.battery), 4),
                loaded=bool(agent.carrying_shelf is not None),
                priority_label=task.label[0] if task else None,
                target_kind=target_kind,
                nearby_agents=tuple(peers[:3]),
            )
        )
    return scenarios


def parse_semantic_response(content: str) -> tuple[float, str, float, str]:
    """Accept exactly the frozen Phase 4 dual-semantic response schema."""
    payload = _json_object(content, _SEMANTIC_KEYS)
    if set(payload) != _SEMANTIC_KEYS:
        raise ValueError("Semantic response must contain only the four frozen keys.")
    task_commitment = _bounded_score(payload, "task_commitment")
    local_assertiveness = _bounded_score(payload, "local_assertiveness")
    task_reason = _nonempty_reason(payload, "task_reason")
    coordination_reason = _nonempty_reason(payload, "coordination_reason")
    return (
        task_commitment,
        task_reason,
        local_assertiveness,
        coordination_reason,
    )


def _bounded_score(payload: dict[str, Any], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a numeric value.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{key} must be within [0, 1].")
    return float(value)


def _nonempty_reason(payload: dict[str, Any], key: str) -> str:
    reason = payload[key]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return reason.strip()


def parse_priority_adjustments(content: str) -> tuple[PriorityAdjustment, ...]:
    """Parse label-only model output; TaskQueue performs final atomic checks."""
    payload = _json_object(content, _ADJUSTMENT_KEYS)
    if set(payload) != _ADJUSTMENT_KEYS or not isinstance(payload["adjustments"], list):
        raise ValueError("Priority response must contain only an adjustments list.")
    adjustments = []
    for item in payload["adjustments"]:
        if not isinstance(item, dict) or set(item) != {"task", "new_label", "reason"}:
            raise ValueError("Each adjustment requires task, new_label, and reason.")
        if not all(isinstance(item[key], str) and item[key] for key in item):
            raise ValueError("Priority adjustment fields must be non-empty strings.")
        adjustments.append(
            PriorityAdjustment(item["task"], item["new_label"], item["reason"])
        )
    return tuple(adjustments)


def write_labelled_scenarios(
    path: str | Path, records: Iterable[LabelledScenario]
) -> int:
    """Atomically persist an offline JSONL label dataset."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record_list = list(records)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in record_list:
            stream.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return len(record_list)


def append_labelled_scenario(path: str | Path, record: LabelledScenario) -> None:
    """Append one validated label to a resumable JSONL checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_labelled_scenarios(path: str | Path) -> list[LabelledScenario]:
    """Load and validate an offline teacher dataset before any training starts."""
    source = Path(path)
    records = []
    lines = source.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            scenario = EngagementScenario(
                scenario_id=payload["scenario_id"],
                observation_version=payload["observation_version"],
                scenario_type=payload.get("scenario_type", "normal_transport"),
                observation=tuple(float(item) for item in payload["observation"]),
                agent_id=int(payload["agent_id"]),
                battery=float(payload["battery"]),
                loaded=bool(payload["loaded"]),
                priority_label=payload["priority_label"],
                target_kind=payload["target_kind"],
                nearby_agents=tuple(payload["nearby_agents"]),
            )
            label = SemanticPreferenceLabel(
                scenario_id=scenario.scenario_id,
                observation_version=scenario.observation_version,
                task_commitment=float(payload["task_commitment"]),
                local_assertiveness=float(payload["local_assertiveness"]),
                model=str(payload["model"]),
                created_at=str(payload["created_at"]),
            )
            records.append(
                LabelledScenario(
                    scenario,
                    label,
                    str(payload["task_reason"]),
                    str(payload["coordination_reason"]),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid label record at {source}:{line_number}") from exc
    if not records:
        raise ValueError("Offline semantic dataset must contain at least one record.")
    observation_lengths = {len(record.scenario.observation) for record in records}
    if len(observation_lengths) != 1:
        raise ValueError(
            "Offline semantic dataset has inconsistent observation sizes."
        )
    return records


def label_scenarios(
    scenarios: Iterable[EngagementScenario], provider: TeacherProvider
) -> list[LabelledScenario]:
    """Call an explicit provider and convert its bounded output into labels."""
    created_at = datetime.now(timezone.utc).isoformat()
    records = []
    for scenario in scenarios:
        task, task_reason, local, coordination_reason = provider.label_semantics(
            scenario
        )
        label = SemanticPreferenceLabel(
            scenario_id=scenario.scenario_id,
            observation_version=scenario.observation_version,
            task_commitment=task,
            local_assertiveness=local,
            model=provider.name,
            created_at=created_at,
        )
        records.append(
            LabelledScenario(
                scenario,
                label,
                task_reason,
                coordination_reason,
            )
        )
    return records


def _scenario_id(
    observation: Sequence[float], agent_id: int, scenario_type: str
) -> str:
    encoded = ",".join(f"{value:.6f}" for value in observation).encode("ascii")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"{OBSERVATION_VERSION}-{scenario_type}-agv{agent_id}-{digest}"


def _json_object(
    content: str, expected_keys: frozenset[str] | None = None
) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content was empty.")
    candidate = _strip_json_code_fence(content.strip())
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        # Permit a short explanation around one JSON object, but never accept
        # arbitrary fields: the schema validators still enforce the contract.
        payload = _embedded_json_object(candidate, expected_keys)
        if payload is None:
            preview = candidate[:200].replace("\n", " ")
            raise ValueError(
                f"LLM response is not valid JSON (preview={preview!r})."
            ) from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    return payload


def _strip_json_code_fence(candidate: str) -> str:
    if not candidate.startswith("```"):
        return candidate
    lines = candidate.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _bounded_response_text(candidate: str) -> str | None:
    for expected_keys in (_SEMANTIC_KEYS, _ADJUSTMENT_KEYS):
        payload = _embedded_json_object(candidate, expected_keys)
        if payload is not None:
            return json.dumps(payload, ensure_ascii=False)
    return None


def _embedded_json_object(
    candidate: str, expected_keys: frozenset[str] | None = None
) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and (
            expected_keys is None or set(payload) == expected_keys
        ):
            return payload
    return None
