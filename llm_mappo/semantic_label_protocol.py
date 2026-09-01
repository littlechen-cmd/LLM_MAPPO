"""Owner-run, auditable semantic-view-v3 label collection primitives."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from llm_mappo.semantic_v3 import SemanticRecordV3
from llm_mappo.semantic_v3 import SemanticViewV3
from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.phase2 import Phase2Warehouse

FLASH_GO = "FLASH_GO"
REGENERATE_FULL_PILOT_WITH_PRO = "REGENERATE_FULL_PILOT_WITH_PRO"
DATASET_NO_GO = "DATASET_NO_GO"
FINGERPRINT_PAUSED = "FINGERPRINT_PAUSED"
STRATA = (
    "normal_transport", "priority_conflict", "narrow_corridor_yield",
    "low_battery_diversion", "station_exit_congestion",
)
SYSTEM_PROMPT = (
    "You are a JSON-only warehouse semantic teacher. Evaluate only the supplied "
    "semantic state. Never output actions, paths, assignments, right-of-way "
    "rulings, station controls, or changes to task labels."
)
USER_PREFIX = """Return one JSON object with exactly these six keys: task_persistence,
task_persistence_reason, yielding_preference, yielding_preference_reason,
coordination_risk, coordination_risk_reason. Each score must be a finite number
in [0,1]. Each reason must be a non-empty string of at most 1000 characters.

task_persistence is how reasonable it is to keep the current transport task.
yielding_preference is a tendency to voluntarily delay or cede local passage.
coordination_risk is the risk of local conflict, congestion, deadlock, or failure.
Use the facts jointly and interpolate continuously: 0.00=no basis, 0.25=weak or
minor, 0.50=balanced or material, 0.75=strong or high, 1.00=overwhelming or
near-certain. The scores are not complements or aliases. Do not invent facts,
IDs, actions, paths, assignments, priorities, right-of-way rulings, station
controls, or task-label changes. Do not emit markdown or text outside the JSON.

Expected JSON shape:
{"task_persistence":0.0,"task_persistence_reason":"...","yielding_preference":0.0,"yielding_preference_reason":"...","coordination_risk":0.0,"coordination_risk_reason":"..."}

SEMANTIC_STATE="""


@dataclass(frozen=True)
class SemanticScenarioAttempt:
    """A pre-generated deterministic attempt; stratum remains provenance only."""

    scenario_id: str
    content_hash: str
    stratum: str
    semantic_view: Mapping[str, Any]
    vector: list[float]

    def __post_init__(self) -> None:
        if self.stratum not in STRATA:
            raise ValueError("Unsupported semantic scenario stratum.")
        if len(self.vector) != 61:
            raise ValueError("semantic-view-v3 attempt requires a 61D vector.")


@dataclass(frozen=True)
class SemanticPrompt:
    system_text: str
    user_text: str
    system_sha256: str
    user_sha256: str
    semantic_view_sha256: str


def build_semantic_prompt(semantic_view: Mapping[str, Any]) -> SemanticPrompt:
    """Construct v4 prompt, explicitly excluding layout identifiers."""
    expected = {"semantic_view_version", "layout_hash", "focal", "neighbors"}
    if set(semantic_view) != expected:
        raise ValueError("semantic-view-v3 JSON schema is incompatible.")
    state = _deidentified_view(semantic_view)
    if state["semantic_view_version"] != "semantic-view-v3":
        raise ValueError("Only semantic-view-v3 can enter the label prompt.")
    canonical = _canonical_json(state)
    user_text = USER_PREFIX + canonical
    return SemanticPrompt(
        SYSTEM_PROMPT, user_text, _digest(SYSTEM_PROMPT), _digest(user_text),
        _digest(canonical),
    )


def require_deepseek_api_key() -> str:
    """Read an owner-injected secret from the process environment only."""
    value = os.environ.get("DEEPSEEK_API_KEY")
    if not value or not value.strip():
        raise RuntimeError("DEEPSEEK_API_KEY is required in the process environment.")
    return value


def decide_pilot_model(review: Mapping[str, Any]) -> str:
    """Apply only the frozen Flash systematic-failure rule."""
    try:
        records = int(review["records"])
        valid = int(review["valid_records"])
        valid_by = review["valid_by_stratum"]
        errors = int(review["substantive_errors"])
        errors_by = review["substantive_by_stratum"]
        critical = int(review["critical_errors"])
        disagreement = int(review["anchor_disagreements"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Pilot review is missing frozen decision fields.") from error
    if records != 60 or set(valid_by) != set(STRATA):
        return DATASET_NO_GO
    failed = (
        valid < 57 or any(int(valid_by[name]) < 11 for name in STRATA)
        or errors > 6 or any(int(errors_by.get(name, 0)) > 2 for name in STRATA)
        or critical != 0 or disagreement > 12
    )
    return REGENERATE_FULL_PILOT_WITH_PRO if failed else FLASH_GO


def build_blind_review_pack(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select exactly 20 records per stratum without exposing provenance IDs."""
    selected = []
    for stratum in STRATA:
        candidates = [record for record in records if record.get("stratum") == stratum]
        ranked = sorted(
            candidates,
            key=lambda record: _digest(
                "20260820|" + str(record.get("content_hash", "")) + "|"
                + str(record.get("scenario_id", ""))
            ),
        )
        if len(ranked) < 20:
            raise ValueError("Blind review requires 20 records in every stratum.")
        for ordinal, record in enumerate(ranked[:20]):
            selected.append({
                "blind_id": f"review-{stratum}-{ordinal:02d}",
                "semantic_view_version": record.get("semantic_view_version", "semantic-view-v3"),
                "vector": record.get("vector"),
                "scores": record.get("scores"),
                "reasons": record.get("reasons"),
                "content_hash": record.get("content_hash"),
            })
    return selected


def build_pilot_review_pack(records: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a blinded 60-record pack plus an owner-only strata/key mapping."""
    if len(records) != 60 or any(sum(item.get("stratum") == name for item in records) != 12
                              for name in STRATA):
        raise ValueError("Pilot review requires the full frozen 60-record matrix.")
    pack, key = [], []
    for ordinal, record in enumerate(records):
        blind_id = f"pilot-{ordinal:03d}"
        pack.append({
            "blind_id": blind_id, "semantic_view": record.get("semantic_view"),
            "scores": record.get("scores"), "reasons": record.get("reasons"),
            "content_hash": record.get("content_hash"),
        })
        key.append({"blind_id": blind_id, "stratum": record.get("stratum"),
                    "scenario_id": record.get("scenario_id")})
    return pack, key


def validate_formal_dataset(
    records: list[Mapping[str, Any]], *, review_verdicts: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a dataset-level Go/No-Go receipt; never repair individual records."""
    reasons = []
    if len(records) != 800:
        reasons.append("record_count")
    by_stratum = {name: [item for item in records if item.get("stratum") == name]
                  for name in STRATA}
    if any(len(items) != 160 for items in by_stratum.values()):
        reasons.append("stratum_count")
    ids = [item.get("scenario_id") for item in records]
    hashes = [item.get("content_hash") for item in records]
    if len(set(ids)) != len(records) or len(set(hashes)) != len(records):
        reasons.append("identity_uniqueness")
    valid = [item for item in records if item.get("validity") == 1]
    if len(valid) < 784:
        reasons.append("overall_validity")
    if any(sum(item.get("validity") == 1 for item in items) < 152
           for items in by_stratum.values()):
        reasons.append("stratum_validity")
    backends = {tuple(item["backend_tuple"]) for item in valid
                if isinstance(item.get("backend_tuple"), list)}
    if len(backends) != 1:
        reasons.append("backend_fingerprint")
    critical = int(review_verdicts.get("critical_errors", 0))
    substantive = int(review_verdicts.get("substantive_errors", 0))
    substantive_by = review_verdicts.get("substantive_by_stratum", {})
    if critical != 0:
        reasons.append("review_critical")
    if substantive > 5 or any(int(substantive_by.get(name, 0)) > 2 for name in STRATA):
        reasons.append("review_substantive")
    return {
        "schema": "semantic-formal-gate-v1", "gate": "GO" if not reasons else DATASET_NO_GO,
        "reasons": reasons, "record_count": len(records), "valid_count": len(valid),
        "content_sha256": _digest("\n".join(sorted(str(item) for item in records))),
    }


def generate_semantic_attempts(mode: str, *, per_stratum: int | None = None) -> list[SemanticScenarioAttempt]:
    """Build deterministic environment snapshots without a planner or LLM call."""
    if mode not in {"pilot", "formal"}:
        raise ValueError("Scenario mode must be pilot or formal.")
    quota = per_stratum if per_stratum is not None else (12 if mode == "pilot" else 160)
    if quota < 1:
        raise ValueError("Scenario quota must be positive.")
    base_seeds = (410, 411, 412) if mode == "pilot" else tuple(range(500, 510))
    attempts = []
    for rank, stratum in enumerate(STRATA):
        for index in range(quota):
            per_seed = 4 if mode == "pilot" else 16
            base_seed = base_seeds[(index // per_seed) % len(base_seeds)]
            derived_seed = base_seed * 100000 + rank * 1000 + (index % per_seed)
            environment = Phase2Warehouse(
                n_agents=5, max_steps=1000, env_id="llm-mappo-medium-3ag-v1",
                charge_threshold=0.30, charge_release_threshold=0.80,
                battery_cost_scale=1.10, deadlock_steps=180,
                batch_interval=40, batch_size_range=(4, 8), request_queue_size=8,
                task_completion_target=50, initial_priority_label="A",
                observation_schema=ObservationSchema.DIRECT_GOAL_V1,
            )
            environment.reset(seed=derived_seed)
            if stratum != "normal_transport":
                _inject_label_only_stratum(environment, stratum, index)
            view = _environment_semantic_view(environment, 0)
            snapshot = _canonical_json(view.json_view)
            layout_hash = environment.env.shadow_layout_hash()
            scenario_id = _digest(
                "semantic-scenario-v3|" + layout_hash + "|" + str(derived_seed) + "|" + _digest(snapshot)
            )
            content_hash = _digest(snapshot + "|" + json.dumps(view.vector.tolist()))
            attempts.append(SemanticScenarioAttempt(
                scenario_id=scenario_id, content_hash=content_hash, stratum=stratum,
                semantic_view=view.json_view, vector=view.vector.tolist(),
            ))
    identities = [item.scenario_id for item in attempts]
    if len(set(identities)) != len(identities):
        raise RuntimeError("Deterministic label generator produced duplicate scenario IDs.")
    return attempts


def _inject_label_only_stratum(
    environment: Phase2Warehouse, stratum: str, candidate_index: int
) -> None:
    """Construct a seed-indexed physical candidate; never step a policy or A*."""
    from llm_mappo.phase4 import _inject_controlled_scenario

    _inject_controlled_scenario(environment, stratum)
    # The historical helper deterministically chose its first legal geometry.
    # A frozen candidate index must also alter observable physical state; otherwise
    # distinct scenario IDs can silently carry identical 61D semantic content.
    peer = environment.env.agents[1]
    peer.battery = 0.20 + 0.004 * candidate_index


def _environment_semantic_view(environment: Phase2Warehouse, focal_index: int) -> SemanticViewV3:
    warehouse = environment.env
    agent = warehouse.agents[focal_index]
    _, target_kind = environment._target_for_agent(agent.id)
    focal = _semantic_agent_state(environment, agent, target_kind)
    peers = [
        _semantic_agent_state(environment, peer, environment._target_for_agent(peer.id)[1])
        for peer in warehouse.agents if peer.id != agent.id
    ]
    return SemanticViewV3.from_state(
        warehouse.shadow_layout_hash(), warehouse.grid_size[1], warehouse.grid_size[0],
        focal, peers,
    )


def _semantic_agent_state(environment, agent, target_kind: str) -> dict[str, Any]:
    warehouse = environment.env
    direction = agent.dir.name.lower()
    deltas = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    forward = deltas[direction]
    right = deltas[{"up": "right", "right": "down", "down": "left", "left": "up"}[direction]]
    def highway(delta):
        x, y = agent.x + delta[0], agent.y + delta[1]
        return 0 <= x < warehouse.grid_size[1] and 0 <= y < warehouse.grid_size[0] and warehouse._is_highway(x, y)
    return {
        "position": (agent.x, agent.y), "orientation": direction,
        "battery_ratio": float(agent.battery), "loaded": agent.carrying_shelf is not None,
        "dead": bool(agent.dead), "priority_present": warehouse.task_queue.task_for_agent(agent.id) is not None,
        "priority_rank": 0.0, "target_kind": target_kind,
        "on_highway": warehouse._is_highway(agent.x, agent.y),
        "at_charging_station": (agent.x, agent.y) in warehouse.charging_stations,
        "at_picking_station": (agent.x, agent.y) in warehouse.picking_stations,
        "adjacent_highway": {"forward": highway(forward), "right": highway(right),
                             "backward": highway((-forward[0], -forward[1])),
                             "left": highway((-right[0], -right[1]))},
    }


class FormalLabelSession:
    """Append-only evidence writer with fail-closed formal fingerprint handling."""

    def __init__(self, output_directory: str | Path, request_model: str, *, mode: str):
        if mode not in {"pilot", "formal"}:
            raise ValueError("Label session mode must be pilot or formal.")
        if request_model not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
            raise ValueError("Label session model is not frozen.")
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.request_model, self.mode = request_model, mode
        self._manifest_path = self.output_directory / "manifest.json"
        self._records_path = self.output_directory / "records.jsonl"
        self._backend_tuple = None
        self._status = "running"
        self._write_manifest()

    def consume_response(self, attempt: SemanticScenarioAttempt,
                         response: Mapping[str, Any]) -> dict[str, Any]:
        if self._status != "running":
            raise RuntimeError(f"Label session is not runnable: {self._status}")
        prompt = build_semantic_prompt(attempt.semantic_view)
        record = self._parse_response(attempt, prompt, response)
        backend = record.get("backend_tuple")
        if self.mode == "formal" and record["validity"] == 1:
            if self._backend_tuple is None:
                self._backend_tuple = tuple(backend)
            elif tuple(backend) != self._backend_tuple:
                self._status = FINGERPRINT_PAUSED
                self._write_manifest()
                raise RuntimeError(FINGERPRINT_PAUSED)
        _append_json_line(self._records_path, record)
        self._write_manifest()
        return record

    def _parse_response(self, attempt, prompt, response):
        status = int(response.get("status", 0))
        body = response.get("body", "")
        raw_body = body if isinstance(body, str) else ""
        result = {
            "scenario_id": attempt.scenario_id, "content_hash": attempt.content_hash,
            "stratum": attempt.stratum, "semantic_view_version": "semantic-view-v3",
            "semantic_view": _deidentified_view(attempt.semantic_view),
            "vector": attempt.vector,
            "prompt": {"version": "semantic-prompt-v4-directional-rubric",
                       "system_sha256": prompt.system_sha256,
                       "user_sha256": prompt.user_sha256,
                       "semantic_view_sha256": prompt.semantic_view_sha256},
            "http": {"status": status, "headers": _redact_headers(response.get("headers", {}))},
            "raw_response": raw_body, "timestamp_unix": time.time(), "validity": 0,
            "failure_reason": "unparsed_response",
        }
        if status != 200:
            result["failure_reason"] = "retryable_http" if status == 429 or status >= 500 else "nonretryable_http"
            return result
        try:
            parsed = json.loads(raw_body)
            choice = parsed["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") != "stop" or not isinstance(content, str):
                raise ValueError("Provider completion is invalid.")
            semantic = SemanticRecordV3.parse_response(content)
            model, fingerprint = parsed["model"], parsed["system_fingerprint"]
            if not isinstance(model, str) or not isinstance(fingerprint, str):
                raise ValueError("Provider identity is absent.")
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            result["failure_reason"] = "schema_or_content_invalid"
            result["parse_error_type"] = type(error).__name__
            return result
        result.update({
            "validity": 1, "scores": semantic.scores.tolist(),
            "reasons": list(semantic.reasons),
            "backend_tuple": [self.request_model, model, fingerprint],
            "response": {"id": parsed.get("id"), "model": model,
                         "system_fingerprint": fingerprint,
                         "created": parsed.get("created"),
                         "finish_reason": choice.get("finish_reason"),
                         "usage": parsed.get("usage")},
            "failure_reason": None,
        })
        return result

    def _write_manifest(self) -> None:
        _atomic_json(self._manifest_path, {
            "schema": "semantic-label-session-v1", "mode": self.mode,
            "request_model": self.request_model, "status": self._status,
            "frozen_backend_tuple": list(self._backend_tuple) if self._backend_tuple else None,
            "records_path": self._records_path.name,
        })


def _redact_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in headers.items()
            if str(key).lower() != "authorization"}


def _append_json_line(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(
            payload, ensure_ascii=False, sort_keys=True,
            default=lambda item: item.item(),
        ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      default=lambda item: item.item())


def _deidentified_view(semantic_view: Mapping[str, Any]) -> dict[str, Any]:
    return {key: semantic_view[key] for key in (
        "semantic_view_version", "focal", "neighbors"
    )}
