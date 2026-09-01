import json

import pytest

from llm_mappo.semantic_label_protocol import (
    FLASH_GO,
    FINGERPRINT_PAUSED,
    FormalLabelSession,
    SEMANTIC_PROMPT_VERSION,
    SemanticScenarioAttempt,
    build_semantic_prompt,
    build_blind_review_pack,
    decide_pilot_model,
    generate_semantic_attempts,
    _inject_label_only_stratum,
    validate_formal_dataset,
)
from llm_mappo.optimization_observation import ObservationSchema
from llm_mappo.phase2 import Phase2Warehouse


def _attempt(identifier="scenario-1", stratum="normal_transport"):
    return SemanticScenarioAttempt(
        scenario_id=identifier,
        content_hash=f"content-{identifier}",
        stratum=stratum,
        semantic_view={
            "semantic_view_version": "semantic-view-v3",
            "layout_hash": "not-for-prompt",
            "focal": {"battery_ratio": 0.5},
            "neighbors": [],
        },
        vector=[0.0] * 61,
    )


def _valid_response(fingerprint="fp-a"):
    return {
        "status": 200,
        "headers": {"x-request-id": "r-1"},
        "body": json.dumps({
            "id": "response-1",
            "model": "deepseek-v4-flash",
            "system_fingerprint": fingerprint,
            "created": 1,
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "task_persistence": 0.2,
                "task_persistence_reason": "A task is active.",
                "yielding_preference": 0.3,
                "yielding_preference_reason": "A peer is nearby.",
                "coordination_risk": 0.4,
                "coordination_risk_reason": "The area is constrained.",
            })}}],
        }),
    }


def test_prompt_uses_only_the_deidentified_semantic_view():
    prompt = build_semantic_prompt(_attempt().semantic_view)

    assert "not-for-prompt" not in prompt.user_text
    for forbidden in ("scenario_type", "RuleKD", "A*", "reward", "Student"):
        assert forbidden not in prompt.user_text
    assert "SEMANTIC_STATE=" in prompt.user_text
    assert "battery_ratio" in prompt.user_text


def test_v5_prompt_defines_the_three_previously_ambiguous_state_fields():
    prompt = build_semantic_prompt(_attempt().semantic_view)

    assert SEMANTIC_PROMPT_VERSION == "semantic-prompt-v5-state-contract"
    assert "priority_rank means higher priority" in prompt.user_text
    assert "adjacent_highway describes static map connectivity only" in prompt.user_text
    assert "not occupancy, blockage, or a planned path" in prompt.user_text
    assert "target_kind=charging is a temporary energy diversion" in prompt.user_text
    assert "original assigned transport task" in prompt.user_text
    assert "For task_persistence, consider whether an active transport task exists" in prompt.user_text
    assert "loaded versus empty status, relative task priority" in prompt.user_text
    assert "For coordination_risk, consider neighbor" in prompt.user_text
    assert "density, narrow or station bottlenecks" in prompt.user_text


def test_session_never_accepts_a_key_argument_or_persists_it(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "temporary-secret")
    session = FormalLabelSession(tmp_path, "deepseek-v4-flash", mode="pilot")
    record = session.consume_response(_attempt(), _valid_response())

    assert record["validity"] == 1
    written = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*") if path.is_file()
    )
    assert "temporary-secret" not in written
    with pytest.raises(TypeError):
        FormalLabelSession(tmp_path, "deepseek-v4-flash", mode="pilot", api_key="x")


def test_formal_session_atomically_pauses_on_fingerprint_change(tmp_path):
    session = FormalLabelSession(tmp_path, "deepseek-v4-flash", mode="formal")
    session.consume_response(_attempt("scenario-1"), _valid_response("fp-a"))
    with pytest.raises(RuntimeError, match=FINGERPRINT_PAUSED):
        session.consume_response(_attempt("scenario-2"), _valid_response("fp-b"))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == FINGERPRINT_PAUSED
    assert manifest["frozen_backend_tuple"][2] == "fp-a"


def test_session_serializes_a_real_generated_view_without_losing_the_first_record(tmp_path):
    generated = generate_semantic_attempts("pilot", per_stratum=1)[0]
    session = FormalLabelSession(tmp_path, "deepseek-v4-flash", mode="pilot")
    session.consume_response(generated, _valid_response())

    assert len((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_pilot_switches_to_pro_only_for_preregistered_flash_failure():
    review = {
        "records": 60,
        "valid_records": 60,
        "valid_by_stratum": {name: 12 for name in (
            "normal_transport", "priority_conflict", "narrow_corridor_yield",
            "low_battery_diversion", "station_exit_congestion",
        )},
        "substantive_errors": 0,
        "substantive_by_stratum": {},
        "critical_errors": 0,
        "anchor_disagreements": 0,
    }
    assert decide_pilot_model(review) == FLASH_GO
    review["critical_errors"] = 1
    assert decide_pilot_model(review) == "REGENERATE_FULL_PILOT_WITH_PRO"


def test_formal_gate_requires_complete_strata_single_backend_and_fixed_blind_pack():
    records = []
    for stratum_index, stratum in enumerate((
        "normal_transport", "priority_conflict", "narrow_corridor_yield",
        "low_battery_diversion", "station_exit_congestion",
    )):
        for index in range(160):
            records.append({
                "scenario_id": f"{stratum}-{index}",
                "content_hash": f"hash-{stratum}-{index}", "stratum": stratum,
                "validity": 1, "vector": [float(stratum_index + index)] * 61,
                "backend_tuple": ["deepseek-v4-flash", "deepseek-v4-flash", "fp-a"],
            })
    pack = build_blind_review_pack(records)
    receipt = validate_formal_dataset(records, review_verdicts={})

    assert len(pack) == 100
    assert all("stratum" not in item and "scenario_id" not in item for item in pack)
    assert receipt["gate"] == "GO"


def test_scenario_generator_is_deterministic_and_never_uses_a_planner():
    first = generate_semantic_attempts("pilot")
    second = generate_semantic_attempts("pilot")

    assert len(first) == len(second) == 60
    assert [item.scenario_id for item in first] == [item.scenario_id for item in second]
    assert len({item.content_hash for item in first}) == 60
    assert all(len(item.vector) == 61 for item in first)


def test_controlled_scenarios_preserve_their_frozen_semantic_invariants():
    attempts = generate_semantic_attempts("pilot")
    grouped = {}
    for attempt in attempts:
        grouped.setdefault(attempt.stratum, []).append(attempt)

    priority = grouped["priority_conflict"]
    assert {item.semantic_view["focal"]["priority_rank"] for item in priority} == {
        0.0, 1.0 / 25.0,
    }
    assert all(any(
        neighbor["mask"] and neighbor["priority_rank"] != item.semantic_view["focal"]["priority_rank"]
        for neighbor in item.semantic_view["neighbors"]
    ) for item in priority)
    assert all(
        not item.semantic_view["focal"]["loaded"]
        and any(neighbor["mask"] and neighbor["loaded"] for neighbor in item.semantic_view["neighbors"])
        for item in grouped["narrow_corridor_yield"]
    )
    assert all(
        sum(bool(value) for value in item.semantic_view["focal"]["adjacent_highway"].values()) == 2
        for item in grouped["narrow_corridor_yield"]
    )
    assert all(
        item.semantic_view["focal"]["battery_ratio"] == 0.15
        and item.semantic_view["focal"]["loaded"]
        and item.semantic_view["focal"]["target_kind"] == "charging"
        for item in grouped["low_battery_diversion"]
    )
    assert all(
        not item.semantic_view["focal"]["at_charging_station"]
        and any(neighbor["mask"] and neighbor["at_charging_station"] for neighbor in item.semantic_view["neighbors"])
        for item in grouped["station_exit_congestion"]
    )


def test_generator_enforces_normal_distance_and_station_first_exit_contract():
    environment = Phase2Warehouse(
        n_agents=5,
        max_steps=1000,
        env_id="llm-mappo-medium-3ag-v1",
        charge_threshold=0.30,
        charge_release_threshold=0.80,
        battery_cost_scale=1.10,
        deadlock_steps=180,
        batch_interval=40,
        batch_size_range=(4, 8),
        request_queue_size=8,
        task_completion_target=50,
        initial_priority_label="A",
        observation_schema=ObservationSchema.DIRECT_GOAL_V1,
    )
    environment.reset(seed=41000000)
    _inject_label_only_stratum(
        environment, "normal_transport", derived_seed=41000000, within_seed_index=0,
    )
    warehouse = environment.env
    focal = warehouse.agents[0]
    assert all(
        abs(focal.x - peer.x) + abs(focal.y - peer.y) > 4
        for peer in warehouse.agents[1:]
    )

    environment.reset(seed=41400000)
    _inject_label_only_stratum(
        environment, "station_exit_congestion", derived_seed=41400000,
        within_seed_index=0,
    )
    focal, station_agent = environment.env.agents[:2]
    stations = set(environment.env.charging_stations) | set(environment.env.picking_stations)
    exits = sorted(
        (
            (x, y)
            for x, y in (
                (station_agent.x - 1, station_agent.y),
                (station_agent.x + 1, station_agent.y),
                (station_agent.x, station_agent.y - 1),
                (station_agent.x, station_agent.y + 1),
            )
            if 0 <= x < environment.env.grid_size[1]
            and 0 <= y < environment.env.grid_size[0]
            and environment.env._is_highway(x, y)
            and (x, y) not in stations
        ),
        key=lambda point: (point[1], point[0]),
    )
    assert (station_agent.x, station_agent.y) in environment.env.charging_stations
    assert (focal.x, focal.y) == exits[0]
