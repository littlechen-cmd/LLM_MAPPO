import csv
import json
from io import BytesIO
from urllib import error

import numpy as np
import pytest
import torch

from eval.audit_phase4_semantics import _automatic_issues, audit_dataset
from llm_mappo.llm_teacher import (
    DeepSeekTeacher,
    EngagementScenario,
    LabelledScenario,
    MockTeacher,
    load_labelled_scenarios,
    parse_priority_adjustments,
    parse_semantic_response,
)
from llm_mappo.mappo import DualHeadMAPPOPolicy, PPOHyperparameters
from llm_mappo.phase2 import Phase2Warehouse
from llm_mappo.phase3_training import Phase3TrainingConfig, train_phase3
from llm_mappo.phase4 import (
    OfflineSemanticTeacher,
    apply_priority_instruction,
    collect_offline_labels,
    collect_stratified_offline_labels,
    repair_offline_labels,
)
from llm_mappo.types import SemanticPreferenceLabel
from visualize import _is_phase3_checkpoint, _load_controller


def test_phase4_rejects_unbounded_llm_json_schemas():
    response = (
        '{"task_commitment":0.6,"task_reason":"continue",'
        '"local_assertiveness":0.2,"coordination_reason":"yield"}'
    )
    assert parse_semantic_response(response) == (0.6, "continue", 0.2, "yield")
    with pytest.raises(ValueError, match="four frozen keys"):
        parse_semantic_response(response[:-1] + ',"action":1}')
    with pytest.raises(ValueError, match="only an adjustments"):
        parse_priority_adjustments('{"actions": [1]}')
    assert parse_semantic_response("```json\n" + response + "\n```") == (
        0.6,
        "continue",
        0.2,
        "yield",
    )
    assert parse_semantic_response(
        'Context: {"scenario_type": "normal_transport"}. Final: '
        + response
    ) == (0.6, "continue", 0.2, "yield")
    with pytest.raises(ValueError, match="content was empty"):
        parse_semantic_response("")


def test_phase4_audit_detects_state_and_score_reason_contradictions():
    scenario = EngagementScenario(
        scenario_id="bad-reason",
        observation_version="phase4-semantic-v2",
        scenario_type="priority_conflict",
        observation=(0.0,),
        agent_id=1,
        battery=1.0,
        loaded=False,
        priority_label="A",
        target_kind="task",
        nearby_agents=(
            {
                "agent_id": 2,
                "distance": 1,
                "loaded": False,
                "battery": 1.0,
                "priority_label": "B",
                "target_kind": "task",
                "at_charging_station": False,
            },
        ),
    )
    label = SemanticPreferenceLabel(
        "bad-reason", "phase4-semantic-v2", 1.0, 0.7, "test", "now"
    )
    record = LabelledScenario(
        scenario,
        label,
        "continue task",
        "Nearby agent 2 is loaded, so focal agent should yield.",
    )
    assert _automatic_issues(record) == [
        "rationale_peer_load_state_mismatch",
        "score_reason_contradiction",
    ]
    load_only = LabelledScenario(
        scenario,
        label,
        "continue task",
        "Must yield to loaded B AGV due to the load rule.",
    )
    assert _automatic_issues(load_only) == ["rationale_peer_load_state_mismatch"]
    hypothetical = LabelledScenario(
        scenario,
        label,
        "continue task",
        "Must yield to loaded B if the load rule applies; otherwise assert priority.",
    )
    assert _automatic_issues(hypothetical) == []


def test_phase4_priority_instruction_keeps_assignment_outside_the_teacher():
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=20,
        priority_schedule=("A", "B", "C"),
        include_priority_features=True,
    )
    try:
        env.reset(seed=3)
        result = apply_priority_instruction(env, MockTeacher(), "B1 to A")

        assert result["updated_labels"] == ["A1", "B1"]
        assert len(result["adjustments"]) == 2
        assert env.env.task_queue.task_for_agent(1).assigned_agent_id == 1
        assert env.env.task_queue.task_for_agent(2).assigned_agent_id == 2
        assert env.env.task_queue.task_for_agent(3).assigned_agent_id == 3
    finally:
        env.close()


def test_phase4_cached_teacher_has_no_provider_at_training_lookup(tmp_path):
    path = tmp_path / "labels.jsonl"
    env = Phase2Warehouse(
        n_agents=3,
        max_steps=20,
        priority_schedule=("A", "B", "C"),
        include_priority_features=True,
    )
    try:
        result = collect_offline_labels(
            env, MockTeacher(), path, seeds=[3], scenarios_per_seed=3
        )
    finally:
        env.close()

    teacher = OfflineSemanticTeacher.from_jsonl(path)
    assert result["records"] == teacher.size == 3
    assert teacher.index.n == 3
    targets = teacher.targets(teacher.observations, neighbours=1)
    assert np.allclose(targets, teacher.preferences)
    assert targets.shape == (3, 2)
    assert teacher.model_names == ("mock-semantic-v2",)


def test_phase4_config_requires_cached_dataset_and_path_teacher():
    config = Phase3TrainingConfig.from_yaml("configs/phase4_llm_distillation.yaml")
    assert config.phase == "4"
    assert config.n_agents == 5
    assert config.batch_size_range == (4, 8)
    assert config.task_completion_target == 50
    assert config.parallel_envs == 12
    assert config.device == "auto"
    assert config.cuda_allow_tf32 is True
    assert config.ppo.reservation_kl_coefficient == 0.05
    assert config.offline_semantic_dataset.endswith(
        "deepseek_medium_5ag_400_v2_repaired_r2.jsonl"
    )

    missing = Phase3TrainingConfig(phase="4", n_agents=5, offline_semantic_dataset=None)
    with pytest.raises(ValueError, match="offline_semantic_dataset"):
        train_phase3(missing)


def test_g2_charging_retrains_are_matched_except_for_energy_pressure():
    control = Phase3TrainingConfig.from_yaml(
        "configs/g2_charging_retrain_control.yaml"
    )
    candidate = Phase3TrainingConfig.from_yaml(
        "configs/g2_charging_retrain_candidate.yaml"
    )
    high_consumption = Phase3TrainingConfig.from_yaml(
        "configs/g2_charging_retrain_candidate_scale120_threshold020.yaml"
    )

    assert control.episodes == candidate.episodes == high_consumption.episodes == 200
    assert (control.battery_cost_scale, control.charge_threshold) == (1.0, 0.2)
    assert (candidate.battery_cost_scale, candidate.charge_threshold) == (1.1, 0.3)
    assert (high_consumption.battery_cost_scale, high_consumption.charge_threshold) == (
        1.2,
        0.2,
    )
    assert {
        control.charge_release_threshold,
        candidate.charge_release_threshold,
        high_consumption.charge_release_threshold,
    } == {0.8}

    allowed_differences = {
        "battery_cost_scale",
        "charge_threshold",
        "output_dir",
    }
    control_fixed = {
        key: value
        for key, value in vars(control).items()
        if key not in allowed_differences
    }
    candidate_fixed = {
        key: value
        for key, value in vars(candidate).items()
        if key not in allowed_differences
    }
    high_consumption_fixed = {
        key: value
        for key, value in vars(high_consumption).items()
        if key not in allowed_differences
    }
    assert control_fixed == candidate_fixed == high_consumption_fixed


def test_phase4_training_aggregates_two_environment_rollouts(tmp_path):
    dataset = tmp_path / "labels.jsonl"
    label_env = Phase2Warehouse(
        n_agents=5, max_steps=8, include_priority_features=True
    )
    try:
        collect_offline_labels(
            label_env,
            MockTeacher(),
            dataset,
            seeds=[3],
            scenarios_per_seed=2,
        )
    finally:
        label_env.close()
    config = Phase3TrainingConfig(
        phase="4",
        seed=3,
        n_agents=5,
        max_steps=2,
        episodes=2,
        parallel_envs=2,
        rollout_steps=2,
        checkpoint_interval=10,
        metrics_write_interval=1,
        output_dir=str(tmp_path / "run"),
        offline_semantic_dataset=str(dataset),
        ppo=PPOHyperparameters(
            update_epochs=1,
            minibatch_steps=2,
            engagement_coefficient=0.1,
            reservation_kl_coefficient=0.05,
        ),
    )
    summary = train_phase3(config)
    assert summary["episodes"] == 2
    assert summary["parallel_envs"] == 2
    assert summary["steps"] == 4
    assert summary["training_elapsed_seconds"] > 0
    assert summary["env_steps_per_second"] > 0
    assert summary["accelerator"]["resolved"] == "cpu"
    runtime = json.loads(
        (tmp_path / "run" / "seed_003" / "runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert runtime["requested"] == runtime["resolved"] == "cpu"
    updates = tmp_path / "run" / "seed_003" / "updates.csv"
    with updates.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["steps"]) for row in rows] == [2, 4]


def test_phase4_label_parser_rejects_task_assignment_fields():
    response = json.dumps(
        {
            "adjustments": [
                {
                    "task": "B1",
                    "new_label": "A1",
                    "reason": "expedite",
                    "agent_id": 3,
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="requires task"):
        parse_priority_adjustments(response)


def test_phase4_checkpoint_uses_the_dual_head_visualization_loader():
    assert _is_phase3_checkpoint({"phase": "4", "model_state": {}})


def test_phase4_visualization_controller_loads_dual_semantic_checkpoint(tmp_path):
    source = DualHeadMAPPOPolicy(615, 5, semantic_dim=2)
    checkpoint = tmp_path / "phase4.pt"
    torch.save(
        {
            "model_state": source.state_dict(),
            "config": {
                "phase": "4",
                "n_agents": 5,
                "max_steps": 2,
                "env_id": "llm-mappo-medium-3ag-v1",
            },
            "actor_observation_dim": 615,
            "episodes": 1,
            "steps": 1,
            "phase": "4",
        },
        checkpoint,
    )

    settings, choose_actions = _load_controller("policy", checkpoint, "unused.yaml")
    env = Phase2Warehouse(
        n_agents=settings.n_agents,
        max_steps=settings.max_steps,
        env_id=settings.env_id,
        include_priority_features=settings.include_priority_features,
    )
    try:
        env.reset(seed=0)
        actions = choose_actions(env, env.action_masks())
    finally:
        env.close()

    assert settings.n_agents == 5
    assert settings.include_priority_features is True
    assert actions.shape == (5,)


def test_phase4_stratified_collection_preserves_each_controlled_scenario(tmp_path):
    env = Phase2Warehouse(
        n_agents=5,
        max_steps=40,
        batch_interval=40,
        batch_size_range=(4, 8),
        initial_priority_label="A",
        request_queue_size=8,
        task_completion_target=50,
        include_priority_features=True,
    )
    quotas = {
        "normal_transport": 5,
        "priority_conflict": 1,
        "narrow_corridor_yield": 1,
        "low_battery_diversion": 1,
        "station_exit_congestion": 1,
    }
    try:
        result = collect_stratified_offline_labels(
            env, MockTeacher(), tmp_path / "stratified.jsonl", [7], quotas
        )
    finally:
        env.close()

    assert result["records"] == 9
    assert result["scenario_counts"] == quotas
    teacher = OfflineSemanticTeacher.from_jsonl(tmp_path / "stratified.jsonl")
    assert teacher.observation_dim == 615
    assert teacher.preferences.shape == (9, 2)
    summary = audit_dataset(
        tmp_path / "stratified.jsonl",
        tmp_path / "review.csv",
        tmp_path / "audit.json",
        sample_rate=1.0,
        expected_total=9,
    )
    assert summary["sample_records"] == 9
    assert summary["automatic_issue_count"] == 0
    records = load_labelled_scenarios(tmp_path / "stratified.jsonl")
    controlled = [
        record for record in records
        if record.scenario.scenario_type != "normal_transport"
    ]
    for record in controlled:
        close_peers = [
            peer for peer in record.scenario.nearby_agents if peer["distance"] <= 1
        ]
        assert [peer["agent_id"] for peer in close_peers] == [2]
        if record.scenario.scenario_type != "station_exit_congestion":
            assert close_peers[0]["at_charging_station"] is False
    station = next(
        record for record in controlled
        if record.scenario.scenario_type == "station_exit_congestion"
    )
    assert station.scenario.nearby_agents[0]["at_charging_station"] is True


def test_phase4_policy_exposes_two_detached_semantic_preferences():
    policy = DualHeadMAPPOPolicy(615, 5, semantic_dim=2)
    observations = np.zeros((5, 615), dtype=np.float32)
    masks = np.ones((5, 5), dtype=bool)
    actions, _, _, semantics = policy.act(observations, masks)
    assert actions.shape == (5,)
    assert semantics.shape == (5, 2)
    assert policy.actor.motion_head.in_features == 66


def test_phase4_targeted_repair_is_resumable_and_preserves_source(tmp_path):
    env = Phase2Warehouse(
        n_agents=5,
        max_steps=40,
        batch_interval=40,
        batch_size_range=(4, 8),
        initial_priority_label="A",
        request_queue_size=8,
        task_completion_target=50,
        include_priority_features=True,
    )
    source = tmp_path / "source.jsonl"
    try:
        collect_stratified_offline_labels(
            env,
            MockTeacher(),
            source,
            [7],
            {
                "normal_transport": 1,
                "priority_conflict": 1,
                "narrow_corridor_yield": 1,
                "low_battery_diversion": 1,
                "station_exit_congestion": 1,
            },
        )
    finally:
        env.close()
    original_bytes = source.read_bytes()
    original = load_labelled_scenarios(source)
    target_ids = [record.scenario.scenario_id for record in original[:2]]
    partial = tmp_path / "repair.partial.jsonl"
    output = tmp_path / "repaired.jsonl"

    class InterruptingTeacher(MockTeacher):
        name = "repair-test"

        def __init__(self):
            self.calls = 0

        def label_semantics(self, scenario):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated repair interruption")
            return super().label_semantics(scenario)

    with pytest.raises(RuntimeError, match="repair interruption"):
        repair_offline_labels(
            source,
            output,
            InterruptingTeacher(),
            target_ids,
            checkpoint_path=partial,
        )
    assert len(load_labelled_scenarios(partial)) == 1

    class CountingTeacher(MockTeacher):
        name = "repair-test"

        def __init__(self):
            self.calls = 0

        def label_semantics(self, scenario):
            self.calls += 1
            return super().label_semantics(scenario)

    resumed = CountingTeacher()
    result = repair_offline_labels(
        source,
        output,
        resumed,
        target_ids,
        checkpoint_path=partial,
    )
    repaired = load_labelled_scenarios(output)
    assert result["records"] == len(original)
    assert result["relabelled_records"] == 2
    assert resumed.calls == 1
    assert source.read_bytes() == original_bytes
    assert not partial.exists()
    assert [item.scenario for item in repaired] == [item.scenario for item in original]
    assert [item.label.model for item in repaired[:2]] == ["repair-test"] * 2
    assert [item.label.model for item in repaired[2:]] == ["mock-semantic-v2"] * 3
    with pytest.raises(ValueError, match="must not overwrite"):
        repair_offline_labels(source, source, MockTeacher(), target_ids)


def test_phase4_initial_a_and_b_batches_respect_the_configured_minimum():
    env = Phase2Warehouse(
        n_agents=5,
        max_steps=100,
        batch_interval=40,
        batch_size_range=(4, 8),
        initial_priority_label="A",
        request_queue_size=8,
        task_completion_target=50,
        include_priority_features=True,
    )
    try:
        env.reset(seed=11)
        labels = [task.label for task in env.env.task_queue.tasks]
        assert sum(label.startswith("A") for label in labels) >= 4
        assert sum(label.startswith("B") for label in labels) >= 4
    finally:
        env.close()


def test_deepseek_teacher_retries_read_timeout(monkeypatch):
    scenario = EngagementScenario(
        scenario_id="s1",
        observation_version="v1",
        scenario_type="normal_transport",
        observation=(0.0,),
        agent_id=1,
        battery=1.0,
        loaded=False,
        priority_label=None,
        target_kind="idle",
        nearby_agents=(),
    )
    responses = [TimeoutError("read timed out"), TimeoutError("read timed out")]
    semantic = json.dumps(
        {
            "task_commitment": 0.6,
            "task_reason": "continue",
            "local_assertiveness": 0.7,
            "coordination_reason": "clear",
        }
    )
    response_body = json.dumps(
        {"choices": [{"message": {"content": semantic}}]}
    ).encode("utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return response_body

    def fake_urlopen(*args, **kwargs):
        if responses:
            value = responses.pop(0)
            if isinstance(value, Exception):
                raise value
        return Response()

    monkeypatch.setattr("llm_mappo.llm_teacher.request.urlopen", fake_urlopen)
    monkeypatch.setattr("llm_mappo.llm_teacher.time.sleep", lambda _: None)
    teacher = DeepSeekTeacher(api_key="test", timeout_seconds=1, max_attempts=3)
    assert teacher.label_semantics(scenario) == (0.6, "continue", 0.7, "clear")
    assert not responses


def test_deepseek_teacher_uses_reasoning_content_when_content_is_empty(monkeypatch):
    semantic = json.dumps(
        {
            "task_commitment": 0.6,
            "task_reason": "continue",
            "local_assertiveness": 0.2,
            "coordination_reason": "yield",
        }
    )
    response_body = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": semantic,
                    }
                }
            ]
        }
    ).encode("utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return response_body

    monkeypatch.setattr(
        "llm_mappo.llm_teacher.request.urlopen", lambda *args, **kwargs: Response()
    )
    teacher = DeepSeekTeacher(api_key="test", max_attempts=1)
    assert parse_semantic_response(teacher._complete("{}")) == (
        0.6,
        "continue",
        0.2,
        "yield",
    )


def test_deepseek_teacher_request_uses_configured_completion_budget():
    teacher = DeepSeekTeacher(api_key="test", max_tokens=640)
    query = teacher._build_request("{}")
    payload = json.loads(query.data.decode("utf-8"))
    assert payload["max_tokens"] == 640
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.0
    assert "reasoning_effort" not in payload

    thinking = DeepSeekTeacher(
        api_key="test",
        model="deepseek-v4-pro",
        max_tokens=4096,
        thinking_enabled=True,
        reasoning_effort="high",
    )
    thinking_payload = json.loads(thinking._build_request("{}").data.decode("utf-8"))
    assert thinking_payload["thinking"] == {"type": "enabled"}
    assert thinking_payload["reasoning_effort"] == "high"
    assert "temperature" not in thinking_payload
    assert thinking.name == "deepseek:deepseek-v4-pro:thinking-high"

    with pytest.raises(ValueError, match="requires thinking_enabled"):
        DeepSeekTeacher(api_key="test", reasoning_effort="high")


def test_deepseek_teacher_prompt_uses_semantic_view_without_raw_observation(
    monkeypatch,
):
    scenario = EngagementScenario(
        scenario_id="s1",
        observation_version="v1",
        scenario_type="priority_conflict",
        observation=(0.123456, 0.654321),
        agent_id=1,
        battery=0.8,
        loaded=True,
        priority_label="A",
        target_kind="delivery",
        nearby_agents=(),
    )
    prompts = []
    teacher = DeepSeekTeacher(api_key="test")
    monkeypatch.setattr(
        teacher,
        "_complete",
        lambda prompt: prompts.append(prompt)
        or json.dumps(
            {
                "task_commitment": 0.8,
                "task_reason": "proceed",
                "local_assertiveness": 0.7,
                "coordination_reason": "clear",
            }
        ),
    )
    assert teacher.label_semantics(scenario) == (0.8, "proceed", 0.7, "clear")
    assert '"scenario_type": "priority_conflict"' in prompts[0]
    assert '"focal_agent"' in prompts[0]
    assert "A > B > C" in prompts[0]
    assert "never means willingness to reach a charging target" in prompts[0]
    assert '"observation":' not in prompts[0]
    assert "0.123456" not in prompts[0]


def test_deepseek_teacher_truncated_reasoning_has_safe_diagnostics(monkeypatch):
    response_body = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": "We need answer only JSON but first reason",
                    },
                }
            ],
            "usage": {"completion_tokens": 96},
        }
    ).encode("utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return response_body

    monkeypatch.setattr(
        "llm_mappo.llm_teacher.request.urlopen", lambda *args, **kwargs: Response()
    )
    teacher = DeepSeekTeacher(api_key="do-not-print-this", max_attempts=1)
    with pytest.raises(ValueError, match="no bounded JSON answer") as raised:
        teacher._complete("{}")
    message = str(raised.value)
    assert '"finish_reason": "length"' in message
    assert '"preview": ""' in message
    assert '"completion_tokens": 96' in message
    assert "We need answer only JSON" in message
    assert "do-not-print-this" not in message


def test_deepseek_teacher_does_not_retry_client_error(monkeypatch):
    calls = []

    def fake_urlopen(*args, **kwargs):
        calls.append(1)
        raise error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            400,
            "invalid model",
            {},
            BytesIO(b'{"error":"model not found"}'),
        )

    monkeypatch.setattr("llm_mappo.llm_teacher.request.urlopen", fake_urlopen)
    teacher = DeepSeekTeacher(api_key="test", timeout_seconds=1, max_attempts=3)
    with pytest.raises(RuntimeError, match="HTTP 400.*deepseek-v4-flash"):
        teacher._complete("{}");
    assert len(calls) == 1


def test_phase4_partial_checkpoint_resumes_without_relabelling(tmp_path):
    class FailingTeacher(MockTeacher):
        def __init__(self):
            self.calls = 0

        def label_semantics(self, scenario):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            return super().label_semantics(scenario)

    output = tmp_path / "labels.jsonl"
    env = Phase2Warehouse(
        n_agents=5,
        max_steps=40,
        batch_interval=40,
        batch_size_range=(4, 8),
        request_queue_size=8,
        task_completion_target=50,
        include_priority_features=True,
    )
    quotas = {name: 1 for name in (
        "normal_transport",
        "priority_conflict",
        "narrow_corridor_yield",
        "low_battery_diversion",
        "station_exit_congestion",
    )}
    try:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            collect_stratified_offline_labels(
                env, FailingTeacher(), output, [7], quotas
            )
        partial = output.with_name(output.name + ".partial.jsonl")
        assert partial.exists()
        result = collect_stratified_offline_labels(
            env, MockTeacher(), output, [7], quotas
        )
    finally:
        env.close()
    assert result["records"] == 5
    assert not partial.exists()
