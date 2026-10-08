"""Independent medium goal-full entry without altering frozen C1 identities."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest
import yaml

from llm_mappo.mappo_baseline.runner import (
    BaselineRun, checkpoint_schema_for_experiment, load_baseline_experiment,
)
from llm_mappo.mappo_baseline.vector_env import new_baseline_environment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/llm_kd/phase_20261007/medium-5ag-baseline-goal-full.yaml"


def _changed_config(tmp_path, section, field, value):
    values = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if section is None:
        values[field] = value
    else:
        values[section][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return path


def test_approved_medium_config_loads_new_contract_and_constructs_real_environment():
    experiment = load_baseline_experiment(CONFIG)
    assert experiment.schema == "core-mappo-209d-goal-full-v1"
    assert experiment.approval_status == "approved"
    assert experiment.environment["profile_id"] == "medium-5ag-goal-full-v1"
    assert experiment.config_sha256 == sha256(CONFIG.read_bytes()).hexdigest()
    assert experiment.training["environment_steps"] == 600000
    environment = new_baseline_environment(experiment.environment)
    try:
        assert environment.n_agents == 5
        assert environment.max_steps == 800
        assert environment.env.charging_contract == "goal-full-v1"
        assert environment.env.charging_rate == 0.1
        assert environment.charge_threshold == 0.3
        assert environment.charge_release_threshold == 1.0
        assert environment.env.grid_size == (24, 20)
    finally:
        environment.close()


@pytest.mark.parametrize("method,namespace", [
    ("MAPPO-FP-DG", "mappo-fp-direct-goal-209d-goal-full-checkpoint-v1"),
    ("MAPPO-FP-DG-LLM-INIT", "mappo-fp-llm-init-209d-goal-full-checkpoint-v1"),
    ("MAPPO-FP-ASTARKD-NO-VALIDITY", "mappo-fp-astar-kd-209d-goal-full-checkpoint-v1"),
    ("MAPPO-FP-ASTARKD-NO-VALIDITY-LLM-INIT",
     "mappo-fp-astar-kd-llm-init-209d-goal-full-checkpoint-v1"),
])
def test_new_checkpoint_namespace_is_independent_for_every_core_method(method, namespace):
    experiment = replace(load_baseline_experiment(CONFIG), method=method)
    assert checkpoint_schema_for_experiment(experiment) == namespace
    old = replace(experiment, schema="core-mappo-209d-delivery-c1-v3")
    assert checkpoint_schema_for_experiment(old) != namespace


@pytest.mark.parametrize("section,field,value", [
    ("environment", "charging_rate", 0.02),
    ("environment", "charge_release_threshold", 0.8),
    ("environment", "charging_contract", "legacy-v1"),
    ("environment", "charge_threshold", 0.2),
    ("environment", "profile_id", "tiny-3ag-goal-full-v1"),
    ("environment", "n_agents", 12),
    ("training", "environment_steps", 800000),
    ("training", "seed", 8),
    ("evaluation", "seeds", [9400, 9401]),
    ("reward_definition", "step_cost", 0.01),
    (None, "schema", "core-mappo-209d-delivery-c1-v3"),
])
def test_new_entry_rejects_unfrozen_values_and_legacy_schema_misuse(
    tmp_path, section, field, value,
):
    with pytest.raises(ValueError):
        load_baseline_experiment(_changed_config(tmp_path, section, field, value))


def test_provisional_goal_full_training_is_rejected_before_creating_artifacts(tmp_path):
    config = _changed_config(tmp_path, None, "approval_status", "provisional")
    experiment = load_baseline_experiment(config)
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="provisional"):
        BaselineRun(experiment, device="cpu", output_root=output)
    assert not output.exists()


@pytest.mark.parametrize("seed", [7, 17, 27, 37, 47, 57, 67, 77])
def test_new_contract_accepts_the_eight_approved_training_seeds(tmp_path, seed):
    experiment = load_baseline_experiment(
        _changed_config(tmp_path, "training", "seed", seed)
    )
    assert experiment.training["seed"] == seed


def test_new_contract_keeps_old_c1_configuration_frozen():
    old_path = ROOT / "configs/core_integration/medium-5ag-baseline.yaml"
    old = load_baseline_experiment(old_path)
    assert old.schema == "core-mappo-209d-delivery-c1-v3"
    environment_bytes = json.dumps(
        dict(old.environment), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    assert sha256(environment_bytes).hexdigest() == (
        "a5c46b1b38a64b33ef55057d1607345ce356f6aca6b7bedfd1e20c37498a9468"
    )
    assert old.environment["charge_release_threshold"] == 0.8
    assert "charging_contract" not in old.environment
    assert checkpoint_schema_for_experiment(old) == (
        "mappo-fp-direct-goal-209d-delivery-c1-checkpoint-v3"
    )


def test_goal_full_astar_method_retains_no_validity_dispatch(tmp_path):
    values = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    values["method"] = "MAPPO-FP-ASTARKD-NO-VALIDITY"
    values["training"]["post_schedule_astar_lambda"] = 0.0
    path = tmp_path / "astar.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    experiment = load_baseline_experiment(path)
    run = BaselineRun(
        experiment, device="cpu", output_root=tmp_path / "run", evaluation_only=True,
    )
    try:
        assert run._astar_kd_enabled
        assert run._astar_kd_variant == "no_validity"
        assert run.post_schedule_astar_lambda == 0.0
    finally:
        run.close()


def test_goal_full_run_manifest_keeps_new_environment_and_reward_identity(tmp_path):
    from llm_mappo.medium5_contract import reward_definition_sha256

    experiment = load_baseline_experiment(CONFIG)
    run = BaselineRun(
        experiment, device="cpu", output_root=tmp_path, evaluation_only=True,
    )
    try:
        manifest = json.loads((run.directory / "run_manifest.json").read_text("utf-8"))
        assert manifest["identity"]["schema"] == "core-mappo-209d-goal-full-v1"
        assert manifest["identity"]["reward_definition_sha256"] == (
            reward_definition_sha256()
        )
        assert manifest["environment"] == dict(experiment.environment)
        assert manifest["identity"]["profile_id"] == "medium-5ag-goal-full-v1"
    finally:
        run.close()
