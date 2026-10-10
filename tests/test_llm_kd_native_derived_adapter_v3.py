"""Native derived admission binds raw lineage before labels or pretraining."""

from hashlib import sha256
import json
from pathlib import Path

import pytest

from llm_mappo.mappo_llm_kd import labeling_run_v3 as labeling
from llm_mappo.mappo_llm_kd import native_derived_adapter_v3 as adapter
from llm_mappo.mappo_llm_kd import pretraining_run as pretraining
from llm_mappo.mappo_llm_kd.pretraining import (
    SEMANTIC_CONTRACT_V3, SEMANTIC_DIMENSIONS_V3, SEMANTIC_SOURCES_V3,
)
from llm_mappo.mappo_llm_kd.source_actor_v3 import canonical_sha256


def _native_artifact(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = dict(verified_source=True,
                  source_checkpoint_sha256=adapter.SOURCE_CHECKPOINT_SHA256,
                  source_code_commit=adapter.SOURCE_COMMIT,
                  source_environment_sha256="native-environment",
                  actor_identity=dict(
                      schema="three-scene-hard-charge-v2", method="MAPPO-FP-DG",
                      seed=7, n_agents=5, profile_id="medium-5ag-hard-charge-v2",
                      charging_queue_policy="battery-priority-yield-v1"))
    environment = dict(profile_id="medium-5ag-hard-charge-v2",
                       scene_id="medium-5ag", n_agents=5, max_steps=800, queue_size=5,
                       charging_rate=0.1, charge_threshold=0.3,
                       charge_release_threshold=1.0, hard_charge_until_full=True)
    rollout = dict(schema="three-scene-hard-charge-v2", environment=environment,
                   environment_sha256="native-environment")
    identity = dict(schema="medium-train-extension-dataset-v1",
                    synthetic_execution=False, source_actor_contract=source,
                    rollout_environment_contract=rollout)
    (dataset / "collection_identity.json").write_text(json.dumps(identity), "utf-8")
    manifest = dict(lineage_schema=adapter.LINEAGE_SCHEMA,
                    collection_identity_sha256=canonical_sha256(identity),
                    source_actor_contract=source, rollout_environment_contract=rollout,
                    environment_sha256=rollout["environment_sha256"])
    return dataset, identity, manifest


@pytest.mark.parametrize("mutation", ["profile", "checkpoint", "synthetic", "lineage"])
def test_native_adapter_rejects_wrong_source_before_raw_verifier(
    tmp_path, monkeypatch, mutation,
):
    dataset, identity, manifest = _native_artifact(tmp_path)
    monkeypatch.setattr(adapter, "verify_medium_extension_dataset",
                        lambda *args, **kwargs: pytest.fail("raw verifier was called"))
    if mutation == "profile":
        identity["rollout_environment_contract"]["environment"]["profile_id"] = "other"
    elif mutation == "checkpoint":
        identity["source_actor_contract"]["source_checkpoint_sha256"] = "0" * 64
    elif mutation == "synthetic":
        identity["synthetic_execution"] = True
    else:
        manifest["lineage_schema"] = "unknown-lineage"
    (dataset / "collection_identity.json").write_text(json.dumps(identity), "utf-8")
    with pytest.raises(ValueError, match="lineage|source"):
        adapter.verify_native_derived_dataset(dataset, manifest)


def test_native_adapter_passes_explicit_relocation_to_raw_verifier(tmp_path, monkeypatch):
    dataset, identity, manifest = _native_artifact(tmp_path)
    calls = []

    def verify(directory, *, require_frozen, artifact_root):
        calls.append((directory, require_frozen, artifact_root))
        return dict(lineage_sha256=canonical_sha256(identity),
                    environment="medium-5ag", total_samples=3000)

    monkeypatch.setattr(adapter, "verify_medium_extension_dataset", verify)
    root = tmp_path / "relocated"
    frozen = adapter.verify_native_derived_dataset(
        dataset, manifest, artifact_root=root,
    )
    assert frozen["total_samples"] == 3000
    assert calls == [(dataset, True, root)]


def test_prepare_checks_native_raw_source_before_provider_or_plan_creation(
    tmp_path, monkeypatch,
):
    dataset, _, manifest = _native_artifact(tmp_path)
    windows = dataset / "windows.jsonl"
    windows.write_text("", "utf-8")
    (dataset / "label_selection_lists.json").write_text("{}", "utf-8")
    (dataset / "state.json").write_text('{"status":"completed"}', "utf-8")
    manifest.update(schema="llm-dataset-build-run-v3",
                    distribution="native-209d-goal-full-main-v3",
                    semantic_contract_version=SEMANTIC_CONTRACT_V3,
                    semantic_dimensions=list(SEMANTIC_DIMENSIONS_V3),
                    semantic_sources=list(SEMANTIC_SOURCES_V3),
                    main_pretraining_eligible=True,
                    windows_sha256=sha256(windows.read_bytes()).hexdigest())
    (dataset / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    calls = []

    def reject(*args, **kwargs):
        calls.append(kwargs["artifact_root"])
        raise ValueError("raw lineage rejected")

    monkeypatch.setattr(labeling, "verify_native_derived_dataset", reject)
    with pytest.raises(ValueError, match="raw lineage"):
        labeling.prepare_labeling_v3(
            dataset, tmp_path / "plan", environment="medium-5ag", providers={},
            pricing_evidence={}, phase_ledger_path=tmp_path / "budget.jsonl",
            source_artifact_root=str(tmp_path),
        )
    assert calls == [str(tmp_path)]
    assert not (tmp_path / "plan").exists()


def test_load_plan_rechecks_derived_lineage_and_verifier_code(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    plan_directory = tmp_path / "labels"
    dataset.mkdir()
    plan_directory.mkdir()
    for name in ("manifest.json", "windows.jsonl", "label_selection_lists.json",
                 "selection_frozen.json"):
        (dataset / name).write_text("{}", "utf-8")
    (dataset / "manifest.json").write_text(
        json.dumps({"lineage_schema": adapter.LINEAGE_SCHEMA}), "utf-8",
    )
    (plan_directory / "requests.jsonl").write_text("", "utf-8")
    calls = []

    def verify(*args, **kwargs):
        calls.append(kwargs["artifact_root"])
        return {"lineage_sha256": "a" * 64}

    monkeypatch.setattr(labeling, "verify_native_derived_dataset", verify)
    monkeypatch.setattr(labeling, "_provider_plans", lambda _: {})
    plan = dict(schema=labeling.LABELING_PLAN_V3,
                semantic_contract_version=SEMANTIC_CONTRACT_V3,
                dataset_directory=str(dataset), environment="medium-5ag",
                lineage_schema=adapter.LINEAGE_SCHEMA, lineage_sha256="a" * 64,
                source_artifact_root=str(tmp_path), providers={},
                provider_plan_sha256=labeling._hash({}),
                maximum_reserve_usd=1.0, job_cap_usd=30.0,
                code_sha256={name: labeling._file_hash(
                    Path(labeling.__file__).with_name(name))
                    for name in ("medium_extension_dataset.py",
                                 "native_derived_adapter_v3.py")})
    for key, name in (
        ("dataset_manifest_sha256", "manifest.json"),
        ("dataset_windows_sha256", "windows.jsonl"),
        ("builder_lists_sha256", "label_selection_lists.json"),
        ("selection_frozen_sha256", "selection_frozen.json"),
    ):
        plan[key] = labeling._file_hash(dataset / name)
    plan["requests_sha256"] = labeling._file_hash(plan_directory / "requests.jsonl")
    (plan_directory / "plan.json").write_text(json.dumps(plan), "utf-8")
    assert labeling._load_plan(plan_directory)["lineage_sha256"] == "a" * 64
    assert calls == [str(tmp_path)]
    plan["code_sha256"]["native_derived_adapter_v3.py"] = "0" * 64
    (plan_directory / "plan.json").write_text(json.dumps(plan), "utf-8")
    with pytest.raises(ValueError, match="code"):
        labeling._load_plan(plan_directory)


def test_pretraining_loader_rechecks_derived_raw_before_tensor_use(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    window = dict(partition="train", episode_id="one", agent_id=1,
                  start_timestep=0, end_timestep=7,
                  observations=[[0.0] * 209 for _ in range(8)],
                  reset_masks=[True] + [False] * 7, stratum="ordinary")
    windows = dataset / "windows.jsonl"
    windows.write_text(json.dumps(window) + "\n", "utf-8")
    manifest = dict(schema="llm-dataset-build-run-v3",
                    distribution="native-209d-goal-full-main-v3",
                    main_pretraining_eligible=True,
                    windows_sha256=sha256(windows.read_bytes()).hexdigest(),
                    semantic_contract_version=SEMANTIC_CONTRACT_V3,
                    semantic_dimensions=list(SEMANTIC_DIMENSIONS_V3),
                    semantic_sources=list(SEMANTIC_SOURCES_V3),
                    lineage_schema=adapter.LINEAGE_SCHEMA)
    (dataset / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    (dataset / "state.json").write_text('{"status":"completed"}', "utf-8")
    calls = []
    monkeypatch.setattr(pretraining, "verify_native_derived_dataset",
                        lambda *args, **kwargs: calls.append(kwargs["artifact_root"]))
    assert len(pretraining._load_eligible_windows(
        dataset, source_artifact_root=str(tmp_path))) == 1
    assert calls == [str(tmp_path)]
    manifest["lineage_schema"] = "unknown"
    (dataset / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    with pytest.raises(ValueError, match="lineage"):
        pretraining._load_eligible_windows(dataset)
