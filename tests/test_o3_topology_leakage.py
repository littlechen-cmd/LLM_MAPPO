"""O3-E runtime and repository leakage guards for held-out topology assets."""

from pathlib import Path

import pytest
import torch
import yaml

from llm_mappo.o3_guard import (
    O3_EFFECTIVE_LAYOUT_HASHES,
    O3_ENVIRONMENT_IDS,
    O3_PROVENANCE_TOKENS,
    reject_o3_environment,
    reject_o3_provenance,
)
from llm_mappo.o3_topologies import make_o3_evaluation_environment
from llm_mappo.optimization_training import OptimizationTrainingConfig
from llm_mappo.phase2_training import Phase2TrainingConfig, train_phase2
from llm_mappo.phase3_training import Phase3TrainingConfig, train_phase3
from llm_mappo.phase4 import collect_offline_labels
from llm_mappo.semantic_v3 import SemanticDatasetV3


def test_o3_guard_rejects_ids_and_nested_provenance_without_a_bypass_flag():
    """Catch a training/data caller accepting any known O3 identifier or hash."""
    for environment_id in O3_ENVIRONMENT_IDS:
        with pytest.raises(ValueError, match="evaluation-only"):
            reject_o3_environment(environment_id, context="test training")
    reject_o3_environment("llm-mappo-medium-3ag-v1", context="test training")

    payload = {"records": [{"layout_hash": O3_EFFECTIVE_LAYOUT_HASHES[0]}]}
    with pytest.raises(ValueError, match="O3 provenance"):
        reject_o3_provenance(payload, context="OOD reference")
    reject_o3_provenance(
        {"layout_hash": "core-layout", "environment_id": "core"},
        context="OOD reference",
    )


def test_all_training_config_types_reject_o3_environment_ids():
    """Catch one legacy or optimization parser bypassing the central guard."""
    environment_id = O3_ENVIRONMENT_IDS[0]
    with pytest.raises(ValueError, match="evaluation-only"):
        Phase2TrainingConfig(env_id=environment_id)
    with pytest.raises(ValueError, match="evaluation-only"):
        Phase3TrainingConfig(env_id=environment_id)

    payload = yaml.safe_load(
        Path("configs/optimization/o1_functional_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["environment_id"] = environment_id
    with pytest.raises(ValueError, match="evaluation-only"):
        OptimizationTrainingConfig.from_mapping(payload)


def test_training_entry_rechecks_a_mutated_config_before_writing(tmp_path):
    """Catch callers mutating a validated dataclass and reaching training side effects."""
    config = Phase2TrainingConfig(
        episodes=1,
        output_dir=str(tmp_path / "phase2"),
    )
    config.env_id = O3_ENVIRONMENT_IDS[0]
    with pytest.raises(ValueError, match="evaluation-only"):
        train_phase2(config)
    assert not (tmp_path / "phase2").exists()

    phase3 = Phase3TrainingConfig(
        episodes=1,
        output_dir=str(tmp_path / "phase3"),
    )
    phase3.env_id = O3_ENVIRONMENT_IDS[1]
    with pytest.raises(ValueError, match="evaluation-only"):
        train_phase3(phase3)
    assert not (tmp_path / "phase3").exists()


def test_label_and_ood_entry_points_reject_o3_provenance_before_output(tmp_path):
    """Catch held-out maps entering label generation or OOD reference fitting."""
    environment = make_o3_evaluation_environment(O3_ENVIRONMENT_IDS[0])

    class ProviderMustNotRun:
        def label(self, *args, **kwargs):
            raise AssertionError("O3 label provider must not run")

    try:
        with pytest.raises(ValueError, match="evaluation-only"):
            collect_offline_labels(
                environment,
                ProviderMustNotRun(),
                tmp_path / "labels.jsonl",
                seeds=[9301],
                scenarios_per_seed=1,
            )
        assert not (tmp_path / "labels.jsonl").exists()
    finally:
        environment.close()

    record = {
        "validity": 1,
        "layout_hash": O3_EFFECTIVE_LAYOUT_HASHES[0],
        "vector": [0.0] * 61,
        "scores": [0.5, 0.5, 0.5],
    }
    with pytest.raises(ValueError, match="O3 provenance"):
        SemanticDatasetV3.from_records([record])


def test_o3_factory_does_not_touch_checkpoint_optimizer_or_online_llm(monkeypatch):
    """Catch the evaluation-only construction path acquiring learning dependencies."""
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("O3 factory must not load a checkpoint")
        ),
    )
    monkeypatch.setattr(
        torch.optim,
        "Adam",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("O3 factory must not create an optimizer")
        ),
    )
    environment = make_o3_evaluation_environment(O3_ENVIRONMENT_IDS[1])
    try:
        assert environment.reset(seed=9302).shape == (5, 613)
    finally:
        environment.close()


def test_trainable_assets_contain_o3_provenance_only_in_governance_manifest():
    """Catch held-out IDs, hashes, or paths leaking into trainable assets."""
    allowed = Path("configs/g3_experiment_manifest.yaml")
    offenders = []
    for root_name in ("configs", "train", "scripts", "prompts", "datasets"):
        root = Path(root_name)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path == allowed
                or path.suffix
                not in {".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml"}
            ):
                continue
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in O3_PROVENANCE_TOKENS):
                offenders.append(str(path))
    assert offenders == []


def test_manifest_requires_interface_revalidation_after_o1_no_go():
    """Catch a future manifest treating stale O3 interface evidence as valid."""
    manifest = Path("docs/evidence/o3-topology-evidence-v1.json").read_text(
        encoding="utf-8"
    )
    assert "retain map bytes" in manifest
    assert "invalidate and rerun all O3 interface evidence" in manifest


def test_o3_execution_code_contains_only_test_seeds_and_no_formal_seed_range():
    """Catch formal held-out seeds entering O3 construction or local smoke code."""
    implementation = Path("llm_mappo/o3_topologies.py").read_text(encoding="utf-8")
    interface_tests = Path("tests/test_o3_topology_interfaces.py").read_text(
        encoding="utf-8"
    )
    assert "9301" not in implementation and "9302" not in implementation
    assert "200" not in implementation and "209" not in implementation
    assert "9301" in interface_tests and "9302" in interface_tests
    assert "200" not in interface_tests and "209" not in interface_tests


def test_rejected_layouts_and_same_map_eight_agv_are_not_active_resources():
    """Catch superseded pressure-layout assets returning to executable paths."""
    active = [
        path
        for root in (Path("rware/layouts"), Path("configs"))
        for path in root.rglob("*")
        if path.is_file()
    ]
    names = "\n".join(str(path).lower() for path in active)
    assert "rejected" not in names
    assert "8ag" not in names
