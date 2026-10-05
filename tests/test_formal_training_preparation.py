"""Formal launch readiness must reflect real artifacts and selected seeds."""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from llm_mappo.mappo_baseline.runner import load_baseline_experiment
from tests.test_llm_kd_initialization import _initialization


ROOT = Path("configs/formal_environment_training")


@pytest.mark.parametrize("method", ["baseline", "astar-kd"])
def test_approved_scene_loads_with_exact_registered_seed(tmp_path, method):
    values = yaml.safe_load((ROOT / f"tiny-3ag-{method}.yaml").read_text())
    values["approval_status"] = "approved"
    values["training"]["seed"] = 77
    if method == "astar-kd":
        values["training"]["astar_teacher_approval"] = "approved"
        values["training"]["astar_teacher_version"] = "astar-geometric-cost-v2"
    path = tmp_path / "approved.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    experiment = load_baseline_experiment(path)
    assert experiment.approval_status == "approved"
    assert experiment.training["seed"] == 77
    assert experiment.environment["n_agents"] == 3


def test_approved_llm_scene_requires_authenticated_real_artifact(tmp_path):
    values = yaml.safe_load((ROOT / "tiny-3ag-llm-kd.yaml").read_text())
    values["approval_status"] = "approved"
    path = tmp_path / "approved.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a real representation"):
        load_baseline_experiment(path)
    initialization, _ = _initialization(tmp_path)
    values["training"]["representation_initialization"] = initialization
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    experiment = load_baseline_experiment(path)
    assert experiment.training["representation_initialization"] == initialization
    initialization["artifact_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="digest does not match"):
        load_baseline_experiment(path)


def _prepare(output, *extra):
    result = subprocess.run(
        [sys.executable, "-m", "scripts.prepare_formal_training_configs",
         "--output-directory", str(output), *extra],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def test_preparation_generates_48_runnable_and_48_pending_without_fake_files(tmp_path):
    manifest = _prepare(tmp_path / "prepared")
    ready = [run for run in manifest["runs"] if run["status"] == "ready"]
    pending = [run for run in manifest["runs"] if run["status"] == "dependency_missing"]
    assert len(ready) == 48
    assert len(pending) == 48
    assert {run["seed"] for run in ready} == {7, 17, 27, 37, 47, 57, 67, 77}
    assert all(Path(run["config_path"]).is_file() for run in ready)
    assert all(run["config_path"] is None for run in pending)
    assert len(list((tmp_path / "prepared").rglob("*.yaml"))) == 48
    for run in ready:
        assert load_baseline_experiment(run["config_path"]).approval_status == "approved"


def test_preparation_binds_same_seed_representation_across_scenes_and_llm_methods(tmp_path):
    initialization, _ = _initialization(tmp_path)
    representation_root = tmp_path / "representations"
    destination = representation_root / "seed_0007" / "representation.pt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(Path(initialization["artifact_path"]).read_bytes())
    manifest = _prepare(
        tmp_path / "prepared", "--seeds", "7",
        "--representation-directory", str(representation_root),
    )
    assert len(manifest["runs"]) == 12
    assert all(run["status"] == "ready" for run in manifest["runs"])
    hashes = set()
    for run in manifest["runs"]:
        experiment = load_baseline_experiment(run["config_path"])
        if run["method"].endswith("LLM-INIT"):
            hashes.add(experiment.training["representation_initialization"]["artifact_sha256"])
    assert hashes == {initialization["artifact_sha256"]}


def test_durability_protocol_keeps_checkpoint_identity_and_changes_only_eval():
    from llm_mappo.formal_training import make_durability_experiment

    source = load_baseline_experiment(ROOT / "formal-8ag-baseline.yaml")
    derived = make_durability_experiment(source)
    assert derived.config_sha256 == source.config_sha256
    assert derived.environment == source.environment
    assert derived.training == source.training
    assert derived.evaluation["seeds"] == list(range(400, 410))
    assert derived.evaluation["environment_overrides"] == {"max_steps": 5000}
    assert source.evaluation["environment_overrides"] == {}
    with pytest.raises(ValueError, match="formal-8ag"):
        make_durability_experiment(load_baseline_experiment(ROOT / "tiny-3ag-baseline.yaml"))


def test_preparation_emits_paired_pretraining_configs_without_running_it(tmp_path):
    from llm_mappo.mappo_llm_kd.pretraining_run import load_pretraining_run_config

    output = tmp_path / "prepared"
    manifest = _prepare(
        output, "--dataset-directory", str(tmp_path / "dataset"),
        "--label-directory", str(tmp_path / "labels"),
    )
    assert len(manifest["pretraining_runs"]) == 8
    for run in manifest["pretraining_runs"]:
        config = load_pretraining_run_config(run["config_path"])
        assert config.approval_status == "approved"
        assert config.seed == run["seed"]
        assert config.batch_size == 256
        assert config.max_epochs == 100
        assert Path(config.dataset_directory) == tmp_path / "dataset"
        assert run["run_id"] == f"seed_{run['seed']:04d}"
    assert not (tmp_path / "dataset").exists()
