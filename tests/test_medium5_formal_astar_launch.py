"""A* formal launch gates without training or reserved evaluation."""

import copy
import importlib
from pathlib import Path
import json

import pytest
import yaml


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    launcher = importlib.import_module("scripts.run_medium5_formal_astar_kd")
    baseline = tmp_path / "configs/training_v1/medium-5ag-formal"
    baseline.mkdir(parents=True)
    directory = tmp_path / "astar"
    directory.mkdir()
    template_path = Path("configs/training_v1/medium-5ag-formal/seed_0007.yaml")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    for seed in launcher.base.TRAIN_SEEDS:
        template["training"]["seed"] = seed
        (baseline / f"seed_{seed:04d}.yaml").write_text(
            yaml.safe_dump(template), encoding="utf-8",
        )
        astar = copy.deepcopy(template)
        astar["method"] = "MAPPO-FP-ASTARKD-NO-VALIDITY"
        astar["artifact_root"] = launcher.ARTIFACT_ROOT
        astar["training"].update(
            astar_teacher_approval="approved",
            astar_teacher_version=launcher.TEACHER_VERSION,
            astar_schedule_steps=600000, post_schedule_astar_lambda=0.0,
        )
        (directory / f"seed_{seed:04d}.yaml").write_text(
            yaml.safe_dump(astar), encoding="utf-8",
        )
    monkeypatch.setattr(launcher.base, "_verify_imports", lambda root: None)
    monkeypatch.setattr(launcher.base, "_git_snapshot", lambda root: {
        "code_commit": "a" * 40, "dirty_training_paths": [], "dirty_paths": [],
    })
    return launcher, tmp_path, directory


def test_launcher_module_exists():
    assert importlib.util.find_spec("scripts.run_medium5_formal_astar_kd") is not None


def test_dryrun_seed7_never_constructs_learner(prepared, monkeypatch):
    launcher, root, directory = prepared
    def prohibited(*args, **kwargs):
        raise AssertionError("dryrun constructed learner")
    monkeypatch.setattr(launcher, "AStarKDRun", prohibited)
    result = launcher.launch(directory, [7], device="cpu", run=False, root=root)
    assert result["runs"][0]["seed"] == 7
    assert result["astar_schedule_steps"] == 600000
    assert result["pure_ppo_tail_steps"] == 300000
    assert not result["launch_blockers"]

@pytest.mark.parametrize("section,field,value", [
    ("training", "entropy_coefficient", 0.02),
    ("environment", "deadlock_steps", 190),
    ("training", "astar_schedule_steps", 900000),
    ("training", "post_schedule_astar_lambda", 0.1),
])

def test_contract_rejects_unpaired_or_no_pure_tail(prepared, section, field, value):
    launcher, root, directory = prepared
    path = directory / "seed_0007.yaml"
    data = yaml.safe_load(path.read_text())
    data[section][field] = value
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.review_configs(directory, [7], root=root)


def test_remaining_seeds_are_blocked_without_review(prepared):
    launcher, root, directory = prepared
    result = launcher.launch(directory,[17], device="cpu", run=False, root=root)
    assert any("seed 7" in item for item in result["launch_blockers"])
    with pytest.raises(ValueError, match="seed 7"):
        launcher.launch(directory,[17], device="cpu", run=True, root=root)


def test_seed7_cannot_be_batched_with_remaining(prepared):
    launcher, root, directory = prepared
    with pytest.raises(ValueError, match="separately"):
        launcher.launch(directory,[7, 17], device="cpu", run=False, root=root)


def test_explicit_seed7_run_records_and_releases_lock(prepared, monkeypatch):
    launcher, root, directory = prepared
    class FakeRun:
        def __init__(self, experiment, device):
            self.directory = root / launcher.ARTIFACT_ROOT / "seed_0007_test"
            self.directory.mkdir(parents=True)
        def train(self):
            pass
    monkeypatch.setattr(launcher, "AStarKDRun", FakeRun)
    result = launcher.launch(directory,[7], device="cpu", run=True, root=root)
    assert result["status"] == "complete"
    assert result["runs"][0]["status"] == "complete"
    assert not list(root.rglob(".medium5-astar-kd.lock"))


@pytest.fixture
def completed_seed7(prepared):
    launcher, root, directory = prepared
    experiment = launcher.review_configs(directory, [7], root=root)[0][1]
    path = root / launcher.ARTIFACT_ROOT / "seed_0007_reviewed"
    path.mkdir(parents=True)
    identity = {
        "schema": launcher.SCHEMA, "method": experiment.method, "seed": 7,
        "config_sha256": experiment.config_sha256, "code_commit": "a" * 40,
    }
    evaluation = {"episodes": [{"seed": seed, "steps": 800}
                               for seed in launcher.EVALUATION_SEEDS]}
    manifest = {
        "identity": identity, "environment": dict(experiment.environment),
        "training": dict(experiment.training),
        "evaluation": dict(experiment.evaluation),
    }
    summary = {"identity": identity, "global_environment_steps": 900000,
               "updates": 879, "evaluation": evaluation}
    (path / "evaluation").mkdir()
    for relative, values in [
        ("run_manifest.json", manifest), ("state.json", {"status": "complete"}),
        ("summary.json", summary), ("evaluation/evaluation.json", evaluation),
    ]:
        (path / relative).write_text(json.dumps(values), encoding="utf-8")
    (path / "checkpoint_final.pt").write_bytes(b"test checkpoint")
    return launcher, root, directory, path


def test_valid_review_allows_remaining_dryrun(completed_seed7):
    launcher, root, directory, path = completed_seed7
    result = launcher.launch(
        directory, [17, 27], device="cpu", run=False, root=root,
        seed7_reviewed_run=path, confirm_seed7_review=True,
    )
    assert not result["launch_blockers"]
    attestation = result["seed7_review_attestation"]
    assert attestation["owner_confirmed_review"]
    assert len(attestation["final_checkpoint_sha256"]) == 64


@pytest.mark.parametrize("relative,field,value", [
    ("state.json", "status", "failed"),
    ("summary.json", "global_environment_steps", 600000),
    ("summary.json", "updates", 800),
    ("summary.json", "total_elapsed_seconds", float("nan")),
])

def test_invalid_seed7_evidence_blocks_remaining(completed_seed7, relative,
                                               field, value):
    launcher, root, directory, path = completed_seed7
    source = path / relative
    data = json.loads(source.read_text())
    data[field] = value
    source.write_text(json.dumps(data), encoding="utf-8")
    result = launcher.launch(
        directory, [17], device="cpu", run=False, root=root,
        seed7_reviewed_run=path, confirm_seed7_review=True,
    )
    assert result["launch_blockers"]


def test_confirmation_alone_cannot_bypass_gate(prepared):
    launcher, root, directory = prepared
    result = launcher.launch(
        directory, [17], device="cpu", run=False, root=root,
        confirm_seed7_review=True,
    )
    assert result["launch_blockers"]
