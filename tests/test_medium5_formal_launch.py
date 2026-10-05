"""Check the formal batch contract without training or reserved evaluation."""

from pathlib import Path

import pytest
import yaml

from scripts import run_medium5_formal_baseline as launcher


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    directory = tmp_path / "configs"
    directory.mkdir()
    template = yaml.safe_load(Path(
        "configs/formal_environment_training/medium-5ag-baseline.yaml"
    ).read_text(encoding="utf-8"))
    for seed in launcher.TRAIN_SEEDS:
        template["approval_status"] = "approved"
        template["training"]["seed"] = seed
        (directory / f"seed_{seed:04d}.yaml").write_text(
            yaml.safe_dump(template), encoding="utf-8",
        )
    monkeypatch.setattr(launcher, "_verify_imports", lambda root: None)
    monkeypatch.setattr(launcher, "_git_snapshot", lambda root: {
        "code_commit": "a" * 40, "dirty_paths": [], "dirty_training_paths": [],
    })
    return tmp_path, directory


def test_dry_run_reviews_all_eight_without_training(prepared, monkeypatch):
    root, directory = prepared

    def prohibited(*args, **kwargs):
        raise AssertionError("Dry-run must not construct a learner.")

    monkeypatch.setattr(launcher, "BaselineRun", prohibited)
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: False)
    result = launcher.launch(
        directory, list(launcher.TRAIN_SEEDS), device="cuda", run=False, root=root,
    )
    assert result["status"] == "dry_run"
    assert result["actor_observation_dimension"] == 209
    assert result["fp_state_dimension"] == 689
    assert len(result["runs"]) == 8
    assert {row["environment_steps"] for row in result["runs"]} == {900_000}
    assert {row["evaluation_instances"] for row in result["runs"]} == {200}
    assert result["launch_blockers"] == [
        "Requested CUDA device is unavailable; no training started."
    ]
    assert Path(result["manifest_path"]).exists()
    assert not (root / launcher.ARTIFACT_ROOT).exists()


@pytest.mark.parametrize("seeds", [[], [7, 7], [8], [7, 87]])
def test_unique_registered_seed_subset(prepared, seeds):
    root, directory = prepared
    with pytest.raises(ValueError, match="unique nonempty"):
        launcher.review_configs(directory, seeds, root=root)


@pytest.mark.parametrize("section,field,value", [
    (None, "approval_status", "provisional"),
    ("training", "seed", 17),
    ("training", "environment_steps", 600_000),
    ("evaluation", "episodes_per_base", 1),
    ("evaluation", "environment_overrides", {"max_steps": 900}),
    ("evaluation", "replenishment_mode", "fixed_batch"),
])
def test_rejects_changed_formal_contract(prepared, section, field, value):
    root, directory = prepared
    path = directory / "seed_0007.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    (data if section is None else data[section])[field] = value
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.review_configs(directory, [7], root=root)


def test_seed_contracts_may_differ_only_in_seed(prepared):
    root, directory = prepared
    path = directory / "seed_0017.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["environment"]["profile_id"] = "a-different-profile"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="differ beyond"):
        launcher.review_configs(directory, [7, 17], root=root)


def test_run_requires_available_cuda(prepared, monkeypatch):
    root, directory = prepared
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA device is unavailable"):
        launcher.launch(directory, [7], device="cuda", run=True, root=root)


@pytest.mark.parametrize("status", ["complete", "running", "failed"])
def test_existing_run_is_never_automatically_repeated(prepared, status):
    root, directory = prepared
    existing = root / launcher.ARTIFACT_ROOT / "seed_0007_20260101T000000Z"
    existing.mkdir(parents=True)
    (existing / "state.json").write_text(
        '{"status": "' + status + '"}', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="already has a run"):
        launcher.launch(directory, [7], device="cpu", run=True, root=root)


def test_run_refuses_dirty_training_code(prepared, monkeypatch):
    root, directory = prepared
    monkeypatch.setattr(launcher, "_git_snapshot", lambda root: {
        "code_commit": "a" * 40, "dirty_paths": ["llm_mappo/planner.py"],
        "dirty_training_paths": ["llm_mappo/planner.py"],
    })
    with pytest.raises(ValueError, match="Commit training code/config"):
        launcher.launch(directory, [7], device="cpu", run=True, root=root)


def test_explicit_run_is_serial_and_records_each_run(prepared, monkeypatch):
    root, directory = prepared
    events = []

    class FakeRun:
        def __init__(self, experiment, *, device):
            self.seed = experiment.training["seed"]
            self.directory = root / launcher.ARTIFACT_ROOT / f"seed_{self.seed:04d}_test"
            self.directory.mkdir(parents=True)
            events.append(("construct", self.seed, device))

        def train(self):
            events.append(("train", self.seed))

    monkeypatch.setattr(launcher, "BaselineRun", FakeRun)
    result = launcher.launch(directory, [7, 17], device="cpu", run=True, root=root)
    assert events == [("construct", 7, "cpu"), ("train", 7),
                      ("construct", 17, "cpu"), ("train", 17)]
    assert result["status"] == "complete"
    assert all(row["status"] == "complete" for row in result["runs"])
    assert not Path(result["manifest_path"]).parent.parent.joinpath(
        ".medium5-baseline.lock"
    ).exists()


def test_run_failure_stops_batch_and_releases_lock(prepared, monkeypatch):
    root, directory = prepared
    calls = []

    class FailedRun:
        def __init__(self, experiment, *, device):
            calls.append(experiment.training["seed"])
            self.directory = root / "failed_run"

        def train(self):
            raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(launcher, "BaselineRun", FailedRun)
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        launcher.launch(directory, [7, 17], device="cpu", run=True, root=root)
    assert calls == [7]
    assert not list(root.rglob(".medium5-baseline.lock"))


def test_import_guard_rejects_another_worktree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Cross-worktree import"):
        launcher._verify_imports(tmp_path)
