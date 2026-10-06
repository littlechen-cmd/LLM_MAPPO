"""Parallel cutoff diagnostic contracts without real learners."""

import importlib
from pathlib import Path

import pytest
import yaml


def test_diagnostic_launcher_exists():
    assert importlib.util.find_spec("scripts.run_medium5_astar_tail_diagnostic")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    launcher = importlib.import_module("scripts.run_medium5_astar_tail_diagnostic")
    baseline_dir = tmp_path / "configs/training_v1/medium-5ag-formal"
    baseline_dir.mkdir(parents=True)
    original = Path("configs/training_v1/medium-5ag-formal/seed_0007.yaml")
    baseline_raw = yaml.safe_load(original.read_text(encoding="utf-8"))
    (baseline_dir / original.name).write_text(
        yaml.safe_dump(baseline_raw), encoding="utf-8",
    )
    directory = tmp_path / "configs/tail"
    directory.mkdir(parents=True)
    for group, stop in launcher.GROUPS.items():
        data = yaml.safe_load(original.read_text(encoding="utf-8"))
        data.update(schema=launcher.TAIL_SCHEMA,
                    method="MAPPO-FP-ASTARKD-NO-VALIDITY",
                    artifact_root=f"{launcher.ARTIFACT_ROOT}/{group}")
        data["training"].update(
            astar_teacher_approval="approved",
            astar_teacher_version=launcher.TEACHER_VERSION,
            astar_schedule_steps=600000, post_schedule_astar_lambda=0.01,
            astar_stop_steps=stop, paired_training_seeds=[7],
        )
        data["evaluation"] = dict(launcher.EVALUATION)
        (directory / f"{group}.yaml").write_text(
            yaml.safe_dump(data), encoding="utf-8",
        )
    monkeypatch.setattr(launcher.base, "_verify_imports", lambda root: None)
    monkeypatch.setattr(launcher.base, "_git_snapshot", lambda root: {
        "code_commit": "a" * 40, "dirty_training_paths": [], "dirty_paths": [],
    })
    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return launcher, tmp_path, directory


def test_dryrun_does_not_construct_or_spawn(prepared, monkeypatch):
    launcher, root, directory = prepared
    def prohibited(*args, **kwargs):
        raise AssertionError("dryrun spawned training")
    monkeypatch.setattr(launcher.subprocess, "Popen", prohibited)
    result = launcher.launch(directory, device="cuda", run=False, root=root)
    assert len(result["groups"]) == 4
    assert result["status"] == "dry_run"
    assert not result["launch_blockers"]


@pytest.mark.parametrize("section,field,value", [
    ("training", "astar_schedule_steps", 450000),
    ("training", "post_schedule_astar_lambda", 0.0),
    ("training", "astar_stop_steps", 600000),
    ("training", "entropy_coefficient", 0.02),
    ("evaluation", "seeds", [200000]),
])
def test_reject_changed_diagnostic_contract(prepared, section, field, value):
    launcher, root, directory = prepared
    source = directory / "cutoff_450k.yaml"
    data = yaml.safe_load(source.read_text())
    data[section][field] = value
    source.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.load_tail_experiment(source, "cutoff_450k", root=root)


def test_four_processes_spawn_before_any_wait(prepared, monkeypatch):
    launcher, root, directory = prepared
    events = []
    class Process:
        def __init__(self, command, **kwargs):
            self.group = command[command.index("--group") + 1]
            self.pid = len(events) + 1
            events.append(("spawn", self.group))
        def wait(self):
            events.append(("wait", self.group))
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    result = launcher.launch(directory, device="cuda", run=True, root=root)
    assert [event[0] for event in events] == ["spawn"] * 4 + ["wait"] * 4
    assert result["status"] == "complete"
    assert not list(root.rglob(".astar-tail.lock"))


@pytest.mark.parametrize("blocker", ["dirty", "duplicate", "cuda", "gpu"])
def test_launch_blockers(prepared, monkeypatch, blocker):
    launcher, root, directory = prepared
    if blocker == "dirty":
        monkeypatch.setattr(launcher.base, "_git_snapshot", lambda root: {
            "code_commit": "a" * 40, "dirty_training_paths": ["scripts/x.py"],
        })
    elif blocker == "duplicate":
        (root / launcher.ARTIFACT_ROOT / "cutoff_450k/seed_0007_old").mkdir(
            parents=True,
        )
    elif blocker == "cuda":
        monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError):
        launcher.launch(directory, device="cuda", run=True, root=root)


def test_failed_group_does_not_cancel_other_groups(prepared, monkeypatch):
    launcher, root, directory = prepared
    events = []

    class Process:
        def __init__(self, command, **kwargs):
            self.group = command[command.index("--group") + 1]
            self.pid = len(events) + 1
            events.append(("spawn", self.group))

        def wait(self):
            events.append(("wait", self.group))
            return 1 if self.group == "cutoff_450k" else 0

        def poll(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    result = launcher.launch(directory, device="cuda", run=True, root=root)
    assert result["status"] == "failed"
    assert [row["status"] for row in result["groups"]] == [
        "failed", "complete", "complete", "complete",
    ]
    assert len(events) == 8
    assert not list(root.rglob(".astar-tail.lock"))


def test_interrupt_terminates_only_started_children(prepared, monkeypatch):
    launcher, root, directory = prepared
    events = []

    class Process:
        def __init__(self, command, **kwargs):
            self.group = command[command.index("--group") + 1]
            self.pid = len(events) + 1
            self.finished = False
            events.append(("spawn", self.group))

        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            self.finished = True
            events.append(("reap", self.group))
            return 0

        def poll(self):
            return 0 if self.finished else None

        def terminate(self):
            events.append(("terminate", self.group))

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    with pytest.raises(KeyboardInterrupt):
        launcher.launch(directory, device="cuda", run=True, root=root)
    assert [event[0] for event in events] == (
        ["spawn"] * 4 + ["terminate"] * 4 + ["reap"] * 4
    )
    assert not list(root.rglob(".astar-tail.lock"))


def test_worker_termination_raises_for_runner_cleanup(prepared):
    launcher, _, _ = prepared
    with pytest.raises(KeyboardInterrupt):
        launcher._interrupt_worker(None, None)


def test_cleanup_kills_only_owned_group_after_timeout(prepared, monkeypatch):
    launcher, _, _ = prepared
    events = []

    class Process:
        pid = 12345

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            if timeout:
                raise launcher.subprocess.TimeoutExpired("test", timeout)
            events.append("reap")
            return 0

        def kill(self):
            events.append("kill")

    monkeypatch.setattr(launcher.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(launcher, "OWN_PROCESS_GROUPS", True, raising=False)
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: events.append(
        (pid, sig)), raising=False)
    launcher._stop_children([Process()])
    assert (12345, launcher.signal.SIGKILL) in events
    assert events[-1] == "reap"
