"""The owner must be able to launch and resume the whole O2 matrix once."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _identity(group="MAPPO-DG", seed=107):
    return {
        "code_commit": "abc123",
        "config_sha256": "config",
        "seed": seed,
        "group": group,
        "o1_code_commit": "o1-commit",
        "o1_summary_sha256": "o1-summary",
    }


def _run_artifact(root, name, identity, status, checkpoint=None):
    directory = root / name
    directory.mkdir()
    (directory / "run_manifest.json").write_text(
        json.dumps({"identity": identity, "diagnostic_only": False}),
        encoding="utf-8",
    )
    (directory / "state.json").write_text(
        json.dumps({"status": status}), encoding="utf-8"
    )
    if checkpoint:
        (directory / checkpoint).touch()
    return directory


def test_o2_matrix_cli_is_one_owner_entry_point():
    result = subprocess.run(
        [sys.executable, "scripts/run_o2_matrix.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--o1-run" in result.stdout
    assert "--output-root" in result.stdout
    assert "--device" in result.stdout
    assert "--run" not in result.stdout
    assert "--smoke-steps" not in result.stdout


def test_o2_matrix_starts_skips_and_resumes_matching_formal_runs(tmp_path):
    from scripts.run_o2_matrix import resolve_run_action

    identity = _identity()
    assert resolve_run_action(tmp_path, identity) == ("start", None)

    complete = _run_artifact(
        tmp_path, "complete", identity, "complete", "checkpoint_final.pt"
    )
    assert resolve_run_action(tmp_path, identity) == ("skip", complete)

    (complete / "run_manifest.json").unlink()
    running = _run_artifact(
        tmp_path, "running", identity, "running", "checkpoint_latest.pt"
    )
    assert resolve_run_action(tmp_path, identity) == ("resume", running)


def test_o2_matrix_fails_closed_on_failed_or_ambiguous_artifacts(tmp_path):
    from scripts.run_o2_matrix import resolve_run_action

    identity = _identity()
    _run_artifact(tmp_path, "failed", identity, "failed")
    with pytest.raises(RuntimeError, match="failed"):
        resolve_run_action(tmp_path, identity)

    (tmp_path / "failed" / "run_manifest.json").unlink()
    _run_artifact(tmp_path, "first", identity, "complete", "checkpoint_final.pt")
    _run_artifact(tmp_path, "second", identity, "complete", "checkpoint_final.pt")
    with pytest.raises(RuntimeError, match="multiple"):
        resolve_run_action(tmp_path, identity)


def test_o2_matrix_builds_isolated_child_commands_in_frozen_order(tmp_path):
    from llm_mappo.o2_contract import O2ExperimentConfig
    from scripts.run_o2_matrix import build_run_command, matrix_specs

    config_path = ROOT / "configs/optimization/o2_calibration.yaml"
    config = O2ExperimentConfig.from_yaml(config_path)
    specs = matrix_specs(config)

    assert [(spec.group, spec.seed) for spec in specs] == [
        ("MAPPO-DG", 107),
        ("MAPPO-DG", 117),
        ("MAPPO-DG", 127),
        ("RC-AStarKD", 107),
        ("RC-AStarKD", 117),
        ("RC-AStarKD", 127),
    ]
    command = build_run_command(
        python_executable="/canonical/python",
        config_path=config_path,
        o1_run=tmp_path / "o1",
        output_root=tmp_path / "matrix",
        run_spec=specs[-1],
        device="cuda:0",
        resume_directory=tmp_path / "resume",
    )
    assert command[0] == "/canonical/python"
    assert Path(command[1]).name == "run_o2_calibration.py"
    assert command[command.index("--run") + 1] == "RC-AStarKD:127"
    assert command[command.index("--resume") + 1] == str(tmp_path / "resume")
    assert "--smoke-steps" not in command
