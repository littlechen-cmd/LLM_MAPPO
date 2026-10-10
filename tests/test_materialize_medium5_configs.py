"""Seed-specific materialization must preserve owner approval status."""

from pathlib import Path
import subprocess
import sys

from llm_mappo.mappo_baseline.runner import load_baseline_experiment


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_medium5_seed_configs.py"
BASELINE = ROOT / "configs" / "core_integration" / "medium-5ag-baseline.yaml"


def test_materializer_creates_three_valid_approved_seed_configs(tmp_path):
    """Catch generated variants changing the reviewed approval or contract."""

    output = tmp_path / "seed-configs"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(BASELINE),
            "--output-directory",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    paths = sorted(output.glob("*.yaml"))
    assert len(paths) == 3
    experiments = [load_baseline_experiment(path) for path in paths]
    assert [item.training["seed"] for item in experiments] == [7, 17, 27]
    assert {item.approval_status for item in experiments} == {"approved"}
    assert len({item.config_sha256 for item in experiments}) == 3


def test_materializer_refuses_to_overwrite_an_existing_directory(tmp_path):
    """Catch a repeated materialization replacing a reviewed seed config."""

    output = tmp_path / "seed-configs"
    output.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(BASELINE),
            "--output-directory",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not list(output.iterdir())
