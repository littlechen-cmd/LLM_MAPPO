"""The owner-facing O2 entry point must expose only the frozen modes."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_o2_cli_exposes_receipted_run_and_diagnostic_smoke_modes():
    result = subprocess.run(
        [sys.executable, "scripts/run_o2_calibration.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--o1-run" in result.stdout
    assert "--smoke-steps" in result.stdout
