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


def test_o2_manifest_device_provenance_is_explicit_for_cpu_smoke():
    from scripts.run_o2_calibration import device_provenance

    assert device_provenance("cpu") == {
        "logical_device": "cpu",
        "cuda_available": False,
        "device_name": None,
        "torch": __import__("torch").__version__,
    }
