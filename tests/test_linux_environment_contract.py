import sys

from scripts import verify_linux_environment
from scripts.verify_linux_environment import (
    EnvironmentContract,
    PackageProbe,
    verify_environment,
)


def _contract():
    return EnvironmentContract(
        python_version="3.10.19",
        torch_version="2.10.0+cu128",
        packages={"numpy": "2.2.6", "pytest": "9.0.2"},
    )


def test_verify_environment_accepts_exact_linux_contract():
    report = verify_environment(
        _contract(),
        python_version="3.10.19",
        package_probe=lambda name: PackageProbe(name, {
            "torch": "2.10.0+cu128",
            "numpy": "2.2.6",
            "pytest": "9.0.2",
        }[name]),
        torch_probe=lambda: (True, "12.8"),
        editable_project_probe=lambda: True,
        freeze_lines=["numpy==2.2.6", "pytest==9.0.2"],
    )

    assert report.passed is True
    assert report.torch_cuda == "12.8"
    assert report.editable_project is True
    assert len(report.freeze_sha256) == 64
    assert report.to_dict()["pass"] is True
    assert "passed" not in report.to_dict()


def test_verify_environment_fails_closed_for_wrong_version_or_cuda():
    report = verify_environment(
        _contract(),
        python_version="3.10.18",
        package_probe=lambda name: PackageProbe(name, "missing"),
        torch_probe=lambda: (False, None),
        editable_project_probe=lambda: False,
        freeze_lines=[],
    )

    assert report.passed is False
    assert "python_version" in report.failures
    assert "torch_version" in report.failures
    assert "cuda_available" in report.failures
    assert "editable_project" in report.failures


def test_pip_freeze_invokes_the_active_interpreter(monkeypatch):
    calls = []

    class Result:
        stdout = "numpy==2.2.6\n"

    monkeypatch.setattr(
        verify_linux_environment.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or Result(),
    )

    assert verify_linux_environment._pip_freeze() == ("numpy==2.2.6",)
    assert calls == [
        (
            [sys.executable, "-m", "pip", "freeze"],
            {"check": True, "capture_output": True, "text": True},
        )
    ]
