"""Read-only verification for the frozen P1 Linux Python/CUDA environment."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class PackageProbe:
    """The version returned by a package metadata probe."""

    name: str
    version: str


@dataclass(frozen=True)
class EnvironmentContract:
    """Versions and capabilities required by the approved P1 contract."""

    python_version: str
    torch_version: str
    packages: Mapping[str, str]


@dataclass(frozen=True)
class EnvironmentReport:
    """Serializable outcome of one read-only environment inspection."""

    python: str
    torch: str
    torch_cuda: Optional[str]
    packages: Mapping[str, str]
    editable_project: bool
    freeze_sha256: str
    passed: bool
    failures: Tuple[str, ...]

    def to_dict(self) -> Mapping[str, object]:
        """Return JSON-safe fields without creating or changing an environment."""
        payload = asdict(self)
        payload["pass"] = payload.pop("passed")
        return payload


def default_contract() -> EnvironmentContract:
    """Return the exact P1 Linux package contract."""
    return EnvironmentContract(
        python_version="3.10.19",
        torch_version="2.10.0+cu128",
        packages={
            "numpy": "2.2.6",
            "scipy": "1.15.3",
            "gymnasium": "1.2.3",
            "PyYAML": "6.0.3",
            "psutil": "7.2.2",
            "matplotlib": "3.10.9",
            "networkx": "3.4.2",
            "pyglet": "1.5.31",
            "pytest": "9.0.2",
            "flake8": "7.3.0",
            "tensorboard": "2.21.0",
            "build": "1.5.0",
            "Pillow": "12.1.1",
        },
    )


def _freeze_sha256(lines: Iterable[str]) -> str:
    canonical = "\n".join(sorted(line.strip() for line in lines if line.strip()))
    return hashlib.sha256((canonical + "\n").encode("utf-8")).hexdigest()


def _safe_package_version(
    package_probe: Callable[[str], PackageProbe], name: str
) -> str:
    try:
        return package_probe(name).version
    except Exception:
        return "unavailable"


def _verify_packages(
    contract: EnvironmentContract, package_probe: Callable[[str], PackageProbe]
) -> Tuple[str, Mapping[str, str], Tuple[str, ...]]:
    torch = _safe_package_version(package_probe, "torch")
    failures = []
    if torch != contract.torch_version:
        failures.append("torch_version")

    packages = {}
    for name, expected in contract.packages.items():
        found = _safe_package_version(package_probe, name)
        packages[name] = found
        if found != expected:
            failures.append("package:" + name)
    return torch, packages, tuple(failures)


def _safe_torch_probe(
    torch_probe: Callable[[], Tuple[bool, Optional[str]]]
) -> Tuple[bool, Optional[str]]:
    try:
        return torch_probe()
    except Exception:
        return False, None


def _safe_editable_project_probe(editable_project_probe: Callable[[], bool]) -> bool:
    try:
        return bool(editable_project_probe())
    except Exception:
        return False


def verify_environment(
    contract: EnvironmentContract,
    *,
    python_version: str,
    package_probe: Callable[[str], PackageProbe],
    torch_probe: Callable[[], Tuple[bool, Optional[str]]],
    editable_project_probe: Callable[[], bool],
    freeze_lines: Iterable[str],
) -> EnvironmentReport:
    """Fail closed when any required P1 environment property is unavailable."""
    failures = []

    if python_version != contract.python_version:
        failures.append("python_version")

    torch, packages, package_failures = _verify_packages(contract, package_probe)
    failures.extend(package_failures)

    cuda_available, torch_cuda = _safe_torch_probe(torch_probe)
    if not cuda_available:
        failures.append("cuda_available")

    editable_project = _safe_editable_project_probe(editable_project_probe)
    if not editable_project:
        failures.append("editable_project")

    return EnvironmentReport(
        python=python_version,
        torch=torch,
        torch_cuda=torch_cuda,
        packages=packages,
        editable_project=editable_project,
        freeze_sha256=_freeze_sha256(freeze_lines),
        passed=not failures,
        failures=tuple(failures),
    )


def _metadata_probe(name: str) -> PackageProbe:
    return PackageProbe(name=name, version=importlib.metadata.version(name))


def _torch_probe() -> Tuple[bool, Optional[str]]:
    import torch

    return bool(torch.cuda.is_available()), torch.version.cuda


def _editable_project_probe() -> bool:
    return (
        importlib.util.find_spec("llm_mappo") is not None
        and importlib.util.find_spec("rware") is not None
    )


def _pip_freeze() -> Tuple[str, ...]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(result.stdout.splitlines())


def _constraints_names(path: Path) -> Tuple[str, ...]:
    return tuple(
        line.split("==", 1)[0].strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--write-freeze", type=Path)
    args = parser.parse_args()

    contract = default_contract()
    constraint_names = set(_constraints_names(args.constraints))
    required_names = {"torch", *(name.lower() for name in contract.packages)}
    if constraint_names != required_names:
        raise SystemExit("constraint file does not match the frozen P1 contract")

    freeze_lines = _pip_freeze()
    report = verify_environment(
        contract,
        python_version=platform.python_version(),
        package_probe=_metadata_probe,
        torch_probe=_torch_probe,
        editable_project_probe=_editable_project_probe,
        freeze_lines=freeze_lines,
    )
    payload = report.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.report is not None:
        _write_json(args.report, payload)
    if args.write_freeze is not None:
        args.write_freeze.parent.mkdir(parents=True, exist_ok=True)
        args.write_freeze.write_text("\n".join(freeze_lines) + "\n", encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
