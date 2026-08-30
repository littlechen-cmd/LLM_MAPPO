"""Run, resume, and aggregate the complete frozen O2 matrix in one owner job."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from llm_mappo.o2_contract import (
    O2ExperimentConfig,
    O2RunSpec,
    expand_o2_matrix,
    verify_o1_authorization,
)
from llm_mappo.o2_device import device_provenance


ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUNNER = ROOT / "scripts" / "run_o2_calibration.py"
ANALYZER = ROOT / "scripts" / "analyze_o2_calibration.py"


def matrix_specs(config: O2ExperimentConfig) -> tuple[O2RunSpec, ...]:
    """Return the frozen execution order for the one owner job."""
    return expand_o2_matrix(config)


def build_run_command(
    *,
    python_executable: str,
    config_path: str | Path,
    o1_run: str | Path,
    output_root: str | Path,
    run_spec: O2RunSpec,
    device: str,
    resume_directory: str | Path | None = None,
) -> list[str]:
    """Build one isolated child command without diagnostic overrides."""
    command = [
        python_executable,
        str(SINGLE_RUNNER),
        "--config",
        str(config_path),
        "--o1-run",
        str(o1_run),
        "--output-root",
        str(output_root),
        "--run",
        f"{run_spec.group}:{run_spec.seed}",
        "--device",
        device,
    ]
    if resume_directory is not None:
        command.extend(("--resume", str(resume_directory)))
    return command


def resolve_run_action(
    matrix_root: str | Path, expected_identity: Mapping[str, Any]
) -> tuple[str, Path | None]:
    """Select start, skip, or exact checkpoint resume for one formal member."""
    root = Path(matrix_root)
    if not root.exists():
        return "start", None
    matches = _matching_runs(root, expected_identity)
    if len(matches) > 1:
        raise RuntimeError("O2 matrix contains multiple matching formal runs.")
    if not matches:
        return "start", None
    return _action_for_state(matches[0])


def _matching_runs(
    root: Path, expected_identity: Mapping[str, Any]
) -> list[Path]:
    matches = []
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    for directory in directories:
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Unreadable O2 run manifest: {directory}") from error
        if (
            manifest.get("identity") == dict(expected_identity)
            and manifest.get("diagnostic_only") is False
        ):
            matches.append(directory)
    return matches


def _action_for_state(directory: Path) -> tuple[str, Path]:
    try:
        state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unreadable O2 run state: {directory}") from error
    status = state.get("status")
    if status == "complete":
        if not (directory / "checkpoint_final.pt").is_file():
            raise RuntimeError("Completed O2 run is missing its final checkpoint.")
        return "skip", directory
    if status == "running":
        if not (directory / "checkpoint_latest.pt").is_file():
            raise RuntimeError(
                "Interrupted O2 run has no resumable update-boundary checkpoint."
            )
        return "resume", directory
    if status == "failed":
        raise RuntimeError("Matching formal O2 run is failed and cannot auto-resume.")
    raise RuntimeError(f"Unsupported O2 run status: {status!r}.")


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Frozen O2 YAML contract.")
    parser.add_argument("--o1-run", required=True, help="Completed receipted O1 run.")
    parser.add_argument("--output-root", required=True, help="O2 artifact root.")
    parser.add_argument("--device", default="cuda:0", help="Torch logical device.")
    return parser.parse_args(argv)


def _code_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


@contextmanager
def _matrix_lock(path: Path):
    """Hold one cross-platform advisory lock for the complete matrix process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            handle.close()
            message = "Another O2 matrix owner job is already running."
            raise RuntimeError(message) from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            handle.close()
    else:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            message = "Another O2 matrix owner job is already running."
            raise RuntimeError(message) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _run_identity(
    *,
    code_commit: str,
    config: O2ExperimentConfig,
    o1: Mapping[str, Any],
    run_spec: O2RunSpec,
) -> dict[str, Any]:
    return {
        "code_commit": code_commit,
        "config_sha256": config.sha256(),
        "seed": run_spec.seed,
        "group": run_spec.group,
        "o1_code_commit": o1["code_commit"],
        "o1_summary_sha256": o1["summary_sha256"],
    }


def _matrix_manifest(
    *,
    code_commit: str,
    config: O2ExperimentConfig,
    o1: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "o2-matrix-manifest-v1",
        "code_commit": code_commit,
        "config_sha256": config.sha256(),
        "o1_code_commit": o1["code_commit"],
        "o1_summary_sha256": o1["summary_sha256"],
        "device": dict(provenance),
        "runs": [
            {"group": item.group, "seed": item.seed}
            for item in matrix_specs(config)
        ],
    }


def _prepare_matrix_root(root: Path, manifest: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "matrix_manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(manifest):
            raise RuntimeError("Existing O2 matrix manifest does not match this job.")
    else:
        _atomic_json(path, manifest)


def run_matrix(arguments: argparse.Namespace) -> dict[str, Any]:  # noqa: C901
    """Run all six isolated children and aggregate the formal Gate evidence."""
    if arguments.device != "cuda:0":
        raise ValueError("Formal O2 matrix device must be logical cuda:0.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("Formal O2 matrix requires CUDA_VISIBLE_DEVICES=0.")
    provenance = device_provenance(arguments.device)
    config_path = Path(arguments.config).resolve()
    o1_run = Path(arguments.o1_run).resolve()
    output_root = Path(arguments.output_root).resolve()
    config = O2ExperimentConfig.from_yaml(config_path)
    o1 = verify_o1_authorization(o1_run)
    code_commit = _code_commit()
    matrix_root = output_root / (
        f"formal_{code_commit[:8]}_{config.sha256()[:8]}"
    )
    manifest = _matrix_manifest(
        code_commit=code_commit,
        config=config,
        o1=o1,
        provenance=provenance,
    )
    matrix_root.mkdir(parents=True, exist_ok=True)
    state_path = matrix_root / "matrix_state.json"

    with _matrix_lock(matrix_root / "matrix.lock"):
        _prepare_matrix_root(matrix_root, manifest)
        completed: list[str] = []
        try:
            for run_spec in matrix_specs(config):
                label = f"{run_spec.group}:{run_spec.seed}"
                identity = _run_identity(
                    code_commit=code_commit,
                    config=config,
                    o1=o1,
                    run_spec=run_spec,
                )
                action, directory = resolve_run_action(matrix_root, identity)
                _atomic_json(
                    state_path,
                    {
                        "schema": "o2-matrix-state-v1",
                        "status": "running",
                        "active_run": label,
                        "active_action": action,
                        "completed_runs": completed,
                    },
                )
                if action != "skip":
                    command = build_run_command(
                        python_executable=sys.executable,
                        config_path=config_path,
                        o1_run=o1_run,
                        output_root=matrix_root,
                        run_spec=run_spec,
                        device=arguments.device,
                        resume_directory=directory if action == "resume" else None,
                    )
                    result = subprocess.run(command, cwd=ROOT, check=False)
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"O2 child {label} exited with {result.returncode}."
                        )
                    action, directory = resolve_run_action(matrix_root, identity)
                    if action != "skip" or directory is None:
                        raise RuntimeError(f"O2 child {label} did not complete cleanly.")
                completed.append(label)

            gate_path = matrix_root / "o2_gate_summary.json"
            analysis = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    "--config",
                    str(config_path),
                    "--runs-root",
                    str(matrix_root),
                    "--output",
                    str(gate_path),
                ],
                cwd=ROOT,
                check=False,
            )
            if analysis.returncode not in (0, 2) or not gate_path.is_file():
                raise RuntimeError(
                    "O2 aggregate analyzer did not produce a Gate result."
                )
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            state = {
                "schema": "o2-matrix-state-v1",
                "status": "complete",
                "gate_pass": bool(gate.get("gate_pass")),
                "completed_runs": completed,
                "gate_summary": str(gate_path),
            }
            _atomic_json(state_path, state)
            return {"matrix_directory": str(matrix_root), **state}
        except Exception as error:
            _atomic_json(
                state_path,
                {
                    "schema": "o2-matrix-state-v1",
                    "status": "failed",
                    "completed_runs": completed,
                    "reason": type(error).__name__,
                    "detail": str(error),
                },
            )
            raise


def main(argv: Sequence[str] | None = None) -> None:
    result = run_matrix(_arguments(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
