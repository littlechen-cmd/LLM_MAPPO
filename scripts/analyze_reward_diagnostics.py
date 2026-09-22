"""Create a deterministic, read-only Group 2 reward-diagnostic report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_mappo.reward_diagnostic_analysis import build_diagnostic_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        action="append",
        required=True,
        help="Completed diagnostic run directory; repeat once per training seed.",
    )
    parser.add_argument(
        "--output-directory",
        required=True,
        help="Report-only directory outside every input run directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_directories = [Path(value).resolve() for value in args.run_directory]
    output_directory = Path(args.output_directory).resolve()
    for directory in run_directories:
        try:
            output_directory.relative_to(directory)
        except ValueError:
            continue
        raise ValueError("Analysis output directory must be outside every input run.")
    report = build_diagnostic_report(run_directories)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "reward_diagnostic_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_directory / "reward_diagnostic_report.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )


def _markdown_report(report: dict) -> str:
    lines = [
        "# Reward diagnostic report",
        "",
        f"Evidence completeness: `{report['evidence_complete']}`",
        "",
        "This report is descriptive evidence, not a performance claim.",
    ]
    for run in report["runs"]:
        identity = run["identity"]
        lines.extend(
            [
                "",
                f"## Training seed {identity['seed']}",
                "",
                f"- Artifact: `{run['artifact_path']}`",
                f"- Source commit: `{identity.get('code_commit')}`",
                f"- Config SHA-256: `{identity.get('config_sha256')}`",
                "- Reward-definition SHA-256: "
                f"`{identity.get('reward_definition_sha256')}`",
                f"- Evaluation seeds: `{run['evaluation_seeds']}`",
                f"- Final checkpoint SHA-256: `{run['checkpoint_sha256']}`",
                "- Episodes / updates: "
                f"`{run['episode_count']} / {run['update_count']}`",
                f"- Total environment steps: `{run['total_environment_steps']}`",
                f"- Completed tasks: `{run['completed_tasks']}`",
                "- Trend window: "
                f"`{run['trend']['window_size']}` "
                f"(reduced: `{run['trend']['reduced_window']}`)",
                "",
                "### Event rates",
                "",
                "| Event | Total | Per 1,000 steps | Per completed task |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, values in run["events"].items():
            lines.append(
                "| "
                f"{name} | {values['total']} | "
                f"{values['per_1000_environment_steps']} | "
                f"{values['per_completed_task']} |"
            )
        lines.extend(
            [
                "",
                "### Reward ledger",
                "",
                "| Component | Total | Per 1,000 steps | Per completed task |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, values in run["ledger"].items():
            lines.append(
                "| "
                f"{name} | {values['total']} | "
                f"{values['per_1000_environment_steps']} | "
                f"{values['per_completed_task']} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
