"""Prepare approved per-seed training configs; report missing LLM dependencies."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import yaml

from llm_mappo.formal_training import (
    TRAIN_SEEDS, authenticate_scene_representation,
)
from llm_mappo.mappo_astar_kd.teacher import TEACHER_VERSION
from llm_mappo.mappo_baseline.runner import load_baseline_experiment


def prepare_configs(  # noqa: C901
    output_directory: Path, *, template_directory: Path,
    representation_directory: Path, seeds: list[int],
    dataset_directory: Path | None = None, label_directory: Path | None = None,
) -> dict:
    """Emit ready configs only; never manufacture artifacts for pending methods."""

    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= set(TRAIN_SEEDS):
        raise ValueError("Use a unique subset of the eight registered training seeds.")
    if (dataset_directory is None) != (label_directory is None):
        raise ValueError("Provide dataset and label directories together.")
    torch.set_num_threads(4)
    templates = sorted(template_directory.glob("*.yaml"))
    if len(templates) != 12:
        raise ValueError("Expected exactly twelve three-scene method templates.")
    output_directory.mkdir(parents=True, exist_ok=True)
    representations = {}
    for seed in seeds:
        artifact = representation_directory / f"seed_{seed:04d}" / "representation.pt"
        if not artifact.exists():
            continue
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        values = {
            "schema": "mappo-representation-initialization-v1",
            "artifact_path": str(artifact.resolve()),
            "artifact_sha256": sha256(artifact.read_bytes()).hexdigest(),
            "identity": payload["identity"],
        }
        representations[seed] = authenticate_scene_representation(values)
    runs = []
    for template in templates:
        load_baseline_experiment(template)
        for seed in seeds:
            values = yaml.safe_load(template.read_text(encoding="utf-8"))
            method = values["method"]
            scene = values["environment"]["scene_id"]
            destination = output_directory / f"seed_{seed:04d}" / template.name
            row = {
                "scene_id": scene, "method": method, "seed": seed,
                "environment_steps": values["training"]["environment_steps"],
                "template_sha256": sha256(template.read_bytes()).hexdigest(),
                "status": "ready", "config_path": str(destination.resolve()),
                "astar_teacher_version": (
                    TEACHER_VERSION if "ASTARKD" in method else None
                ),
            }
            if method.endswith("LLM-INIT") and seed not in representations:
                row.update({
                    "status": "dependency_missing", "config_path": None,
                    "required_artifact": str((
                        representation_directory / f"seed_{seed:04d}"
                        / "representation.pt"
                    ).resolve()),
                })
                # A prior generated config must not outlive its removed dependency.
                if destination.exists():
                    destination.unlink()
                runs.append(row)
                continue
            values["approval_status"] = "approved"
            values["training"]["seed"] = seed
            if "ASTARKD" in method:
                values["training"].update({
                    "astar_teacher_approval": "approved",
                    "astar_teacher_version": TEACHER_VERSION,
                })
            if method.endswith("LLM-INIT"):
                values["training"]["representation_initialization"] = (
                    representations[seed]
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
                encoding="utf-8", newline="\n",
            )
            experiment = load_baseline_experiment(destination)
            row["config_sha256"] = experiment.config_sha256
            runs.append(row)
    manifest = {
        "schema": "three-scene-approved-config-preparation-v1",
        "astar_teacher_version": TEACHER_VERSION,
        "astar_teacher_variant": "no_validity",
        "astar_teacher_semantics": (
            "Semantic validity checks disabled; technical invalid labels remain "
            "excluded by existing validity/scope masks."
        ),
        "runs": runs,
        "pretraining_runs": [],
    }
    if dataset_directory is not None:
        from llm_mappo.mappo_llm_kd.pretraining_run import load_pretraining_run_config

        for seed in seeds:
            destination = output_directory / "pretraining" / f"seed_{seed:04d}.yaml"
            values = {
                "schema": "semantic-pretraining-runner-v1",
                "dataset_directory": str(dataset_directory.resolve()),
                "label_directory": str(label_directory.resolve()),
                "artifact_root": str(representation_directory.resolve()),
                "approval_status": "approved", "seed": seed,
                "batch_size": 256, "max_epochs": 100, "learning_rate": 0.001,
                "dimension_weights": [1.0, 1.0, 1.0, 1.0],
                "early_stopping_patience": 10, "early_stopping_min_delta": 0.0001,
            }
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                yaml.safe_dump(values, sort_keys=False), encoding="utf-8", newline="\n"
            )
            load_pretraining_run_config(destination)
            manifest["pretraining_runs"].append({
                "seed": seed, "run_id": f"seed_{seed:04d}",
                "config_path": str(destination.resolve()),
            })
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-directory", type=Path,
                        default=Path("configs/formal_environment_training"))
    parser.add_argument(
        "--output-directory", type=Path,
        default=Path("artifacts/formal_environment_training/approved_configs"),
    )
    parser.add_argument(
        "--representation-directory", type=Path,
        default=Path("artifacts/formal_environment_training/representations"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    parser.add_argument("--dataset-directory", type=Path)
    parser.add_argument("--label-directory", type=Path)
    arguments = parser.parse_args()
    manifest = prepare_configs(
        arguments.output_directory, template_directory=arguments.template_directory,
        representation_directory=arguments.representation_directory,
        seeds=arguments.seeds, dataset_directory=arguments.dataset_directory,
        label_directory=arguments.label_directory,
    )
    ready = sum(run["status"] == "ready" for run in manifest["runs"])
    print(json.dumps({
        "ready": ready, "dependency_missing": len(manifest["runs"]) - ready,
        "manifest": str((arguments.output_directory / "manifest.json").resolve()),
    }))


if __name__ == "__main__":
    main()
