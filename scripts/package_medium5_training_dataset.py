"""Package validated medium-5ag training windows and both semantic teachers."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from llm_mappo.mappo_llm_kd.pretraining_run import (  # noqa: E402
    _attach_scores, _load_eligible_windows, _load_scores,
    _validate_label_artifact, _window_id,
)


PARTITION_COUNTS = {"train": 4200, "validation": 900, "test": 900}
_BUNDLE_MANIFEST = "medium5-training-bundle-manifest.json"
_LAUNCH_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".yaml", ".yml"}
_SECRET_NAMES = {".env", "credentials.json", "secrets.json", "token.json"}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sources(dataset, labels, rules):  # noqa: C901
    windows = _load_eligible_windows(dataset)
    if len(windows) != 6000 or Counter(
        window["partition"] for window in windows
    ) != PARTITION_COUNTS:
        raise ValueError("Dataset must contain exactly 4200/900/900 training windows.")
    if len({_window_id(window) for window in windows}) != 6000:
        raise ValueError("Dataset window IDs must be unique.")
    _validate_label_artifact(dataset, labels)
    label_manifest = _read_json(labels / "manifest.json")
    if (
        label_manifest.get("schema") != "llm-dataset-label-run-v1"
        or label_manifest.get("stage") != "full"
        or label_manifest.get("pretraining_eligible") is not True
        or label_manifest.get("quality_metrics", {}).get("go") is not True
        or label_manifest.get("partition_counts") != PARTITION_COUNTS
        or label_manifest.get("successful_records") != 6000
    ):
        raise ValueError("LLM labels must be completed, full, eligible, and quality Go.")
    if label_manifest.get("attempts_sha256") != _digest(labels / "attempts.jsonl"):
        raise ValueError("LLM attempt log differs from the final manifest.")
    if _read_json(labels / "full-metrics.json") != label_manifest["quality_metrics"]:
        raise ValueError("Final quality metrics differ from the full manifest.")
    _validate_label_artifact(dataset, rules)
    rule_manifest = _read_json(rules / "manifest.json")
    if (
        rule_manifest.get("teacher_type") != "deterministic-observable-rule-v1"
        or rule_manifest.get("successful_records") != 6000
        or rule_manifest.get("label_counts") != PARTITION_COUNTS
        or rule_manifest.get("pretraining_eligible") is not True
    ):
        raise ValueError("Rule comparator must contain all 6000 same-window labels.")
    for directory in (labels, rules):
        scores = _load_scores(directory / "labels.jsonl")
        if len(scores) != 6000 or len(_attach_scores(windows, scores)) != 6000:
            raise ValueError("Teacher scores must join exactly 6000 dataset windows.")
    dataset_manifest = _read_json(dataset / "manifest.json")
    if label_manifest.get("windows_sha256") != dataset_manifest.get("windows_sha256"):
        raise ValueError("LLM label manifest is bound to different windows.")
    if rule_manifest.get("source_windows_sha256") != dataset_manifest.get(
        "windows_sha256"
    ):
        raise ValueError("Rule comparator is bound to different windows.")
    identity = _read_json(labels / "identity.json")
    for key in ("dataset_manifest_sha256", "windows_sha256", "label_config_sha256",
                "semantic_contract_version"):
        if identity.get(key) != label_manifest.get(key):
            raise ValueError("LLM identity and final manifest differ.")
    return dataset_manifest, label_manifest, rule_manifest, identity


def _member_name(path):
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ValueError("Bundle members must be regular source files.")
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError("Bundle sources must be inside this repository.") from error
    if resolved.name.lower() in _SECRET_NAMES:
        raise ValueError("A credential file cannot be a bundle member.")
    if any(part in {".git", "__pycache__"} for part in relative.parts):
        raise ValueError("Repository internals cannot be bundle members.")
    return relative.as_posix()


def _collect_members(dataset, labels, rules, delivery_readme):  # noqa: C901
    members = {}

    def add(path):
        members[_member_name(path)] = path

    for path in dataset.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            add(path)
    episodes = dataset / "episodes"
    if episodes.exists():
        for path in episodes.rglob("*.json"):
            add(path)
    for path in labels.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            add(path)
    for path in rules.rglob("*"):
        if path.is_file():
            if path.suffix.lower() in {".npz", ".npy"}:
                raise ValueError("Rule directories must not contain raw rollout arrays.")
            add(path)
    parent_dataset = dataset.parent / "main-c6d2df435f63"
    if not parent_dataset.is_dir():
        raise ValueError("Frozen parent dataset is required for budget verification.")
    for filename in ("manifest.json", "state.json", "windows.jsonl"):
        add(parent_dataset / filename)
    for path in parent_dataset.iterdir():
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            add(path)
    for path in (parent_dataset / "episodes").rglob("*.json"):
        add(path)
    for path in (parent_dataset / "labels").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
            add(path)
    for namespace in ("326af258ba60", "6bb5154b6308"):
        old_labels = dataset / "labels" / namespace / "deepseek-official-flash"
        if not old_labels.is_dir():
            raise ValueError("Quarantined label lineage is required for verification.")
        for filename in ("identity.json", "attempts.jsonl"):
            add(old_labels / filename)
        for path in old_labels.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}:
                add(path)
    label_root = dataset / "labels"
    for path in label_root.iterdir():
        name = path.name.lower()
        if path.is_file() and path.suffix.lower() == ".json" and (
            "authorization" in name or name.startswith("launch-")
        ):
            add(path)
        elif path.is_dir() and name.startswith("launch-code-"):
            for source in path.rglob("*"):
                if source.is_file() and source.suffix.lower() in _LAUNCH_SUFFIXES:
                    add(source)
        elif path.is_file() and name.startswith("launch-code-") and (
            name.endswith((".zip", ".tar", ".tar.gz", ".tgz"))
        ):
            add(path)
    for path in (REPOSITORY_ROOT / "configs" / "llm_kd").glob(
        "*medium-5ag-dataset-labeling*.yaml"
    ):
        add(path)
    add(REPOSITORY_ROOT / "specs" / "2026-09-30-medium-5ag-llm-dataset"
        / "observable-coordination-v2.md")
    for filename in ("pretraining.py", "pretraining_run.py", "medium_prompt.py"):
        add(REPOSITORY_ROOT / "llm_mappo" / "mappo_llm_kd" / filename)
    add(REPOSITORY_ROOT / "scripts" / "package_medium5_training_dataset.py")
    if delivery_readme is not None:
        add(delivery_readme)
    return dict(sorted(members.items()))


def package_training_dataset(dataset, labels, rules, output_zip, delivery_readme=None):
    """Validate before writing; create a new ZIP without modifying source files."""

    dataset, labels, rules, output_zip = (
        Path(path).resolve() for path in (dataset, labels, rules, output_zip)
    )
    if output_zip.exists():
        raise FileExistsError("Output ZIP already exists; choose a new output path.")
    sources = _validate_sources(dataset, labels, rules)
    members = _collect_members(dataset, labels, rules, delivery_readme)
    for directory in (dataset, labels, rules):
        for filename in ("manifest.json", "state.json"):
            if _member_name(directory / filename) not in members:
                raise ValueError("Bundle source manifests and states are required.")
    hashes = {name: _digest(path) for name, path in members.items()}
    source_manifest, label_manifest, rule_manifest, identity = sources
    manifest = {
        "schema": "medium5-training-bundle-v1",
        "partition_counts": PARTITION_COUNTS,
        "training_windows": 6000,
        "llm_teacher_records": 6000,
        "rule_comparator_records": 6000,
        "dataset_directory": _member_name(dataset / "manifest.json").rsplit("/", 1)[0],
        "label_directory": _member_name(labels / "manifest.json").rsplit("/", 1)[0],
        "rule_directory": _member_name(rules / "manifest.json").rsplit("/", 1)[0],
        "source_identities": {
            "dataset_manifest_sha256": _digest(dataset / "manifest.json"),
            "windows_sha256": source_manifest["windows_sha256"],
            "llm_identity": identity,
            "llm_manifest_sha256": _digest(labels / "manifest.json"),
            "llm_labels_sha256": label_manifest["labels_sha256"],
            "rule_manifest_sha256": _digest(rules / "manifest.json"),
            "rule_labels_sha256": rule_manifest["labels_sha256"],
            "rule_code_sha256": rule_manifest.get("rule_code_sha256"),
        },
        "rule_comparator_is_gold": False,
        "kd_performance_evidence": False,
        "confirmed_global_deadlock_truth": False,
        "raw_episode_rollouts_included": False,
        "prior_label_audit_lineage_included": True,
        "failed_quarantined_versions_are_provenance_only": True,
        "qualified_training_label_directory": (
            _member_name(labels / "manifest.json").rsplit("/", 1)[0]
        ),
        "prior_lineage_directories": [
            str((dataset.parent / "main-c6d2df435f63").relative_to(REPOSITORY_ROOT)
                .as_posix()),
            str((dataset / "labels" / "326af258ba60").relative_to(REPOSITORY_ROOT)
                .as_posix()),
            str((dataset / "labels" / "6bb5154b6308").relative_to(REPOSITORY_ROOT)
                .as_posix()),
        ],
        "prior_budget_chain_location": (
            _member_name(labels / "attempts.jsonl")
        ),
        "bundle_manifest_excluded_from_member_hashes": True,
        "members": [
            {"path": name, "sha256": hashes[name], "size_bytes": path.stat().st_size}
            for name, path in members.items()
        ],
    }
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    target = output_zip.open("xb")
    try:
        with target, zipfile.ZipFile(
            target, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=6, allowZip64=True,
        ) as archive:
            for name, path in members.items():
                digest = sha256()
                with path.open("rb") as source, archive.open(
                    name, "w", force_zip64=True,
                ) as destination:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        destination.write(chunk)
                if digest.hexdigest() != hashes[name]:
                    raise ValueError("A source file changed while packaging.")
            archive.writestr(_BUNDLE_MANIFEST, json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2,
            ) + "\n")
    except BaseException:
        output_zip.unlink(missing_ok=True)
        raise
    return {
        "output_zip": str(output_zip),
        "zip_sha256": _digest(output_zip),
        "size_bytes": output_zip.stat().st_size,
        "member_count": len(members) + 1,
        "partition_counts": PARTITION_COUNTS,
        "dataset_directory": str(dataset),
        "label_directory": str(labels),
        "rule_directory": str(rules),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-directory", required=True, type=Path)
    parser.add_argument("--label-directory", required=True, type=Path)
    parser.add_argument("--rule-directory", required=True, type=Path)
    parser.add_argument("--output-zip", required=True, type=Path)
    parser.add_argument("--delivery-readme", type=Path)
    args = parser.parse_args()
    result = package_training_dataset(
        args.dataset_directory, args.label_directory, args.rule_directory,
        args.output_zip, args.delivery_readme,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
