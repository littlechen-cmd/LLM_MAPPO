from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from llm_mappo.e1_protocol import (
    E1_FORMAL_ENVIRONMENT_STEPS,
    expand_e1_formal_matrix,
    load_e1_governance_manifest,
    resolve_e1_run,
    validate_e1_governance_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/g3_experiment_manifest.yaml"


def test_e1_expands_the_preregistered_65_run_matrix_without_duplicates():
    manifest = load_e1_governance_manifest(MANIFEST_PATH)
    runs = expand_e1_formal_matrix(manifest)

    assert len(runs) == 65
    assert len({run.identity for run in runs}) == 65
    assert Counter(run.group for run in runs) == {
        "MAPPO-DG": 8,
        "RC-AStarKD": 8,
        "LLMKD": 8,
        "RC-AStarKD+LLMKD": 8,
        "Fixed-AStarKD+LLMKD": 8,
        "QMIX-DG": 8,
        "RuleKD-v3": 8,
        "ShuffleKD-v3": 3,
        "NoOOD-v1": 3,
        "NoGoalHint-v1": 3,
    }
    assert {run.real_environment_steps for run in runs} == {
        E1_FORMAL_ENVIRONMENT_STEPS
    }
    assert {run.checkpoint_rule for run in runs} == {"checkpoint_final.pt"}
    assert all(run.artifact_path.startswith("artifacts/optimization/e2_formal/")
               for run in runs)


def test_e1_resolves_every_nonfirst_formal_seed_without_smoke_substitution():
    runs = expand_e1_formal_matrix(load_e1_governance_manifest(MANIFEST_PATH))

    selected = resolve_e1_run(runs, "MAPPO-DG:17", smoke=False)

    assert selected.identity == "MAPPO-DG:seed017"
    assert selected.real_environment_steps == E1_FORMAL_ENVIRONMENT_STEPS


def test_e1_governance_records_selected_route_and_exploratory_o3_execution():
    manifest = load_e1_governance_manifest(MANIFEST_PATH)
    validate_e1_governance_manifest(manifest)

    raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 10
    assert raw["status"] == "d1_optimization_selected_e1_implementation_in_progress"
    assert raw["freeze_blockers"] == ["E1 selected-route protocol freeze"]
    assert raw["training"]["formal_environment_steps"] == 150000
    assert raw["evaluation"]["o3_exploratory_matrix"]["default_state"] == "execute"
    assert raw["evaluation"]["o3_exploratory_matrix"]["total_episodes"] == 6400


def test_e1_governance_rejects_noncanonical_training_gpu():
    manifest = deepcopy(load_e1_governance_manifest(MANIFEST_PATH))
    manifest["training"]["execution_gpu"]["physical_index"] = 1

    with pytest.raises(ValueError, match="execution GPU"):
        validate_e1_governance_manifest(manifest)


def test_e1_validator_writes_a_machine_readable_65_run_matrix(tmp_path):
    output = tmp_path / "e1-matrix.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_e1_protocol.py",
            "--manifest",
            str(MANIFEST_PATH),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_pass"] is True
    assert payload["formal_run_count"] == 65
    assert payload["o3_exploratory_total_episodes"] == 6400
    assert len(payload["runs"]) == 65


def test_e1_terms_are_defined_in_the_canonical_terminology():
    terminology = (ROOT / "terminology.md").read_text(encoding="utf-8")

    for term in (
        "Formal Run Matrix",
        "Seed Block",
        "GPU Provenance",
        "Dataset-level Gate",
        "System Fingerprint",
        "Resume Identity",
    ):
        assert term in terminology
