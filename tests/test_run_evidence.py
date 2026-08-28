import errno
from pathlib import Path

import pytest

from llm_mappo.run_evidence import (
    ExternalGpuInterferenceError,
    RunIdentity,
    classify_failure,
    load_valid_shard,
    verify_o1_gate_receipt,
    write_new_atomic_text,
    write_o1_gate_receipt,
    write_shard,
)


def _identity():
    return RunIdentity(
        code_commit="abc",
        config_sha256="config",
        immutable_machine_sha256="machine",
        environment_sha256="environment",
    )


def test_shards_are_reused_only_when_the_identity_and_schema_match(tmp_path):
    path = tmp_path / "repeat.json"
    write_shard(path, {"schema": "runtime-v1", "seconds": 1.0}, _identity())

    assert load_valid_shard(path, _identity(), "runtime-v1")["seconds"] == 1.0
    with pytest.raises(ValueError, match="identity"):
        load_valid_shard(path, RunIdentity("other", "config", "machine", "environment"), "runtime-v1")
    with pytest.raises(ValueError, match="schema"):
        load_valid_shard(path, _identity(), "memory-v1")


def test_only_predeclared_infrastructure_failures_are_resumable():
    assert classify_failure(KeyboardInterrupt()) == "infrastructure"
    assert classify_failure(SystemExit(143)) == "infrastructure"
    assert classify_failure(ExternalGpuInterferenceError()) == "infrastructure"
    assert classify_failure(OSError(errno.ENOSPC, "disk full")) == "infrastructure"
    assert classify_failure(RuntimeError("CUDA out of memory")) == "algorithm"
    assert classify_failure(ValueError("NaN")) == "algorithm"


def test_o1_receipt_is_written_only_for_a_passing_gate(tmp_path):
    receipt = tmp_path / "o1_gate_receipt.json"
    summary = {"gate_pass": True, "runtime_gate_pass": True, "memory_gate_pass": True}
    write_o1_gate_receipt(summary, _identity(), receipt)

    verified = verify_o1_gate_receipt(receipt, _identity())
    assert verified["next_required_phase"] == "O2"
    with pytest.raises(ValueError, match="identity"):
        verify_o1_gate_receipt(receipt, RunIdentity("x", "config", "machine", "environment"))
    with pytest.raises(ValueError, match="Gate did not pass"):
        write_o1_gate_receipt({"gate_pass": False}, _identity(), tmp_path / "failed.json")


def test_immutable_text_artifact_refuses_overwrite(tmp_path):
    path = tmp_path / "runtime.csv"
    write_new_atomic_text(path, "header\nvalue\n")

    assert path.read_text(encoding="utf-8") == "header\nvalue\n"
    with pytest.raises(FileExistsError):
        write_new_atomic_text(path, "other\n")
