"""Snapshot boundary and completed-episode integrity regression tests."""

import io
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_hard_charge_dataset_snapshot import (
    check_charging_holds, digest, extract_snapshot, verify_receipt,
)


def test_archive_rejects_parent_escape(tmp_path):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        item = tarfile.TarInfo("../outside.py")
        item.size = 1
        archive.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe"):
        extract_snapshot(output.getvalue(), tmp_path)
    assert not (tmp_path.parent / "outside.py").exists()


def test_receipt_rejects_changed_array(tmp_path):
    path = tmp_path / "seed_10000.npz"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        verify_receipt(path, {"seed": 10000, "npz_sha256": "0" * 64},
                       "identity", 10000)


def test_receipt_rejects_fake_reset_alignment(tmp_path):
    path = tmp_path / "seed_10000.npz"
    np.savez_compressed(path, observations=np.zeros((5, 8, 209), dtype=np.float32),
                        reset_masks=np.zeros((5, 8), dtype=np.bool_))
    receipt = dict(schema="goal-full-complete-episode-v3", seed=10000,
                   npz_sha256=digest(path), collection_identity_sha256="identity",
                   boundary="truncated", steps=8,
                   episode_id="collection-seed-10000-episode-0")
    with pytest.raises(ValueError, match="reset alignment"):
        verify_receipt(path, receipt, "identity", 10000)


def test_charging_hold_requires_noop_and_full_release():
    before = {1: ((0, 0), 0, 0.95)}
    environment = SimpleNamespace(
        env=SimpleNamespace(agents=[SimpleNamespace(x=0, y=0, dir=0, battery=1.0)]),
        _charging_holds={})
    mask = np.asarray([[True, False, False, False, False]])
    assert check_charging_holds(before, environment, [0], mask) == ([], 1)
    violations, releases = check_charging_holds(before, environment, [1], mask)
    assert len(violations) == 1 and releases == 0
