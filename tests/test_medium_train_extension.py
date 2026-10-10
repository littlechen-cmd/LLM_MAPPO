"""Train-only collection bounds and immutable parent provenance."""

import pytest
import numpy as np
from types import SimpleNamespace

from scripts.run_medium_train_extension import batch_seed_start, check_parent_reference


def write_episode(directory, seed, identity_sha):
    from scripts.run_medium_train_extension import helpers

    path = directory / "episodes" / f"seed_{seed:05d}.npz"
    path.parent.mkdir(exist_ok=True)
    reset = np.zeros((5, 8), dtype=np.bool_)
    reset[:, 0] = True
    np.savez_compressed(path, observations=np.zeros((5, 8, 209), dtype=np.float32),
                        reset_masks=reset)
    receipt = dict(schema="goal-full-complete-episode-v3", seed=seed,
                   episode_id=f"collection-seed-{seed}-episode-0", steps=8,
                   boundary="truncated", npz_sha256=helpers.digest(path),
                   collection_identity_sha256=identity_sha,
                   charging_validation=dict(violations=0))
    helpers.atomic(path.with_suffix(".json"), receipt)
    return receipt


def make_parent_reference(directory):
    from scripts.run_medium_train_extension import helpers

    receipts = {}
    for seed in range(10000, 10120):
        receipt = write_episode(directory, seed, "synthetic-parent-identity")
        receipts[receipt["episode_id"]] = receipt
    return dict(run_directory=str(directory), files_sha256={},
                collection_identity_sha256="synthetic-parent-identity",
                episode_receipts_sha256=helpers.canonical(receipts))


def test_extension_seed_bounds():
    assert [batch_seed_start(i) for i in range(1, 7)] == [
        10120, 10160, 10200, 10240, 10280, 10320]
    for invalid in (0, 7, True, 1.0):
        with pytest.raises(ValueError, match="batch"):
            batch_seed_start(invalid)


def test_parent_reference_rejects_changed_heldout_file(tmp_path):
    from scripts.run_medium_train_extension import helpers

    path = tmp_path / "windows.jsonl"
    path.write_text("original\n", encoding="utf-8")
    reference = make_parent_reference(tmp_path)
    reference["files_sha256"]["windows.jsonl"] = helpers.digest(path)
    check_parent_reference(reference)
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parent"):
        check_parent_reference(reference)


def test_native_batch_uses_new_seed_and_recovers_complete_receipt(tmp_path, monkeypatch):
    from scripts.run_medium_train_extension import collect_extension_native, helpers
    import llm_mappo.mappo_baseline.vector_env as vector
    import llm_mappo.mappo_baseline.structured_observation as structured
    import llm_mappo.mappo_llm_kd.collector as collector

    calls = []

    class Environment:
        _charging_holds = {}
        env = SimpleNamespace(agents=[])

        def reset(self, seed):
            calls.append(seed)

        def action_masks(self):
            return np.ones((5, 5), dtype=np.bool_)

        def step(self, actions):
            return SimpleNamespace(terminated=False, truncated=True,
                                   metrics=SimpleNamespace(deadlocked=False,
                                                           as_dict=lambda: {}))

        def close(self):
            pass

    class Policy:
        def __init__(self, *args):
            pass

        def start_episode(self, seed):
            pass

        def act(self, rows, mask, hidden):
            return np.zeros(5, dtype=np.int64), hidden

    monkeypatch.setattr(vector, "new_baseline_environment", lambda values: Environment())
    monkeypatch.setattr(structured, "build_structured_observations",
                        lambda *args, **kwargs: np.zeros((5, 209), dtype=np.float32))
    monkeypatch.setattr(collector, "FrozenMAPPOPolicy", Policy)
    (tmp_path / "episodes").mkdir()
    helpers.atomic(tmp_path / "collection_identity.json", dict(
        collection_contract=dict(batch_index=1, seed_start=10120),
        source_actor_contract=dict(source_checkpoint_sha256="synthetic-only")))
    experiment = SimpleNamespace(environment={})
    collect_extension_native(tmp_path, experiment, None, 10120, 1)
    receipt_path = tmp_path / "episodes/seed_10120.json"
    receipt = helpers.read(receipt_path)
    assert receipt["seed"] == 10120
    assert receipt["charging_validation"]["violations"] == 0
    receipt_path.unlink()
    collect_extension_native(tmp_path, experiment, None, 10120, 1)
    assert helpers.read(receipt_path) == receipt
    assert calls == [10120]
    assert not (tmp_path / "episodes/seed_10000.npz").exists()
    receipt["charging_validation"]["violations"] = 1
    helpers.atomic(receipt_path, receipt)
    with pytest.raises(ValueError, match="hard-charge validation"):
        collect_extension_native(tmp_path, experiment, None, 10120, 1)
    assert calls == [10120]


def test_resume_rejects_smaller_completed_prefix(tmp_path):
    from scripts.run_medium_train_extension import check_requested_prefix, helpers

    identity = dict(collection_contract=dict(batch_index=1, seed_start=10120))
    helpers.atomic(tmp_path / "collection_identity.json", identity)
    for seed in (10120, 10121):
        write_episode(tmp_path, seed, helpers.canonical(identity))
    check_requested_prefix(tmp_path, 2)
    with pytest.raises(ValueError, match="smaller"):
        check_requested_prefix(tmp_path, 1)


def test_sparse_prefix_requires_valid_completed_intent(tmp_path):
    from scripts.run_medium_train_extension import check_requested_prefix, helpers

    identity = dict(collection_contract=dict(batch_index=1, seed_start=10120))
    identity_sha = helpers.canonical(identity)
    helpers.atomic(tmp_path / "collection_identity.json", identity)
    write_episode(tmp_path, 10120, identity_sha)
    write_episode(tmp_path, 10122, identity_sha)
    with pytest.raises(ValueError, match="smaller"):
        check_requested_prefix(tmp_path, 2)
    with pytest.raises(ValueError, match="Sparse"):
        check_requested_prefix(tmp_path, 3)
    completion = write_episode(tmp_path, 10121, identity_sha)
    (tmp_path / "episodes/seed_10121.json").unlink()
    helpers.atomic(tmp_path / "episodes/seed_10121.intent.json", dict(
        seed=10121, collection_identity_sha256=identity_sha, completion=completion))
    check_requested_prefix(tmp_path, 3)


@pytest.mark.parametrize("target", ["receipt", "array"])
def test_parent_reference_rechecks_original_episode_bytes(tmp_path, target):
    from scripts.run_medium_train_extension import helpers

    reference = make_parent_reference(tmp_path)
    check_parent_reference(reference)
    if target == "receipt":
        path = tmp_path / "episodes/seed_10000.json"
        receipt = helpers.read(path)
        receipt["changed_metric"] = True
        helpers.atomic(path, receipt)
    else:
        (tmp_path / "episodes/seed_10000.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        check_parent_reference(reference)
