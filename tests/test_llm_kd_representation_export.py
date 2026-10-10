from hashlib import sha256

import pytest
import torch

from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_llm_kd.representation_export import (
    export_actor_representation,
    import_actor_representation,
)


def _identity():
    return {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": 209,
        "ontology_schema": "agent-centric-four-score-ontology-v1",
        "semantic_dimensions": [
            "mission_urgency",
            "concession_appropriateness",
            "interaction_risk",
            "deadlock_severity",
        ],
        "actor_architecture": "mappo-actor-encoder-gru-v1",
        "hidden_dim": 64,
        "dataset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
    }


def _actor(seed, action_dim=5):
    torch.manual_seed(seed)
    return MAPPOActor(209, action_dim, 64)


def _state(actor):
    return {key: value.detach().clone() for key, value in actor.state_dict().items()}


def _assert_state_equal(expected, actor):
    actual = actor.state_dict()
    assert tuple(actual) == tuple(expected)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


def _import(path, actor, *, identity=None, file_sha256=None):
    return import_actor_representation(
        path,
        actor,
        expected_identity=_identity() if identity is None else identity,
        expected_artifact_sha256=(
            sha256(path.read_bytes()).hexdigest()
            if file_sha256 is None
            else file_sha256
        ),
    )


def test_export_contains_only_exact_representation_and_identity(tmp_path):
    actor = _actor(11)
    path = tmp_path / "representation.pt"
    receipt = export_actor_representation(path, actor, identity=_identity())
    payload = torch.load(path, weights_only=True)

    assert tuple(payload) == (
        "export_schema",
        "identity",
        "payload_content_sha256",
        "actor_representation",
    )
    assert tuple(payload["identity"]) == tuple(_identity())
    assert len(payload["payload_content_sha256"]) == 64
    assert receipt["file_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert payload["actor_representation"]
    assert all(
        key.startswith(("encoder.", "gru."))
        for key in payload["actor_representation"]
    )
    forbidden_top_level = {
        "semantic_head", "policy_head", "critic", "optimizer", "rng", "progress"
    }
    assert forbidden_top_level.isdisjoint(payload)
    assert not any(
        "action_head" in key or "semantic_head" in key
        for key in payload["actor_representation"]
    )


def test_round_trip_preserves_representation_and_fresh_policy_head(tmp_path):
    source = _actor(12)
    target = _actor(13)
    fresh_policy = target.action_head.state_dict()
    observations = torch.randn(3, 2, 209)
    hidden = torch.randn(2, 64)
    masks = torch.ones(3, 2)
    expected_embedding, _ = source._sequence(observations, hidden, masks)
    path = tmp_path / "representation.pt"

    receipt = export_actor_representation(path, source, identity=_identity())
    result = _import(path, target, file_sha256=receipt["file_sha256"])
    actual_embedding, _ = target._sequence(observations, hidden, masks)

    assert result["identity"] == _identity()
    assert result["payload_content_sha256"]
    assert result["file_sha256"] == receipt["file_sha256"]
    for key, value in source.state_dict().items():
        if key.startswith(("encoder.", "gru.")):
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    _assert_state_equal(fresh_policy, target.action_head)
    torch.testing.assert_close(actual_embedding, expected_embedding, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(export_schema="legacy"),
        lambda value: value.update(extra="forbidden"),
        lambda value: value.pop("identity"),
        lambda value: value["identity"].pop("config_sha256"),
        lambda value: value["identity"].update(extra="forbidden"),
        lambda value: value["identity"].update(observation_dim=208),
        lambda value: value["identity"].update(dataset_sha256="c" * 64),
        lambda value: value["actor_representation"].pop(
            next(iter(value["actor_representation"]))
        ),
        lambda value: value["actor_representation"].update(extra=torch.zeros(1)),
        lambda value: value["actor_representation"].__setitem__(
            next(iter(value["actor_representation"])), torch.zeros(1)
        ),
        lambda value: value["actor_representation"].__setitem__(
            next(iter(value["actor_representation"])),
            next(iter(value["actor_representation"].values())).to(torch.float64),
        ),
        lambda value: next(iter(value["actor_representation"].values()))
        .view(-1)
        .__setitem__(0, float("nan")),
        lambda value: value.update(payload_content_sha256="0" * 64),
    ],
)
def test_invalid_import_is_rejected_without_changing_any_actor_parameter(
    tmp_path, mutation
):
    path = tmp_path / "representation.pt"
    export_actor_representation(path, _actor(20), identity=_identity())
    payload = torch.load(path, weights_only=True)
    mutation(payload)
    torch.save(payload, path)
    target = _actor(21)
    before = _state(target)

    with pytest.raises(ValueError):
        _import(path, target)

    _assert_state_equal(before, target)


def test_import_rejects_incompatible_actor_and_unsafe_file_atomically(tmp_path):
    path = tmp_path / "representation.pt"
    export_actor_representation(path, _actor(30), identity=_identity())
    incompatible = MAPPOActor(208, 5, 64)
    before = _state(incompatible)
    with pytest.raises(ValueError):
        _import(path, incompatible)
    _assert_state_equal(before, incompatible)

    path.write_bytes(b"not a safe torch artifact")
    target = _actor(31)
    before = _state(target)
    with pytest.raises(ValueError, match="safely loaded"):
        _import(path, target)
    _assert_state_equal(before, target)


def test_import_requires_a_valid_externally_pinned_file_digest(tmp_path):
    path = tmp_path / "representation.pt"
    export_actor_representation(path, _actor(32), identity=_identity())
    target = _actor(33)
    before = _state(target)

    with pytest.raises(TypeError, match="expected_artifact_sha256"):
        import_actor_representation(path, target, expected_identity=_identity())
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        import_actor_representation(
            path,
            target,
            expected_identity=_identity(),
            expected_artifact_sha256="not-a-digest",
        )
    _assert_state_equal(before, target)


def test_external_digest_rejects_self_consistent_tampering_before_torch_load(
    tmp_path, monkeypatch
):
    import llm_mappo.mappo_llm_kd.representation_export as module

    original = tmp_path / "approved.pt"
    receipt = export_actor_representation(original, _actor(34), identity=_identity())
    payload = torch.load(original, weights_only=True)
    first_tensor = next(iter(payload["actor_representation"].values()))
    first_tensor.view(-1)[0] += 1.0
    payload["payload_content_sha256"] = module._payload_content_digest(
        payload["identity"], payload["actor_representation"]
    )
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    target = _actor(35)
    before = _state(target)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("torch.load must not run before file authentication")

    monkeypatch.setattr(module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="file digest"):
        import_actor_representation(
            tampered,
            target,
            expected_identity=_identity(),
            expected_artifact_sha256=receipt["file_sha256"],
        )
    _assert_state_equal(before, target)


def test_authenticated_byte_snapshot_prevents_path_replacement_race(
    tmp_path, monkeypatch
):
    import llm_mappo.mappo_llm_kd.representation_export as module

    approved_actor = _actor(36)
    approved = tmp_path / "approved.pt"
    receipt = export_actor_representation(
        approved, approved_actor, identity=_identity()
    )
    malicious = tmp_path / "replacement.pt"
    export_actor_representation(malicious, _actor(37), identity=_identity())
    malicious_bytes = malicious.read_bytes()
    original_torch_load = module.torch.load
    target = _actor(38)

    def replace_path_then_load(source, *args, **kwargs):
        assert hasattr(source, "read")
        approved.write_bytes(malicious_bytes)
        return original_torch_load(source, *args, **kwargs)

    monkeypatch.setattr(module.torch, "load", replace_path_then_load)
    _import(
        approved,
        target,
        file_sha256=receipt["file_sha256"],
    )

    for key, value in approved_actor.state_dict().items():
        if key.startswith(("encoder.", "gru.")):
            torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)

    next_target = _actor(39)
    before = _state(next_target)
    with pytest.raises(ValueError, match="file digest"):
        _import(
            approved,
            next_target,
            file_sha256=receipt["file_sha256"],
        )
    _assert_state_equal(before, next_target)


def test_atomic_export_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "representation.pt"
    previous = b"previous-valid-bytes"
    path.write_bytes(previous)
    import llm_mappo.mappo_llm_kd.representation_export as module

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        export_actor_representation(path, _actor(40), identity=_identity())
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_export_receipt_is_bound_to_published_bytes_not_post_replace_path(
    tmp_path, monkeypatch
):
    import llm_mappo.mappo_llm_kd.representation_export as module

    replacement_path = tmp_path / "replacement.pt"
    export_actor_representation(
        replacement_path, _actor(41), identity=_identity()
    )
    replacement_bytes = replacement_path.read_bytes()
    destination = tmp_path / "published.pt"
    original_replace = module.os.replace
    published_digest = []

    def replace_then_attack(source, target):
        published_digest.append(sha256(source.read_bytes()).hexdigest())
        original_replace(source, target)
        destination.write_bytes(replacement_bytes)

    monkeypatch.setattr(module.os, "replace", replace_then_attack)
    receipt = export_actor_representation(
        destination, _actor(42), identity=_identity()
    )

    assert receipt["file_sha256"] == published_digest[0]
    assert receipt["file_sha256"] != sha256(destination.read_bytes()).hexdigest()
    target = _actor(43)
    before = _state(target)
    with pytest.raises(ValueError, match="file digest"):
        _import(
            destination,
            target,
            file_sha256=receipt["file_sha256"],
        )
    _assert_state_equal(before, target)


def test_public_api_exports_representation_functions():
    import llm_mappo.mappo_llm_kd as llm_kd

    assert llm_kd.export_actor_representation is export_actor_representation
    assert llm_kd.import_actor_representation is import_actor_representation
    assert "export_actor_representation" in llm_kd.__all__
    assert "import_actor_representation" in llm_kd.__all__
