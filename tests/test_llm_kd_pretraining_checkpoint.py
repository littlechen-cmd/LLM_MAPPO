import copy
import random

import numpy as np
import pytest
import torch

from llm_mappo.mappo_baseline.networks import MAPPOActor
from llm_mappo.mappo_llm_kd.pretraining import (
    SemanticPretrainingModel,
    validate_pretraining_identity,
)
from llm_mappo.mappo_llm_kd.pretraining_checkpoint import (
    load_pretraining_checkpoint,
    save_pretraining_checkpoint,
)


def _identity():
    identity = {
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
    validate_pretraining_identity(identity)
    return identity


def _bundle(seed=11):
    torch.manual_seed(seed)
    model = SemanticPretrainingModel(
        MAPPOActor(observation_dim=209, action_dim=5, hidden_dim=64)
    )
    optimizer = torch.optim.Adam(
        list(model.actor.encoder.parameters())
        + list(model.actor.gru.parameters())
        + list(model.semantic_head.parameters()),
        lr=1e-3,
    )
    return model, optimizer


def _step(model, optimizer):
    optimizer.zero_grad()
    loss = sum(
        parameter.square().mean()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    loss.backward()
    optimizer.step()


def _save(path, model, optimizer, *, progress=None):
    save_pretraining_checkpoint(
        path,
        model,
        optimizer,
        identity=_identity(),
        progress=(
            {"global_step": 7, "epoch": 2, "batch_in_epoch": 3}
            if progress is None
            else progress
        ),
        early_stopping={
            "best_validation_loss": 0.125,
            "bad_epochs": 1,
            "patience": 5,
            "minimum_improvement": 1e-4,
        },
    )


def _snapshot(model, optimizer):
    return (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().clone(),
        (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    )


def _assert_unchanged(before, model, optimizer):
    (
        model_state,
        optimizer_state,
        python_state,
        numpy_state,
        torch_state,
        cuda_state,
    ) = before
    for key, value in model_state.items():
        assert torch.equal(value, model.state_dict()[key])
    assert _nested_equal(optimizer_state, optimizer.state_dict())
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    if cuda_state is not None:
        current_cuda = torch.cuda.get_rng_state_all()
        assert len(current_cuda) == len(cuda_state)
        assert all(
            torch.equal(current, expected)
            for current, expected in zip(current_cuda, cuda_state)
        )


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def test_checkpoint_round_trip_restores_next_step_and_random_streams(tmp_path):
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    model, optimizer = _bundle(seed=104)
    _step(model, optimizer)
    path = tmp_path / "semantic-pretraining.pt"
    _save(path, model, optimizer)

    expected_random = (random.random(), np.random.random(), torch.rand(3))
    _step(model, optimizer)
    expected_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if key.startswith(("actor.encoder.", "actor.gru.", "semantic_head."))
    }

    restored, restored_optimizer = _bundle(seed=999)
    result = load_pretraining_checkpoint(
        path, restored, restored_optimizer, expected_identity=_identity()
    )
    actual_random = (random.random(), np.random.random(), torch.rand(3))
    _step(restored, restored_optimizer)

    assert result["identity"] == _identity()
    assert result["progress"] == {
        "global_step": 7, "epoch": 2, "batch_in_epoch": 3
    }
    assert actual_random[0] == expected_random[0]
    assert actual_random[1] == expected_random[1]
    torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)
    for key, value in expected_state.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(checkpoint_schema="legacy-semantic-v3"),
        lambda payload: payload.update(extra="forbidden"),
        lambda payload: payload.pop("early_stopping"),
        lambda payload: payload["identity"].update(observation_dim=208),
        lambda payload: payload["identity"].update(semantic_dimensions=["old"] * 3),
        lambda payload: payload["actor_representation"].pop(
            next(iter(payload["actor_representation"]))
        ),
        lambda payload: payload["semantic_head"].update(extra=torch.zeros(1)),
        lambda payload: payload["actor_representation"].update(
            encoder_1_weight=torch.zeros(1)
        ),
        lambda payload: payload["semantic_head"].__setitem__(
            "weight", torch.zeros(4, 63)
        ),
        lambda payload: payload["semantic_head"]["weight"].__setitem__(
            (0, 0), float("nan")
        ),
        lambda payload: payload["optimizer"].update(extra="forbidden"),
        lambda payload: payload["rng"].pop("torch_cpu"),
    ],
)
def test_invalid_checkpoint_is_rejected_before_any_state_changes(tmp_path, mutation):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "valid.pt"
    _save(path, source, source_optimizer)
    payload = torch.load(path, weights_only=True)
    mutation(payload)
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)

    target, target_optimizer = _bundle(seed=55)
    before = _snapshot(target, target_optimizer)
    with pytest.raises(ValueError):
        load_pretraining_checkpoint(
            broken, target, target_optimizer, expected_identity=_identity()
        )
    _assert_unchanged(before, target, target_optimizer)


def test_checkpoint_contains_representation_but_not_policy_head(tmp_path):
    model, optimizer = _bundle()
    path = tmp_path / "checkpoint.pt"
    _save(
        path,
        model,
        optimizer,
        progress={"global_step": 0, "epoch": 0, "batch_in_epoch": 0},
    )
    payload = torch.load(path, weights_only=True)

    assert set(payload) == {
        "checkpoint_schema",
        "identity",
        "actor_representation",
        "semantic_head",
        "optimizer",
        "rng",
        "progress",
        "early_stopping",
    }
    assert all(
        key.startswith(("encoder.", "gru."))
        for key in payload["actor_representation"]
    )
    assert not any("action_head" in key for key in payload["actor_representation"])


def test_save_is_atomic_and_does_not_leave_temporary_file(tmp_path):
    model, optimizer = _bundle()
    path = tmp_path / "checkpoint.pt"
    _save(
        path,
        model,
        optimizer,
        progress={"global_step": 0, "epoch": 0, "batch_in_epoch": 0},
    )

    assert path.is_file()
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_rejects_optimizer_that_tracks_the_policy_head(tmp_path):
    model, _ = _bundle()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="representation and SemanticHead"):
        _save(tmp_path / "forbidden.pt", model, optimizer)


def test_checkpoint_api_is_exported_by_the_isolated_package():
    import llm_mappo.mappo_llm_kd as llm_kd

    assert llm_kd.save_pretraining_checkpoint is save_pretraining_checkpoint
    assert llm_kd.load_pretraining_checkpoint is load_pretraining_checkpoint
    assert "save_pretraining_checkpoint" in llm_kd.__all__
    assert "load_pretraining_checkpoint" in llm_kd.__all__


def test_load_requires_expected_identity_and_rejects_mismatch_atomically(tmp_path):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)
    target, target_optimizer = _bundle(seed=44)
    before = _snapshot(target, target_optimizer)
    mismatched = {**_identity(), "dataset_sha256": "c" * 64}

    with pytest.raises(TypeError, match="expected_identity"):
        load_pretraining_checkpoint(path, target, target_optimizer)
    _assert_unchanged(before, target, target_optimizer)
    with pytest.raises(ValueError, match="identity"):
        load_pretraining_checkpoint(
            path,
            target,
            target_optimizer,
            expected_identity=mismatched,
        )
    _assert_unchanged(before, target, target_optimizer)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_optimizer_hyperparameters_are_validated_before_restore(tmp_path, bad_value):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)
    payload = torch.load(path, weights_only=True)
    payload["optimizer"]["state_dict"]["param_groups"][0]["lr"] = bad_value
    torch.save(payload, path)
    target, target_optimizer = _bundle(seed=45)
    before = _snapshot(target, target_optimizer)

    with pytest.raises(ValueError, match="hyperparameter"):
        load_pretraining_checkpoint(
            path, target, target_optimizer, expected_identity=_identity()
        )
    _assert_unchanged(before, target, target_optimizer)


def test_optimizer_hyperparameters_must_exactly_match_expected_config(tmp_path):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)
    target, target_optimizer = _bundle(seed=46)
    target_optimizer.param_groups[0]["lr"] = 2e-3
    before = _snapshot(target, target_optimizer)

    with pytest.raises(ValueError, match="hyperparameter"):
        load_pretraining_checkpoint(
            path, target, target_optimizer, expected_identity=_identity()
        )
    _assert_unchanged(before, target, target_optimizer)


def test_swapped_same_shape_optimizer_parameter_ids_are_rejected_atomically(
    tmp_path,
):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)
    payload = torch.load(path, weights_only=True)
    parameter_ids = payload["optimizer"]["state_dict"]["param_groups"][0][
        "params"
    ]
    parameter_ids[0], parameter_ids[1] = parameter_ids[1], parameter_ids[0]
    torch.save(payload, path)
    target, target_optimizer = _bundle(seed=47)
    before = _snapshot(target, target_optimizer)

    with pytest.raises(ValueError, match="identifier sequence"):
        load_pretraining_checkpoint(
            path, target, target_optimizer, expected_identity=_identity()
        )
    _assert_unchanged(before, target, target_optimizer)


@pytest.mark.parametrize("failure_point", ["save", "replace"])
def test_failed_atomic_overwrite_keeps_old_bytes_and_removes_temp(
    tmp_path, monkeypatch, failure_point
):
    model, optimizer = _bundle()
    path = tmp_path / "checkpoint.pt"
    old_bytes = b"previous-good-checkpoint"
    path.write_bytes(old_bytes)

    if failure_point == "save":
        def fail_save(payload, stream):
            stream.write(b"partial")
            raise OSError("simulated save failure")

        monkeypatch.setattr(torch, "save", fail_save)
    else:
        import llm_mappo.mappo_llm_kd.pretraining_checkpoint as checkpoint

        def fail_replace(source, destination):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(checkpoint.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        _save(
            path,
            model,
            optimizer,
            progress={"global_step": 0, "epoch": 0, "batch_in_epoch": 0},
        )

    assert path.read_bytes() == old_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_stepped_checkpoint_rejects_missing_optimizer_state_atomically(tmp_path):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)
    payload = torch.load(path, weights_only=True)
    state = payload["optimizer"]["state_dict"]["state"]
    state.pop(next(iter(state)))
    torch.save(payload, path)
    target, target_optimizer = _bundle(seed=48)
    before = _snapshot(target, target_optimizer)

    with pytest.raises(ValueError, match="complete optimizer state"):
        load_pretraining_checkpoint(
            path, target, target_optimizer, expected_identity=_identity()
        )
    _assert_unchanged(before, target, target_optimizer)


def test_zero_step_checkpoint_requires_and_accepts_empty_optimizer_state(tmp_path):
    source, source_optimizer = _bundle()
    progress = {"global_step": 0, "epoch": 0, "batch_in_epoch": 0}
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer, progress=progress)
    target, target_optimizer = _bundle(seed=49)

    result = load_pretraining_checkpoint(
        path, target, target_optimizer, expected_identity=_identity()
    )

    assert result["progress"] == progress
    assert target_optimizer.state_dict()["state"] == {}


def test_zero_step_checkpoint_rejects_prepopulated_optimizer_state_atomically(
    tmp_path,
):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)
    payload = torch.load(path, weights_only=True)
    payload["progress"] = {
        "global_step": 0,
        "epoch": 0,
        "batch_in_epoch": 0,
    }
    torch.save(payload, path)
    target, target_optimizer = _bundle(seed=50)
    before = _snapshot(target, target_optimizer)

    with pytest.raises(ValueError, match="zero-step"):
        load_pretraining_checkpoint(
            path, target, target_optimizer, expected_identity=_identity()
        )
    _assert_unchanged(before, target, target_optimizer)


@pytest.mark.parametrize(
    "bad_step",
    [
        torch.tensor(1, dtype=torch.int64),
        torch.tensor(-1.0, dtype=torch.float32),
        torch.tensor(float("nan"), dtype=torch.float32),
    ],
)
def test_optimizer_step_tensor_contract_is_strict_and_atomic(tmp_path, bad_step):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)
    payload = torch.load(path, weights_only=True)
    first_state = next(
        iter(payload["optimizer"]["state_dict"]["state"].values())
    )
    first_state["step"] = bad_step
    torch.save(payload, path)
    target, target_optimizer = _bundle(seed=51)
    before = _snapshot(target, target_optimizer)

    with pytest.raises(ValueError, match="step"):
        load_pretraining_checkpoint(
            path, target, target_optimizer, expected_identity=_identity()
        )
    _assert_unchanged(before, target, target_optimizer)


@pytest.mark.parametrize("bad_id", [True, np.int64(0)])
def test_optimizer_state_mapping_ids_must_be_builtin_integers_atomically(
    tmp_path, bad_id
):
    source, source_optimizer = _bundle()
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)
    payload = torch.load(path, weights_only=True)
    state = payload["optimizer"]["state_dict"]["state"]
    original_id = next(iter(state))
    state[bad_id] = state.pop(original_id)
    torch.save(payload, path)
    target, target_optimizer = _bundle(seed=52)
    before = _snapshot(target, target_optimizer)

    with pytest.raises(ValueError):
        load_pretraining_checkpoint(
            path, target, target_optimizer, expected_identity=_identity()
        )
    _assert_unchanged(before, target, target_optimizer)


def test_multiple_optimizer_groups_apply_their_own_amsgrad_contract(tmp_path):
    torch.manual_seed(53)
    source = SemanticPretrainingModel(
        MAPPOActor(observation_dim=209, action_dim=5, hidden_dim=64)
    )
    encoder = list(source.actor.encoder.parameters())
    remainder = (
        list(source.actor.gru.parameters())
        + list(source.semantic_head.parameters())
    )
    source_optimizer = torch.optim.Adam(
        [
            {"params": encoder, "amsgrad": False},
            {"params": remainder, "amsgrad": True},
        ],
        lr=1e-3,
    )
    _step(source, source_optimizer)
    path = tmp_path / "checkpoint.pt"
    _save(path, source, source_optimizer)

    torch.manual_seed(54)
    target = SemanticPretrainingModel(
        MAPPOActor(observation_dim=209, action_dim=5, hidden_dim=64)
    )
    target_encoder = list(target.actor.encoder.parameters())
    target_remainder = (
        list(target.actor.gru.parameters())
        + list(target.semantic_head.parameters())
    )
    target_optimizer = torch.optim.Adam(
        [
            {"params": target_encoder, "amsgrad": False},
            {"params": target_remainder, "amsgrad": True},
        ],
        lr=1e-3,
    )

    load_pretraining_checkpoint(
        path, target, target_optimizer, expected_identity=_identity()
    )

    groups = target_optimizer.state_dict()["param_groups"]
    state = target_optimizer.state_dict()["state"]
    assert all("max_exp_avg_sq" not in state[key] for key in groups[0]["params"])
    assert all("max_exp_avg_sq" in state[key] for key in groups[1]["params"])
