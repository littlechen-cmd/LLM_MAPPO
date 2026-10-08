"""Continuous, partially observed v3 semantic pretraining contracts."""

from hashlib import sha256
import json

import numpy as np
import pytest
import torch

from llm_mappo.mappo_baseline.networks import MAPPOActor
import llm_mappo.mappo_llm_kd.pretraining as contracts
import llm_mappo.mappo_llm_kd.pretraining_run as runner
from llm_mappo.mappo_llm_kd.initialization import (
    initialize_final_actor,
    load_representation_initialization,
    REPRESENTATION_INITIALIZATION_V1,
)
from llm_mappo.mappo_llm_kd.representation_export import export_actor_representation


def _identity():
    return {
        "observation_schema": "direct-goal-observation-v2",
        "observation_dim": 209,
        "ontology_schema": "agent-centric-priority-coordination-four-score-v3",
        "semantic_dimensions": (
            "goal-priority", "yielding-preference", "local-constraint",
            "motion-stagnation",
        ),
        "actor_architecture": "mappo-actor-encoder-gru-v1",
        "hidden_dim": 64,
        "dataset_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "semantic_contract_version": "observable-priority-coordination-four-score-v3",
        "semantic_sources": ("llm", "llm", "llm", "observable-rule-v3"),
        "loss_aggregation": "per-dimension-known-mean-then-weighted-mean",
    }


def test_rank3_loss_normalizes_each_dimension_and_masks_unknown_gradient():
    predictions = torch.tensor(
        [[[0.5, 0.8, 0.1, 0.9]], [[0.5, 0.4, 0.1, 0.9]]],
        requires_grad=True,
    )
    targets = torch.tensor([[[0.37, 0.0, 0.0, 0.0]], [[0.63, 0.0, 0.0, 0.0]]])
    mask = torch.tensor([[[True, False, True, False]],
                         [[True, False, False, False]]])
    loss = contracts.masked_semantic_loss(
        predictions, targets, mask, torch.tensor([1.0, 1.0, 1.0, 0.25])
    )
    # Per-dimension MSEs are .0169 and .01; the other two dimensions are absent.
    assert loss.item() == pytest.approx(0.01345)
    loss.backward()
    assert torch.count_nonzero(predictions.grad[~mask]).item() == 0
    assert predictions.grad[0, 0, 0].item() == pytest.approx(0.065)
    assert predictions.grad[1, 0, 0].item() == pytest.approx(-0.065)


def test_rank3_batch_allows_unknown_final_targets_but_rejects_padding_targets():
    values = dict(
        observations=torch.zeros(1, 2, 209),
        valid_masks=torch.tensor([[False, True]]),
        reset_masks=torch.tensor([[False, True]]),
        targets=torch.zeros(1, 2, 4), initial_hidden=torch.zeros(1, 64),
        target_masks=torch.zeros(1, 2, 4, dtype=torch.bool),
    )
    batch = contracts.PretrainingBatch(**values)
    assert not batch.target_masks.any()
    values["target_masks"][0, 0, 0] = True
    with pytest.raises(ValueError, match="padded"):
        contracts.PretrainingBatch(**values)


@pytest.mark.parametrize("mutation", [None, "dimensions", "sources", "contract", "json"])
def test_v3_identity_and_encoder_only_initialization(tmp_path, mutation):
    identity = _identity()
    if mutation and mutation != "json":
        key = {"dimensions": "semantic_dimensions", "sources": "semantic_sources",
               "contract": "semantic_contract_version"}[mutation]
        identity[key] = {
            "dimensions": contracts.SEMANTIC_DIMENSIONS,
            "sources": ("llm",) * 4,
            "contract": contracts.SEMANTIC_CONTRACT_V2,
        }[mutation]
        with pytest.raises(ValueError):
            contracts.validate_pretraining_identity(identity)
        return
    source = MAPPOActor(209, 5, 64)
    target = MAPPOActor(209, 5, 64)
    before = {name: value.clone() for name, value in target.state_dict().items()}
    path = tmp_path / "v3-representation.pt"
    exported = export_actor_representation(path, source, identity=identity)
    initialization = load_representation_initialization({
        "schema": REPRESENTATION_INITIALIZATION_V1,
        "artifact_path": str(path), "artifact_sha256": exported["file_sha256"],
        "identity": json.loads(json.dumps(identity)) if mutation == "json" else identity,
    })
    initialize_final_actor(target, initialization)
    for name, value in target.state_dict().items():
        expected = source.state_dict()[name] if name.startswith(
            ("encoder.", "gru.")
        ) else before[name]
        assert torch.equal(value, expected)


def _sample(index, scores, masks):
    return {
        "partition": "validation", "episode_id": f"episode-{index}",
        "agent_id": 1, "start_timestep": 0, "end_timestep": 0,
        "observations": [np.zeros(209).tolist()], "reset_masks": [True],
        "stratum": "fixture", "scores": scores, "known_masks": masks,
    }


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_v3_validation_aggregates_per_dimension_across_entire_partition(batch_size):
    model = contracts.SemanticPretrainingModel(MAPPOActor(209, 5, 64))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    samples = [
        _sample(0, [0.37, 0, 0, 0], [True, False, False, False]),
        _sample(1, [0.63, 0, 0, 0], [True, False, False, False]),
        _sample(2, [0, 1, 0, 0], [False, True, False, False]),
    ]
    actual = runner._evaluate(
        model, samples, batch_size, torch.device("cpu"),
        torch.tensor([1.0, 1.0, 1.0, 0.25]),
    )
    assert actual == pytest.approx((0.0169 + 0.25) / 2)


def _write_v3_artifacts(tmp_path):
    dataset = tmp_path / "dataset-v3"
    labels = dataset / "labels" / "v3"
    labels.mkdir(parents=True)
    names = _identity()["semantic_dimensions"]
    sources = dict(zip(names, _identity()["semantic_sources"]))
    windows = []
    records = []
    for index in range(5):
        scores = [0.37, None, 0.25, 0.5] if index == 0 else [0.37, 0.63, 0.25, 0.5]
        window = _sample(index, [], [])
        del window["scores"], window["known_masks"]
        window["partition"] = "train" if index < 3 else (
            "validation" if index == 3 else "test"
        )
        windows.append(window)
        records.append({
            "status": "ok", "view_id": runner._window_id(window),
            "scores": dict(zip(names, scores)),
            "known_masks": dict(zip(names, [v is not None for v in scores])),
            "sources": sources,
            "evidence": {name: [
                "history[0].self.battery"
            ] for i, name in enumerate(names)},
            "unknown_reasons": {name: "ambiguous-peer-association" if scores[i]
                                is None else None for i, name in enumerate(names)},
            "receipt": {"semantic_contract_version":
                        _identity()["semantic_contract_version"]},
        })
    # Held-out values must never be interpreted during fitting/model selection.
    records[-1]["scores"] = "held-out-test-must-not-be-parsed"
    windows_path = dataset / "windows.jsonl"
    windows_path.write_text("".join(json.dumps(w) + "\n" for w in windows), "utf-8")
    manifest = {
        "schema": "llm-dataset-build-run-v3",
        "distribution": "native-209d-goal-full-main-v3",
        "main_pretraining_eligible": True,
        "windows_sha256": sha256(windows_path.read_bytes()).hexdigest(),
        "semantic_contract_version": _identity()["semantic_contract_version"],
        "semantic_dimensions": list(names),
        "semantic_sources": list(_identity()["semantic_sources"]),
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    (dataset / "state.json").write_text(json.dumps({"status": "completed"}), "utf-8")
    labels_path = labels / "labels.jsonl"
    labels_path.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")
    label_manifest = {
        "schema": "llm-dataset-label-run-v3",
        "dataset_manifest_sha256": sha256(
            (dataset / "manifest.json").read_bytes()
        ).hexdigest(),
        "labels_sha256": sha256(labels_path.read_bytes()).hexdigest(),
        "successful_records": 5, "pretraining_eligible": True,
        "semantic_contract_version": _identity()["semantic_contract_version"],
        "semantic_dimensions": list(names),
        "semantic_sources": list(_identity()["semantic_sources"]),
    }
    (labels / "manifest.json").write_text(json.dumps(label_manifest), "utf-8")
    (labels / "state.json").write_text(json.dumps({"status": "completed"}), "utf-8")
    return dataset, labels


def _v3_config(tmp_path, dataset, labels):
    return runner.PretrainingRunConfig(
        schema=runner.PRETRAINING_RUNNER_V1,
        dataset_directory=str(dataset), label_directory=str(labels),
        artifact_root=str(tmp_path / "runs"), config_sha256="a" * 64,
        approval_status="approved", seed=7, batch_size=1, max_epochs=1,
        learning_rate=0.001, dimension_weights=(1.0, 1.0, 1.0, 0.25),
        early_stopping_patience=10, early_stopping_min_delta=0.0001,
    )


def test_v3_real_loop_preserves_continuous_scores_and_binds_identity(tmp_path):
    dataset, labels = _write_v3_artifacts(tmp_path)
    config = _v3_config(tmp_path, dataset, labels)
    result = runner.run_semantic_pretraining(config, device="cpu", run_id="v3")
    manifest = result["manifest"]
    assert manifest["global_step"] == 3
    assert manifest["skipped_unknown_batches"] == 0
    assert manifest["history"][0]["skipped_unknown_batches"] == 0
    assert manifest["identity"]["semantic_dimensions"] == (
        "goal-priority", "yielding-preference", "local-constraint", "motion-stagnation",
    )
    assert manifest["identity"]["semantic_sources"] == (
        "llm", "llm", "llm", "observable-rule-v3",
    )
    assert manifest["identity"]["loss_aggregation"] == (
        "per-dimension-known-mean-then-weighted-mean"
    )
    values = runner._load_scores(
        labels / "labels.jsonl", contract=_identity()["semantic_contract_version"],
        view_ids={"train:episode-1:agent-1:start-0:end-0"},
    )
    assert values["train:episode-1:agent-1:start-0:end-0"]["scores"][:2] == (
        0.37, 0.63,
    )


def test_all_unknown_batch_skip_is_defensive_and_resume_retains_skip_count(
    tmp_path, monkeypatch,
):
    dataset, labels = _write_v3_artifacts(tmp_path)
    config = _v3_config(tmp_path, dataset, labels)
    original_scores = runner._load_scores

    def all_unknown_fixture(*args, **kwargs):
        # Real artifacts remain valid: this deliberately unreachable label shape
        # tests the batch scheduler's defensive branch after strict loading.
        scores = original_scores(*args, **kwargs)
        scores["train:episode-0:agent-1:start-0:end-0"] = {
            "scores": (0.0,) * 4, "known_masks": (False,) * 4,
        }
        return scores

    monkeypatch.setattr(runner, "_load_scores", all_unknown_fixture)
    original_save = runner.save_pretraining_checkpoint

    def interrupt_after_first_known_update(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if kwargs["progress"] == {
            "global_step": 1, "epoch": 1, "batch_in_epoch": 2,
        }:
            raise RuntimeError("interrupted after skipped batch")

    monkeypatch.setattr(runner, "save_pretraining_checkpoint",
                        interrupt_after_first_known_update)
    with pytest.raises(RuntimeError, match="skipped batch"):
        runner.run_semantic_pretraining(config, device="cpu", run_id="skipped")
    monkeypatch.setattr(runner, "save_pretraining_checkpoint", original_save)
    result = runner.run_semantic_pretraining(
        config, device="cpu", resume=tmp_path / "runs" / "skipped",
    )
    assert result["manifest"]["global_step"] == 2
    assert result["manifest"]["skipped_unknown_batches"] == 1
    assert result["manifest"]["history"][0]["skipped_unknown_batches"] == 1
    model = contracts.SemanticPretrainingModel(MAPPOActor(209, 5, 64))
    optimizer = torch.optim.Adam(
        list(model.actor.encoder.parameters()) + list(model.actor.gru.parameters())
        + list(model.semantic_head.parameters()), lr=0.001,
    )
    identity_as_json = json.loads(json.dumps(result["manifest"]["identity"]))
    restored = runner.load_pretraining_checkpoint(
        tmp_path / "runs" / "skipped" / "checkpoint_latest.pt", model, optimizer,
        expected_identity=identity_as_json,
    )
    assert restored["progress"]["global_step"] == 2


@pytest.mark.parametrize("mutation", ["unknown-value", "missing-known", "source",
                                      "receipt", "nan", "goal-unknown", "reason",
                                      "evidence-count", "unknown-evidence", "duplicate"])
def test_v3_loader_rejects_invalid_known_masks_sources_and_contract(tmp_path, mutation):
    _, labels = _write_v3_artifacts(tmp_path)
    path = labels / "labels.jsonl"
    records = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    record = records[1]
    if mutation == "unknown-value":
        record["known_masks"]["goal-priority"] = False
    elif mutation == "missing-known":
        record["scores"]["goal-priority"] = None
    elif mutation == "source":
        record["sources"]["motion-stagnation"] = "llm"
    elif mutation == "receipt":
        record["receipt"]["semantic_contract_version"] = contracts.SEMANTIC_CONTRACT_V2
    elif mutation == "nan":
        record["scores"]["goal-priority"] = float("nan")
    elif mutation in {"goal-unknown", "reason", "unknown-evidence"}:
        name = "goal-priority" if mutation == "goal-unknown" else "yielding-preference"
        record["scores"][name] = None
        record["known_masks"][name] = False
        record["unknown_reasons"][name] = (
            "ambiguous-peer-association" if mutation == "unknown-evidence"
            else "insufficient-observation"
        )
        if mutation == "unknown-evidence":
            record["evidence"][name] = []
    elif mutation == "duplicate":
        record["evidence"]["goal-priority"] *= 2
    else:
        record["evidence"]["goal-priority"] *= 5
    path.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")
    with pytest.raises(ValueError):
        runner._load_scores(
            path, contract=_identity()["semantic_contract_version"],
            view_ids={record["view_id"]},
        )
