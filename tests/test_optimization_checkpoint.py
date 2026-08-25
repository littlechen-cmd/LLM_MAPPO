import copy

import pytest
import torch
from torch import nn

from llm_mappo.optimization_checkpoint import (
    O0CheckpointV1,
    load_o0_checkpoint,
    save_o0_checkpoint,
)
from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor


def _bundle():
    return nn.ModuleDict({"actor": O0StudentActor(), "critic": O0CentralizedCritic()})


def test_o0_checkpoint_round_trips_only_with_strict_contract(tmp_path):
    model = _bundle()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    path = tmp_path / "o0.pt"
    metadata = O0CheckpointV1.default_metadata(
        global_env_steps=12,
        update_count=1,
        completed_episodes=0,
        ema_state={"count": 0, "mean": 0.0, "m2": 0.0, "initialized": False},
        provenance={"config_hash": "abc", "layout_hash": "def"},
    )
    save_o0_checkpoint(path, model, optimizer, metadata)

    restored = _bundle()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=3e-4)
    loaded = load_o0_checkpoint(path, restored, restored_optimizer)

    assert loaded["metadata"] == metadata
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])

    broken = copy.deepcopy(torch.load(path, weights_only=False))
    del broken["metadata"]["ema_state"]
    broken_path = tmp_path / "broken.pt"
    torch.save(broken, broken_path)
    with pytest.raises(ValueError, match="ema_state"):
        load_o0_checkpoint(broken_path, _bundle(), None)


def test_o0_checkpoint_rejects_legacy_payload_before_model_loading(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"phase": "4", "model_state": {}}, path)

    with pytest.raises(ValueError, match="checkpoint_schema"):
        load_o0_checkpoint(path, _bundle(), None)


def test_o0_checkpoint_metadata_freezes_teacher_and_calibration_parameters():
    metadata = O0CheckpointV1.default_metadata(
        global_env_steps=0,
        update_count=0,
        completed_episodes=0,
        ema_state={"count": 0, "mean": 0.0, "m2": 0.0, "initialized": False},
        provenance={"config_hash": "abc", "layout_hash": "def"},
    )

    assert metadata["k_motion"] == 12
    assert metadata["h_reward"] == 12
    assert metadata["expansion_budget"] == 512
    assert metadata["calibration_modulus"] == 16
    assert metadata["ema_decay"] == 0.99
    assert metadata["ema_minimum_scale"] == 1e-3
    assert metadata["ema_initialization_samples"] == 64
