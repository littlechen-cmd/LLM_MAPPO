"""O2 evidence must stay compact, atomic and identity-bound."""

import json

import pytest
import torch


def test_o2_evidence_rejects_existing_run_and_full_state_event(tmp_path):
    from llm_mappo.o2_evidence import O2EvidenceWriter

    run_directory = tmp_path / "run"
    writer = O2EvidenceWriter.create(
        run_directory, {"group": "MAPPO-DG", "seed": 107}
    )
    writer.write_teacher_step_count(
        {"real_env_steps": 1, "teacher_queries": 0, "shadow_calls": 0}
    )
    writer.write_event({"kind": "selected", "real_env_steps": 1})
    writer.close(summary={"real_env_steps": 1})

    assert json.loads((run_directory / "state.json").read_text()) == {
        "status": "complete"
    }
    with pytest.raises(FileExistsError):
        O2EvidenceWriter.create(run_directory, {"group": "MAPPO-DG"})
    with pytest.raises(ValueError, match="full-array"):
        writer.write_event({"observations": [0.0] * 100})


def test_o2_checkpoint_requires_empty_rollout_and_matching_identity(tmp_path):
    from llm_mappo.o2_evidence import load_o2_checkpoint, save_o2_checkpoint

    actor = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()))
    checkpoint = tmp_path / "checkpoint.pt"
    identity = {"code_commit": "abc", "config_sha256": "def", "seed": 107}

    with pytest.raises(ValueError, match="empty rollout"):
        save_o2_checkpoint(
            checkpoint,
            identity=identity,
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            schedule_state={"global_env_steps": 1},
            calibration_state=None,
            trainer_state={},
            rollout_empty=False,
        )
    save_o2_checkpoint(
        checkpoint,
        identity=identity,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
        schedule_state={"global_env_steps": 1},
        calibration_state=None,
        trainer_state={"episode_index": 0},
        rollout_empty=True,
    )
    restored = load_o2_checkpoint(
        checkpoint,
        expected_identity=identity,
        actor=actor,
        critic=critic,
        optimizer=optimizer,
    )
    assert restored["trainer_state"] == {"episode_index": 0}
    with pytest.raises(ValueError, match="identity"):
        load_o2_checkpoint(
            checkpoint,
            expected_identity={**identity, "seed": 117},
            actor=actor,
            critic=critic,
            optimizer=optimizer,
        )


def test_throughput_grid_uses_zero_before_first_completed_episode():
    from llm_mappo.o2_evidence import compute_throughput_grid

    grid = compute_throughput_grid(
        [
            {
                "real_env_steps": 12,
                "cumulative_completed_tasks": 3,
                "cumulative_episode_steps": 12,
            },
            {
                "real_env_steps": 20,
                "cumulative_completed_tasks": 5,
                "cumulative_episode_steps": 20,
            },
        ],
        [0, 10, 20],
    )

    assert grid == [
        {"real_env_steps": 0, "throughput": 0.0},
        {"real_env_steps": 10, "throughput": 0.0},
        {"real_env_steps": 20, "throughput": 250.0},
    ]
