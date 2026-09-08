"""Behavior contracts for Phase 1 recurrent MAPPO networks."""

import torch
import numpy as np

from llm_mappo.mappo_baseline.networks import MAPPOActor, MAPPOFPCritic
from llm_mappo.mappo_baseline.storage import MAPPORolloutStorage


def test_actor_keeps_one_hidden_state_for_each_worker_and_agent() -> None:
    actor = MAPPOActor(observation_dim=7, action_dim=5, hidden_dim=64)
    observations = torch.randn(2, 3, 7)
    hidden = torch.zeros(2, 3, 64)

    logits, next_hidden = actor(observations, hidden)

    assert logits.shape == (2, 3, 5)
    assert next_hidden.shape == (2, 3, 64)
    assert not torch.equal(next_hidden[0, 0], next_hidden[1, 2])


def test_critic_keeps_separate_recurrent_state_from_actor_state() -> None:
    critic = MAPPOFPCritic(state_dim=11, hidden_dim=64)
    states = torch.randn(2, 3, 11)
    hidden = torch.zeros(2, 3, 64)

    values, next_hidden = critic(states, hidden)

    assert values.shape == (2, 3)
    assert next_hidden.shape == (2, 3, 64)


def test_rollout_records_hidden_state_before_each_recurrent_action() -> None:
    storage = MAPPORolloutStorage(
        rollout_length=1,
        num_workers=1,
        num_agents=2,
        observation_dim=3,
        state_dim=4,
        action_dim=5,
        recurrent_hidden_dim=64,
    )
    storage.insert(
        observations=np.zeros((1, 2, 3), dtype=np.float32),
        fp_states=np.zeros((1, 2, 4), dtype=np.float32),
        actions=np.zeros((1, 2), dtype=np.int64),
        action_masks=np.ones((1, 2, 5), dtype=bool),
        log_probs=np.zeros((1, 2), dtype=np.float32),
        values=np.zeros((1, 2), dtype=np.float32),
        rewards=np.zeros((1, 2), dtype=np.float32),
        dones=np.zeros((1, 2), dtype=bool),
        active_masks=np.ones((1, 2), dtype=np.float32),
        actor_hidden_states=np.zeros((1, 2, 64), dtype=np.float32),
        critic_hidden_states=np.zeros((1, 2, 64), dtype=np.float32),
    )

    batch = storage.compute_returns(
        next_values=np.zeros((1, 2), dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert batch.actor_hidden_states.shape == (1, 1, 2, 64)
    assert batch.critic_hidden_states.shape == (1, 1, 2, 64)


def test_recurrent_sequence_resets_hidden_state_at_episode_boundary() -> None:
    torch.manual_seed(9)
    actor = MAPPOActor(observation_dim=7, action_dim=5, hidden_dim=64)
    sequence = torch.randn(2, 1, 1, 7)
    initial_hidden = torch.randn(1, 1, 64)
    continuation = torch.tensor([[[1.0]], [[0.0]]])

    logits, _ = actor.forward_sequence(sequence, initial_hidden, continuation)
    reset_logits, _ = actor(sequence[1], torch.zeros(1, 1, 64))

    torch.testing.assert_close(logits[1], reset_logits)
