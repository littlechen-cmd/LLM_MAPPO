import torch

from llm_mappo.optimization_student import O0CentralizedCritic, O0StudentActor


def _has_grad(module):
    return any(parameter.grad is not None for parameter in module.parameters())


def test_o0_student_uses_the_frozen_actor_and_critic_shapes():
    actor = O0StudentActor()
    critic = O0CentralizedCritic()
    physical = torch.zeros((2, 5, 613))
    semantic = torch.zeros((2, 5, 61))

    output = actor(physical, semantic)

    assert output.action_logits.shape == (2, 5, 5)
    assert output.motion_logits.shape == (2, 5, 3)
    assert output.semantic_scores.shape == (2, 5, 3)
    assert critic(physical).shape == (2,)


def test_o0_student_enforces_gradient_ownership():
    physical = torch.randn((2, 613))
    semantic = torch.randn((2, 61))

    actor = O0StudentActor()
    actor(physical, semantic).action_logits.sum().backward()
    assert _has_grad(actor.motion_encoder)
    assert _has_grad(actor.semantic_adapter)
    assert _has_grad(actor.action_head)
    assert not _has_grad(actor.motion_prior_head)
    assert not _has_grad(actor.semantic_encoder)
    assert not _has_grad(actor.semantic_head)

    actor = O0StudentActor()
    actor(physical, semantic).motion_logits.sum().backward()
    assert _has_grad(actor.motion_encoder)
    assert _has_grad(actor.motion_prior_head)
    assert not _has_grad(actor.semantic_encoder)
    assert not _has_grad(actor.semantic_head)
    assert not _has_grad(actor.semantic_adapter)
    assert not _has_grad(actor.action_head)

    actor = O0StudentActor()
    target = torch.zeros((2, 3))
    torch.nn.functional.mse_loss(actor(physical, semantic).semantic_scores, target).backward()
    assert _has_grad(actor.semantic_encoder)
    assert _has_grad(actor.semantic_head)
    assert not _has_grad(actor.motion_encoder)
    assert not _has_grad(actor.motion_prior_head)
    assert not _has_grad(actor.semantic_adapter)
    assert not _has_grad(actor.action_head)
