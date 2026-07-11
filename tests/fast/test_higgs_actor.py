import pytest
import torch

from miles_plugins.omni.higgs_actor import (
    _FUSED_EMBED_KEY,
    backbone_parameter_to_checkpoint_name,
    build_full_server_weights,
    clipped_grpo_loss,
    selected_codebook_logprobs,
)


def test_selected_codebook_logprobs_scores_every_codebook():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    fused_weight = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    codes = torch.tensor([[0, 2], [1, 0]])

    actual = selected_codebook_logprobs(
        hidden,
        fused_weight,
        codes,
        num_codebooks=2,
        codebook_vocab=3,
        temperature=0.5,
    )

    logits = torch.nn.functional.linear(hidden, fused_weight).view(2, 2, 3) / 0.5
    expected = torch.log_softmax(logits, dim=-1).gather(-1, codes.unsqueeze(-1)).squeeze(-1)
    assert actual.shape == (2, 2)
    assert torch.allclose(actual, expected)


def test_clipped_grpo_loss_backpropagates_through_all_unmasked_codebooks():
    current = torch.tensor([[-0.2, -0.3], [-0.4, -0.5]], requires_grad=True)
    old = torch.tensor([[-0.25, -0.35], [-0.45, -0.55]])
    mask = torch.tensor([[True, True], [True, False]])

    loss = clipped_grpo_loss(current, old, mask, advantage=0.7, clip_eps=0.2)
    loss.backward()

    assert current.grad is not None
    assert torch.all(current.grad[mask] != 0)
    assert current.grad[~mask].item() == 0


@pytest.mark.parametrize(
    ("actor_name", "checkpoint_name"),
    [
        ("embed_tokens.weight", "tied.embedding.text_embedding.weight"),
        ("layers.2.self_attn.q_proj.weight", "body.layers.2.self_attn.q_proj.weight"),
        ("norm.weight", "body.norm.weight"),
    ],
)
def test_backbone_parameter_to_checkpoint_name(actor_name, checkpoint_name):
    assert backbone_parameter_to_checkpoint_name(actor_name) == checkpoint_name


def test_build_full_server_weights_includes_backbone_and_tied_codebook_weight():
    tensors = {
        "embed_tokens.weight": torch.randn(3, 2),
        "layers.0.self_attn.q_proj.weight": torch.randn(2, 2),
        "norm.weight": torch.randn(2),
    }

    class FakeBackbone:
        def named_parameters(self):
            return iter(tensors.items())

    fused_weight = torch.randn(6, 2)
    weights = build_full_server_weights(FakeBackbone(), fused_weight)

    assert set(weights) == {
        "tied.embedding.text_embedding.weight",
        "body.layers.0.self_attn.q_proj.weight",
        "body.norm.weight",
        _FUSED_EMBED_KEY,
    }
    assert weights[_FUSED_EMBED_KEY] is fused_weight
