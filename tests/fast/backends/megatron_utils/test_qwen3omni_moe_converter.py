"""Qwen3-Omni thinker Megatron->HF converter: thinker.* naming + dispatch contract."""

from types import SimpleNamespace

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-cpu", labels=[])

from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from miles.backends.megatron_utils.megatron_to_hf.qwen3omni_moe import convert_qwen3omni_moe_to_hf


def _args():
    # head_dim=kv_channels=2, value_num_per_group=heads//groups=2
    return SimpleNamespace(hidden_size=8, num_attention_heads=4, num_query_groups=2, kv_channels=2)


def _names(name, param):
    return [x for x, _ in convert_qwen3omni_moe_to_hf(_args(), name, param)]


def test_thinker_converter_simple_params():
    assert _names("module.module.embedding.word_embeddings.weight", torch.zeros(10, 8)) == [
        "thinker.model.embed_tokens.weight"
    ]
    assert _names("module.module.output_layer.weight", torch.zeros(10, 8)) == ["thinker.lm_head.weight"]
    assert _names("module.module.decoder.final_layernorm.weight", torch.zeros(8)) == ["thinker.model.norm.weight"]
    assert _names("module.module.decoder.layers.3.mlp.router.weight", torch.zeros(128, 8)) == [
        "thinker.model.layers.3.mlp.gate.weight"
    ]
    assert _names("module.module.decoder.layers.3.self_attention.q_layernorm.weight", torch.zeros(2)) == [
        "thinker.model.layers.3.self_attn.q_norm.weight"
    ]


def test_thinker_converter_fused_splits():
    conv, args = convert_qwen3omni_moe_to_hf, _args()

    # qkv: (groups*(vpg+2)*head_dim, hidden) = (16, 8) -> q(8,8) k(4,8) v(4,8)
    out = conv(args, "module.module.decoder.layers.1.self_attention.linear_qkv.weight", torch.zeros(16, 8))
    assert {n: tuple(t.shape) for n, t in out} == {
        "thinker.model.layers.1.self_attn.q_proj.weight": (8, 8),
        "thinker.model.layers.1.self_attn.k_proj.weight": (4, 8),
        "thinker.model.layers.1.self_attn.v_proj.weight": (4, 8),
    }

    # expert fc1: chunk into gate/up
    out = conv(args, "module.module.decoder.layers.2.mlp.experts.linear_fc1.weight5", torch.zeros(12, 8))
    assert {n: tuple(t.shape) for n, t in out} == {
        "thinker.model.layers.2.mlp.experts.5.gate_proj.weight": (6, 8),
        "thinker.model.layers.2.mlp.experts.5.up_proj.weight": (6, 8),
    }


def test_thinker_converter_never_emits_body_prefix():
    # body.* is the Higgs-TTS namespace; Qwen3-Omni thinker load_weights expects thinker.*/HF names
    for name, param in [
        ("module.module.embedding.word_embeddings.weight", torch.zeros(10, 8)),
        ("module.module.decoder.layers.0.mlp.experts.linear_fc2.weight0", torch.zeros(8, 6)),
    ]:
        for hf_name, _ in convert_qwen3omni_moe_to_hf(_args(), name, param):
            assert hf_name.startswith("thinker."), hf_name
            assert not hf_name.startswith("body."), hf_name


def test_dispatch_routes_qwen3omni_before_generic_qwen3():
    # "qwen3omni_moe" does not contain "qwen3moe"; without the dedicated branch it would fall
    # through to the dense qwen2 converter and crash on MoE params
    out = _convert_to_hf_core(
        _args(), "qwen3omni_moe", "module.module.embedding.word_embeddings.weight", torch.zeros(10, 8)
    )
    assert [n for n, _ in out] == ["thinker.model.embed_tokens.weight"]


def test_dispatch_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        _convert_to_hf_core(_args(), "unknown_model_xyz", "module.module.x", torch.zeros(1))
