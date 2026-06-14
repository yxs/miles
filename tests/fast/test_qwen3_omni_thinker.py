"""Qwen3-Omni thinker bridge: extraction name/config logic + the body.* converter contract."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _load_extract_tool():
    path = _REPO / "tools" / "extract_qwen3_omni_thinker.py"
    spec = importlib.util.spec_from_file_location("extract_qwen3_omni_thinker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_map_thinker_param_name_keeps_text_backbone():
    m = _load_extract_tool().map_thinker_param_name
    assert m("thinker.model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert m("thinker.lm_head.weight") == "lm_head.weight"
    assert m("thinker.model.norm.weight") == "model.norm.weight"
    assert m("thinker.model.layers.7.mlp.experts.3.down_proj.weight") == "model.layers.7.mlp.experts.3.down_proj.weight"
    assert m("thinker.model.layers.0.self_attn.q_norm.weight") == "model.layers.0.self_attn.q_norm.weight"


def test_map_thinker_param_name_normalizes_language_model_infix():
    m = _load_extract_tool().map_thinker_param_name
    assert m("thinker.model.language_model.layers.0.mlp.gate.weight") == "model.layers.0.mlp.gate.weight"


def test_map_thinker_param_name_drops_non_thinker_and_non_text():
    m = _load_extract_tool().map_thinker_param_name
    assert m("talker.model.layers.0.self_attn.q_proj.weight") is None
    assert m("code2wav.decoder.weight") is None
    assert m("thinker.audio_tower.layers.0.fc1.weight") is None
    assert m("thinker.visual.blocks.0.attn.qkv.weight") is None


def test_synthesize_thinker_config_stamps_qwen3_moe():
    synth = _load_extract_tool().synthesize_thinker_config
    omni = {
        "model_type": "qwen3_omni_moe",
        "thinker_config": {
            "eos_token_id": 151645,
            "text_config": {"vocab_size": 152064, "num_experts": 128, "tie_word_embeddings": False},
        },
    }
    cfg = synth(omni)
    assert cfg["model_type"] == "qwen3_moe"
    assert cfg["architectures"] == ["Qwen3MoeForCausalLM"]
    assert cfg["num_experts"] == 128
    assert cfg["vocab_size"] == 152064
    assert cfg["tie_word_embeddings"] is False
    assert cfg["eos_token_id"] == 151645


def test_synthesize_thinker_config_flat_fallback():
    cfg = _load_extract_tool().synthesize_thinker_config({"vocab_size": 100, "num_experts": 8})
    assert cfg["model_type"] == "qwen3_moe"
    assert cfg["vocab_size"] == 100
    assert cfg["tie_word_embeddings"] is False


def _args():
    # head_dim=kv_channels=2, value_num_per_group=heads//groups=2
    return SimpleNamespace(hidden_size=8, num_attention_heads=4, num_query_groups=2, kv_channels=2)


def _conv():
    return pytest.importorskip(
        "miles.backends.megatron_utils.megatron_to_hf.qwen3omni_moe"
    ).convert_qwen3omni_moe_to_hf


def test_body_converter_simple_params():
    torch = pytest.importorskip("torch")
    conv, args = _conv(), _args()
    names = lambda n, p: [x for x, _ in conv(args, n, p)]
    assert names("module.module.embedding.word_embeddings.weight", torch.zeros(10, 8)) == ["body.model.embed_tokens.weight"]
    assert names("module.module.output_layer.weight", torch.zeros(10, 8)) == ["body.lm_head.weight"]
    assert names("module.module.decoder.final_layernorm.weight", torch.zeros(8)) == ["body.model.norm.weight"]
    assert names("module.module.decoder.layers.3.mlp.router.weight", torch.zeros(128, 8)) == ["body.model.layers.3.mlp.gate.weight"]
    assert names("module.module.decoder.layers.3.self_attention.q_layernorm.weight", torch.zeros(2)) == ["body.model.layers.3.self_attn.q_norm.weight"]
    assert names("module.module.decoder.layers.3.self_attention.k_layernorm.weight", torch.zeros(2)) == ["body.model.layers.3.self_attn.k_norm.weight"]


def test_body_converter_fused_splits():
    torch = pytest.importorskip("torch")
    conv, args = _conv(), _args()

    # qkv: (groups*(vpg+2)*head_dim, hidden) = (16, 8) -> q(8,8) k(4,8) v(4,8)
    out = conv(args, "module.module.decoder.layers.1.self_attention.linear_qkv.weight", torch.zeros(16, 8))
    assert {n: tuple(t.shape) for n, t in out} == {
        "body.model.layers.1.self_attn.q_proj.weight": (8, 8),
        "body.model.layers.1.self_attn.k_proj.weight": (4, 8),
        "body.model.layers.1.self_attn.v_proj.weight": (4, 8),
    }
    # expert fc1: gate+up fused on dim 0 -> chunk(2)
    out = conv(args, "module.module.decoder.layers.1.mlp.experts.linear_fc1.weight5", torch.zeros(8, 8))
    assert {n: tuple(t.shape) for n, t in out} == {
        "body.model.layers.1.mlp.experts.5.gate_proj.weight": (4, 8),
        "body.model.layers.1.mlp.experts.5.up_proj.weight": (4, 8),
    }
    out = conv(args, "module.module.decoder.layers.1.mlp.experts.linear_fc2.weight5", torch.zeros(8, 4))
    assert [n for n, _ in out] == ["body.model.layers.1.mlp.experts.5.down_proj.weight"]


def test_body_converter_dispatch_routes_qwen3omni():
    torch = pytest.importorskip("torch")
    mod = pytest.importorskip("miles.backends.megatron_utils.megatron_to_hf")
    out = mod._convert_to_hf_core(
        _args(), "qwen3omni_moe", "module.module.decoder.layers.0.mlp.router.weight", torch.zeros(128, 8)
    )
    assert [n for n, _ in out] == ["body.model.layers.0.mlp.gate.weight"]


# --- rollout adapter (miles.rollout.generate_hub.omni_thinker.generate) ---


def _omni_mod():
    return pytest.importorskip("miles.rollout.generate_hub.omni_thinker")


def _omni_input(monkeypatch, omni, *, replay=False, indexer_replay=False, multimodal=None, metadata=None):
    """A generate() input with post/prompt-id mocked; returns (input, sample, captured)."""
    Sample = pytest.importorskip("miles.utils.types").Sample
    captured = {}

    async def fake_post(url, payload):
        captured["url"], captured["payload"] = url, payload
        return {
            "text": "hello",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 11], [-0.2, 22]],
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cached_tokens": 0,
            },
        }

    monkeypatch.setattr(omni, "post", fake_post)
    monkeypatch.setattr(omni, "compute_prompt_ids_from_sample", lambda state, sample: [1, 2, 3])

    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        use_rollout_routing_replay=replay,
        use_rollout_indexer_replay=indexer_replay,
        rollout_max_response_len=128,
        rollout_max_context_len=0,
        sglang_speculative_algorithm=None,
    )
    sample = Sample(status=Sample.Status.PENDING, metadata=metadata or {}, multimodal_inputs=multimodal)
    inp = SimpleNamespace(
        args=args, sample=sample, sampling_params={"temperature": 1.0, "max_new_tokens": 64}, state=None
    )
    return inp, sample, captured


def test_generate_builds_text_only_omni_payload_and_parses_response(monkeypatch):
    omni = _omni_mod()
    inp, sample, captured = _omni_input(monkeypatch, omni, metadata={"task": "math"})

    asyncio.run(omni.generate(inp))

    p = captured["payload"]
    assert captured["url"].endswith("/generate")
    assert p["input_ids"] == [1, 2, 3]
    assert p["output_modalities"] == ["text"]
    assert p["return_omni_rollout"] is False
    assert p["return_routed_experts"] is False
    assert p["return_indexer_topk"] is False
    assert p["sampling_params"]["repetition_penalty"] == 1.0
    assert p["metadata"] == {"task": "math"}
    # response parse: tokens = input_ids + decoded ids, logprobs aligned, status from finish_reason
    assert sample.tokens == [1, 2, 3, 11, 22]
    assert sample.rollout_log_probs == [-0.1, -0.2]
    assert sample.response == "hello"
    assert sample.status.name == "COMPLETED"


def test_generate_rejects_moe_replay_for_omni(monkeypatch):
    omni = _omni_mod()
    inp, _, _ = _omni_input(monkeypatch, omni, replay=True)
    with pytest.raises(AssertionError, match="replay"):
        asyncio.run(omni.generate(inp))


def test_generate_rejects_indexer_replay_for_omni(monkeypatch):
    omni = _omni_mod()
    inp, _, _ = _omni_input(monkeypatch, omni, indexer_replay=True)
    with pytest.raises(AssertionError, match="replay"):
        asyncio.run(omni.generate(inp))


def test_generate_rejects_multimodal_on_text_only_path(monkeypatch):
    omni = _omni_mod()
    inp, _, _ = _omni_input(monkeypatch, omni, multimodal={"images": ["img"]})
    with pytest.raises(AssertionError, match="text-only"):
        asyncio.run(omni.generate(inp))
