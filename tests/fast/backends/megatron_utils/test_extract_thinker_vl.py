"""Pseudo-Qwen3-VL extraction: thinker (visual+text) masquerading as qwen3_vl_moe."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

_REPO = Path(__file__).resolve().parents[4]


def _load_tool():
    path = _REPO / "tools" / "extract_qwen3_omni_thinker.py"
    spec = importlib.util.spec_from_file_location("extract_qwen3_omni_thinker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vl_name_map_text_and_visual():
    m = _load_tool().map_thinker_param_name_vl
    assert m("thinker.lm_head.weight") == "lm_head.weight"
    assert m("thinker.model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert m("thinker.model.layers.3.self_attn.q_proj.weight") == "model.language_model.layers.3.self_attn.q_proj.weight"
    assert m("thinker.model.layers.3.mlp.gate.weight") == "model.language_model.layers.3.mlp.gate.weight"
    assert m("thinker.visual.patch_embed.proj.weight") == "model.visual.patch_embed.proj.weight"
    assert m("thinker.visual.blocks.7.attn.qkv.weight") == "model.visual.blocks.7.attn.qkv.weight"
    # merger renames: ln_q -> norm, mlp.0/2 -> linear_fc1/2, merger_list -> deepstack_merger_list
    assert m("thinker.visual.merger.ln_q.weight") == "model.visual.merger.norm.weight"
    assert m("thinker.visual.merger.mlp.0.bias") == "model.visual.merger.linear_fc1.bias"
    assert m("thinker.visual.merger.mlp.2.weight") == "model.visual.merger.linear_fc2.weight"
    assert m("thinker.visual.merger_list.1.ln_q.weight") == "model.visual.deepstack_merger_list.1.norm.weight"
    assert m("thinker.visual.merger_list.2.mlp.2.bias") == "model.visual.deepstack_merger_list.2.linear_fc2.bias"
    # drops
    assert m("thinker.audio_tower.layers.0.fc1.weight") is None
    assert m("talker.model.norm.weight") is None
    assert m("code2wav.decoder.weight") is None


def test_vl_expert_fusion_hub_layout():
    tool = _load_tool()
    # E=2 experts, I=3 (moe ffn), H=4 (hidden)
    gate = [torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100 * e for e in range(2)]
    up = [torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100 * e + 50 for e in range(2)]
    down = [torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100 * e for e in range(2)]

    gate_up_fused, down_fused = tool.fuse_layer_experts_vl(gate, up, down)

    # hub layout: gate_up [E, H, 2I], down [E, I, H]
    assert gate_up_fused.shape == (2, 4, 6)
    assert down_fused.shape == (2, 3, 4)
    for e in range(2):
        expected = torch.cat([gate[e], up[e]], dim=0).T  # [H, 2I]
        assert torch.equal(gate_up_fused[e], expected)
        assert torch.equal(down_fused[e], down[e].T)


def test_vl_config_synthesis():
    synth = _load_tool().synthesize_vl_config
    omni = {
        "model_type": "qwen3_omni_moe",
        "thinker_config": {
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "position_id_per_seconds": 13,
            "audio_token_id": 151675,
            "tie_word_embeddings": False,
            "text_config": {
                "model_type": "qwen3_omni_moe_text",
                "vocab_size": 152064,
                "hidden_size": 2048,
                "num_hidden_layers": 48,
                "rope_theta": 1000000,
                "max_position_embeddings": 65536,
                "rope_scaling": {"interleaved": True, "mrope_interleaved": True, "mrope_section": [24, 20, 20], "type": "default"},
                "num_experts": 128,
            },
            "vision_config": {
                "model_type": "qwen3_omni_moe_vision_encoder",
                "depth": 27,
                "hidden_size": 1152,
                "deepstack_visual_indexes": [8, 16, 24],
                "apply_vit_abs_pos_embed": True,
                "image_size": 768,
                "spatial_patch_size": 16,
                "in_chans": 3,
                "tokens_per_second": 2,
                "out_hidden_size": 2048,
                "spatial_merge_size": 2,
                "patch_size": 16,
                "temporal_patch_size": 2,
            },
            "audio_config": {"d_model": 1280},
        },
    }

    cfg = synth(omni)

    assert cfg["model_type"] == "qwen3_vl_moe"
    assert cfg["architectures"] == ["Qwen3VLMoeForConditionalGeneration"]
    assert cfg["tie_word_embeddings"] is False
    for key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        assert cfg[key] == omni["thinker_config"][key]
    text = cfg["text_config"]
    assert text["model_type"] == "qwen3_vl_moe_text"
    assert text["vocab_size"] == 152064 and text["rope_theta"] == 1000000
    assert text["max_position_embeddings"] == 65536
    assert text["rope_scaling"] == {"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True}
    assert text["bos_token_id"] == 151643 and text["eos_token_id"] == 151645
    vision = cfg["vision_config"]
    assert vision["model_type"] == "qwen3_vl_moe"
    assert vision["num_position_embeddings"] == 2304  # (image_size // patch_size) ** 2
    for gone in ("apply_vit_abs_pos_embed", "image_size", "spatial_patch_size", "in_chans", "tokens_per_second"):
        assert gone not in vision
    assert "audio_config" not in cfg
    # sideband for the trainer-side omni video rope (no VL slot for it)
    assert cfg["omni_sideband"] == {"position_id_per_seconds": 13, "audio_token_id": 151675}


def _tiny_omni_src(src: Path):
    src.mkdir(parents=True, exist_ok=True)
    thinker = {
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "position_id_per_seconds": 13,
        "audio_token_id": 151675,
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_omni_moe_text",
            "vocab_size": 32,
            "num_experts": 2,
            "rope_scaling": {"interleaved": True, "mrope_interleaved": True, "mrope_section": [24, 20, 20], "type": "default"},
        },
        "vision_config": {"model_type": "x", "image_size": 32, "patch_size": 16},
    }
    with open(src / "config.json", "w") as f:
        json.dump({"thinker_config": thinker}, f)
    tensors = {
        "thinker.model.embed_tokens.weight": torch.randn(32, 4),
        "thinker.lm_head.weight": torch.randn(32, 4),
        "thinker.visual.merger.ln_q.weight": torch.randn(6),
        "thinker.audio_tower.proj1.weight": torch.zeros(2, 2),
    }
    for e in range(2):
        tensors[f"thinker.model.layers.0.mlp.experts.{e}.gate_proj.weight"] = torch.randn(3, 4)
        tensors[f"thinker.model.layers.0.mlp.experts.{e}.up_proj.weight"] = torch.randn(3, 4)
        tensors[f"thinker.model.layers.0.mlp.experts.{e}.down_proj.weight"] = torch.randn(4, 3)
    save_file(tensors, src / "model.safetensors", metadata={"format": "pt"})
    return tensors


def test_extract_vl_roundtrip(tmp_path):
    tool = _load_tool()
    src, dst = tmp_path / "src", tmp_path / "dst"
    tensors = _tiny_omni_src(src)

    tool.extract(src, dst, shard_size_gb=1.0, variant="vl")

    with open(dst / "config.json") as f:
        cfg = json.load(f)
    assert cfg["model_type"] == "qwen3_vl_moe"

    out = {}
    with safe_open(dst / "model.safetensors", framework="pt") as reader:
        for key in reader.keys():
            out[key] = reader.get_tensor(key)

    assert "model.language_model.embed_tokens.weight" in out
    assert "lm_head.weight" in out
    assert "model.visual.merger.norm.weight" in out
    assert not any("audio_tower" in k for k in out)
    gate_up = out["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    down = out["model.language_model.layers.0.mlp.experts.down_proj"]
    assert gate_up.shape == (2, 4, 6) and down.shape == (2, 3, 4)
    expected_e1 = torch.cat(
        [
            tensors["thinker.model.layers.0.mlp.experts.1.gate_proj.weight"],
            tensors["thinker.model.layers.0.mlp.experts.1.up_proj.weight"],
        ],
        dim=0,
    ).T
    assert torch.equal(gate_up[1], expected_e1)
    # per-expert 2D names must NOT leak through
    assert not any(".experts.0." in k or ".experts.1." in k for k in out)


def test_extract_text_variant_unchanged(tmp_path):
    tool = _load_tool()
    src, dst = tmp_path / "src", tmp_path / "dst"
    _tiny_omni_src(src)

    tool.extract(src, dst, shard_size_gb=1.0, variant="text")

    with open(dst / "config.json") as f:
        assert json.load(f)["model_type"] == "qwen3_moe"
    with safe_open(dst / "model.safetensors", framework="pt") as reader:
        keys = list(reader.keys())
    assert "model.embed_tokens.weight" in keys
    assert any(".experts.0.gate_proj" in k for k in keys)  # text variant keeps per-expert 2D
    assert not any("visual" in k for k in keys)
