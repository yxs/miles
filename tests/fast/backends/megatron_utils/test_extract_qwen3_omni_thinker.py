"""Qwen3-Omni thinker extraction tool: name/config mapping + streaming shard writes."""

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
    assert (
        m("thinker.model.layers.7.mlp.experts.3.down_proj.weight") == "model.layers.7.mlp.experts.3.down_proj.weight"
    )
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


def _write_omni_src(src: Path, tensors: dict[str, torch.Tensor]):
    src.mkdir(parents=True, exist_ok=True)
    with open(src / "config.json", "w") as f:
        json.dump({"thinker_config": {"text_config": {"vocab_size": 32}}}, f)
    save_file(tensors, src / "model.safetensors", metadata={"format": "pt"})


def _read_all(dst: Path) -> dict[str, torch.Tensor]:
    index_path = dst / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        weight_map, shards = None, ["model.safetensors"]
    out = {}
    for shard in shards:
        with safe_open(dst / shard, framework="pt") as r:
            for k in r.keys():
                out[k] = r.get_tensor(k)
                if weight_map is not None:
                    assert weight_map[k] == shard
    return out


def test_extract_single_shard_roundtrip(tmp_path):
    tool = _load_extract_tool()
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_omni_src(
        src,
        {
            "thinker.model.embed_tokens.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "thinker.audio_tower.layers.0.fc1.weight": torch.zeros(2, 2),
            "talker.model.norm.weight": torch.zeros(2),
        },
    )

    total = tool.extract(src, dst, shard_size_gb=1.0)

    assert total == 1
    assert (dst / "model.safetensors").exists()
    assert not (dst / "model.safetensors.index.json").exists()
    out = _read_all(dst)
    assert set(out) == {"model.embed_tokens.weight"}
    assert torch.equal(out["model.embed_tokens.weight"], torch.arange(6, dtype=torch.float32).reshape(2, 3))
    with open(dst / "config.json") as f:
        assert json.load(f)["model_type"] == "qwen3_moe"


def test_extract_multi_shard_writes_streaming_index(tmp_path):
    tool = _load_extract_tool()
    src, dst = tmp_path / "src", tmp_path / "dst"
    tensors = {f"thinker.model.layers.{i}.mlp.gate.weight": torch.full((64, 64), float(i)) for i in range(4)}
    _write_omni_src(src, tensors)

    # 64*64*4B = 16KiB per tensor; 20KiB shards force one tensor per shard (flush at >=1 tensor)
    total = tool.extract(src, dst, shard_size_gb=20 * 1024 / (1024**3))

    assert total == 4
    index_path = dst / "model.safetensors.index.json"
    assert index_path.exists()
    with open(index_path) as f:
        index = json.load(f)
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in tensors.values())
    assert len(set(index["weight_map"].values())) > 1, "expected multiple shards"
    out = _read_all(dst)
    for i in range(4):
        assert torch.equal(out[f"model.layers.{i}.mlp.gate.weight"], torch.full((64, 64), float(i)))


def test_extract_fails_loud_when_no_thinker_tensors(tmp_path):
    tool = _load_extract_tool()
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_omni_src(src, {"talker.model.norm.weight": torch.zeros(2)})

    with pytest.raises(ValueError, match="no thinker tensors"):
        tool.extract(src, dst, shard_size_gb=1.0)
