"""Extract a standalone thinker checkpoint from a full Qwen3-Omni model.

Two variants, both keyed on model_type so miles' AutoBridge can load them:

- ``text`` (default): thinker text backbone + lm_head as a plain Qwen3-MoE
  (model_type=qwen3_moe, per-expert 2D weights). Audio/visual towers are dropped on
  purpose: the trainer loads the audio tower frozen from the original omni checkpoint
  (see the audio-injection plugin), and the rollout server keeps its own copies.

- ``vl``: thinker visual tower + text backbone masquerading as Qwen3-VL-MoE
  (model_type=qwen3_vl_moe). Names are mechanically compatible except three merger
  renames (ln_q->norm, mlp.0/2->linear_fc1/2, merger_list->deepstack_merger_list) and
  the MoE experts, which the VL hub format fuses to 3D hub layout
  (gate_up_proj [E,H,2I], down_proj [E,I,H]). This unlocks miles' existing
  bridge-based Qwen3-VL training path (packed mrope + deepstack + CP) for image/video
  input. The audio tower is dropped; omni-only rope fields ride along under
  ``omni_sideband`` in config.json for the trainer-side video-rope override.

    python tools/extract_qwen3_omni_thinker.py --src <omni> --dst <thinker> [--variant vl]
"""

from __future__ import annotations

import re

_NON_TEXT_THINKER_SUBMODULES = ("audio_tower.", "visual.", "model.audio_tower.", "model.visual.")

_VL_MERGER_RENAMES = (
    (re.compile(r"^visual\.merger\.ln_q\."), "visual.merger.norm."),
    (re.compile(r"^visual\.merger\.mlp\.0\."), "visual.merger.linear_fc1."),
    (re.compile(r"^visual\.merger\.mlp\.2\."), "visual.merger.linear_fc2."),
    (re.compile(r"^visual\.merger_list\.(\d+)\.ln_q\."), r"visual.deepstack_merger_list.\1.norm."),
    (re.compile(r"^visual\.merger_list\.(\d+)\.mlp\.0\."), r"visual.deepstack_merger_list.\1.linear_fc1."),
    (re.compile(r"^visual\.merger_list\.(\d+)\.mlp\.2\."), r"visual.deepstack_merger_list.\1.linear_fc2."),
)

_EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def map_thinker_param_name(name: str) -> str | None:
    """Full omni param name -> standalone Qwen3-MoE name, or None to drop."""
    if not name.startswith("thinker."):
        return None
    rest = name[len("thinker.") :]
    if rest.startswith(_NON_TEXT_THINKER_SUBMODULES):
        return None
    return rest.replace("model.language_model.", "model.")


def map_thinker_param_name_vl(name: str) -> str | None:
    """Full omni param name -> pseudo Qwen3-VL-MoE name, or None to drop.

    Expert weights keep their per-expert names here; `extract(variant="vl")` fuses them
    into the 3D hub tensors afterwards.
    """
    if not name.startswith("thinker."):
        return None
    rest = name[len("thinker.") :].replace("model.language_model.", "model.")
    if rest.startswith(("audio_tower.", "model.audio_tower.")):
        return None
    if rest == "lm_head.weight":
        return rest
    if rest.startswith("visual."):
        for pattern, replacement in _VL_MERGER_RENAMES:
            if pattern.search(rest):
                rest = pattern.sub(replacement, rest)
                break
        return f"model.{rest}"
    if rest.startswith("model."):
        return rest.replace("model.", "model.language_model.", 1)
    return None


def fuse_layer_experts_vl(gates, ups, downs):
    """Per-expert 2D weights -> fused hub-layout tensors.

    gate/up: [I, H] each -> gate_up_proj [E, H, 2I]; down: [H, I] -> down_proj [E, I, H].
    Hub layout is what transformers (Transpose check_dims), Megatron-Bridge import, and
    sglang's fused loaders all expect.
    """
    import torch

    gate_up = torch.stack([torch.cat([g, u], dim=0).T for g, u in zip(gates, ups, strict=True)], dim=0)
    down = torch.stack([d.T for d in downs], dim=0)
    return gate_up.contiguous(), down.contiguous()


def synthesize_thinker_config(omni_config: dict) -> dict:
    """Plain Qwen3-MoE config dict from the full omni config."""
    thinker = omni_config.get("thinker_config", omni_config)
    cfg = dict(thinker.get("text_config", thinker))
    cfg["model_type"] = "qwen3_moe"
    cfg["architectures"] = ["Qwen3MoeForCausalLM"]
    cfg.setdefault("tie_word_embeddings", False)
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if k not in cfg and k in thinker:
            cfg[k] = thinker[k]
    return cfg


# vision fields whose omni spelling has no qwen3_vl_moe slot (dropped after use)
_VL_VISION_DROPPED = ("apply_vit_abs_pos_embed", "image_size", "spatial_patch_size", "in_chans", "tokens_per_second")


def synthesize_vl_config(omni_config: dict) -> dict:
    """Pseudo Qwen3-VL-MoE config from the full omni config.

    Omni numerics are preserved (vocab 152064, rope_theta 1e6, max_position_embeddings
    65536); rope_scaling is normalized to the VL whitelist; the vision config gains
    num_position_embeddings (the pos_embed table size, (image_size/patch_size)^2) and
    drops omni-only spellings. Fields with no VL slot that the trainer still needs
    (omni video rope) ride under `omni_sideband`.
    """
    thinker = omni_config.get("thinker_config", omni_config)

    text = dict(thinker["text_config"])
    text["model_type"] = "qwen3_vl_moe_text"
    rope_scaling = dict(text.get("rope_scaling") or {})
    text["rope_scaling"] = {
        "rope_type": rope_scaling.get("rope_type") or rope_scaling.get("type", "default"),
        "mrope_section": rope_scaling["mrope_section"],
        "mrope_interleaved": rope_scaling.get("mrope_interleaved", True),
    }
    text.setdefault("bos_token_id", 151643)
    text.setdefault("eos_token_id", 151645)

    vision = dict(thinker["vision_config"])
    vision["model_type"] = "qwen3_vl_moe"
    if "num_position_embeddings" not in vision and "image_size" in vision:
        patch = vision.get("patch_size") or vision.get("spatial_patch_size")
        vision["num_position_embeddings"] = (vision["image_size"] // patch) ** 2
    for key in _VL_VISION_DROPPED:
        vision.pop(key, None)

    cfg = {
        "model_type": "qwen3_vl_moe",
        "architectures": ["Qwen3VLMoeForConditionalGeneration"],
        "tie_word_embeddings": thinker.get("tie_word_embeddings", False),
        "text_config": text,
        "vision_config": vision,
        "omni_sideband": {
            "position_id_per_seconds": thinker.get("position_id_per_seconds"),
            "audio_token_id": thinker.get("audio_token_id"),
        },
    }
    for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        cfg[k] = thinker[k]
    return cfg


class _VlExpertAccumulator:
    """Collects per-expert 2D weights per layer and emits fused hub-layout tensors."""

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self._layers: dict[int, dict[str, dict[int, object]]] = {}

    def offer(self, vl_name: str, tensor) -> list[tuple[str, object]]:
        match = _EXPERT_RE.match(vl_name.replace("model.language_model.", "model.", 1))
        if match is None:
            return [(vl_name, tensor)]
        layer, expert, kind = int(match.group(1)), int(match.group(2)), match.group(3)
        buckets = self._layers.setdefault(layer, {"gate_proj": {}, "up_proj": {}, "down_proj": {}})
        buckets[kind][expert] = tensor
        if all(len(buckets[k]) == self.num_experts for k in ("gate_proj", "up_proj", "down_proj")):
            del self._layers[layer]
            order = range(self.num_experts)
            gate_up, down = fuse_layer_experts_vl(
                [buckets["gate_proj"][e] for e in order],
                [buckets["up_proj"][e] for e in order],
                [buckets["down_proj"][e] for e in order],
            )
            prefix = f"model.language_model.layers.{layer}.mlp.experts"
            return [(f"{prefix}.gate_up_proj", gate_up), (f"{prefix}.down_proj", down)]
        return []

    def assert_drained(self):
        assert not self._layers, f"incomplete expert layers left over: {sorted(self._layers)}"


def extract(src, dst, shard_size_gb: float = 5.0, variant: str = "text") -> int:
    """Stream thinker tensors from src into sharded safetensors under dst.

    Each shard is written (and its buffer released) as soon as it fills, so peak RAM is
    one shard (plus, for the vl variant, at most a few layers of in-flight expert
    weights). Returns the number of tensors written.
    """
    import json
    import os
    import shutil
    from pathlib import Path

    from safetensors import safe_open
    from safetensors.torch import save_file

    assert variant in ("text", "vl"), variant
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(src / "config.json") as f:
        omni_config = json.load(f)
    synthesize = synthesize_thinker_config if variant == "text" else synthesize_vl_config
    out_config = synthesize(omni_config)
    with open(dst / "config.json", "w") as f:
        json.dump(out_config, f, indent=2)

    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.json",
        "chat_template.jinja",
        # processor artifacts: the trainer loads AutoProcessor from this dir to expand
        # audio/vision placeholders exactly like the rollout server
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    ):
        if (src / fname).exists():
            shutil.copy2(src / fname, dst / fname)

    # chat_template.json is a processor-level file; AutoTokenizer only auto-loads the
    # .jinja variant, and the trainer templates via the tokenizer
    if not (src / "chat_template.jinja").exists() and (src / "chat_template.json").exists():
        with open(src / "chat_template.json") as f:
            (dst / "chat_template.jinja").write_text(json.load(f)["chat_template"])

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    map_name = map_thinker_param_name if variant == "text" else map_thinker_param_name_vl
    accumulator = None
    if variant == "vl":
        thinker = omni_config.get("thinker_config", omni_config)
        accumulator = _VlExpertAccumulator(num_experts=thinker["text_config"]["num_experts"])

    shard_size_bytes = int(shard_size_gb * (1024**3))
    out_index: dict[str, str] = {}
    shard_names: list[str] = []
    buf: dict = {}
    buf_bytes = 0
    total_size = 0
    total_kept = 0

    def flush():
        nonlocal buf, buf_bytes, total_size
        if not buf:
            return
        shard_name = f"model-{len(shard_names) + 1:05d}.safetensors"
        save_file(buf, dst / shard_name, metadata={"format": "pt"})
        shard_names.append(shard_name)
        for k, t in buf.items():
            out_index[k] = shard_name
            total_size += t.numel() * t.element_size()
        buf, buf_bytes = {}, 0

    def emit(out_name, tensor):
        nonlocal buf_bytes, total_kept
        buf[out_name] = tensor
        buf_bytes += tensor.numel() * tensor.element_size()
        total_kept += 1
        if buf_bytes >= shard_size_bytes:
            flush()

    for shard_file in shard_files:
        with safe_open(src / shard_file, framework="pt") as reader:
            for key in reader.keys():
                new_key = map_name(key)
                if new_key is None:
                    continue
                tensor = reader.get_tensor(key)
                if accumulator is not None:
                    for out_name, out_tensor in accumulator.offer(new_key, tensor):
                        emit(out_name, out_tensor)
                else:
                    emit(new_key, tensor)
    if accumulator is not None:
        accumulator.assert_drained()
    flush()

    if total_kept == 0:
        raise ValueError(f"no thinker tensors found under {src} (wrong --src?)")

    if len(shard_names) == 1:
        os.rename(dst / shard_names[0], dst / "model.safetensors")
    else:
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": out_index}, f, indent=2)

    print(f"[done] {total_kept} thinker tensors ({variant}) -> {dst}")
    return total_kept


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--shard-size-gb", type=float, default=5.0)
    parser.add_argument("--variant", choices=["text", "vl"], default="text")
    args = parser.parse_args()
    extract(args.src, args.dst, shard_size_gb=args.shard_size_gb, variant=args.variant)


if __name__ == "__main__":
    main()
