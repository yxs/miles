"""Extract a standalone Qwen3-MoE thinker checkpoint from a full Qwen3-Omni model.

miles loads HF via AutoBridge keyed on model_type; the composite omni checkpoint has
no bridge, but the thinker text backbone is a plain Qwen3-MoE. This writes a
self-contained HF dir (thinker text + lm_head, renamed, model_type=qwen3_moe) for the
existing Qwen3MoEBridge.

    python tools/extract_qwen3_omni_thinker.py --src <omni> --dst <thinker>
"""

from __future__ import annotations

_NON_TEXT_THINKER_SUBMODULES = ("audio_tower.", "visual.", "model.audio_tower.", "model.visual.")


def map_thinker_param_name(name: str) -> str | None:
    """Full omni param name -> standalone Qwen3-MoE name, or None to drop."""
    if not name.startswith("thinker."):
        return None
    rest = name[len("thinker.") :]
    if rest.startswith(_NON_TEXT_THINKER_SUBMODULES):
        return None
    return rest.replace("model.language_model.", "model.")


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


def main() -> None:
    import argparse
    import json
    import shutil
    from pathlib import Path

    from safetensors import safe_open
    from safetensors.torch import save_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--shard-size-gb", type=float, default=5.0)
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(src / "config.json") as f:
        omni_config = json.load(f)
    with open(dst / "config.json", "w") as f:
        json.dump(synthesize_thinker_config(omni_config), f, indent=2)

    for fname in (
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "generation_config.json", "chat_template.json", "chat_template.jinja",
    ):
        if (src / fname).exists():
            shutil.copy2(src / fname, dst / fname)

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    shard_size_bytes = int(args.shard_size_gb * (1024 ** 3))
    out_index: dict[str, str] = {}
    out_shards: list[tuple[str, dict]] = []
    buf: dict = {}
    buf_bytes = 0
    total_kept = 0

    def flush():
        nonlocal buf, buf_bytes
        if not buf:
            return
        shard_name = f"model-{len(out_shards) + 1:05d}.safetensors"
        out_shards.append((shard_name, buf))
        for k in buf:
            out_index[k] = shard_name
        buf, buf_bytes = {}, 0

    for shard_file in shard_files:
        with safe_open(src / shard_file, framework="pt") as reader:
            for key in reader.keys():
                new_key = map_thinker_param_name(key)
                if new_key is None:
                    continue
                tensor = reader.get_tensor(key)
                buf[new_key] = tensor
                buf_bytes += tensor.numel() * tensor.element_size()
                total_kept += 1
                if buf_bytes >= shard_size_bytes:
                    flush()
    flush()

    if len(out_shards) == 1:
        save_file(out_shards[0][1], dst / "model.safetensors", metadata={"format": "pt"})
    else:
        total_size = sum(t.numel() * t.element_size() for _, ts in out_shards for t in ts.values())
        for shard_name, tensors in out_shards:
            save_file(tensors, dst / shard_name, metadata={"format": "pt"})
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": out_index}, f, indent=2)

    print(f"[done] {total_kept} thinker tensors -> {dst}")


if __name__ == "__main__":
    main()
