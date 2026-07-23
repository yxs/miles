"""Extract a standalone Qwen3-MoE thinker checkpoint from a full Qwen3-Omni model.

miles loads HF via AutoBridge keyed on model_type; the composite omni checkpoint has
no bridge, but the thinker text backbone is a plain Qwen3-MoE. This writes a
self-contained HF dir (thinker text + lm_head, renamed, model_type=qwen3_moe) for the
existing Qwen3MoEBridge. Audio/visual towers are dropped here on purpose: the trainer
loads them frozen from the original omni checkpoint (see the audio-injection wrapper),
and the rollout server keeps its own copies.

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


def extract(src, dst, shard_size_gb: float = 5.0) -> int:
    """Stream thinker tensors from src into sharded safetensors under dst.

    Each shard is written (and its buffer released) as soon as it fills, so peak RAM is
    one shard, not the whole ~60 GB thinker. Returns the number of kept tensors.
    """
    import json
    import os
    import shutil
    from pathlib import Path

    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(src / "config.json") as f:
        omni_config = json.load(f)
    with open(dst / "config.json", "w") as f:
        json.dump(synthesize_thinker_config(omni_config), f, indent=2)

    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.json",
        "chat_template.jinja",
    ):
        if (src / fname).exists():
            shutil.copy2(src / fname, dst / fname)

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

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

    if total_kept == 0:
        raise ValueError(f"no thinker tensors found under {src} (wrong --src?)")

    if len(shard_names) == 1:
        os.rename(dst / shard_names[0], dst / "model.safetensors")
    else:
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": out_index}, f, indent=2)

    print(f"[done] {total_kept} thinker tensors -> {dst}")
    return total_kept


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--shard-size-gb", type=float, default=5.0)
    args = parser.parse_args()
    extract(args.src, args.dst, shard_size_gb=args.shard_size_gb)


if __name__ == "__main__":
    main()
