"""Extract the Qwen3-Omni thinker submodule into a standalone HF checkpoint.

The full Qwen3-Omni checkpoint stores the thinker / talker / code2wav stacks under
prefixed keys (``thinker.*`` / ``talker.*`` / ``code2wav.*``). For thinker-only RL
training, we strip the ``thinker.`` prefix and write a standalone
``Qwen3OmniMoeThinkerForConditionalGeneration`` checkpoint. Its config carries a
``vision_config``, so the FSDP actor's ``get_model_cls()`` loads it via
``AutoModelForImageTextToText`` (text-only input just runs the LM → logits).

Streaming + RAM-bounded: tensors are copied shard-by-shard and flushed once a size
budget is reached, so the 30B model never needs to fit in memory at once.

Usage::

    python -m miles_plugins.omni.extract_thinker --src <full_omni_ckpt> --dst <thinker_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict

PREFIX = "thinker."

# Tokenizer / aux files copied verbatim so the standalone dir is self-contained.
AUX_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
]

DEFAULT_SHARD_BYTES = 5 * 1024**3


def extract_thinker(src: str, dst: str, shard_bytes: int = DEFAULT_SHARD_BYTES) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    os.makedirs(dst, exist_ok=True)
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    input_shards = sorted(set(weight_map.values()))

    buf: "OrderedDict[str, object]" = OrderedDict()
    buf_bytes = 0
    out_shards: list[str] = []
    out_weight_map: dict[str, str] = {}
    total_bytes = 0

    def flush() -> None:
        nonlocal buf, buf_bytes
        if not buf:
            return
        name = f"model-{len(out_shards) + 1:05d}.safetensors"
        save_file(buf, os.path.join(dst, name), metadata={"format": "pt"})
        for key in buf:
            out_weight_map[key] = name
        out_shards.append(name)
        buf = OrderedDict()
        buf_bytes = 0

    for shard in input_shards:
        with safe_open(os.path.join(src, shard), framework="pt") as f:
            for key in f.keys():
                if not key.startswith(PREFIX):
                    continue
                tensor = f.get_tensor(key)
                new_key = key[len(PREFIX) :]
                buf[new_key] = tensor
                nbytes = tensor.numel() * tensor.element_size()
                buf_bytes += nbytes
                total_bytes += nbytes
                if buf_bytes >= shard_bytes:
                    flush()
    flush()

    # Rename single-shard output to the conventional unsharded filename.
    if len(out_shards) == 1:
        only = out_shards[0]
        os.replace(os.path.join(dst, only), os.path.join(dst, "model.safetensors"))
        out_weight_map = {k: "model.safetensors" for k in out_weight_map}
    json.dump(
        {"metadata": {"total_size": total_bytes}, "weight_map": out_weight_map},
        open(os.path.join(dst, "model.safetensors.index.json"), "w"),
    )

    full_cfg = json.load(open(os.path.join(src, "config.json")))
    thinker_cfg = full_cfg["thinker_config"]
    thinker_cfg["architectures"] = ["Qwen3OmniMoeThinkerForConditionalGeneration"]
    json.dump(thinker_cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)

    for fn in AUX_FILES:
        src_path = os.path.join(src, fn)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst)

    summary = {
        "tensors": len(out_weight_map),
        "total_gb": round(total_bytes / 1e9, 2),
        "shards": len(out_weight_map and set(out_weight_map.values())),
        "dst": dst,
    }
    print(f"extracted thinker: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="full Qwen3-Omni checkpoint dir")
    parser.add_argument("--dst", required=True, help="output thinker checkpoint dir")
    parser.add_argument("--shard-bytes", type=int, default=DEFAULT_SHARD_BYTES)
    args = parser.parse_args()
    extract_thinker(args.src, args.dst, args.shard_bytes)


if __name__ == "__main__":
    main()
