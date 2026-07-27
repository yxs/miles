"""Megatron->HF broadcast converter for the Qwen3-Omni thinker (Qwen3-MoE text).

Same conversion as qwen3moe, but prefixes every HF name with `thinker.` — the namespace
the sglang-omni thinker stage strips in `load_weights` (names under `audio_tower.` /
`visual.` / `talker.` / `code2wav.` are ignored there, so pushing only the text backbone
is safe). Selected via `--model-name qwen3omni_moe`. The thinker is untied.
"""

from .qwen3moe import convert_qwen3moe_to_hf

THINKER_PREFIX = "thinker."


def convert_qwen3omni_moe_to_hf(args, name, param):
    return [(f"{THINKER_PREFIX}{n}", t) for n, t in convert_qwen3moe_to_hf(args, name, param)]
