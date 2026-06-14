"""Megatron->HF broadcast converter for the Qwen3-Omni thinker (Qwen3-MoE text).

Same conversion as qwen3moe, but prefixes every HF name with `body.` (the namespace
the sglang-omni weight-receive side demuxes on). Selected via `--model-name
qwen3omni_moe`. Nothing is dropped: the thinker is untied; `tied.*` is talker-side.
"""

from .qwen3moe import convert_qwen3moe_to_hf

BODY_PREFIX = "body."


def convert_qwen3omni_moe_to_hf(args, name, param):
    return [(f"{BODY_PREFIX}{n}", t) for n, t in convert_qwen3moe_to_hf(args, name, param)]
