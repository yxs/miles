from argparse import Namespace

import torch

from .qwen2 import convert_qwen2_to_hf


def convert_higgs_to_hf(args: Namespace, name: str, param: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Convert the TP-gathered Higgs policy weights to canonical checkpoint names."""

    if name == "module.module.codec_embeddings.weight":
        expected_shape = (
            args.higgs_num_codebooks * args.higgs_codebook_vocab_size,
            args.hidden_size,
        )
        if tuple(param.shape) != expected_shape:
            raise ValueError(
                "Higgs codec embedding shape mismatch: " f"expected {expected_shape}, got {tuple(param.shape)}"
            )
        return [("tied.embedding.modality_embeddings.0.embedding.weight", param)]

    converted = convert_qwen2_to_hf(args, name, param)
    renamed: list[tuple[str, torch.Tensor]] = []
    for hf_name, tensor in converted:
        if hf_name == "model.embed_tokens.weight":
            canonical_name = "tied.embedding.text_embedding.weight"
        elif hf_name == "model.norm.weight":
            canonical_name = "body.norm.weight"
        elif hf_name.startswith("model.layers."):
            canonical_name = "body.layers." + hf_name.removeprefix("model.layers.")
        elif hf_name == "lm_head.weight":
            raise ValueError("the tied Higgs text head must not be a separate Megatron parameter")
        else:
            raise ValueError(f"unsupported Higgs policy parameter mapping: {name!r} -> {hf_name!r}")
        renamed.append((canonical_name, tensor))
    return renamed


__all__ = ["convert_higgs_to_hf"]
