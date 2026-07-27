"""Frozen-audio-encoder embedding injection for the Qwen3-Omni thinker text backbone.

The trainer holds only the extracted Qwen3-MoE text model; audio placeholder tokens in a
multimodal prompt would otherwise embed as ordinary vocab rows and the recomputed
logprobs would be garbage. This patch fills those positions from the omni checkpoint's
frozen audio tower at forward time, mirroring HF `get_audio_features` +
`masked_scatter`. Audio-only prompts keep plain sequential position ids (HF
`get_rope_index` only takes the mrope branch when image/video grids are present), so the
text backbone's RoPE needs no change — images/videos are NOT supported here and are
rejected loudly.

The tower is deliberately not a registered submodule: it stays out of DDP buckets, the
optimizer, and (dist-)checkpoints; it is lazily loaded once per process and shared.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

AUDIO_KWARG_KEYS = ("input_features", "feature_attention_mask", "audio_feature_lengths")
_UNSUPPORTED_MM_KEYS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "video_second_per_grid",
)

_ENCODER_CACHE: dict[tuple[str, str, torch.dtype], torch.nn.Module] = {}


def resolve_audio_token_id(omni_config: dict) -> int:
    thinker = omni_config.get("thinker_config", omni_config)
    audio_token_id = thinker.get("audio_token_id")
    assert audio_token_id is not None, "audio_token_id not found in omni config (thinker_config)"
    return int(audio_token_id)


def load_frozen_audio_encoder(omni_checkpoint: str | Path, device, dtype) -> torch.nn.Module:
    """Build the audio tower from `thinker.audio_tower.*` tensors of the full omni checkpoint.

    Reads only the tower's tensors (safetensors are lazy), so the 60 GB text/talker
    weights never touch RAM.
    """
    from safetensors import safe_open
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

    src = Path(omni_checkpoint)
    with open(src / "config.json") as f:
        omni_config = json.load(f)
    thinker = omni_config.get("thinker_config", omni_config)
    audio_config = Qwen3OmniMoeAudioEncoderConfig(**thinker["audio_config"])

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted({shard for name, shard in weight_map.items() if name.startswith("thinker.audio_tower.")})
    else:
        shard_files = ["model.safetensors"]

    prefix = "thinker.audio_tower."
    state_dict = {}
    for shard_file in shard_files:
        with safe_open(src / shard_file, framework="pt") as reader:
            for key in reader.keys():
                if key.startswith(prefix):
                    state_dict[key[len(prefix) :]] = reader.get_tensor(key)
    assert state_dict, f"no {prefix}* tensors found under {src}"

    encoder = Qwen3OmniMoeAudioEncoder(audio_config)
    encoder.load_state_dict(state_dict, strict=True)
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.requires_grad_(False)
    encoder.eval()
    logger.info(
        f"loaded frozen Qwen3-Omni audio tower from {src} ({sum(p.numel() for p in encoder.parameters()):,} params)"
    )
    return encoder


def compute_audio_embeddings(encoder, input_features, feature_attention_mask, audio_feature_lengths) -> torch.Tensor:
    """Mirror HF Qwen3OmniMoeThinker.get_audio_features: [num_audios, mel, T] -> [total_audio_tokens, output_dim]."""
    with torch.no_grad():
        if feature_attention_mask is not None:
            feature_lens = feature_attention_mask.sum(dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            assert audio_feature_lengths is not None, "need feature_attention_mask or audio_feature_lengths"
            feature_lens = audio_feature_lengths
            input_features = input_features.permute(0, 2, 1).reshape(-1, input_features.shape[1]).permute(1, 0)
        encoder_param = next(encoder.parameters())
        input_features = input_features.to(device=encoder_param.device, dtype=encoder_param.dtype)
        feature_lens = feature_lens.to(encoder_param.device)
        return encoder(input_features, feature_lens=feature_lens).last_hidden_state


def scatter_audio_embeddings(
    hidden, input_ids, audio_embeds, audio_token_id, sp_rank: int = 0, sp_size: int = 1
) -> torch.Tensor:
    """Replace hidden rows at audio placeholder positions, out of place.

    hidden: [s_local, b, h] (mcore embedding layout); input_ids: [b, s_global];
    audio_embeds: [n_global, h]. Requires the packed b == 1 layout: masked_scatter fills
    s-major, which would interleave samples for b > 1.

    With sequence parallelism the embedding output is the rank's contiguous chunk
    (s_local = s_global / sp_size, rows [sp_rank*s_local, (sp_rank+1)*s_local)); the
    mask and the encoder outputs are sliced to that window, so every rank scatters
    exactly its own audio rows.
    """
    assert hidden.dim() == 3 and input_ids.dim() == 2, f"{hidden.shape=} {input_ids.shape=}"
    assert (
        hidden.size(1) == 1 and input_ids.size(0) == 1
    ), f"audio injection requires the packed [1, s] layout, got {input_ids.shape}"
    mask = input_ids[0] == audio_token_id  # [s_global]
    num_positions = int(mask.sum())
    assert num_positions == audio_embeds.size(0), (
        f"audio placeholder/embedding mismatch: {num_positions} audio tokens in input_ids "
        f"vs {audio_embeds.size(0)} encoder outputs"
    )
    if sp_size > 1:
        s_local = hidden.size(0)
        assert (
            s_local * sp_size == mask.numel()
        ), f"sequence-parallel chunking mismatch: local {s_local} x sp {sp_size} != global {mask.numel()}"
        start = sp_rank * s_local
        prior = int(mask[:start].sum())
        mask = mask[start : start + s_local]
        audio_embeds = audio_embeds[prior : prior + int(mask.sum())]
    audio_embeds = audio_embeds.to(device=hidden.device, dtype=hidden.dtype)
    return hidden.masked_scatter(mask.view(-1, 1, 1).to(hidden.device), audio_embeds)


def install_audio_injection(model, args, encoder_loader=None, audio_token_id: int | None = None):
    """Patch `model.forward` to inject frozen-audio-tower embeddings via decoder_input.

    Text-only batches pass through untouched (the encoder is not even loaded). Non-first
    PP stages swallow the audio kwargs and pass through. Keeps the module identity
    unchanged, so DDP buckets, the optimizer, and checkpoints see the same model.
    """
    checkpoint_path = getattr(args, "qwen3_omni_audio_encoder_path", None)
    if audio_token_id is None:
        with open(Path(checkpoint_path) / "config.json") as f:
            audio_token_id = resolve_audio_token_id(json.load(f))
    if encoder_loader is None:

        def encoder_loader(device, dtype):
            key = (str(checkpoint_path), str(device), dtype)
            if key not in _ENCODER_CACHE:
                _ENCODER_CACHE[key] = load_frozen_audio_encoder(checkpoint_path, device=device, dtype=dtype)
            return _ENCODER_CACHE[key]

    orig_forward = model.forward

    def forward(*fargs, **kwargs):
        audio_kwargs = {k: kwargs.pop(k) for k in AUDIO_KWARG_KEYS if k in kwargs}
        unsupported = [k for k in _UNSUPPORTED_MM_KEYS if k in kwargs]
        assert not unsupported, (
            f"qwen3_omni_thinker audio injection is audio-only; got unsupported multimodal keys {unsupported} "
            "(image/video need the mrope + deepstack path)"
        )
        input_features = audio_kwargs.get("input_features")
        if input_features is None or not getattr(model, "pre_process", True):
            return orig_forward(*fargs, **kwargs)

        assert not fargs, "audio injection expects keyword-only forward calls"
        assert "decoder_input" not in kwargs, "decoder_input already set upstream"
        assert getattr(args, "context_parallel_size", 1) == 1, "audio injection requires context_parallel_size == 1"

        input_ids = kwargs["input_ids"]
        # with sequence parallelism the embedding output is this rank's contiguous chunk
        sp_rank, sp_size = 0, 1
        if getattr(args, "sequence_parallel", False) and getattr(args, "tensor_model_parallel_size", 1) > 1:
            from megatron.core import parallel_state as mpu

            sp_rank = mpu.get_tensor_model_parallel_rank()
            sp_size = mpu.get_tensor_model_parallel_world_size()
        hidden = model.embedding(input_ids=input_ids, position_ids=kwargs.get("position_ids"))
        encoder = encoder_loader(device=hidden.device, dtype=hidden.dtype)
        # NOTE: every TP rank re-encodes the whole audio and then scatters only its own SP
        # chunk -> TP_size x redundant tower forward. Fine here (frozen ~0.6B tower, small
        # audio); if audio grows or TP scales, encode once on rank 0 and broadcast.
        audio_embeds = compute_audio_embeddings(
            encoder,
            input_features,
            audio_kwargs.get("feature_attention_mask"),
            audio_kwargs.get("audio_feature_lengths"),
        )
        kwargs["decoder_input"] = scatter_audio_embeddings(
            hidden, input_ids, audio_embeds, audio_token_id, sp_rank=sp_rank, sp_size=sp_size
        )
        return orig_forward(**kwargs)

    model.forward = forward
    return model
