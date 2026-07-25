"""Omni video TM-RoPE + kwarg hygiene for the pseudo-Qwen3-VL thinker.

The pseudo-VL checkpoint (tools/extract_qwen3_omni_thinker.py --variant vl) trains on
miles' bridge Qwen3-VL path. Image/text positions are identical between the omni and VL
formulas, but video differs: the omni server rolls out with TM-RoPE
(t_index = k * second_per_grid * position_id_per_seconds, one contiguous block per
video), while the VL/bridge formula splits per frame around timestamp text tokens the
omni processor never emits — silently wrong positions after frame 1. This plugin swaps
the video branch to the omni formula and keeps delegating image-only segments to the
bridge implementation.

It also pops `video_second_per_grid` (omni processor output with no slot in the bridge
forward) and un-freezes the language model on providers whose defaults freeze it.

Install via `install_omni_vl(args)` — wired behind --qwen3-omni-vl in model_provider.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_tls = threading.local()
_PATCHED = "_miles_omni_vl_patched"


def omni_video_rope_index(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    video_second_per_grid: torch.Tensor | None,
    position_id_per_seconds: float,
) -> torch.Tensor:
    """Qwen3-Omni thinker TM-RoPE positions for one [1, s] row (text/image/video, no audio).

    Port of HF Qwen3OmniMoe get_rope_index minus the audio branches: video temporal
    index strides by second_per_grid * position_id_per_seconds within one contiguous
    block; images use t = const; text runs sequentially from the running max + 1.
    """
    ids = input_ids[0]
    tokens = ids.tolist()
    device = input_ids.device
    image_idx = video_idx = 0
    st = 0
    pos_chunks: list[torch.Tensor] = []

    def _vision_block(st_idx: float, grid, t_index: torch.Tensor) -> torch.Tensor:
        t, h, w = int(grid[0]), int(grid[1]) // spatial_merge_size, int(grid[2]) // spatial_merge_size
        t_pos = t_index.view(t, 1).expand(t, h * w).flatten()
        h_pos = torch.arange(h).view(1, h, 1).expand(t, h, w).flatten()
        w_pos = torch.arange(w).view(1, 1, w).expand(t, h, w).flatten()
        return torch.stack([t_pos, h_pos.float(), w_pos.float()]) + st_idx

    while st < len(tokens):
        st_idx = pos_chunks[-1].max().item() + 1 if pos_chunks else 0
        try:
            next_vision = tokens.index(vision_start_token_id, st)
        except ValueError:
            next_vision = len(tokens)

        text_len = next_vision - st
        if text_len > 0:
            pos_chunks.append(torch.arange(text_len).view(1, -1).expand(3, -1).float() + st_idx)
            st_idx += text_len
        if next_vision >= len(tokens):
            break

        # vision_start token itself is ordinary text
        pos_chunks.append(torch.arange(1).view(1, -1).expand(3, -1).float() + st_idx)
        st_idx += 1
        marker = tokens[next_vision + 1]
        if marker == image_token_id:
            grid = image_grid_thw[image_idx]
            t_index = (torch.arange(int(grid[0])) * 1 * position_id_per_seconds).float()
            block = _vision_block(st_idx, grid, t_index)
            image_idx += 1
        elif marker == video_token_id:
            grid = video_grid_thw[video_idx]
            second = float(video_second_per_grid[video_idx])
            t_index = (torch.arange(int(grid[0])) * second * position_id_per_seconds).float()
            block = _vision_block(st_idx, grid, t_index)
            video_idx += 1
        else:
            raise AssertionError(f"vision_start not followed by an image/video token: {marker}")
        pos_chunks.append(block)
        st = next_vision + 1 + block.shape[1]

    positions = torch.cat(pos_chunks, dim=1)
    assert positions.shape[1] == len(tokens), f"{positions.shape=} vs {len(tokens)=}"
    return positions.view(3, 1, -1).to(device)


def _make_omni_aware_rope_index(orig_get_rope_index, position_id_per_seconds: float):
    """Bridge-signature get_rope_index that reroutes video segments to the omni formula.

    Consumes `video_second_per_grid` sequentially from _tls (segments are processed in
    order by the qwen3_vl packed-mrope patch, so a running cursor stays aligned with the
    grid slices it hands us).
    """

    def omni_aware(merge, img_id, vid_id, vstart, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        if video_grid_thw is None or video_grid_thw.numel() == 0:
            return orig_get_rope_index(
                merge, img_id, vid_id, vstart, input_ids,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw, attention_mask=attention_mask,
            )
        seconds_all = getattr(_tls, "video_second_per_grid", None)
        assert seconds_all is not None, (
            "video tokens reached the trainer without video_second_per_grid; the omni processor "
            "always emits it - check the multimodal_train_inputs plumbing"
        )
        cursor = getattr(_tls, "video_cursor", 0)
        n_videos = video_grid_thw.shape[0]
        seconds = seconds_all[cursor : cursor + n_videos]
        assert seconds.numel() == n_videos, f"video_second_per_grid exhausted: {cursor=} {n_videos=} {seconds_all.numel()=}"
        _tls.video_cursor = cursor + n_videos
        positions = omni_video_rope_index(
            merge, img_id, vid_id, vstart, input_ids,
            image_grid_thw, video_grid_thw, seconds, position_id_per_seconds,
        )
        return positions, None

    return omni_aware


def resolve_position_id_per_seconds(hf_checkpoint: str | Path) -> float:
    """The pseudo-VL config has no VL slot for this; the extractor stashes it in omni_sideband."""
    with open(Path(hf_checkpoint) / "config.json") as f:
        config = json.load(f)
    sideband = config.get("omni_sideband") or {}
    value = sideband.get("position_id_per_seconds")
    assert value is not None, (
        f"omni_sideband.position_id_per_seconds missing from {hf_checkpoint}/config.json - "
        "re-extract with tools/extract_qwen3_omni_thinker.py --variant vl"
    )
    return float(value)


def install_omni_vl(args) -> None:
    """Install the omni video rope override + video_second_per_grid pop on the bridge model."""
    import importlib

    import miles_plugins.models.qwen3_vl as qwen3_vl_patch

    model_mod = importlib.import_module("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")
    if getattr(model_mod, _PATCHED, False):
        return

    position_id_per_seconds = resolve_position_id_per_seconds(args.hf_checkpoint)

    orig_build = qwen3_vl_patch._build_packed_positions

    def build_with_omni_video(model, parsed, kwargs, orig_get_rope_index):
        return orig_build(model, parsed, kwargs, _make_omni_aware_rope_index(orig_get_rope_index, position_id_per_seconds))

    qwen3_vl_patch._build_packed_positions = build_with_omni_video

    Qwen3VLModel = model_mod.Qwen3VLModel
    inner_forward = Qwen3VLModel.forward

    def forward(self, *fargs, **kwargs):
        seconds = kwargs.pop("video_second_per_grid", None)
        _tls.video_second_per_grid = seconds
        _tls.video_cursor = 0
        try:
            return inner_forward(self, *fargs, **kwargs)
        finally:
            _tls.video_second_per_grid = None
            _tls.video_cursor = 0

    Qwen3VLModel.forward = forward
    setattr(model_mod, _PATCHED, True)
    logger.info(f"omni VL patch installed (position_id_per_seconds={position_id_per_seconds})")


def unfreeze_provider(provider) -> None:
    """Some megatron-bridge versions default Qwen3VLMoEModelProvider to a frozen LM;
    RL must train the backbone, so force-clear every freeze knob that exists."""
    for field in ("freeze_language_model", "freeze_vision_model", "freeze_vision_projection"):
        if getattr(provider, field, False):
            logger.warning(f"pseudo-VL provider had {field}=True; clearing it for RL training")
            setattr(provider, field, False)
