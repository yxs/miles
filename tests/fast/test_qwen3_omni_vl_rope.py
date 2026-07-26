"""Omni TM-RoPE port for the pseudo-VL trainer: parity with the HF reference."""

from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-a-cpu", labels=[])

from miles_plugins.models.qwen3_omni_thinker_vl import _make_omni_aware_rope_index, _tls, omni_video_rope_index

MERGE = 2
IMG, VID, VSTART, VEND = 151655, 151656, 151652, 151653
POS_PER_SEC = 13.0


def _hf_reference():
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoePreTrainedModelForConditionalGeneration,
    )

    ref = Qwen3OmniMoePreTrainedModelForConditionalGeneration.__new__(
        Qwen3OmniMoePreTrainedModelForConditionalGeneration
    )
    config = SimpleNamespace(
        image_token_id=IMG,
        video_token_id=VID,
        audio_token_id=151675,
        vision_start_token_id=VSTART,
        audio_start_token_id=151669,
        position_id_per_seconds=POS_PER_SEC,
        vision_config=SimpleNamespace(spatial_merge_size=MERGE),
    )
    object.__setattr__(ref, "config", config)
    object.__setattr__(ref, "spatial_merge_size", MERGE)
    return ref


def _build_ids(segments) -> tuple[torch.Tensor, list, list, list]:
    """segments: list of ('text', n) | ('image', (t,h,w)) | ('video', (t,h,w), seconds)."""
    ids, image_grids, video_grids, seconds = [], [], [], []
    text_token = 7
    for seg in segments:
        if seg[0] == "text":
            ids += [text_token] * seg[1]
        elif seg[0] == "image":
            t, h, w = seg[1]
            image_grids.append([t, h, w])
            ids += [VSTART] + [IMG] * (t * h * w // MERGE**2) + [VEND]
        else:
            t, h, w = seg[1]
            video_grids.append([t, h, w])
            seconds.append(seg[2])
            ids += [VSTART] + [VID] * (t * h * w // MERGE**2) + [VEND]
    return torch.tensor([ids]), image_grids, video_grids, seconds


@pytest.mark.parametrize(
    "segments",
    [
        [("text", 5), ("video", (4, 4, 6), 0.5), ("text", 3)],
        [("text", 2), ("image", (1, 4, 4)), ("text", 1), ("video", (2, 6, 4), 2.0), ("text", 4)],
        [("video", (2, 4, 4), 1.0), ("video", (3, 4, 4), 0.25), ("text", 2)],
        [("text", 6)],
        [("image", (1, 6, 6)), ("text", 2)],
    ],
)
def test_omni_video_rope_matches_hf_reference(segments):
    input_ids, image_grids, video_grids, seconds = _build_ids(segments)
    image_grid_thw = torch.tensor(image_grids) if image_grids else None
    video_grid_thw = torch.tensor(video_grids) if video_grids else None
    second_per_grid = torch.tensor(seconds) if seconds else None

    ours = omni_video_rope_index(
        MERGE, IMG, VID, VSTART, input_ids, image_grid_thw, video_grid_thw, second_per_grid, POS_PER_SEC
    )

    ref = _hf_reference()
    expected, _ = ref.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=torch.ones_like(input_ids),
        use_audio_in_video=False,
        audio_seqlens=None,
        second_per_grids=second_per_grid,
    )

    assert ours.shape == expected.shape, f"{ours.shape=} {expected.shape=}"
    assert torch.allclose(
        ours.float(), expected.float()
    ), f"positions diverge from the HF omni reference for {segments}"


def test_omni_aware_rope_delegates_images_and_consumes_seconds_in_order():
    calls = []

    def fake_orig(merge, img_id, vid_id, vstart, ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        calls.append("orig")
        return torch.zeros(3, 1, ids.shape[1]), None

    omni_aware = _make_omni_aware_rope_index(fake_orig, POS_PER_SEC)
    _tls.video_second_per_grid = torch.tensor([0.5, 2.0])
    _tls.video_cursor = 0
    try:
        # image-only segment -> delegate to the bridge implementation
        ids_img, image_grids, _, _ = _build_ids([("image", (1, 4, 4)), ("text", 1)])
        omni_aware(MERGE, IMG, VID, VSTART, ids_img, torch.tensor(image_grids), None, None)
        assert calls == ["orig"]

        # two video segments consume seconds sequentially
        ids_v1, _, grids_v1, _ = _build_ids([("video", (2, 4, 4), 0.5)])
        pos1, _ = omni_aware(MERGE, IMG, VID, VSTART, ids_v1, None, torch.tensor(grids_v1), None)
        assert _tls.video_cursor == 1
        ids_v2, _, grids_v2, _ = _build_ids([("video", (2, 4, 4), 2.0)])
        pos2, _ = omni_aware(MERGE, IMG, VID, VSTART, ids_v2, None, torch.tensor(grids_v2), None)
        assert _tls.video_cursor == 2
        # different seconds -> different temporal strides
        assert not torch.equal(pos1, pos2)
        # stride check: frame 2 of video 1 starts at t = 0.5 * 13 relative to block start
        t_axis = pos1[0, 0]
        block = t_axis[1:9]  # exactly the 8 video tokens (skip vision_start, stop before vision_end)
        assert block.max() - block.min() == pytest.approx(0.5 * POS_PER_SEC)
    finally:
        _tls.video_second_per_grid = None
        _tls.video_cursor = 0


def test_omni_aware_rope_fails_loud_without_seconds():
    omni_aware = _make_omni_aware_rope_index(lambda *a, **k: (None, None), POS_PER_SEC)
    _tls.video_second_per_grid = None
    ids, _, grids, _ = _build_ids([("video", (2, 4, 4), 0.5)])
    with pytest.raises(AssertionError, match="video_second_per_grid"):
        omni_aware(MERGE, IMG, VID, VSTART, ids, None, torch.tensor(grids), None)


def test_pseudo_vl_to_omni_server_names_roundtrip_the_extraction_map():
    import importlib.util
    from pathlib import Path

    from miles_plugins.models.qwen3_omni_thinker_vl import pseudo_vl_to_omni_server_name

    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("extract_tool", repo / "tools" / "extract_qwen3_omni_thinker.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)

    omni_names = [
        "thinker.lm_head.weight",
        "thinker.model.embed_tokens.weight",
        "thinker.model.layers.3.self_attn.q_proj.weight",
        "thinker.visual.patch_embed.proj.weight",
        "thinker.visual.blocks.7.attn.qkv.weight",
        "thinker.visual.merger.ln_q.weight",
        "thinker.visual.merger.mlp.0.bias",
        "thinker.visual.merger.mlp.2.weight",
        "thinker.visual.merger_list.1.ln_q.weight",
        "thinker.visual.merger_list.2.mlp.2.bias",
    ]
    for omni in omni_names:
        vl = tool.map_thinker_param_name_vl(omni)
        assert pseudo_vl_to_omni_server_name(vl) == omni, f"{omni} -> {vl} did not roundtrip"

    # fused experts map onto the sglang fused loader names (thinker. + model.)
    assert (
        pseudo_vl_to_omni_server_name("model.language_model.layers.0.mlp.experts.gate_up_proj")
        == "thinker.model.layers.0.mlp.experts.gate_up_proj"
    )
    assert pseudo_vl_to_omni_server_name("not_a_model_tensor") is None


def test_resolve_position_id_per_seconds(tmp_path):
    import json

    from miles_plugins.models.qwen3_omni_thinker_vl import resolve_position_id_per_seconds

    (tmp_path / "config.json").write_text(json.dumps({"omni_sideband": {"position_id_per_seconds": 13}}))
    assert resolve_position_id_per_seconds(tmp_path) == 13.0

    (tmp_path / "config.json").write_text(json.dumps({}))
    with pytest.raises(AssertionError, match="omni_sideband"):
        resolve_position_id_per_seconds(tmp_path)
