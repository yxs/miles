"""Multimodal payload benchmark: bundle synthesis + measurement plumbing."""

import importlib.util
from pathlib import Path

import torch

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-cpu", labels=[])

_REPO = Path(__file__).resolve().parents[2]


def _load_tool():
    path = _REPO / "examples" / "omni_thinker" / "bench_mm_payload.py"
    spec = importlib.util.spec_from_file_location("bench_mm_payload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_video_bundle_shapes_follow_qwen_patch_geometry():
    tool = _load_tool()
    # 16 frames @ 448x448, patch 16, temporal_patch 2, merge 2 -> grid t=8, h=w=28
    bundle = tool.make_video_bundle(num_frames=16, height=448, width=448)

    grid = bundle["video_grid_thw"]
    assert grid.tolist() == [[8, 28, 28]]
    # flattened patches x (3 * temporal_patch * patch^2)
    assert bundle["pixel_values_videos"].shape == (8 * 28 * 28, 3 * 2 * 16 * 16)
    assert bundle["video_second_per_grid"].shape == (1,)


def test_image_and_audio_bundles():
    tool = _load_tool()
    image = tool.make_image_bundle(height=448, width=448)
    assert image["image_grid_thw"].tolist() == [[1, 28, 28]]
    assert image["pixel_values"].shape == (1 * 28 * 28, 3 * 2 * 16 * 16)

    audio = tool.make_audio_bundle(seconds=10.0)
    assert audio["input_features"].shape[1] == 128  # mel bins
    assert audio["feature_attention_mask"].shape == audio["input_features"].shape[:1] + audio["input_features"].shape[2:]


def test_measure_roundtrip_reports_size_and_integrity():
    tool = _load_tool()
    bundle = {"pixel_values": torch.randn(64, 32)}

    report = tool.measure_roundtrip(bundle)

    assert report["payload_bytes"] > 64 * 32 * 4  # base64 expands raw fp32
    assert report["serialize_ms"] >= 0 and report["deserialize_ms"] >= 0
    assert report["roundtrip_equal"] is True
