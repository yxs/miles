"""Benchmark the processed-multimodal wire contract (serialize -> JSON -> deserialize).

Quantifies the base64-tensor-in-JSON costs of `multimodal_train_inputs` per modality —
the known heavy case is video (pixel_values_videos grows with frames x resolution) — so
the transport decision (keep JSON vs move to a binary/shared-memory path) is made on
numbers, not vibes.

    python examples/omni_thinker/bench_mm_payload.py [--frames 8 16 32] [--seconds 10 30]
    python examples/omni_thinker/bench_mm_payload.py --server http://<ip>:<port>  # + HTTP POST timing
"""

import argparse
import json
import time

import torch

from miles.rollout.generate_utils.generate_endpoint_utils import serialize_multimodal_train_inputs

PATCH = 16
TEMPORAL_PATCH = 2
MERGE = 2  # spatial merge; grid counts are pre-merge patches per qwen vision geometry
MEL_BINS = 128
FRAMES_PER_SECOND_MEL = 100  # whisper-style feature extractor: 10ms hop


def make_video_bundle(num_frames: int, height: int, width: int) -> dict[str, torch.Tensor]:
    grid_t = num_frames // TEMPORAL_PATCH
    grid_h, grid_w = height // PATCH, width // PATCH
    patch_dim = 3 * TEMPORAL_PATCH * PATCH * PATCH
    return {
        "pixel_values_videos": torch.randn(grid_t * grid_h * grid_w, patch_dim, dtype=torch.float32),
        "video_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long),
        "video_second_per_grid": torch.tensor([0.5], dtype=torch.float32),
    }


def make_image_bundle(height: int, width: int) -> dict[str, torch.Tensor]:
    grid_h, grid_w = height // PATCH, width // PATCH
    patch_dim = 3 * TEMPORAL_PATCH * PATCH * PATCH
    return {
        "pixel_values": torch.randn(grid_h * grid_w, patch_dim, dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, grid_h, grid_w]], dtype=torch.long),
    }


def make_audio_bundle(seconds: float) -> dict[str, torch.Tensor]:
    frames = int(seconds * FRAMES_PER_SECOND_MEL)
    return {
        "input_features": torch.randn(1, MEL_BINS, frames, dtype=torch.float32),
        "feature_attention_mask": torch.ones(1, frames, dtype=torch.long),
    }


def _deserialize(bundle: dict) -> dict[str, torch.Tensor]:
    """Mirror the server decode (sglang-omni preprocessor.py)."""
    import pybase64

    out = {}
    for name, spec in bundle["tensors"].items():
        raw = bytearray(pybase64.b64decode(spec["data"]))
        out[name] = torch.frombuffer(raw, dtype=getattr(torch, spec["dtype"])).reshape(spec["shape"])
    return out


def measure_roundtrip(tensors: dict[str, torch.Tensor], server: str | None = None) -> dict:
    t0 = time.perf_counter()
    bundle = serialize_multimodal_train_inputs(tensors)
    t1 = time.perf_counter()
    payload = json.dumps(bundle)
    t2 = time.perf_counter()
    decoded_bundle = json.loads(payload)
    restored = _deserialize(decoded_bundle)
    t3 = time.perf_counter()

    report = {
        "raw_bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
        "payload_bytes": len(payload),
        "serialize_ms": (t1 - t0) * 1e3,
        "json_dump_ms": (t2 - t1) * 1e3,
        "deserialize_ms": (t3 - t2) * 1e3,
        "roundtrip_equal": all(torch.equal(restored[k], tensors[k].cpu()) for k in tensors),
    }

    if server:
        import requests

        body = {
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "multimodal_train_inputs": bundle,
        }
        t4 = time.perf_counter()
        resp = requests.post(f"{server}/generate", json=body, timeout=600)
        report["http_ms"] = (time.perf_counter() - t4) * 1e3
        report["http_status"] = resp.status_code
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--seconds", type=float, nargs="+", default=[10.0, 30.0])
    parser.add_argument("--server", type=str, default=None, help="optional http://ip:port for POST timing")
    args = parser.parse_args()

    rows = [("image", f"{args.resolution}px", make_image_bundle(args.resolution, args.resolution))]
    rows += [
        ("video", f"{n}f@{args.resolution}px", make_video_bundle(n, args.resolution, args.resolution))
        for n in args.frames
    ]
    rows += [("audio", f"{s:.0f}s", make_audio_bundle(s)) for s in args.seconds]

    header = (
        f"{'modality':8s} {'case':12s} {'raw_MB':>8s} {'json_MB':>8s} {'ser_ms':>8s} {'dump_ms':>8s} {'deser_ms':>9s}"
    )
    print(header)
    for modality, case, tensors in rows:
        r = measure_roundtrip(tensors, server=args.server)
        line = (
            f"{modality:8s} {case:12s} {r['raw_bytes'] / 2**20:8.1f} {r['payload_bytes'] / 2**20:8.1f} "
            f"{r['serialize_ms']:8.1f} {r['json_dump_ms']:8.1f} {r['deserialize_ms']:9.1f}"
        )
        if args.server:
            line += f"  http={r['http_ms']:.0f}ms({r['http_status']})"
        assert r["roundtrip_equal"]
        print(line)


if __name__ == "__main__":
    main()
