import asyncio
import base64
import sys
from types import SimpleNamespace

import pytest
import torch

from miles.rollout.generate_utils.generate_endpoint_utils import serialize_multimodal_train_inputs
from miles.utils.types import Sample


def test_serialize_audio_video_processor_tensors():
    inputs = {
        "input_features": torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
        "feature_attention_mask": torch.ones((1, 2), dtype=torch.long),
        "pixel_values_videos": torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3),
        "video_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "video_second_per_grid": torch.tensor([0.5], dtype=torch.float32),
    }

    bundle = serialize_multimodal_train_inputs(inputs)

    assert bundle["version"] == 1
    assert set(bundle) == {"version", "tensors"}
    assert set(bundle["tensors"]) == set(inputs)
    for name, tensor in inputs.items():
        encoded = bundle["tensors"][name]
        assert encoded["dtype"] == str(tensor.dtype).removeprefix("torch.")
        assert encoded["shape"] == list(tensor.shape)
        decoded = torch.frombuffer(
            bytearray(base64.b64decode(encoded["data"])),
            dtype=getattr(torch, encoded["dtype"]),
        ).reshape(encoded["shape"])
        assert torch.equal(decoded, tensor)


def test_qwen_omni_media_extraction_and_tensor_normalization(monkeypatch):
    from miles.utils import processing_utils

    prompt = [{"role": "user", "content": [{"type": "audio", "audio": "a.wav"}]}]
    captured = {}

    def fake_process_mm_info(conversations, **kwargs):
        captured["conversations"], captured["kwargs"] = conversations, kwargs
        return ["audio samples"], ["image"], ["video frames"]

    monkeypatch.setitem(
        sys.modules,
        "qwen_omni_utils",
        SimpleNamespace(process_mm_info=fake_process_mm_info),
    )
    processor = object.__new__(processing_utils.Qwen3OmniMoeProcessor)
    processor.image_processor = SimpleNamespace(patch_size=16)

    media = processing_utils.process_vision_info(prompt, processor)
    train_inputs = processing_utils.extract_multimodal_train_inputs(
        {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "pixel_values_videos": torch.ones((1, 2)),
            "video_second_per_grid": [0.5],
        }
    )

    assert media == {
        "audio": ["audio samples"],
        "images": ["image"],
        "videos": ["video frames"],
    }
    assert captured == {
        "conversations": prompt,
        "kwargs": {"use_audio_in_video": False, "image_patch_size": 16},
    }
    assert torch.equal(train_inputs["video_second_per_grid"], torch.tensor([0.5]))


def test_sglang_omni_adapter_sends_processed_audio_video(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    captured = {}

    async def fake_post(url, payload, headers=None):
        captured.update(url=url, payload=payload, headers=headers)
        return {}

    async def fake_update(*args, **kwargs):
        return None

    monkeypatch.setattr(sglang_omni, "post", fake_post)
    monkeypatch.setattr(sglang_omni, "update_sample_from_response", fake_update)
    monkeypatch.setattr(
        sglang_omni,
        "compute_prompt_ids_from_sample",
        lambda state, sample: [1, 2, 3],
    )

    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_policy="round_robin",
        rollout_max_response_len=128,
        rollout_max_context_len=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
    )
    sample = Sample(
        multimodal_inputs={"audio": ["audio"], "videos": ["video"]},
        multimodal_train_inputs={
            "input_features": torch.ones((1, 2, 3)),
            "pixel_values_videos": torch.ones((2, 2, 3)),
        },
    )
    generate_input = SimpleNamespace(
        args=args,
        sample=sample,
        sampling_params={"temperature": 1.0, "max_new_tokens": 16},
        state=None,
    )

    asyncio.run(sglang_omni.generate(generate_input))

    assert captured["url"].endswith("/generate")
    assert captured["payload"]["input_ids"] == [1, 2, 3]
    assert set(captured["payload"]["multimodal_train_inputs"]["tensors"]) == {
        "input_features",
        "pixel_values_videos",
    }
    assert captured["headers"] is None


def _adapter_args() -> SimpleNamespace:
    return SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_policy="round_robin",
        rollout_max_response_len=128,
        rollout_max_context_len=0,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
    )


def test_sglang_omni_adapter_rejects_multimodal_without_train_inputs(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    monkeypatch.setattr(
        sglang_omni,
        "compute_prompt_ids_from_sample",
        lambda state, sample: [1, 2, 3],
    )
    generate_input = SimpleNamespace(
        args=_adapter_args(),
        sample=Sample(multimodal_inputs={"audio": ["audio"]}),
        sampling_params={"temperature": 1.0, "max_new_tokens": 16},
        state=None,
    )

    with pytest.raises(ValueError, match="requires processor-produced"):
        asyncio.run(sglang_omni.generate(generate_input))


def test_sglang_omni_adapter_truncates_exhausted_resume(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    async def fail_post(url, payload, headers=None):
        raise AssertionError("post must not run for an exhausted resume")

    monkeypatch.setattr(sglang_omni, "post", fail_post)
    monkeypatch.setattr(
        sglang_omni,
        "compute_prompt_ids_from_sample",
        lambda state, sample: [1, 2, 3],
    )
    sample = Sample(response="hi", tokens=[1, 2, 3, 4, 5])
    generate_input = SimpleNamespace(
        args=_adapter_args(),
        sample=sample,
        sampling_params={"temperature": 1.0, "max_new_tokens": 2},
        state=None,
    )

    output = asyncio.run(sglang_omni.generate(generate_input))

    assert sample.status == Sample.Status.TRUNCATED
    assert output.samples is sample


def test_sglang_omni_adapter_resumes_partial_rollout(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    captured = {}

    async def fake_post(url, payload, headers=None):
        captured["payload"] = payload
        return {}

    async def fake_update(*args, **kwargs):
        return None

    monkeypatch.setattr(sglang_omni, "post", fake_post)
    monkeypatch.setattr(sglang_omni, "update_sample_from_response", fake_update)
    monkeypatch.setattr(
        sglang_omni,
        "compute_prompt_ids_from_sample",
        lambda state, sample: [1, 2, 3],
    )
    sample = Sample(response="hi", tokens=[1, 2, 3, 4, 5])
    sampling_params = {"temperature": 1.0, "max_new_tokens": 16}
    generate_input = SimpleNamespace(
        args=_adapter_args(),
        sample=sample,
        sampling_params=sampling_params,
        state=None,
    )

    asyncio.run(sglang_omni.generate(generate_input))

    assert captured["payload"]["input_ids"] == [1, 2, 3, 4, 5]
    assert sampling_params["max_new_tokens"] == 14
