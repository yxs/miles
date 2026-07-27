from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="stage-b-cpu", labels=[])

import asyncio
import base64
import sys
from types import SimpleNamespace

import pytest
import torch
from tests.fast.fixtures.generation_fixtures import generation_env, make_sample, run_generate

from miles.rollout.generate_utils.generate_endpoint_utils import serialize_multimodal_train_inputs
from miles.utils.test_utils.mock_sglang_server import ProcessResult
from miles.utils.types import Sample

_ = generation_env

PROMPT_TOKENS = [3838, 374, 220, 16, 10, 22, 30]  # Qwen3-0.6B "What is 1+7?"
RESPONSE_TOKENS = [59, 79075, 90, 23, 92]  # "\\boxed{8}"
RESPONSE_LOG_PROBS = [-0.0, -0.0078125, -0.015625, -0.0234375, -0.03125]
SAMPLING_PARAMS = {"max_new_tokens": 16, "temperature": 0.7}


@pytest.fixture
def variant():
    return "sglang_omni"


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


def test_abort_external_omni_requests_uses_pause_abort_continue(monkeypatch):
    # a bare omni server has neither the router /workers listing nor /abort_request;
    # pause(mode=abort) + continue is the omni-native equivalent
    from miles.rollout.generate_utils import generate_endpoint_utils as geu

    calls = []

    async def fake_post(url, payload, **kwargs):
        calls.append((url, payload))
        return {}

    monkeypatch.setattr(geu, "post", fake_post)
    args = SimpleNamespace(
        rollout_external_admin_api="sglang-omni",
        rollout_external_engine_addrs=["10.0.0.1:30111", "10.0.0.2:30111"],
    )

    asyncio.run(geu.abort_external_omni_requests(args))

    assert calls == [
        ("http://10.0.0.1:30111/pause_generation", {"mode": "abort"}),
        ("http://10.0.0.1:30111/continue_generation", {}),
        ("http://10.0.0.2:30111/pause_generation", {"mode": "abort"}),
        ("http://10.0.0.2:30111/continue_generation", {}),
    ]


def test_is_omni_external_admin(monkeypatch):
    from miles.rollout.generate_utils import generate_endpoint_utils as geu

    assert geu.is_omni_external_admin(SimpleNamespace(rollout_external_admin_api="sglang-omni")) is True
    assert geu.is_omni_external_admin(SimpleNamespace(rollout_external_admin_api="sglang")) is False
    assert geu.is_omni_external_admin(SimpleNamespace()) is False


class TestHarness:
    """Through the real parse_args + mock sglang server harness (generation_fixtures)."""

    def test_basic_generation_request_shape_and_sample(self, generation_env):
        result = run_generate(generation_env, make_sample(), dict(SAMPLING_PARAMS), variant="sglang_omni")

        [request] = result.requests
        assert request["input_ids"] == PROMPT_TOKENS
        assert request["output_modalities"] == ["text"]
        assert request["return_omni_rollout"] is False
        assert request["return_logprob"] is True
        assert request["sampling_params"]["repetition_penalty"] == 1.0
        assert request["sampling_params"]["max_new_tokens"] == SAMPLING_PARAMS["max_new_tokens"]
        sample = result.sample
        assert sample.status == Sample.Status.COMPLETED
        assert sample.tokens == PROMPT_TOKENS + RESPONSE_TOKENS
        assert sample.rollout_log_probs == RESPONSE_LOG_PROBS
        assert sample.response == "\\boxed{8}"

    def test_resumed_partial_rollout_decrements_budget(self, generation_env):
        partial_tokens = [59, 79075]  # "\\boxed"
        generation_env.mock_server.process_fn = lambda _: ProcessResult(text="\\boxed", finish_reason="abort")
        sample = make_sample()
        result1 = run_generate(generation_env, sample, dict(SAMPLING_PARAMS), variant="sglang_omni")
        assert result1.sample.status == Sample.Status.ABORTED
        assert result1.sample.tokens == PROMPT_TOKENS + partial_tokens

        generation_env.mock_server.process_fn = lambda _: ProcessResult(text="{8}", finish_reason="stop")
        result2 = run_generate(generation_env, result1.sample, dict(SAMPLING_PARAMS), variant="sglang_omni")
        [request] = result2.requests
        assert request["input_ids"] == PROMPT_TOKENS + partial_tokens
        assert request["sampling_params"]["max_new_tokens"] == SAMPLING_PARAMS["max_new_tokens"] - len(partial_tokens)
        assert result2.sample.status == Sample.Status.COMPLETED
        assert result2.sample.tokens == PROMPT_TOKENS + partial_tokens + [90, 23, 92]
        assert result2.sample.rollout_log_probs == [-0.0, -0.0078125, -0.0, -0.0078125, -0.015625]


def test_sglang_omni_adapter_sets_omni_contract_fields(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    captured = {}

    async def fake_post(url, payload, headers=None):
        captured["payload"] = payload
        return {}

    async def fake_update(*args, **kwargs):
        return None

    monkeypatch.setattr(sglang_omni, "post", fake_post)
    monkeypatch.setattr(sglang_omni, "update_sample_from_response", fake_update)
    monkeypatch.setattr(sglang_omni, "compute_prompt_ids_from_sample", lambda state, sample: [1, 2, 3])
    generate_input = SimpleNamespace(
        args=_adapter_args(),
        sample=Sample(),
        sampling_params={"temperature": 1.0, "max_new_tokens": 16, "repetition_penalty": 1.4},
        state=None,
    )

    asyncio.run(sglang_omni.generate(generate_input))

    payload = captured["payload"]
    assert payload["output_modalities"] == ["text"]
    assert payload["return_omni_rollout"] is False
    # trainer recompute cannot replay a repetition penalty; the adapter pins it to 1.0
    assert payload["sampling_params"]["repetition_penalty"] == 1.0
    assert "metadata" not in payload


def test_sglang_omni_adapter_filters_sampling_params_to_omni_schema(monkeypatch):
    # the omni RolloutSamplingParams is extra="forbid": miles' detok-only keys 422 the request
    from miles.rollout.generate_hub import sglang_omni

    captured = {}

    async def fake_post(url, payload, headers=None):
        captured["payload"] = payload
        return {}

    async def fake_update(*args, **kwargs):
        return None

    monkeypatch.setattr(sglang_omni, "post", fake_post)
    monkeypatch.setattr(sglang_omni, "update_sample_from_response", fake_update)
    monkeypatch.setattr(sglang_omni, "compute_prompt_ids_from_sample", lambda state, sample: [1, 2, 3])
    generate_input = SimpleNamespace(
        args=_adapter_args(),
        sample=Sample(),
        sampling_params={
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": 16,
            "stop": None,
            "stop_token_ids": None,
            "skip_special_tokens": False,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        },
        state=None,
    )

    asyncio.run(sglang_omni.generate(generate_input))

    sp = captured["payload"]["sampling_params"]
    for detok_key in ("skip_special_tokens", "no_stop_trim", "spaces_between_special_tokens"):
        assert detok_key not in sp
    assert sp["temperature"] == 1.0 and sp["max_new_tokens"] == 16 and sp["top_k"] == -1
    assert sp["repetition_penalty"] == 1.0


def test_sglang_omni_adapter_rejects_unknown_sampling_keys(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    monkeypatch.setattr(sglang_omni, "compute_prompt_ids_from_sample", lambda state, sample: [1, 2, 3])
    generate_input = SimpleNamespace(
        args=_adapter_args(),
        sample=Sample(),
        sampling_params={"temperature": 1.0, "max_new_tokens": 16, "min_new_tokens": 4},
        state=None,
    )

    with pytest.raises(AssertionError, match="min_new_tokens"):
        asyncio.run(sglang_omni.generate(generate_input))


def test_sglang_omni_adapter_forwards_sample_metadata(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    captured = {}

    async def fake_post(url, payload, headers=None):
        captured["payload"] = payload
        return {}

    async def fake_update(*args, **kwargs):
        return None

    monkeypatch.setattr(sglang_omni, "post", fake_post)
    monkeypatch.setattr(sglang_omni, "update_sample_from_response", fake_update)
    monkeypatch.setattr(sglang_omni, "compute_prompt_ids_from_sample", lambda state, sample: [1, 2, 3])
    generate_input = SimpleNamespace(
        args=_adapter_args(),
        sample=Sample(metadata={"task": "avqa"}),
        sampling_params={"temperature": 1.0, "max_new_tokens": 16},
        state=None,
    )

    asyncio.run(sglang_omni.generate(generate_input))

    assert captured["payload"]["metadata"] == {"task": "avqa"}


def test_sglang_omni_adapter_rejects_rollout_replay(monkeypatch):
    from miles.rollout.generate_hub import sglang_omni

    monkeypatch.setattr(sglang_omni, "compute_prompt_ids_from_sample", lambda state, sample: [1, 2, 3])
    for flag in ("use_rollout_routing_replay", "use_rollout_indexer_replay"):
        args = _adapter_args()
        setattr(args, flag, True)
        generate_input = SimpleNamespace(
            args=args,
            sample=Sample(),
            sampling_params={"temperature": 1.0, "max_new_tokens": 16},
            state=None,
        )
        with pytest.raises(AssertionError, match="replay"):
            asyncio.run(sglang_omni.generate(generate_input))


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
