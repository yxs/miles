"""Integration tests for the loadable OmniGenerateFn hook.

Loads the class through the same path-string loader rollout uses, stubs the HTTP
transport, and asserts the exact request emitted to the omni ``/generate`` endpoint plus
the resulting sample. Exercises the highest-risk path (the real ``__call__``), unlike the
pure-helper tests in test_omni_rollout_contract.py.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import miles_plugins.omni.omni_generate_fn as omni_mod
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.utils.types import Sample

_HOOK_PATH = "miles_plugins.omni.omni_generate_fn.OmniGenerateFn"


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def _fake_state(*, max_context_len=0):
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=8000,
        rollout_max_response_len=128,
        rollout_max_context_len=max_context_len,
        sglang_speculative_algorithm=None,
    )
    return SimpleNamespace(args=args, tokenizer=_FakeTokenizer(), processor=None)


def _canned_response():
    return {
        "text": "hello",
        "audio": {"format": "wav", "data": "<b64>"},
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [[-0.1, 10], [-0.2, 11]],
            "completion_tokens": 2,
            "weight_version": "7",
            "cached_tokens": 0,
            "prompt_tokens": 3,
        },
    }


def test_omni_generate_fn_emits_payload_and_applies_response(monkeypatch):
    captured = {}

    async def fake_post(url, payload, **kwargs):
        captured["url"] = url
        captured["payload"] = payload
        return _canned_response()

    monkeypatch.setattr(omni_mod, "post", fake_post)

    fn = load_generate_function(_HOOK_PATH)
    assert fn is not None

    sample = Sample(prompt="hi", index=5, group_index=2)
    inp = GenerateFnInput(
        state=_fake_state(),
        sample=sample,
        sampling_params={
            "temperature": 0.7,
            "skip_special_tokens": True,  # dropped by the omni schema
            "sampling_seed": 9,  # aliased -> seed
            "max_new_tokens": 64,
        },
        evaluation=False,
    )

    out = asyncio.run(fn(inp))
    result_sample = out.samples

    payload = captured["payload"]
    assert captured["url"] == "http://127.0.0.1:8000/generate"
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["return_logprob"] is True
    assert payload["sampling_params"] == {"temperature": 0.7, "seed": 9, "max_new_tokens": 64}
    assert payload["metadata"] == {"group_index": 2, "index": 5}
    assert "audio_data" not in payload  # no input audio on this sample

    assert result_sample.tokens == [1, 2, 3, 10, 11]
    assert result_sample.response_length == 2
    assert result_sample.rollout_log_probs == [-0.1, -0.2]
    assert result_sample.response == "hello"
    # generated audio is reward-facing -> metadata, never multimodal_train_inputs
    assert result_sample.metadata["generated_audio"] == {"format": "wav", "data": "<b64>"}
    assert result_sample.multimodal_train_inputs is None
    assert result_sample.weight_versions == ["7"]
    assert result_sample.status == Sample.Status.COMPLETED


def test_omni_generate_fn_truncates_when_no_context_budget(monkeypatch):
    async def fail_post(url, payload, **kwargs):
        raise AssertionError("post must not be called when there is no token budget")

    monkeypatch.setattr(omni_mod, "post", fail_post)

    fn = load_generate_function(_HOOK_PATH)
    sample = Sample(prompt="hi")
    inp = GenerateFnInput(
        state=_fake_state(max_context_len=3),  # prompt is 3 tokens -> 0 budget left
        sample=sample,
        sampling_params={"max_new_tokens": 64},
        evaluation=False,
    )

    out = asyncio.run(fn(inp))
    assert out.samples.status == Sample.Status.TRUNCATED


def test_omni_generate_fn_encodes_input_audio(monkeypatch):
    captured = {}

    async def fake_post(url, payload, **kwargs):
        captured["payload"] = payload
        return _canned_response()

    monkeypatch.setattr(omni_mod, "post", fake_post)

    fn = load_generate_function(_HOOK_PATH)
    sample = Sample(prompt="hi")
    sample.multimodal_inputs = {"audios": [(np.zeros(160, dtype=np.float32), 16000)]}
    inp = GenerateFnInput(
        state=_fake_state(),
        sample=sample,
        sampling_params={"max_new_tokens": 32},
        evaluation=False,
    )

    asyncio.run(fn(inp))
    audio_data = captured["payload"]["audio_data"]
    assert len(audio_data) == 1
    assert audio_data[0].startswith("data:audio/wav;base64,")


def test_omni_generate_fn_resume_keeps_loss_mask_aligned(monkeypatch):
    async def fake_post(url, payload, **kwargs):
        # resume turn: only the newly generated tokens come back
        return {
            "text": " more",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.3, 20], [-0.4, 21]],
                "completion_tokens": 2,
                "cached_tokens": 0,
                "prompt_tokens": 5,
            },
        }

    monkeypatch.setattr(omni_mod, "post", fake_post)

    fn = load_generate_function(_HOOK_PATH)
    sample = Sample(prompt="hi")
    # simulate a partial rollout whose off-policy response was pre-masked by generate_and_rm
    sample.tokens = [1, 2, 3, 10, 11]  # prompt [1,2,3] + old response [10,11]
    sample.response = "old"
    sample.response_length = 2
    sample.loss_mask = [0, 0]  # off-policy tokens masked off
    sample.rollout_log_probs = [-0.1, -0.2]
    inp = GenerateFnInput(
        state=_fake_state(),
        sample=sample,
        sampling_params={"max_new_tokens": 64},
        evaluation=False,
    )

    out = asyncio.run(fn(inp))
    s = out.samples
    assert s.tokens == [1, 2, 3, 10, 11, 20, 21]
    assert s.response_length == 4
    # new on-policy tokens are trainable; mask stays aligned with response_length
    assert s.loss_mask == [0, 0, 1, 1]
    assert len(s.loss_mask) == s.response_length
