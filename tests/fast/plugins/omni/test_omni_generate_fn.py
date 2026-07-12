from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.utils.types import Sample
from miles_plugins.omni import omni_generate_fn
from miles_plugins.omni.omni_generate_fn import (
    OmniGenerateFn,
    build_higgs_generate_payload,
    build_zero_shot_higgs_prompt_ids,
    neutral_higgs_sampling_params,
)


class FakeHiggsTokenizer:
    def get_added_vocab(self) -> dict[str, int]:
        return {"<|tts|>": 100, "<|text|>": 101, "<|audio|>": 102}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "speak"
        assert add_special_tokens is False
        return [7, 8]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.7),
        ("top_p", 0.9),
        ("top_k", 10),
        ("min_p", 0.1),
        ("repetition_penalty", 1.1),
    ],
)
def test_neutral_sampling_rejects_behavior_changes(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="Higgs RL requires"):
        neutral_higgs_sampling_params({field: value})


def test_payload_requests_only_structured_audio() -> None:
    payload = build_higgs_generate_payload(
        [1, 2],
        {
            "temperature": 1,
            "top_p": 1.0,
            "top_k": -1,
            "sampling_seed": 12,
            "max_new_tokens": 8,
            "skip_special_tokens": False,
        },
    )

    assert payload["input_ids"] == [1, 2]
    assert payload["output_modalities"] == ["audio"]
    assert payload["return_logprob"] is True
    assert payload["return_omni_rollout"] is True
    assert payload["stream"] is False
    assert payload["sampling_params"] == {
        "temperature": 1.0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8,
        "seed": 12,
    }
    assert "metadata" not in payload


def test_zero_shot_prompt_matches_higgs_server_contract() -> None:
    assert build_zero_shot_higgs_prompt_ids(FakeHiggsTokenizer(), "speak") == [100, 101, 7, 8, 102]


def test_zero_shot_prompt_requires_higgs_special_tokens() -> None:
    tokenizer = FakeHiggsTokenizer()
    tokenizer.get_added_vocab = lambda: {"<|tts|>": 100, "<|text|>": 101}

    with pytest.raises(ValueError, match="missing Higgs TTS specials"):
        build_zero_shot_higgs_prompt_ids(tokenizer, "speak")


@pytest.mark.asyncio
async def test_generate_keeps_prompt_and_audio_actions_separate(monkeypatch, higgs_response: dict) -> None:
    seen: dict = {}
    response = deepcopy(higgs_response)
    response["meta_info"]["prompt_tokens"] = 5

    async def fake_post(url: str, payload: dict):
        seen.update(url=url, payload=payload)
        return response

    monkeypatch.setattr(omni_generate_fn, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_response_len=64,
        rollout_max_context_len=128,
    )
    state = SimpleNamespace(args=args, processor=None, tokenizer=FakeHiggsTokenizer())
    sample = Sample(prompt="speak", tokens=[100, 101, 7, 8, 102])

    output = await OmniGenerateFn()(
        GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params={"temperature": 1.0, "top_p": 1.0, "top_k": -1},
            evaluation=False,
        )
    )

    generated = output.samples
    assert generated is sample
    assert generated.tokens == [100, 101, 7, 8, 102]
    assert generated.response == ""
    assert generated.response_length == 0
    assert generated.rollout_log_probs is None
    assert generated.loss_mask is None
    assert generated.action_trace is not None
    assert generated.decoded_audio is not None
    assert generated.weight_versions == ["7"]
    assert generated.status is Sample.Status.COMPLETED
    assert seen["payload"]["input_ids"] == [100, 101, 7, 8, 102]
    assert "metadata" not in seen["payload"]


@pytest.mark.asyncio
async def test_generate_builds_zero_shot_prompt_for_normal_empty_token_sample(
    monkeypatch, higgs_response: dict
) -> None:
    seen: dict = {}
    response = deepcopy(higgs_response)
    response["meta_info"]["prompt_tokens"] = 5

    async def fake_post(url: str, payload: dict):
        seen.update(url=url, payload=payload)
        return response

    monkeypatch.setattr(omni_generate_fn, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        rollout_max_response_len=64,
        rollout_max_context_len=128,
    )
    state = SimpleNamespace(args=args, processor=None, tokenizer=FakeHiggsTokenizer())
    sample = Sample(prompt="speak")

    output = await OmniGenerateFn()(GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False))

    assert output.samples.tokens == [100, 101, 7, 8, 102]
    assert seen["payload"]["input_ids"] == [100, 101, 7, 8, 102]


@pytest.mark.asyncio
async def test_generate_rejects_reference_media_before_http(monkeypatch) -> None:
    async def unexpected_post(url: str, payload: dict):
        pytest.fail("unsupported media must be rejected before HTTP")

    monkeypatch.setattr(omni_generate_fn, "post", unexpected_post)
    args = SimpleNamespace(sglang_router_ip="localhost", sglang_router_port=1)
    state = SimpleNamespace(args=args, processor=None, tokenizer=FakeHiggsTokenizer())
    sample = Sample(prompt="speak", multimodal_inputs={"audios": ["https://example.test/reference.wav"]})

    with pytest.raises(ValueError, match="zero-shot text-to-audio only"):
        await OmniGenerateFn()(GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False))


@pytest.mark.asyncio
async def test_generate_rejects_noncanonical_pretokenized_prompt() -> None:
    args = SimpleNamespace(sglang_router_ip="localhost", sglang_router_port=1)
    state = SimpleNamespace(args=args, processor=None, tokenizer=FakeHiggsTokenizer())
    sample = Sample(prompt="speak", tokens=[7, 8])

    with pytest.raises(ValueError, match="canonical zero-shot encoding"):
        await OmniGenerateFn()(GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False))


@pytest.mark.parametrize(
    "sample",
    [
        Sample(response="partial", response_length=1, tokens=[1]),
        Sample(rollout_log_probs=[]),
        Sample(loss_mask=[]),
        Sample(weight_versions=["6"]),
    ],
)
@pytest.mark.asyncio
async def test_generate_rejects_partial_or_resumed_samples(sample: Sample) -> None:
    args = SimpleNamespace(sglang_router_ip="localhost", sglang_router_port=1)
    state = SimpleNamespace(args=args, processor=None, tokenizer=None)
    with pytest.raises(ValueError, match="partial token|partial text|fresh sample|text loss"):
        await OmniGenerateFn()(GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False))


@pytest.mark.asyncio
async def test_generate_rejects_aborted_sample_with_partial_tokens() -> None:
    args = SimpleNamespace(sglang_router_ip="localhost", sglang_router_port=1)
    state = SimpleNamespace(args=args, processor=None, tokenizer=FakeHiggsTokenizer())
    sample = Sample(prompt="speak", status=Sample.Status.ABORTED, tokens=[100])

    with pytest.raises(ValueError, match="partial token"):
        await OmniGenerateFn()(GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False))


@pytest.mark.asyncio
async def test_generate_allows_clean_aborted_retry(monkeypatch, higgs_response: dict) -> None:
    response = deepcopy(higgs_response)
    response["meta_info"]["prompt_tokens"] = 5

    async def fake_post(url: str, payload: dict):
        return response

    monkeypatch.setattr(omni_generate_fn, "post", fake_post)
    args = SimpleNamespace(
        sglang_router_ip="localhost",
        sglang_router_port=1,
        rollout_max_response_len=64,
        rollout_max_context_len=128,
    )
    state = SimpleNamespace(args=args, processor=None, tokenizer=FakeHiggsTokenizer())
    sample = Sample(prompt="speak", status=Sample.Status.ABORTED)

    output = await OmniGenerateFn()(GenerateFnInput(state=state, sample=sample, sampling_params={}, evaluation=False))

    assert output.samples.status is Sample.Status.COMPLETED
