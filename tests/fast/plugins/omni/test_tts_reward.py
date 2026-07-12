from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from miles.utils.types import DecodedAudio, Sample
from miles_plugins.omni import tts_reward
from miles_plugins.omni.tts_reward import (
    INVALID_AUDIO_REWARD,
    TtsRoundTripReward,
    character_error_rate,
    compute_tts_reward,
)

from .conftest import wav_base64


def test_character_error_rate_is_normalized_and_bounded() -> None:
    assert character_error_rate("Hello, world!", "hello world") == 0.0
    assert character_error_rate("abc", "xyzxyz") == 1.0


def test_tts_reward_scores_valid_wav_without_retaining_components(monkeypatch) -> None:
    reward = TtsRoundTripReward()
    monkeypatch.setattr(reward, "transcribe", lambda waveform, sample_rate: "hello world")
    audio = DecodedAudio(data=wav_base64(), format="wav", sample_rate=24000)

    assert reward.score(audio, "Hello, world!") == 1.0


def test_tts_reward_defaults_to_cpu_when_actor_has_no_gpu(monkeypatch) -> None:
    monkeypatch.delenv("MILES_TTS_ASR_DEVICE", raising=False)

    assert TtsRoundTripReward().device == "cpu"


def test_tts_reward_surfaces_asr_infrastructure_failures(monkeypatch) -> None:
    reward = TtsRoundTripReward()

    def fail_transcription(waveform, sample_rate):
        raise RuntimeError("ASR model unavailable")

    monkeypatch.setattr(reward, "transcribe", fail_transcription)
    audio = DecodedAudio(data=wav_base64(), format="wav", sample_rate=24000)

    with pytest.raises(RuntimeError, match="ASR model unavailable"):
        reward.score(audio, "hello")


def test_whisper_receives_attention_mask_and_transcribe_task() -> None:
    calls: dict = {}

    class FakeProcessor:
        def __call__(self, waveform, **kwargs):
            calls["processor"] = kwargs
            return SimpleNamespace(
                input_features=torch.ones(1, 80, 4),
                attention_mask=torch.ones(1, 4, dtype=torch.long),
            )

        def batch_decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return ["hello"]

    class FakeModel:
        dtype = torch.float32

        def generate(self, features, **kwargs):
            calls["generate"] = kwargs
            return torch.tensor([[1]])

    reward = TtsRoundTripReward(device="cpu")
    reward._processor = FakeProcessor()
    reward._model = FakeModel()

    assert reward.transcribe(np.ones(16000, dtype=np.float32), 16000) == "hello"
    assert calls["processor"]["return_attention_mask"] is True
    assert calls["generate"]["task"] == "transcribe"
    assert torch.equal(calls["generate"]["attention_mask"], torch.ones(1, 4, dtype=torch.long))


@pytest.mark.asyncio
async def test_reward_releases_audio_after_scoring(monkeypatch) -> None:
    reward = TtsRoundTripReward()
    monkeypatch.setattr(reward, "transcribe", lambda waveform, sample_rate: "hello")
    monkeypatch.setattr(tts_reward, "_SHARED_REWARD", reward)
    sample = Sample(
        prompt="hello",
        decoded_audio=DecodedAudio(data=wav_base64(), format="wav", sample_rate=24000),
    )

    assert await compute_tts_reward(SimpleNamespace(), sample) == 1.0
    assert sample.decoded_audio is None


@pytest.mark.asyncio
async def test_reward_compares_asr_to_the_text_sent_for_generation(monkeypatch) -> None:
    reward = TtsRoundTripReward()
    monkeypatch.setattr(reward, "transcribe", lambda waveform, sample_rate: "spoken prompt")
    monkeypatch.setattr(tts_reward, "_SHARED_REWARD", reward)
    sample = Sample(
        prompt="spoken prompt",
        label="unrelated dataset label",
        decoded_audio=DecodedAudio(data=wav_base64(), format="wav", sample_rate=24000),
    )

    assert await compute_tts_reward(SimpleNamespace(), sample) == 1.0


@pytest.mark.asyncio
async def test_reward_releases_audio_even_when_scorer_raises(monkeypatch) -> None:
    class RaisingReward:
        def score(self, audio, target_text):
            raise RuntimeError("ASR failed")

    monkeypatch.setattr(tts_reward, "_SHARED_REWARD", RaisingReward())
    sample = Sample(
        prompt="hello",
        decoded_audio=DecodedAudio(data=wav_base64(), format="wav", sample_rate=24000),
    )

    with pytest.raises(RuntimeError, match="ASR failed"):
        await compute_tts_reward(SimpleNamespace(), sample)
    assert sample.decoded_audio is None


@pytest.mark.asyncio
async def test_batch_reward_releases_unscored_audio_after_failure(monkeypatch) -> None:
    class RaisingReward:
        def score(self, audio, target_text):
            raise RuntimeError("ASR failed")

    monkeypatch.setattr(tts_reward, "_SHARED_REWARD", RaisingReward())
    samples = [
        Sample(
            prompt="hello",
            decoded_audio=DecodedAudio(data=wav_base64(), format="wav", sample_rate=24000),
        )
        for _ in range(2)
    ]

    with pytest.raises(RuntimeError, match="ASR failed"):
        await compute_tts_reward(SimpleNamespace(), samples)
    assert all(sample.decoded_audio is None for sample in samples)


@pytest.mark.asyncio
async def test_invalid_audio_is_rejected_and_released(monkeypatch) -> None:
    monkeypatch.setattr(tts_reward, "_SHARED_REWARD", TtsRoundTripReward())
    sample = Sample(
        prompt="hello",
        decoded_audio=DecodedAudio(data="not-base64", format="wav", sample_rate=24000),
    )

    assert await compute_tts_reward(SimpleNamespace(), sample) == INVALID_AUDIO_REWARD
    assert sample.decoded_audio is None
