"""ASR round-trip reward for decoded Higgs WAV output."""

from __future__ import annotations

import base64
import binascii
import io
import math
import os
import re
import wave
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from miles.utils.types import DecodedAudio, Sample

INVALID_AUDIO_REWARD = -1.0
_TEXT_CHARS = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_asr_text(text: str) -> str:
    return _TEXT_CHARS.sub("", text.casefold())


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = normalize_asr_text(reference)
    hypothesis = normalize_asr_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0

    previous = list(range(len(hypothesis) + 1))
    for ref_index, reference_char in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hypothesis_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + (reference_char != hypothesis_char),
                )
            )
        previous = current
    return min(1.0, previous[-1] / len(reference))


def decode_wav(audio: DecodedAudio) -> tuple[np.ndarray, int]:
    """Decode the typed server artifact and verify its declared sample rate."""
    encoded = audio.data.split(",", 1)[1] if audio.data.startswith("data:") and "," in audio.data else audio.data
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("decoded audio is not valid base64") from error
    if not raw:
        raise ValueError("decoded WAV is empty")

    try:
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                raise ValueError("decoded WAV must use 16-bit PCM")
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            frame_count = wav_file.getnframes()
            pcm = np.frombuffer(wav_file.readframes(frame_count), dtype="<i2")
    except (EOFError, wave.Error) as error:
        raise ValueError("decoded audio is not a valid WAV file") from error

    if sample_rate != audio.sample_rate:
        raise ValueError("decoded WAV sample rate does not match response metadata")
    if channels <= 0 or frame_count <= 0 or pcm.size != frame_count * channels:
        raise ValueError("decoded WAV contains no complete audio frames")
    waveform = pcm.reshape(frame_count, channels).astype(np.float32).mean(axis=1) / 32768.0
    return waveform, sample_rate


@dataclass
class TtsRoundTripReward:
    asr_model_path: str = field(default_factory=lambda: os.environ.get("MILES_TTS_ASR_MODEL", "openai/whisper-base"))
    device: str = field(default_factory=lambda: os.environ.get("MILES_TTS_ASR_DEVICE", "cpu"))
    min_duration_seconds: float = 0.3
    max_duration_seconds: float = 30.0
    silence_rms_floor: float = 1e-3
    max_clipped_fraction: float = 0.01

    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def _load_asr(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.asr_model_path)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.asr_model_path,
            torch_dtype=dtype,
        ).to(self.device)
        self._model.eval()

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        import torch

        self._load_asr()
        if sample_rate != 16000:
            target_length = max(1, round(waveform.size * 16000 / sample_rate))
            source_positions = np.arange(waveform.size, dtype=np.float64)
            target_positions = np.linspace(0, waveform.size - 1, target_length)
            waveform = np.interp(target_positions, source_positions, waveform).astype(np.float32)
        model_inputs = self._processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = model_inputs.input_features.to(self.device, dtype=self._model.dtype)
        attention_mask = model_inputs.attention_mask.to(self.device)
        with torch.no_grad():
            token_ids = self._model.generate(
                features,
                attention_mask=attention_mask,
                max_new_tokens=128,
                task="transcribe",
            )
        return self._processor.batch_decode(token_ids, skip_special_tokens=True)[0]

    def score(self, audio: DecodedAudio, target_text: str) -> float:
        try:
            waveform, sample_rate = decode_wav(audio)
        except ValueError:
            return INVALID_AUDIO_REWARD

        duration = waveform.size / sample_rate
        if not self.min_duration_seconds <= duration <= self.max_duration_seconds:
            return INVALID_AUDIO_REWARD
        if not bool(np.isfinite(waveform).all()):
            return INVALID_AUDIO_REWARD
        rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
        if not math.isfinite(rms) or rms < self.silence_rms_floor:
            return INVALID_AUDIO_REWARD
        clipped_fraction = float(np.mean(np.abs(waveform) >= (32767.0 / 32768.0)))
        if clipped_fraction > self.max_clipped_fraction:
            return INVALID_AUDIO_REWARD
        normalized_target = normalize_asr_text(target_text)
        if not normalized_target:
            return INVALID_AUDIO_REWARD

        # Model loading and inference failures are infrastructure errors, not bad samples.
        transcript = self.transcribe(waveform, sample_rate)
        reward = 1.0 - character_error_rate(normalized_target, transcript)
        return float(min(1.0, max(0.0, reward)))


_SHARED_REWARD: TtsRoundTripReward | None = None


def _reward_model(args: Any) -> TtsRoundTripReward:
    global _SHARED_REWARD
    if _SHARED_REWARD is None:
        values = vars(args)
        _SHARED_REWARD = TtsRoundTripReward(
            asr_model_path=values.get("tts_asr_model", os.environ.get("MILES_TTS_ASR_MODEL", "openai/whisper-base")),
            device=values.get("tts_asr_device", os.environ.get("MILES_TTS_ASR_DEVICE", "cpu")),
        )
    return _SHARED_REWARD


def _target_text(sample: Sample) -> str:
    if isinstance(sample.prompt, str):
        return sample.prompt
    for message in reversed(sample.prompt):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def _score_and_release(reward_model: TtsRoundTripReward, sample: Sample) -> float:
    try:
        if sample.decoded_audio is None:
            return INVALID_AUDIO_REWARD
        return reward_model.score(sample.decoded_audio, _target_text(sample))
    finally:
        sample.decoded_audio = None


async def compute_tts_reward(
    args: Any,
    sample: Sample | list[Sample],
    **_: Any,
) -> float | list[float]:
    """Miles custom reward hook; decoded waveform bytes never enter training data."""
    reward_model = _reward_model(args)
    if isinstance(sample, list):
        try:
            return [_score_and_release(reward_model, item) for item in sample]
        finally:
            for item in sample:
                item.decoded_audio = None
    return _score_and_release(reward_model, sample)


__all__ = [
    "INVALID_AUDIO_REWARD",
    "TtsRoundTripReward",
    "character_error_rate",
    "compute_tts_reward",
    "decode_wav",
    "normalize_asr_text",
]
