"""Composite reward for TTS RL (GATE-B): ASR round-trip CER + audio-validity guards.

The TTS actor generates speech for a target text. The reward transcribes the generated
audio with an ASR model (Whisper) and scores content correctness via CER, combined with
hard audio-validity guards (decode success, duration bounds, non-silence). A failed decode
yields a deterministic low reward instead of crashing, so the loop never wedges. This is
the DEC-2 composite design: ASR alone rewards transcribable-but-degenerate audio, so the
guards must be present from day one.

Usable two ways:
  - standalone: ``TtsCompositeReward(...).score(audio_b64, target_text)``
  - miles hook: ``--custom-rm-path miles_plugins.omni.tts_reward.compute_tts_reward``
    (reads the generated audio from ``sample.metadata["generated_audio"]`` and the target
    text from ``sample.label``).
"""

from __future__ import annotations

import base64
import io
import os
import re
import wave
from dataclasses import dataclass, field
from typing import Any

# Suggested defaults (tunable). Duration in seconds; energy is RMS of float[-1,1] samples.
MIN_DURATION_S = 0.3
MAX_DURATION_S = 30.0
SILENCE_RMS_FLOOR = 1e-3
DECODE_FAIL_REWARD = -1.0
ASR_WEIGHT = 1.0

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_text(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _char_error_rate(hyp: str, ref: str) -> float:
    """Levenshtein char edit distance / len(ref), clamped to [0, 1]."""
    ref = _normalize_text(ref)
    hyp = _normalize_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return min(1.0, prev[-1] / len(ref))


def _decode_wav_b64(data: str) -> tuple[Any, int]:
    """Decode a base64 (optionally data-URI) WAV string to (float32 mono waveform, sr)."""
    import numpy as np

    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    return pcm, sr


@dataclass
class RewardComponents:
    reward: float
    cer: float | None = None
    duration_s: float | None = None
    rms: float | None = None
    transcript: str = ""
    guard: str = "ok"  # "ok" | "decode_fail" | "too_short" | "too_long" | "silent"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TtsCompositeReward:
    asr_model_path: str = field(default_factory=lambda: os.environ.get("ASR_MODEL", "openai/whisper-base"))
    device: str = field(default_factory=lambda: os.environ.get("ASR_DEVICE", "cuda:0"))
    asr_weight: float = ASR_WEIGHT
    min_duration_s: float = MIN_DURATION_S
    max_duration_s: float = MAX_DURATION_S
    silence_rms_floor: float = SILENCE_RMS_FLOOR

    _model: Any = None
    _processor: Any = None

    def _ensure_asr(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._processor = WhisperProcessor.from_pretrained(self.asr_model_path)
        self._model = (
            WhisperForConditionalGeneration.from_pretrained(self.asr_model_path, dtype=torch.float16)
            .to(self.device)
            .eval()
        )

    def transcribe(self, waveform, sr: int) -> str:
        import torch
        import torchaudio.functional as AF

        self._ensure_asr()
        wav = torch.as_tensor(waveform).float()
        if sr != 16000:
            wav = AF.resample(wav, sr, 16000)
        feats = self._processor(
            wav.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device, dtype=self._model.dtype)
        with torch.no_grad():
            ids = self._model.generate(feats, language="en", task="transcribe", max_new_tokens=128)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def score(self, audio_b64: str | None, target_text: str) -> RewardComponents:
        """Composite reward in [DECODE_FAIL_REWARD, asr_weight]; never raises."""
        if not audio_b64:
            return RewardComponents(reward=DECODE_FAIL_REWARD, guard="decode_fail")
        try:
            wav, sr = _decode_wav_b64(audio_b64)
        except Exception:
            return RewardComponents(reward=DECODE_FAIL_REWARD, guard="decode_fail")

        import numpy as np

        duration = len(wav) / sr if sr else 0.0
        rms = float(np.sqrt(np.mean(wav**2))) if len(wav) else 0.0
        if duration < self.min_duration_s:
            return RewardComponents(reward=DECODE_FAIL_REWARD, duration_s=duration, rms=rms, guard="too_short")
        if duration > self.max_duration_s:
            return RewardComponents(reward=DECODE_FAIL_REWARD, duration_s=duration, rms=rms, guard="too_long")
        if rms < self.silence_rms_floor:
            return RewardComponents(reward=DECODE_FAIL_REWARD, duration_s=duration, rms=rms, guard="silent")

        try:
            transcript = self.transcribe(wav, sr)
        except Exception:
            return RewardComponents(reward=DECODE_FAIL_REWARD, duration_s=duration, rms=rms, guard="decode_fail")

        cer = _char_error_rate(transcript, target_text)
        reward = self.asr_weight * (1.0 - cer)
        return RewardComponents(
            reward=reward, cer=cer, duration_s=duration, rms=rms, transcript=transcript, guard="ok"
        )


_SHARED: TtsCompositeReward | None = None


async def compute_tts_reward(args, sample, **kwargs) -> float:
    """miles --custom-rm-path hook: score generated audio against sample.label."""
    global _SHARED
    if _SHARED is None:
        _SHARED = TtsCompositeReward()
    audio = (sample.metadata or {}).get("generated_audio")
    audio_b64 = audio.get("data") if isinstance(audio, dict) else audio
    comp = _SHARED.score(audio_b64, str(sample.label or ""))
    if isinstance(sample.metadata, dict):
        sample.metadata["tts_reward_components"] = comp.to_dict()
    return comp.reward
