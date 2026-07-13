from __future__ import annotations

import base64
import io
import wave

import pytest


def wav_base64(*, sample_rate: int = 24000, frames: int = 9600, amplitude: int = 4000) -> str:
    samples = int(amplitude).to_bytes(2, "little", signed=True) * frames
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@pytest.fixture
def higgs_response() -> dict:
    actions = [
        [10, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        [1025, 11, 12, 13, 14, 15, 16, 17],
    ]
    action_mask = [
        [True, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ]
    policy_logprobs = [
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
    ]
    return {
        "text": "",
        "audio": {
            "data": wav_base64(),
            "path": None,
            "format": "wav",
            "sample_rate": 24000,
        },
        "meta_info": {
            "finish_reason": {"type": "stop", "length": None},
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "cached_tokens": 1,
            "weight_version": "7",
            "request_metadata": None,
            "output_token_logprobs": [[-1.0, 10], [-2.0, 1025]],
            "output_codebook_tokens": [list(row) for row in actions],
            "omni_rollout": {
                "version": 2,
                "model_family": "higgs_tts",
                "total_action_count": 9,
                "action_streams": [
                    {
                        "name": "higgs_codes",
                        "stage": "tts_engine",
                        "modality": "audio",
                        "action_type": "multi_discrete",
                        "layout": "time_codebook",
                        "shape": [2, 8],
                        "vocab_size": 1026,
                        "actions": [list(row) for row in actions],
                        "policy_logprobs": policy_logprobs,
                        "action_mask": action_mask,
                        "codec_content_mask": action_mask,
                        "channel_ids": list(range(8)),
                    }
                ],
            },
        },
    }
