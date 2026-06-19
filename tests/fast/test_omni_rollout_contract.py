"""Unit tests for the omni rollout contract and the audio rollout-encode helper.

These exercise the miles-side glue for the sglang-omni ``/generate`` backend without any
GPU, network, or model: payload whitelisting, response parsing/alignment, sample
accumulation, and WAV encoding.
"""

import base64
import io
import wave

import numpy as np
import pytest

from miles.utils.processing_utils import encode_audio_for_rollout_engine
from miles.utils.types import Sample
from miles_plugins.omni.rollout_contract import (
    OMNI_SAMPLING_PARAM_KEYS,
    apply_response_to_sample,
    build_generate_payload,
    clean_sampling_params,
    parse_generate_response,
)


def _response(token_logprobs, *, completion_tokens=None, finish="stop", **meta):
    meta_info = {
        "finish_reason": {"type": finish},
        "output_token_logprobs": token_logprobs,
        **meta,
    }
    if completion_tokens is not None:
        meta_info["completion_tokens"] = completion_tokens
    return {"text": meta.pop("text", ""), "meta_info": meta_info}


# --- sampling param whitelisting -------------------------------------------------------


def test_clean_sampling_params_drops_unknown_keys_and_aliases_seed():
    raw = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 128,
        "sampling_seed": 1234,  # legacy alias -> seed
        "skip_special_tokens": True,  # not accepted by omni schema
        "no_stop_trim": False,  # not accepted by omni schema
        "spaces_between_special_tokens": True,  # not accepted by omni schema
        "top_k": None,  # None dropped
    }
    cleaned = clean_sampling_params(raw)
    assert cleaned == {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 128, "seed": 1234}
    assert set(cleaned).issubset(OMNI_SAMPLING_PARAM_KEYS)
    assert "skip_special_tokens" not in cleaned
    assert "sampling_seed" not in cleaned


def test_build_generate_payload_shape_and_metadata():
    payload = build_generate_payload(
        [1, 2, 3],
        {"temperature": 1.0, "skip_special_tokens": True},
        metadata={"index": 5},
        output_modalities=["audio"],
    )
    assert payload["input_ids"] == [1, 2, 3]
    assert payload["return_logprob"] is True
    assert payload["sampling_params"] == {"temperature": 1.0}  # forbidden key removed
    assert payload["metadata"] == {"index": 5}
    assert payload["output_modalities"] == ["audio"]
    # empty metadata must not be emitted
    assert "metadata" not in build_generate_payload([1], {})


# --- response parsing ------------------------------------------------------------------


def test_parse_generate_response_aligns_tokens_and_logprobs():
    resp = _response(
        [[-0.1, 10], [-0.2, 11], [-0.3, 12]],
        completion_tokens=3,
        weight_version="42",
        cached_tokens=7,
        prompt_tokens=9,
    )
    result = parse_generate_response(resp)
    assert result.response_tokens == [10, 11, 12]
    assert result.response_log_probs == [-0.1, -0.2, -0.3]
    assert result.weight_version == "42"
    assert result.cached_tokens == 7 and isinstance(result.cached_tokens, int)
    assert result.completion_tokens == 3


def test_parse_generate_response_captures_audio_and_text():
    resp = _response([[-0.5, 99]], completion_tokens=1, text="hi")
    resp["text"] = "hi"
    resp["audio"] = {"format": "wav", "sample_rate": 24000, "data": "<b64>"}
    result = parse_generate_response(resp)
    assert result.audio == {"format": "wav", "sample_rate": 24000, "data": "<b64>"}
    assert result.text == "hi"


def test_parse_generate_response_empty_completion_is_not_an_error():
    result = parse_generate_response(_response([], completion_tokens=0))
    assert result.response_tokens == []
    assert result.response_log_probs == []


def test_parse_generate_response_length_mismatch_raises():
    with pytest.raises(ValueError, match="completion_tokens"):
        parse_generate_response(_response([[-0.1, 10]], completion_tokens=5))


def test_parse_generate_response_malformed_item_raises():
    with pytest.raises(ValueError, match="malformed"):
        parse_generate_response(_response([[-0.1]], completion_tokens=1))


def test_parse_generate_response_rejects_overlong_logprob_entry():
    # strict contract: each entry must be exactly [log_prob, token_id]
    with pytest.raises(ValueError, match="malformed"):
        parse_generate_response(_response([[-0.1, 10, "extra"]], completion_tokens=1))


def test_parse_generate_response_missing_meta_info_raises():
    with pytest.raises(ValueError, match="meta_info"):
        parse_generate_response({"text": ""})


def test_parse_generate_response_missing_finish_reason_raises():
    with pytest.raises(ValueError, match="finish_reason"):
        parse_generate_response({"meta_info": {"output_token_logprobs": []}})


# --- sample accumulation ---------------------------------------------------------------


def test_apply_response_to_sample_aligns_and_validates():
    sample = Sample(prompt="p", tokens=[])
    prompt_ids = [1, 2, 3]
    result = parse_generate_response(
        _response([[-0.1, 10], [-0.2, 11]], completion_tokens=2, weight_version="3")
    )
    apply_response_to_sample(sample, prompt_ids, result, update_loss_mask=True)

    assert sample.tokens == [1, 2, 3, 10, 11]
    assert sample.response_length == 2
    assert sample.rollout_log_probs == [-0.1, -0.2]
    # miles convention: loss_mask spans only the response tokens
    assert sample.loss_mask == [1, 1]
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length
    sample.validate()  # must not raise


def test_apply_response_to_sample_stores_audio_in_metadata_not_train_inputs():
    sample = Sample(prompt="p", tokens=[])
    result = parse_generate_response(_response([[-0.1, 5]], completion_tokens=1))
    result.audio = {"format": "wav", "data": "<b64>"}
    apply_response_to_sample(sample, [1, 2], result)
    # reward-facing audio lives in metadata; multimodal_train_inputs stays tensor-only
    assert sample.metadata["generated_audio"] == {"format": "wav", "data": "<b64>"}
    assert sample.multimodal_train_inputs is None


def test_apply_response_to_sample_multi_turn_accumulates():
    sample = Sample(prompt="p", tokens=[])
    first = parse_generate_response(_response([[-0.1, 10]], completion_tokens=1))
    apply_response_to_sample(sample, [1, 2], first, update_loss_mask=True)
    # second turn: tokens already present, continue appending
    second = parse_generate_response(_response([[-0.2, 20], [-0.3, 21]], completion_tokens=2))
    apply_response_to_sample(sample, [1, 2], second, update_loss_mask=True)

    assert sample.tokens == [1, 2, 10, 20, 21]
    assert sample.response_length == 3
    assert sample.rollout_log_probs == [-0.1, -0.2, -0.3]
    assert sample.loss_mask == [1, 1, 1]


# --- audio encode helper ---------------------------------------------------------------


def test_encode_audio_for_rollout_engine_roundtrips_wav():
    sampling_rate = 24000
    waveform = np.linspace(-1.0, 1.0, num=480, dtype=np.float32)
    uri = encode_audio_for_rollout_engine(waveform, sampling_rate)
    assert uri.startswith("data:audio/wav;base64,")

    raw = base64.b64decode(uri.split(",", 1)[1])
    with wave.open(io.BytesIO(raw), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == sampling_rate
        assert wav_file.getnframes() == 480


def test_encode_audio_for_rollout_engine_rejects_multichannel():
    with pytest.raises(ValueError, match="mono"):
        encode_audio_for_rollout_engine(np.zeros((2, 100), dtype=np.float32), 16000)


def test_encode_audio_for_rollout_engine_rejects_out_of_range_int():
    with pytest.raises(ValueError, match="int16"):
        encode_audio_for_rollout_engine(np.array([0, 40000, -50000], dtype=np.int32), 16000)


def test_encode_audios_for_rollout_engine_handles_tuples_and_dicts():
    from miles.utils.processing_utils import encode_audios_for_rollout_engine

    audios = [
        (np.zeros(160, dtype=np.float32), 16000),
        {"array": np.zeros(240, dtype=np.int16), "sampling_rate": 24000},
    ]
    uris = encode_audios_for_rollout_engine(audios)
    assert len(uris) == 2
    assert all(u.startswith("data:audio/wav;base64,") for u in uris)
