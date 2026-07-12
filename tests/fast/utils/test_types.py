"""Unit tests for Sample.strip_last_output_tokens."""

from unittest.mock import MagicMock

import numpy
import pytest

from miles.utils.types import DecodedAudio, DiscreteActionStream, RolloutActionTrace, Sample


def _make_action_trace() -> RolloutActionTrace:
    stream = DiscreteActionStream(
        name="higgs_codes",
        stage="tts_engine",
        modality="audio",
        shape=[2, 2],
        vocab_size=8,
        actions=[[0, 3], [4, 5]],
        policy_logprobs=[[0.0, -0.3], [-0.4, -0.5]],
        action_mask=[[False, True], [True, True]],
        codec_content_mask=[[False, True], [True, True]],
        channel_ids=[0, 1],
    )
    return RolloutActionTrace(
        version=2,
        model_family="higgs_tts",
        total_action_count=3,
        action_streams=[stream],
    )


def _make_sample(
    prompt_ids: list[int],
    response_ids: list[int],
    *,
    log_probs: bool = False,
    loss_mask: bool = False,
    routed_experts: bool = False,
    indexer_topk: bool = False,
) -> Sample:
    """Create a Sample with the given prompt + response token IDs."""
    tokens = prompt_ids + response_ids
    s = Sample(
        tokens=tokens,
        response_length=len(response_ids),
        response="dummy",
    )
    if log_probs:
        s.rollout_log_probs = [-0.1] * len(response_ids)
    if loss_mask:
        s.loss_mask = [1] * len(response_ids)
    if routed_experts:
        # shape: (num_tokens - 1, ...)
        s.rollout_routed_experts = numpy.zeros((len(tokens) - 1, 2, 2), dtype=numpy.int32)
    if indexer_topk:
        # shape: (num_tokens - 1, ...)
        s.rollout_indexer_topk = numpy.zeros((len(tokens) - 1, 2, 3), dtype=numpy.int32)
    return s


@pytest.fixture
def tokenizer():
    tok = MagicMock()
    tok.decode = lambda ids: "".join(chr(65 + i) for i in ids)
    return tok


class TestStripLastOutputTokens:
    def test_strip_zero_is_noop(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        original_tokens = list(s.tokens)
        s.strip_last_output_tokens(0, tokenizer)
        assert s.tokens == original_tokens
        assert s.response_length == 3

    def test_strip_basic(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(2, tokenizer)
        assert s.tokens == [1, 2, 3]
        assert s.response_length == 1

    def test_strip_all_response(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(3, tokenizer)
        assert s.tokens == [1, 2]
        assert s.response_length == 0
        assert s.response == ""

    def test_strip_too_many_raises(self, tokenizer):
        s = _make_sample([1, 2], [3, 4])
        with pytest.raises(AssertionError, match="cannot strip 3 tokens"):
            s.strip_last_output_tokens(3, tokenizer)

    def test_strip_truncates_log_probs(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], log_probs=True)
        assert len(s.rollout_log_probs) == 3
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_log_probs) == 1

    def test_strip_truncates_loss_mask(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], loss_mask=True)
        assert len(s.loss_mask) == 3
        s.strip_last_output_tokens(1, tokenizer)
        assert len(s.loss_mask) == 2

    def test_strip_truncates_routed_experts(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], routed_experts=True)
        original_len = len(s.rollout_routed_experts)
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_routed_experts) == original_len - 2

    def test_strip_truncates_indexer_topk(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5], indexer_topk=True)
        original_len = len(s.rollout_indexer_topk)
        s.strip_last_output_tokens(2, tokenizer)
        assert len(s.rollout_indexer_topk) == original_len - 2

    def test_strip_updates_response_text(self, tokenizer):
        s = _make_sample([1, 2], [3, 4, 5])
        s.strip_last_output_tokens(1, tokenizer)
        # response should be re-decoded from the remaining response tokens
        assert s.response == tokenizer.decode(s.tokens[-s.response_length :])

    def test_strip_negative_is_noop(self, tokenizer):
        s = _make_sample([1, 2], [3, 4])
        original_tokens = list(s.tokens)
        s.strip_last_output_tokens(-1, tokenizer)
        assert s.tokens == original_tokens


class TestStructuredActionTypes:
    def test_sample_dict_round_trip_preserves_typed_artifacts(self):
        sample = Sample(
            tokens=[1, 2],
            action_trace=_make_action_trace(),
            decoded_audio=DecodedAudio(data="UklGRg==", format="wav", sample_rate=24000),
        )

        restored = Sample.from_dict(sample.to_dict())

        assert isinstance(restored.action_trace, RolloutActionTrace)
        assert isinstance(restored.action_trace.action_streams[0], DiscreteActionStream)
        assert isinstance(restored.decoded_audio, DecodedAudio)
        assert restored.action_trace == sample.action_trace
        assert restored.decoded_audio == sample.decoded_audio

    def test_action_stream_rejects_shape_mismatch(self):
        data = _make_action_trace().action_streams[0].to_dict()
        data["shape"] = [3, 2]

        with pytest.raises(ValueError, match="declared shape"):
            DiscreteActionStream.from_dict(data)

    def test_action_stream_rejects_nonfinite_sampled_logprob(self):
        data = _make_action_trace().action_streams[0].to_dict()
        data["policy_logprobs"][0][1] = float("nan")

        with pytest.raises(ValueError, match="non-finite"):
            DiscreteActionStream.from_dict(data)

    def test_action_stream_rejects_nonzero_forced_logprob(self):
        data = _make_action_trace().action_streams[0].to_dict()
        data["policy_logprobs"][0][0] = -1.0

        with pytest.raises(ValueError, match="must be zero"):
            DiscreteActionStream.from_dict(data)

    def test_action_stream_rejects_out_of_range_action(self):
        data = _make_action_trace().action_streams[0].to_dict()
        data["actions"][1][1] = data["vocab_size"]

        with pytest.raises(ValueError, match="outside"):
            DiscreteActionStream.from_dict(data)

    def test_action_stream_requires_strict_ordered_channel_ids(self):
        data = _make_action_trace().action_streams[0].to_dict()
        data["channel_ids"] = [False, 1]

        with pytest.raises(ValueError, match="channel_ids"):
            DiscreteActionStream.from_dict(data)

    def test_to_dict_revalidates_mutated_streams(self):
        stream = _make_action_trace().action_streams[0]
        stream.policy_logprobs[0][0] = -1.0

        with pytest.raises(ValueError, match="must be zero"):
            stream.to_dict()

    def test_trace_rejects_incorrect_action_count(self):
        data = _make_action_trace().to_dict()
        data["total_action_count"] = 2

        with pytest.raises(ValueError, match="total_action_count"):
            RolloutActionTrace.from_dict(data)

    def test_trace_rejects_unknown_fields(self):
        data = _make_action_trace().to_dict()
        data["unexpected"] = True

        with pytest.raises(ValueError, match="extra=.*unexpected"):
            RolloutActionTrace.from_dict(data)

    def test_decoded_audio_requires_wav_with_positive_sample_rate(self):
        with pytest.raises(ValueError, match="format"):
            DecodedAudio(data="data", format="mp3", sample_rate=24000)
        with pytest.raises(ValueError, match="sample_rate"):
            DecodedAudio(data="data", format="wav", sample_rate=0)

    def test_reset_for_retry_clears_structured_outputs(self):
        sample = Sample(
            tokens=[1, 2],
            action_trace=_make_action_trace(),
            decoded_audio=DecodedAudio(data="UklGRg==", format="wav", sample_rate=24000),
        )

        sample.reset_for_retry()

        assert sample.action_trace is None
        assert sample.decoded_audio is None
