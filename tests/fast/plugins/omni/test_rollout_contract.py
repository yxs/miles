from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from miles.utils.types import DecodedAudio, RolloutActionTrace
from miles_plugins.omni.rollout_contract import parse_higgs_generate_response


def test_parse_higgs_v2_response_maps_to_domain_types(higgs_response: dict) -> None:
    result = parse_higgs_generate_response(higgs_response, expected_prompt_tokens=3)

    assert isinstance(result.action_trace, RolloutActionTrace)
    assert isinstance(result.decoded_audio, DecodedAudio)
    assert result.action_trace.action_streams[0].shape == [2, 8]
    assert result.action_trace.total_action_count == 9
    assert result.decoded_audio.sample_rate == 24000
    assert result.weight_version == "7"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda body: body["meta_info"]["omni_rollout"]["action_streams"][0].update(shape=[2, 7]),
            "8 codebooks",
        ),
        (
            lambda body: body["meta_info"]["omni_rollout"]["action_streams"][0]["action_mask"][0].__setitem__(1, 0),
            "boolean",
        ),
        (
            lambda body: body["meta_info"]["omni_rollout"]["action_streams"][0]["policy_logprobs"][0].__setitem__(
                1, -1.0
            ),
            "forced Higgs action policy logprob must be zero",
        ),
        (
            lambda body: body["meta_info"]["omni_rollout"]["action_streams"][0]["policy_logprobs"][1].__setitem__(
                0, float("nan")
            ),
            "non-finite",
        ),
        (
            lambda body: body["meta_info"]["omni_rollout"].update(total_action_count=8),
            "total_action_count",
        ),
        (
            lambda body: body["meta_info"].update(completion_tokens=3),
            "completion_tokens",
        ),
        (
            lambda body: body["meta_info"]["output_codebook_tokens"][0].__setitem__(0, 99),
            "output_codebook_tokens",
        ),
        (
            lambda body: body["audio"].update(format="pcm"),
            "wav",
        ),
        (
            lambda body: body["audio"].update(data=""),
            "at least 1 character",
        ),
    ],
)
def test_parse_higgs_v2_response_fails_closed(higgs_response: dict, mutate, message: str) -> None:
    body = copy.deepcopy(higgs_response)
    mutate(body)
    with pytest.raises((ValidationError, ValueError), match=message):
        parse_higgs_generate_response(body)


def test_parse_rejects_prompt_retokenization(higgs_response: dict) -> None:
    with pytest.raises(ValueError, match="exact prompt IDs"):
        parse_higgs_generate_response(higgs_response, expected_prompt_tokens=4)


def test_parse_rejects_unknown_wire_fields(higgs_response: dict) -> None:
    higgs_response["meta_info"]["unexpected"] = "silently dropped"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse_higgs_generate_response(higgs_response)
