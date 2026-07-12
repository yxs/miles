"""Strict Higgs rollout contract for sglang-omni ``POST /generate``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from miles.utils.types import DecodedAudio, DiscreteActionStream, RolloutActionTrace

HIGGS_ROLLOUT_VERSION = 2
HIGGS_MODEL_FAMILY = "higgs_tts"
HIGGS_STREAM_NAME = "higgs_codes"
HIGGS_STREAM_STAGE = "tts_engine"
HIGGS_NUM_CODEBOOKS = 8
HIGGS_CODEBOOK_VOCAB_SIZE = 1026


class _StrictWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class HiggsActionStreamResponse(_StrictWireModel):
    name: Literal["higgs_codes"]
    stage: Literal["tts_engine"]
    modality: Literal["audio"]
    action_type: Literal["multi_discrete"]
    layout: Literal["time_codebook"]
    shape: list[StrictInt]
    vocab_size: Literal[1026]
    actions: list[list[StrictInt]]
    policy_logprobs: list[list[StrictFloat]]
    action_mask: list[list[StrictBool]]
    codec_content_mask: list[list[StrictBool]] | None = None
    channel_ids: list[StrictInt]

    @model_validator(mode="after")
    def validate_lattice(self) -> HiggsActionStreamResponse:
        if len(self.shape) != 2:
            raise ValueError("Higgs action stream shape must be [time, codebooks]")
        length, codebooks = self.shape
        if length <= 0:
            raise ValueError("Higgs action stream must contain at least one row")
        if codebooks != HIGGS_NUM_CODEBOOKS:
            raise ValueError(f"Higgs action stream must contain {HIGGS_NUM_CODEBOOKS} codebooks")
        if self.channel_ids != list(range(HIGGS_NUM_CODEBOOKS)):
            raise ValueError("Higgs channel_ids must be the ordered codebook indices")

        matrices: dict[str, list[list[Any]]] = {
            "actions": self.actions,
            "policy_logprobs": self.policy_logprobs,
            "action_mask": self.action_mask,
        }
        if self.codec_content_mask is not None:
            matrices["codec_content_mask"] = self.codec_content_mask
        for name, matrix in matrices.items():
            if len(matrix) != length or any(len(row) != codebooks for row in matrix):
                raise ValueError(f"{name} must have declared shape {self.shape}")

        for row in range(length):
            for codebook in range(codebooks):
                action = self.actions[row][codebook]
                logprob = self.policy_logprobs[row][codebook]
                sampled = self.action_mask[row][codebook]
                if not 0 <= action < HIGGS_CODEBOOK_VOCAB_SIZE:
                    raise ValueError("Higgs action is outside the codebook vocabulary")
                if sampled and not math.isfinite(logprob):
                    raise ValueError("sampled Higgs action has a non-finite policy logprob")
                if not sampled and logprob != 0.0:
                    raise ValueError("forced Higgs action policy logprob must be zero")
        return self

    def to_domain(self) -> DiscreteActionStream:
        return DiscreteActionStream(
            name=self.name,
            stage=self.stage,
            modality=self.modality,
            action_type=self.action_type,
            layout=self.layout,
            shape=list(self.shape),
            vocab_size=self.vocab_size,
            actions=[list(row) for row in self.actions],
            policy_logprobs=[list(row) for row in self.policy_logprobs],
            action_mask=[list(row) for row in self.action_mask],
            codec_content_mask=(
                [list(row) for row in self.codec_content_mask] if self.codec_content_mask is not None else None
            ),
            channel_ids=list(self.channel_ids),
        )


class HiggsRolloutTraceResponse(_StrictWireModel):
    version: Literal[2]
    model_family: Literal["higgs_tts"]
    total_action_count: StrictInt = Field(ge=1)
    action_streams: list[HiggsActionStreamResponse] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def validate_action_count(self) -> HiggsRolloutTraceResponse:
        count = sum(int(sampled) for stream in self.action_streams for row in stream.action_mask for sampled in row)
        if count != self.total_action_count:
            raise ValueError("total_action_count does not match the Higgs action mask")
        return self

    def to_domain(self) -> RolloutActionTrace:
        return RolloutActionTrace(
            version=self.version,
            model_family=self.model_family,
            total_action_count=self.total_action_count,
            action_streams=[stream.to_domain() for stream in self.action_streams],
        )


class _FinishReasonResponse(_StrictWireModel):
    type: Literal["stop", "length"]
    length: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_length(self) -> _FinishReasonResponse:
        if self.type == "length" and self.length is None:
            raise ValueError("length finish_reason requires its emitted row count")
        if self.type != "length" and self.length is not None:
            raise ValueError("only a length finish_reason may carry a length")
        return self


class _DecodedAudioResponse(_StrictWireModel):
    data: StrictStr = Field(min_length=1)
    path: None = None
    format: Literal["wav"]
    sample_rate: StrictInt = Field(gt=0)

    @field_validator("data")
    @classmethod
    def validate_nonblank_data(cls, data: str) -> str:
        if not data.strip():
            raise ValueError("decoded WAV data must not be blank")
        return data

    def to_domain(self) -> DecodedAudio:
        return DecodedAudio(data=self.data, format=self.format, sample_rate=self.sample_rate)


class _MetaInfoResponse(_StrictWireModel):
    finish_reason: _FinishReasonResponse
    prompt_tokens: StrictInt = Field(ge=0)
    completion_tokens: StrictInt = Field(gt=0)
    cached_tokens: StrictInt = Field(ge=0)
    weight_version: StrictStr = Field(min_length=1)
    request_metadata: None = None
    output_token_logprobs: list[list[StrictFloat | StrictInt]] | None = None
    output_codebook_tokens: list[list[StrictInt]] | None = None
    omni_rollout: HiggsRolloutTraceResponse

    @field_validator("weight_version")
    @classmethod
    def validate_nonblank_weight_version(cls, weight_version: str) -> str:
        if not weight_version.strip():
            raise ValueError("weight_version must not be blank")
        return weight_version

    @model_validator(mode="after")
    def validate_codebook_zero_diagnostics(self) -> _MetaInfoResponse:
        diagnostics = self.output_token_logprobs
        if diagnostics is None:
            return self
        stream = self.omni_rollout.action_streams[0]
        if len(diagnostics) != stream.shape[0]:
            raise ValueError("output_token_logprobs length does not match the Higgs action row count")
        for row, item in enumerate(diagnostics):
            if len(item) != 2:
                raise ValueError("output_token_logprobs entries must be [logprob, token_id]")
            logprob, token_id = item
            if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
                raise ValueError("output_token_logprobs contains a non-finite diagnostic logprob")
            if type(token_id) is not int or token_id != stream.actions[row][0]:
                raise ValueError("output_token_logprobs token does not match the codebook-0 action")
        return self


class HiggsGenerateResponse(_StrictWireModel):
    text: Literal[""]
    audio: _DecodedAudioResponse
    meta_info: _MetaInfoResponse

    @model_validator(mode="after")
    def validate_compatibility_fields(self) -> HiggsGenerateResponse:
        stream = self.meta_info.omni_rollout.action_streams[0]
        if self.meta_info.completion_tokens != stream.shape[0]:
            raise ValueError("completion_tokens does not match the Higgs action row count")
        compatibility_codes = self.meta_info.output_codebook_tokens
        if compatibility_codes is not None and compatibility_codes != stream.actions:
            raise ValueError("output_codebook_tokens does not match the structured Higgs actions")
        if self.meta_info.finish_reason.type == "length" and self.meta_info.finish_reason.length != stream.shape[0]:
            raise ValueError("finish_reason.length does not match the Higgs action row count")
        return self


@dataclass(frozen=True)
class HiggsRolloutResult:
    action_trace: RolloutActionTrace
    decoded_audio: DecodedAudio
    finish_type: Literal["stop", "length"]
    weight_version: str
    prompt_tokens: int
    cached_tokens: int


def parse_higgs_generate_response(
    response: Any,
    *,
    expected_prompt_tokens: int | None = None,
) -> HiggsRolloutResult:
    """Parse one complete Higgs rollout and return domain objects plus status data."""
    parsed = HiggsGenerateResponse.model_validate(response, strict=True)
    if expected_prompt_tokens is not None and parsed.meta_info.prompt_tokens != expected_prompt_tokens:
        raise ValueError("prompt_tokens does not match the exact prompt IDs sent to the inference server")
    return HiggsRolloutResult(
        action_trace=parsed.meta_info.omni_rollout.to_domain(),
        decoded_audio=parsed.audio.to_domain(),
        finish_type=parsed.meta_info.finish_reason.type,
        weight_version=parsed.meta_info.weight_version,
        prompt_tokens=parsed.meta_info.prompt_tokens,
        cached_tokens=parsed.meta_info.cached_tokens,
    )


__all__ = [
    "HIGGS_CODEBOOK_VOCAB_SIZE",
    "HIGGS_MODEL_FAMILY",
    "HIGGS_NUM_CODEBOOKS",
    "HIGGS_ROLLOUT_VERSION",
    "HIGGS_STREAM_NAME",
    "HIGGS_STREAM_STAGE",
    "HiggsGenerateResponse",
    "HiggsRolloutResult",
    "parse_higgs_generate_response",
]
