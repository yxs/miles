import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy
import torch


def _check_dict_keys(data: dict, *, required: set[str], optional: set[str], type_name: str) -> None:
    keys = set(data)
    missing = required - keys
    extra = keys - required - optional
    if missing or extra:
        raise ValueError(f"{type_name} fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}")


@dataclass
class DiscreteActionStream:
    """One aligned time-by-channel multi-discrete policy action stream."""

    name: str
    stage: str
    modality: str
    shape: list[int]
    vocab_size: int
    actions: list[list[int]]
    policy_logprobs: list[list[float]]
    action_mask: list[list[bool]]
    channel_ids: list[int]
    codec_content_mask: list[list[bool]] | None = None
    action_type: str = "multi_discrete"
    layout: str = "time_codebook"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for field_name, value in (
            ("name", self.name),
            ("stage", self.stage),
            ("modality", self.modality),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a nonempty string")
        if self.action_type != "multi_discrete":
            raise ValueError("action_type must be 'multi_discrete'")
        if self.layout != "time_codebook":
            raise ValueError("layout must be 'time_codebook'")
        if not isinstance(self.shape, list) or len(self.shape) != 2 or any(type(x) is not int for x in self.shape):
            raise ValueError("shape must be a two-element integer list [time, channels]")

        length, channels = self.shape
        if length < 0 or channels <= 0:
            raise ValueError("action stream dimensions must be non-negative")
        if type(self.vocab_size) is not int or self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if (
            not isinstance(self.channel_ids, list)
            or any(type(channel_id) is not int for channel_id in self.channel_ids)
            or self.channel_ids != list(range(channels))
        ):
            raise ValueError("channel_ids must be the ordered channel indices")

        matrices = {
            "actions": self.actions,
            "policy_logprobs": self.policy_logprobs,
            "action_mask": self.action_mask,
        }
        if self.codec_content_mask is not None:
            matrices["codec_content_mask"] = self.codec_content_mask
        for matrix_name, matrix in matrices.items():
            if not isinstance(matrix, list) or len(matrix) != length:
                raise ValueError(f"{matrix_name} must have declared shape {self.shape}")
            if any(not isinstance(row, list) or len(row) != channels for row in matrix):
                raise ValueError(f"{matrix_name} must have declared shape {self.shape}")

        for row in range(length):
            for channel in range(channels):
                action = self.actions[row][channel]
                logprob = self.policy_logprobs[row][channel]
                sampled = self.action_mask[row][channel]
                if type(action) is not int or not 0 <= action < self.vocab_size:
                    raise ValueError("action is outside the declared vocabulary")
                if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
                    raise ValueError("policy_logprobs must contain numeric values")
                if type(sampled) is not bool:
                    raise ValueError("action_mask must contain boolean values")
                if sampled and not math.isfinite(logprob):
                    raise ValueError("sampled action has a non-finite policy logprob")
                if not sampled and logprob != 0.0:
                    raise ValueError("forced action policy logprob must be zero")
                if self.codec_content_mask is not None and type(self.codec_content_mask[row][channel]) is not bool:
                    raise ValueError("codec_content_mask must contain boolean values")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "name": self.name,
            "stage": self.stage,
            "modality": self.modality,
            "action_type": self.action_type,
            "layout": self.layout,
            "shape": list(self.shape),
            "vocab_size": self.vocab_size,
            "actions": [list(row) for row in self.actions],
            "policy_logprobs": [list(row) for row in self.policy_logprobs],
            "action_mask": [list(row) for row in self.action_mask],
            "codec_content_mask": (
                [list(row) for row in self.codec_content_mask] if self.codec_content_mask is not None else None
            ),
            "channel_ids": list(self.channel_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscreteActionStream":
        if not isinstance(data, dict):
            raise ValueError("DiscreteActionStream must be constructed from a dictionary")
        _check_dict_keys(
            data,
            required={
                "name",
                "stage",
                "modality",
                "action_type",
                "layout",
                "shape",
                "vocab_size",
                "actions",
                "policy_logprobs",
                "action_mask",
                "channel_ids",
            },
            optional={"codec_content_mask"},
            type_name="DiscreteActionStream",
        )
        return cls(
            name=data["name"],
            stage=data["stage"],
            modality=data["modality"],
            action_type=data["action_type"],
            layout=data["layout"],
            shape=data["shape"],
            vocab_size=data["vocab_size"],
            actions=data["actions"],
            policy_logprobs=data["policy_logprobs"],
            action_mask=data["action_mask"],
            codec_content_mask=data.get("codec_content_mask"),
            channel_ids=data["channel_ids"],
        )


@dataclass
class RolloutActionTrace:
    """Versioned collection of structured policy action streams."""

    version: int
    model_family: str
    total_action_count: int
    action_streams: list[DiscreteActionStream]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.version) is not int or self.version != 2:
            raise ValueError("only rollout action trace version 2 is supported")
        if not isinstance(self.model_family, str) or not self.model_family:
            raise ValueError("model_family must be a nonempty string")
        if type(self.total_action_count) is not int or self.total_action_count < 0:
            raise ValueError("total_action_count must be a non-negative integer")
        if not isinstance(self.action_streams, list) or not self.action_streams:
            raise ValueError("action_streams must contain at least one stream")
        if any(not isinstance(stream, DiscreteActionStream) for stream in self.action_streams):
            raise ValueError("action_streams must contain DiscreteActionStream values")
        for stream in self.action_streams:
            stream.validate()
        actual_count = sum(
            int(sampled) for stream in self.action_streams for row in stream.action_mask for sampled in row
        )
        if actual_count != self.total_action_count:
            raise ValueError("total_action_count does not match action masks")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "model_family": self.model_family,
            "total_action_count": self.total_action_count,
            "action_streams": [stream.to_dict() for stream in self.action_streams],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutActionTrace":
        if not isinstance(data, dict):
            raise ValueError("RolloutActionTrace must be constructed from a dictionary")
        _check_dict_keys(
            data,
            required={"version", "model_family", "total_action_count", "action_streams"},
            optional=set(),
            type_name="RolloutActionTrace",
        )
        if not isinstance(data["action_streams"], list):
            raise ValueError("action_streams must be a list")
        return cls(
            version=data["version"],
            model_family=data["model_family"],
            total_action_count=data["total_action_count"],
            action_streams=[DiscreteActionStream.from_dict(stream) for stream in data["action_streams"]],
        )


@dataclass
class DecodedAudio:
    """Decoded waveform returned for reward evaluation, not policy training."""

    data: str
    format: str
    sample_rate: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.data, str) or not self.data:
            raise ValueError("decoded audio data must be a nonempty string")
        if self.format != "wav":
            raise ValueError("decoded audio format must be 'wav'")
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError("decoded audio sample_rate must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"data": self.data, "format": self.format, "sample_rate": self.sample_rate}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecodedAudio":
        if not isinstance(data, dict):
            raise ValueError("DecodedAudio must be constructed from a dictionary")
        _check_dict_keys(
            data,
            required={"data", "format", "sample_rate"},
            optional=set(),
            type_name="DecodedAudio",
        )
        return cls(data=data["data"], format=data["format"], sample_rate=data["sample_rate"])


@dataclass
class Sample:
    """The sample generated"""

    group_index: int | None = None
    index: int | None = None
    # prompt
    prompt: str | list[dict[str, str]] = ""
    tokens: list[int] = field(default_factory=list)
    multimodal_inputs: dict[str, Any] = None  # raw multimodal data, e.g. images, videos, etc.
    multimodal_train_inputs: dict[str, Any] = None  # processed multimodal data, e.g. pixel_values, etc.
    # response
    response: str = ""
    response_length: int = 0
    label: str | None = None
    reward: float | dict[str, Any] | None = None
    loss_mask: list[int] | None = None
    weight_versions: list[str] = field(default_factory=list)
    rollout_log_probs: list[float] | None = None  # Log probabilities from rollout engine
    rollout_routed_experts: numpy.ndarray | None = (
        None  # Routed experts from rollout engine. shape: (num_tokens-1, num_layers, moe_router_topk), dtype=int32
    )
    rollout_indexer_topk: numpy.ndarray | None = (
        None  # Indexer topk from rollout engine. shape: (num_tokens-1, num_indexer_layers, index_topk), dtype=int32
    )
    remove_sample: bool = False
    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD
    opd_reverse_kl: list[float] | None = None  # Precomputed per-token OPD reverse-KL estimate

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        # Indicates a recoverable or non-critical failure during generation (e.g., tool call failure,
        # external API error, parsing error). Unlike ABORTED, FAILED samples may still contain partial
        # valid output and can be retried or handled gracefully.
        FAILED = "failed"

    status: Status = Status.PENDING

    metadata: dict = field(default_factory=dict)
    generate_function_path: str | None = None
    # metadata used during training, e.g., what loss to use for this sample.
    train_metadata: dict | None = None

    # Session ID for consistent hashing routing (used when router policy is consistent_hashing)
    # TODO: Its definition needs to merge with the session server's session id in the new rollout function.
    session_id: str | None = None

    non_generation_time: float = 0.0  # time spent in non-generation steps

    @dataclass
    class SpecInfo:
        spec_accept_token_num: int = 0
        spec_draft_token_num: int = 0
        spec_verify_ct: int = 0
        completion_token_num: int = 0

        @property
        def spec_accept_rate(self) -> float:
            return self.spec_accept_token_num / self.spec_draft_token_num if self.spec_draft_token_num > 0 else 0.0

        @property
        def spec_accept_length(self) -> float:
            return self.completion_token_num / self.spec_verify_ct if self.spec_verify_ct > 0 else 0.0

        def add(self, meta_info: dict):
            self.spec_accept_token_num += meta_info.get("spec_accept_token_num", 0)
            self.spec_draft_token_num += meta_info.get("spec_draft_token_num", 0)
            self.spec_verify_ct += meta_info.get("spec_verify_ct", 0)
            self.completion_token_num += meta_info.get("completion_tokens", 0)

        def to_dict(self):
            return {
                "spec_accept_token_num": self.spec_accept_token_num,
                "spec_draft_token_num": self.spec_draft_token_num,
                "spec_verify_ct": self.spec_verify_ct,
                "completion_token_num": self.completion_token_num,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.SpecInfo()
            info.spec_accept_token_num = data.get("spec_accept_token_num", 0)
            info.spec_draft_token_num = data.get("spec_draft_token_num", 0)
            info.spec_verify_ct = data.get("spec_verify_ct", 0)
            info.completion_token_num = data.get("completion_token_num", 0)
            return info

    spec_info: SpecInfo = field(default_factory=SpecInfo)

    @dataclass
    class PrefixCacheInfo:
        cached_tokens: int = 0
        total_prompt_tokens: int = 0

        @property
        def prefix_cache_hit_rate(self) -> float:
            return self.cached_tokens / self.total_prompt_tokens if self.total_prompt_tokens > 0 else 0.0

        def add(self, meta_info: dict):
            self.cached_tokens += meta_info.get("cached_tokens", 0)
            # new_tokens = input_tokens - cached_tokens
            self.total_prompt_tokens += meta_info.get("prompt_tokens", 0)

        def to_dict(self):
            return {
                "cached_tokens": self.cached_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
            }

        @staticmethod
        def from_dict(data: dict):
            info = Sample.PrefixCacheInfo()
            info.cached_tokens = data.get("cached_tokens", 0)
            info.total_prompt_tokens = data.get("total_prompt_tokens", 0)
            return info

    prefix_cache_info: PrefixCacheInfo = field(default_factory=PrefixCacheInfo)

    # Structured policy outputs are independent of the legacy text-token fields.
    action_trace: RolloutActionTrace | None = None
    decoded_audio: DecodedAudio | None = None

    def to_dict(self):
        value = self.__dict__.copy()
        value["status"] = self.status.value
        value["spec_info"] = self.spec_info.to_dict()
        value["prefix_cache_info"] = self.prefix_cache_info.to_dict()
        value["action_trace"] = self.action_trace.to_dict() if self.action_trace is not None else None
        value["decoded_audio"] = self.decoded_audio.to_dict() if self.decoded_audio is not None else None
        return value

    @staticmethod
    def from_dict(data: dict):
        data = dict(data)
        data["status"] = Sample.Status(data["status"])
        data["spec_info"] = Sample.SpecInfo.from_dict(data.get("spec_info", {}))
        data["prefix_cache_info"] = Sample.PrefixCacheInfo.from_dict(data.get("prefix_cache_info", {}))
        if data.get("action_trace") is not None:
            data["action_trace"] = RolloutActionTrace.from_dict(data["action_trace"])
        if data.get("decoded_audio") is not None:
            data["decoded_audio"] = DecodedAudio.from_dict(data["decoded_audio"])

        field_names = set(Sample.__dataclass_fields__.keys())
        init_data = {k: v for k, v in data.items() if k in field_names}
        sample = Sample(**init_data)

        for key, value in data.items():
            if key not in field_names:
                setattr(sample, key, value)

        return sample

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]

    @property
    def effective_response_length(self):
        return sum(self.loss_mask) if self.loss_mask is not None else self.response_length

    def validate(self):
        assert self.response_length >= 0, f"response_length must be >= 0, got {self.response_length}"
        assert (
            len(self.tokens) >= self.response_length
        ), f"tokens length ({len(self.tokens)}) must be >= response_length ({self.response_length})"
        if self.loss_mask is not None:
            assert (
                len(self.loss_mask) == self.response_length
            ), f"loss_mask length ({len(self.loss_mask)}) != response_length ({self.response_length})"
        if self.rollout_log_probs is not None:
            assert (
                len(self.rollout_log_probs) == self.response_length
            ), f"rollout_log_probs length ({len(self.rollout_log_probs)}) != response_length ({self.response_length})"
        if self.teacher_log_probs is not None:
            assert (
                len(self.teacher_log_probs) == self.response_length
            ), f"teacher_log_probs length ({len(self.teacher_log_probs)}) != response_length ({self.response_length})"
        if self.opd_reverse_kl is not None:
            assert (
                len(self.opd_reverse_kl) == self.response_length
            ), f"opd_reverse_kl length ({len(self.opd_reverse_kl)}) != response_length ({self.response_length})"
        if self.action_trace is not None:
            self.action_trace.validate()
        if self.decoded_audio is not None:
            self.decoded_audio.validate()
        if self.rollout_routed_experts is not None:
            actual = len(self.rollout_routed_experts)
            expect = len(self.tokens) - 1
            assert actual == expect, f"rollout_routed_experts length ({actual}) != len(tokens) - 1 ({expect})"
        if self.rollout_indexer_topk is not None:
            actual = len(self.rollout_indexer_topk)
            expect = len(self.tokens) - 1
            assert actual == expect, f"rollout_indexer_topk length ({actual}) != len(tokens) - 1 ({expect})"

    def strip_last_output_tokens(self, n: int, tokenizer) -> None:
        """Remove the last *n* output tokens and all associated per-token info."""
        if n <= 0:
            return
        assert (
            n <= self.response_length
        ), f"cannot strip {n} tokens: only {self.response_length} output tokens available"
        self.tokens = self.tokens[:-n]
        self.response_length -= n
        if self.rollout_log_probs is not None:
            self.rollout_log_probs = self.rollout_log_probs[:-n]
        if self.teacher_log_probs is not None:
            self.teacher_log_probs = self.teacher_log_probs[:-n]
        if self.opd_reverse_kl is not None:
            self.opd_reverse_kl = self.opd_reverse_kl[:-n]
        if self.metadata and "opd_student_top_logprobs" in self.metadata:
            self.metadata["opd_student_top_logprobs"] = self.metadata["opd_student_top_logprobs"][:-n]
        if self.loss_mask is not None:
            self.loss_mask = self.loss_mask[:-n]
        self.response = tokenizer.decode(self.tokens[-self.response_length :]) if self.response_length > 0 else ""
        if self.rollout_routed_experts is not None:
            self.rollout_routed_experts = self.rollout_routed_experts[:-n]
        if self.rollout_indexer_topk is not None:
            self.rollout_indexer_topk = self.rollout_indexer_topk[:-n]

    def reset_for_retry(self) -> None:
        """Reset generated outputs so the original prompt can be re-sampled.

        Keeps identity / prompt fields (group_index, index, prompt, label,
        multimodal_inputs, metadata, generate_function_path, session_id) and
        restores everything else to dataclass defaults.
        """
        self.tokens = []
        self.multimodal_train_inputs = None
        self.response = ""
        self.response_length = 0
        self.reward = None
        self.loss_mask = None
        self.weight_versions = []
        self.rollout_log_probs = None
        self.action_trace = None
        self.decoded_audio = None
        self.rollout_routed_experts = None
        self.rollout_indexer_topk = None
        self.status = Sample.Status.ABORTED
        self.non_generation_time = 0.0
        self.spec_info = Sample.SpecInfo()
        self.prefix_cache_info = Sample.PrefixCacheInfo()
        self.remove_sample = False
        self.train_metadata = None

    @property
    def oldest_weight_version(self) -> int | None:
        """Minimum weight version across all turns (generation calls) for this trajectory."""
        numeric = [int(v) for v in self.weight_versions if str(v).isdigit()]
        return min(numeric) if numeric else None

    def update_from_meta_info(self, args, meta_info: dict):
        """
        Update the sample with new information from meta_info returned by the rollout engine.
        And extract
        """
        if args.sglang_speculative_algorithm:
            # cannot directly use spec info from sglang because of partial rollout.
            self.spec_info.add(meta_info=meta_info)

        # Collect prefix cache statistics
        self.prefix_cache_info.add(meta_info=meta_info)

        if "weight_version" in meta_info:
            self.weight_versions.append(meta_info["weight_version"])

        match meta_info["finish_reason"]["type"]:
            case "length":
                self.status = Sample.Status.TRUNCATED
            case "abort":
                self.status = Sample.Status.ABORTED
            case "stop":
                self.status = Sample.Status.COMPLETED


@dataclass(frozen=True)
class ParamInfo:
    name: str
    dtype: torch.dtype
    shape: torch.Size
    attrs: dict
    size: int
    src_rank: int


# A dict-based batch produced along the rollout -> training path
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators (see megatron_utils.actor._get_rollout_data).
RolloutBatch = dict[
    str,
    list[torch.Tensor] | list[int] | list[float] | list[str] | list[RolloutActionTrace],
]


@dataclass
class MultimodalType:
    name: str  # Type identifier used in message content (e.g., "image")
    placeholder: str  # Placeholder token in conversation messages (e.g., "<image>")


class MultimodalTypes:
    IMAGE = MultimodalType(name="image", placeholder="<image>")
    VIDEO = MultimodalType(name="video", placeholder="<video>")
    AUDIO = MultimodalType(name="audio", placeholder="<audio>")

    @classmethod
    def all(cls) -> list[MultimodalType]:
        return [cls.IMAGE, cls.VIDEO, cls.AUDIO]

    @classmethod
    def get(cls, name: str) -> MultimodalType | None:
        return next((m for m in cls.all() if m.name == name), None)
