"""Typed request/response contract for the sglang-omni ``/generate`` rollout endpoint.

The omni backend exposes a stricter rollout schema than the stock sglang ``/generate``:
its sampling params reject unknown keys (``extra="forbid"``), so miles' default sampling
params (which carry keys such as ``skip_special_tokens`` or ``sampling_seed``) must be
whitelisted and aliased before they are sent. The response carries the generated tokens
and their behavior-policy log-probs inside ``meta_info.output_token_logprobs`` (one
``[log_prob, token_id]`` pair per generated token), optional decoded ``audio`` for TTS
rewards, and ``weight_version`` provenance.

All functions here are pure and side-effect free except :func:`apply_response_to_sample`,
which accumulates generated tokens onto a sample following the existing miles rollout
convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from miles.utils.types import Sample

# Sampling-param keys accepted by the omni rollout endpoint. Anything else is dropped so
# the request is not rejected by the backend's strict (extra-forbidding) schema.
OMNI_SAMPLING_PARAM_KEYS: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "stop",
        "stop_token_ids",
        "seed",
        "max_new_tokens",
        "max_tokens",
    }
)

# miles uses some legacy names that map onto the omni schema's canonical fields.
OMNI_SAMPLING_PARAM_ALIASES: dict[str, str] = {"sampling_seed": "seed"}


def clean_sampling_params(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """Project miles sampling params onto the keys the omni endpoint accepts.

    Unknown keys are dropped, known aliases are renamed, and ``None`` values are removed
    so optional fields fall back to backend defaults instead of failing validation.
    """
    cleaned: dict[str, Any] = {}
    for key, value in (sampling_params or {}).items():
        target = OMNI_SAMPLING_PARAM_ALIASES.get(key, key)
        if target in OMNI_SAMPLING_PARAM_KEYS and value is not None:
            cleaned[target] = value
    return cleaned


def build_generate_payload(
    input_ids: list[int],
    sampling_params: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    output_modalities: list[str] | None = None,
    return_logprob: bool = True,
    audio_data: list[str] | None = None,
) -> dict[str, Any]:
    """Build an omni ``/generate`` request body from pre-tokenized inputs.

    The trainer always sends ``input_ids`` (it computes gradients on these exact tokens),
    requests log-probs by default, and may echo ``metadata`` so responses can be matched
    back to a rollout batch.
    """
    payload: dict[str, Any] = {
        "input_ids": list(input_ids),
        "sampling_params": clean_sampling_params(sampling_params),
        "return_logprob": return_logprob,
    }
    if metadata:
        payload["metadata"] = metadata
    if output_modalities is not None:
        payload["output_modalities"] = output_modalities
    if audio_data is not None:
        payload["audio_data"] = audio_data
    return payload


@dataclass
class OmniRolloutResult:
    """Parsed view of an omni ``/generate`` response, ready to apply to a sample."""

    response_tokens: list[int]
    response_log_probs: list[float]
    text: str = ""
    finish_reason: dict[str, Any] = field(default_factory=dict)
    weight_version: str | None = None
    cached_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    audio: dict[str, Any] | None = None


def parse_generate_response(response: dict[str, Any]) -> OmniRolloutResult:
    """Parse an omni ``/generate`` response into :class:`OmniRolloutResult`.

    Raises ``ValueError`` (loudly, never silently truncating) when the response is
    malformed or when the per-token log-prob count disagrees with ``completion_tokens``.
    """
    if "meta_info" not in response:
        raise ValueError("omni /generate response is missing 'meta_info'")
    meta = response["meta_info"]

    token_logprobs = meta.get("output_token_logprobs") or []
    response_tokens: list[int] = []
    response_log_probs: list[float] = []
    for i, item in enumerate(token_logprobs):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(
                f"output_token_logprobs[{i}] is malformed: {item!r}; expected [log_prob, token_id]"
            )
        response_log_probs.append(float(item[0]))
        response_tokens.append(int(item[1]))

    completion_tokens = meta.get("completion_tokens")
    if completion_tokens is not None and len(response_tokens) != completion_tokens:
        raise ValueError(
            f"output_token_logprobs length ({len(response_tokens)}) "
            f"!= completion_tokens ({completion_tokens})"
        )

    if "finish_reason" not in meta:
        raise ValueError("omni /generate meta_info is missing 'finish_reason'")

    return OmniRolloutResult(
        response_tokens=response_tokens,
        response_log_probs=response_log_probs,
        text=response.get("text", "") or "",
        finish_reason=meta["finish_reason"],
        weight_version=meta.get("weight_version"),
        cached_tokens=int(meta.get("cached_tokens") or 0),
        prompt_tokens=int(meta.get("prompt_tokens") or 0),
        completion_tokens=int(completion_tokens if completion_tokens is not None else len(response_tokens)),
        audio=response.get("audio"),
    )


def apply_response_to_sample(
    sample: Sample,
    prompt_ids: list[int],
    result: OmniRolloutResult,
    *,
    update_loss_mask: bool = False,
) -> Sample:
    """Accumulate parsed generation onto ``sample`` (tokens, log-probs, loss mask, audio).

    Follows the miles convention where ``loss_mask`` and ``rollout_log_probs`` span only
    the generated (completion) tokens (length == ``response_length``); the prompt is
    excluded by lying outside the mask rather than by leading zeros. Standard meta_info
    handling (status, weight-version, prefix-cache stats) stays with the caller via the
    existing ``Sample.update_from_meta_info`` so this stays backend-agnostic and testable
    without trainer ``args``.
    """
    if not sample.tokens:
        sample.tokens = list(prompt_ids)

    sample.tokens = sample.tokens + result.response_tokens
    sample.response_length += len(result.response_tokens)
    sample.response += result.text

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += result.response_log_probs

    if update_loss_mask:
        if sample.loss_mask is None:
            sample.loss_mask = []
        sample.loss_mask += [1] * len(result.response_tokens)

    if result.audio is not None:
        if sample.multimodal_train_inputs is None:
            sample.multimodal_train_inputs = {}
        sample.multimodal_train_inputs["audio"] = result.audio

    return sample
