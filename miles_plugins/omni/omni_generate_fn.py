"""Single-turn Higgs audio rollout client for sglang-omni."""

from __future__ import annotations

from typing import Any

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.types import Sample

from .rollout_contract import parse_higgs_generate_response

_FILTER_DEFAULTS: dict[str, float] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
}
_PASSTHROUGH_SAMPLING_KEYS = frozenset({"max_new_tokens", "max_tokens", "seed", "sampling_seed"})
_MILES_PRESENTATION_KEYS = frozenset(
    {"skip_special_tokens", "no_stop_trim", "spaces_between_special_tokens", "stop", "stop_token_ids"}
)
_ZERO_SHOT_SPECIALS = ("<|tts|>", "<|text|>", "<|audio|>")


def neutral_higgs_sampling_params(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the initial unfiltered Higgs RL sampling profile."""
    params = dict(sampling_params)
    allowed = set(_FILTER_DEFAULTS) | {"top_k"} | _PASSTHROUGH_SAMPLING_KEYS | _MILES_PRESENTATION_KEYS
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unsupported Higgs rollout sampling parameters: {sorted(unknown)}")
    if params.get("stop") not in (None, "", []):
        raise ValueError("Higgs RL requires text stop strings to be disabled")
    if params.get("stop_token_ids") not in (None, []):
        raise ValueError("Higgs RL requires text stop token IDs to be disabled")

    normalized: dict[str, Any] = {}
    for name, expected in _FILTER_DEFAULTS.items():
        value = params.get(name, expected)
        if value is None:
            value = expected
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != expected:
            raise ValueError(f"Higgs RL requires {name}={expected}")
        normalized[name] = expected

    top_k = params.get("top_k")
    if isinstance(top_k, bool) or top_k not in (None, 0, -1):
        raise ValueError("Higgs RL requires top_k filtering to be disabled")

    if "seed" in params and "sampling_seed" in params:
        raise ValueError("set only one of seed and sampling_seed")
    if "max_new_tokens" in params and "max_tokens" in params:
        raise ValueError("set only one of max_new_tokens and max_tokens")
    for name in ("max_new_tokens", "max_tokens"):
        value = params.get(name)
        if value is not None:
            if type(value) is not int or value <= 0:
                raise ValueError(f"Higgs rollout {name} must be a positive integer")
            normalized[name] = value
    seed = params.get("seed", params.get("sampling_seed"))
    if seed is not None:
        if type(seed) is not int:
            raise ValueError("Higgs rollout seed must be an integer")
        normalized["seed"] = seed
    return normalized


def build_higgs_generate_payload(
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    """Build one non-streaming, audio-only structured rollout request."""
    if not prompt_ids or any(type(token_id) is not int for token_id in prompt_ids):
        raise ValueError("Higgs rollout prompt_ids must be a nonempty integer list")

    return {
        "input_ids": list(prompt_ids),
        "sampling_params": neutral_higgs_sampling_params(sampling_params),
        "stream": False,
        "output_modalities": ["audio"],
        "return_logprob": True,
        "return_omni_rollout": True,
    }


def build_zero_shot_higgs_prompt_ids(tokenizer: Any, prompt_text: str) -> list[int]:
    """Build the exact Higgs text-to-audio prompt expected by the server."""
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("Higgs zero-shot TTS requires nonempty prompt text")

    vocab = dict(tokenizer.get_added_vocab())
    missing = [token for token in _ZERO_SHOT_SPECIALS if token not in vocab]
    if missing:
        raise ValueError(f"tokenizer is missing Higgs TTS specials: {missing}")

    text_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    if any(type(token_id) is not int for token_id in text_ids):
        raise ValueError("Higgs tokenizer returned non-integer text token IDs")
    return [vocab["<|tts|>"], vocab["<|text|>"], *text_ids, vocab["<|audio|>"]]


class OmniGenerateFn:
    """Miles custom generate function for a fresh Higgs audio trajectory."""

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        sample = input.sample
        _validate_fresh_sample(sample)

        prompt_ids = _prompt_ids(input)
        sampling_params = dict(input.sampling_params)
        _set_generation_budget(input.args, sampling_params, len(prompt_ids))
        payload = build_higgs_generate_payload(prompt_ids, sampling_params)

        url = f"http://{input.args.sglang_router_ip}:{input.args.sglang_router_port}/generate"
        response = await post(url, payload)
        result = parse_higgs_generate_response(
            response,
            expected_prompt_tokens=len(prompt_ids),
        )

        # Audio codebooks are a separate action stream, never text response tokens.
        sample.tokens = list(prompt_ids)
        sample.action_trace = result.action_trace
        sample.decoded_audio = result.decoded_audio
        sample.weight_versions.append(result.weight_version)
        sample.status = Sample.Status.COMPLETED if result.finish_type == "stop" else Sample.Status.TRUNCATED
        sample.prefix_cache_info.cached_tokens += result.cached_tokens
        sample.prefix_cache_info.total_prompt_tokens += result.prompt_tokens
        return GenerateFnOutput(samples=sample)


def _validate_fresh_sample(sample: Sample) -> None:
    if sample.status not in {Sample.Status.PENDING, Sample.Status.ABORTED}:
        raise ValueError("Higgs structured rollouts require a pending sample or a clean retry")
    if sample.status is Sample.Status.ABORTED and sample.tokens:
        raise ValueError("Higgs structured rollouts cannot resume partial token state")
    if sample.response or sample.response_length != 0:
        raise ValueError("Higgs structured rollouts do not support partial text responses")
    if sample.loss_mask is not None or sample.rollout_log_probs is not None:
        raise ValueError("Higgs audio actions must not use text loss/logprob fields")
    if sample.action_trace is not None or sample.decoded_audio is not None or sample.weight_versions:
        raise ValueError("Higgs structured rollouts require a fresh sample")
    if sample.multimodal_inputs:
        raise ValueError("Higgs RL currently supports zero-shot text-to-audio only; reference media is not supported")


def _prompt_ids(input: GenerateFnInput) -> list[int]:
    sample = input.sample
    canonical_ids = build_zero_shot_higgs_prompt_ids(input.state.tokenizer, _prompt_text(sample))
    if sample.tokens:
        prompt_ids = list(sample.tokens)
        if prompt_ids != canonical_ids:
            raise ValueError("pretokenized Higgs prompt does not match the canonical zero-shot encoding")
    else:
        prompt_ids = canonical_ids
    if not prompt_ids or any(type(token_id) is not int for token_id in prompt_ids):
        raise ValueError("tokenization did not produce a nonempty integer prompt")
    return prompt_ids


def _prompt_text(sample: Sample) -> str:
    if isinstance(sample.prompt, str):
        return sample.prompt
    for message in reversed(sample.prompt):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError("Higgs zero-shot TTS requires a string prompt or a user text message")


def _set_generation_budget(args: Any, sampling_params: dict[str, Any], prompt_length: int) -> None:
    arg_values = vars(args)
    max_tokens = sampling_params.pop("max_tokens", None)
    max_new_tokens = sampling_params.get("max_new_tokens", max_tokens)
    if max_tokens is not None and "max_new_tokens" in sampling_params:
        raise ValueError("set only one of max_new_tokens and max_tokens")
    if max_new_tokens is None:
        max_new_tokens = arg_values.get("rollout_max_response_len")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("Higgs rollout max_new_tokens must be a positive integer")

    max_context = arg_values.get("rollout_max_context_len")
    if max_context is not None:
        if type(max_context) is not int or max_context <= prompt_length:
            raise ValueError("Higgs prompt leaves no context budget for an audio rollout")
        max_new_tokens = min(max_new_tokens, max_context - prompt_length)
    sampling_params["max_new_tokens"] = max_new_tokens


__all__ = [
    "OmniGenerateFn",
    "build_higgs_generate_payload",
    "build_zero_shot_higgs_prompt_ids",
    "neutral_higgs_sampling_params",
]
