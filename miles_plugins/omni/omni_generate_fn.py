"""Per-sample generate function that drives rollout against the sglang-omni backend.

Load it via ``--custom-generate-function-path miles_plugins.omni.omni_generate_fn.OmniGenerateFn``.
It mirrors the stock single-turn generate path (including partial-rollout budget and
context-length halting) but speaks the omni ``/generate`` contract: a whitelisted
sampling-param payload, encoded input audio, request ``metadata`` for response matching,
and a response parser that captures generated tokens, behavior-policy log-probs, decoded
audio (for TTS rewards), and ``weight_version`` provenance.
"""

from __future__ import annotations

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import compute_prompt_ids_from_sample
from miles.utils.http_utils import post
from miles.utils.processing_utils import (
    encode_audios_for_rollout_engine,
    encode_image_for_rollout_engine,
    extract_audio_inputs,
)
from miles.utils.types import Sample

from .rollout_contract import apply_response_to_sample, build_generate_payload, parse_generate_response


class OmniGenerateFn:
    """Class-based generate function for omni (Thinker AR / TTS) rollout."""

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        args = input.args
        sample = input.sample
        sampling_params = dict(input.sampling_params)  # copied; max_new_tokens is adjusted below
        assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"

        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        prompt_ids = compute_prompt_ids_from_sample(input.state, sample)
        # Partial-rollout resume: continue from already-generated tokens and shrink the
        # remaining budget by what was already produced. Audio-only rollouts can have
        # empty decoded text, so response_length is the source of truth here.
        generated_token_count = _generated_token_count(sample, prompt_ids)
        if generated_token_count > 0:
            input_ids = sample.tokens
            total_budget = sampling_params.get("max_new_tokens", args.rollout_max_response_len)
            sampling_params["max_new_tokens"] = total_budget - generated_token_count
        else:
            input_ids = prompt_ids

        halt_status = _clamp_max_new_tokens(args, sampling_params, len(input_ids))
        if halt_status is not None:
            sample.status = halt_status
            return GenerateFnOutput(samples=sample)

        payload = build_generate_payload(
            input_ids,
            sampling_params,
            metadata=_request_metadata(sample),
            output_modalities=sample.metadata.get("output_modalities"),
            return_omni_rollout=True,
            images=_encode_input_images(sample),
            audios=_encode_input_audio(sample),
            videos=_encoded_input_videos(sample),
        )

        output = await post(url, payload)

        result = parse_generate_response(output)
        apply_response_to_sample(sample, prompt_ids, result)
        # Reuse the existing meta_info handling for status / weight_version / prefix-cache stats.
        sample.update_from_meta_info(args, output["meta_info"])

        return GenerateFnOutput(samples=sample)


def _generated_token_count(sample: Sample, prompt_ids: list[int]) -> int:
    """Return how many completion tokens have already been generated for resume."""
    if sample.response_length > 0:
        return sample.response_length
    return max(0, len(sample.tokens) - len(prompt_ids))


def _clamp_max_new_tokens(args, sampling_params: dict, prompt_len: int) -> Sample.Status | None:
    """Cap ``max_new_tokens`` by the context budget; return a halt status if none remains.

    Mirrors ``compute_request_payload`` so the omni path enforces the same limits as the
    stock generate path.
    """
    max_new_tokens = sampling_params.get("max_new_tokens")
    if max_new_tokens is None:
        max_new_tokens = args.rollout_max_response_len
    if context_len := getattr(args, "rollout_max_context_len", None):
        max_new_tokens = min(max_new_tokens, context_len - prompt_len)
    if max_new_tokens <= 0:
        return Sample.Status.TRUNCATED
    sampling_params["max_new_tokens"] = max_new_tokens
    return None


def _encode_input_audio(sample: Sample) -> list[str] | None:
    """Encode input-side audio from ``sample.multimodal_inputs`` for the request payload."""
    audios = extract_audio_inputs(sample.multimodal_inputs)
    if not audios:
        return None
    return encode_audios_for_rollout_engine(audios)


def _encode_input_images(sample: Sample) -> list[str] | None:
    """Reuse Miles' standard VLM serializer for the omni request contract."""
    images = (sample.multimodal_inputs or {}).get("images")
    if not images:
        return None
    return [encode_image_for_rollout_engine(image) for image in images]


def _encoded_input_videos(sample: Sample) -> list[str] | None:
    """Forward already transport-safe video references and reject other shapes."""
    videos = (sample.multimodal_inputs or {}).get("videos")
    if not videos:
        return None
    if not all(
        isinstance(video, str)
        and (video.startswith("data:video/") or video.startswith("https://") or video.startswith("http://"))
        for video in videos
    ):
        raise ValueError("Omni rollout video inputs must be data:video URIs or HTTP(S) URLs")
    return list(videos)


def _request_metadata(sample: Sample) -> dict:
    """Identifiers echoed back by the backend so responses can be matched to a rollout."""
    fields = {
        "group_index": sample.group_index,
        "index": sample.index,
        "session_id": sample.session_id,
    }
    return {k: v for k, v in fields.items() if v is not None}
