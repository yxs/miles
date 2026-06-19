"""Per-sample generate function that drives rollout against the sglang-omni backend.

Load it via ``--custom-generate-function-path miles_plugins.omni.omni_generate_fn.OmniGenerateFn``.
It mirrors the stock single-turn generate path but speaks the omni ``/generate`` contract:
a whitelisted sampling-param payload, request ``metadata`` for response matching, and a
response parser that captures generated tokens, behavior-policy log-probs, decoded audio
(for TTS rewards), and ``weight_version`` provenance.
"""

from __future__ import annotations

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import compute_prompt_ids_from_sample
from miles.utils.http_utils import post
from miles.utils.types import Sample

from .rollout_contract import apply_response_to_sample, build_generate_payload, parse_generate_response


class OmniGenerateFn:
    """Class-based generate function for omni (Thinker AR / TTS) rollout."""

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        args = input.args
        sample = input.sample
        sampling_params = input.sampling_params
        assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"

        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        prompt_ids = compute_prompt_ids_from_sample(input.state, sample)
        # Partial-rollout resume: continue from already-generated tokens.
        input_ids = sample.tokens if len(sample.response) > 0 else prompt_ids

        payload = build_generate_payload(
            input_ids,
            sampling_params,
            metadata=_request_metadata(sample),
            output_modalities=sample.metadata.get("output_modalities"),
        )

        output = await post(url, payload)

        result = parse_generate_response(output)
        apply_response_to_sample(sample, prompt_ids, result)
        # Reuse the existing meta_info handling for status / weight_version / prefix-cache stats.
        sample.update_from_meta_info(args, output["meta_info"])

        return GenerateFnOutput(samples=sample)


def _request_metadata(sample: Sample) -> dict:
    """Identifiers echoed back by the backend so responses can be matched to a rollout."""
    fields = {
        "group_index": sample.group_index,
        "index": sample.index,
        "session_id": sample.session_id,
    }
    return {k: v for k, v in fields.items() if v is not None}
