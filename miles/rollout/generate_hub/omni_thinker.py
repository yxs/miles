"""Rollout adapter for the sglang-omni `/generate` endpoint (thinker / text).

Thin wrapper over the default single-turn generate: reuses miles' payload builder and
response parser, adds only the omni-specific fields. Wire via
`--custom-generate-function-path miles.rollout.generate_hub.omni_thinker.generate`.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"
    # omni /generate emits temp-1 (pre-temperature) full-vocab logprobs; the trainer recompute
    # divides logits by rollout_temperature, so they agree only at temp=1.
    assert args.rollout_temperature == 1.0, (
        f"omni rollout logprob is temp-1; rollout_temperature must be 1.0, got {args.rollout_temperature}"
    )
    # text-only path with no MoE/indexer replay yet (server forward-declares both)
    assert not (args.use_rollout_routing_replay or args.use_rollout_indexer_replay), (
        "omni rollout has no routing/indexer replay; unset --use-rollout-routing-replay / --use-rollout-indexer-replay"
    )
    assert not (sample.multimodal_inputs or {}).get("images"), "omni thinker rollout is text-only"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)
    if len(sample.response) > 0:  # partial rollout resume
        input_ids = sample.tokens
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)
        assert sampling_params["max_new_tokens"] >= 0
        if sampling_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    payload, halt_status = compute_request_payload(
        args, input_ids=input_ids, sampling_params=sampling_params, multimodal_inputs=sample.multimodal_inputs
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    payload["output_modalities"] = ["text"]
    payload["return_omni_rollout"] = False
    # rep_penalty=1: the trainer recompute can't replay a repetition penalty (logprobs would diverge)
    payload["sampling_params"]["repetition_penalty"] = 1.0
    if sample.metadata:
        payload["metadata"] = sample.metadata

    output = await post(url, payload)
    await update_sample_from_response(args, sample, payload=payload, output=output)
    return GenerateFnOutput(samples=sample)
