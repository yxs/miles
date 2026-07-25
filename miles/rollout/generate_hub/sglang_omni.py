"""Single-turn rollout using SGLang Omni's processed multimodal contract.

Select with ``--custom-generate-function-path
miles.rollout.generate_hub.sglang_omni.generate``.

Multimodal samples ship the processor-expanded ``input_ids`` plus the serialized
processor tensors (``multimodal_train_inputs``); the server trusts those ids, feeds the
tensors straight into its audio/vision towers, and never re-runs media processing, so
rollout and training share one canonical token sequence. Logprobs come from the vanilla
sglang sampler (post-temperature unless the server sets SGLANG_RETURN_ORIGINAL_LOGPROB),
matching the trainer recompute convention.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput

# RolloutSamplingParams on the omni server is extra="forbid"; this mirrors its schema
_OMNI_SAMPLING_KEYS = frozenset(
    ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "stop", "stop_token_ids", "seed", "max_new_tokens", "max_tokens")
)
# detok/text-shaping flags: they never reach the token or logprob streams the trainer consumes
_DETOK_ONLY_KEYS = frozenset(("skip_special_tokens", "no_stop_trim", "spaces_between_special_tokens"))
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params
    assert sample.status in {
        Sample.Status.PENDING,
        Sample.Status.ABORTED,
    }, f"{sample.status=}"
    # the omni server declares return_routed_experts/return_indexer_topk in its protocol but
    # implements neither replay; fail loud instead of training on silently missing traces
    assert not (
        args.use_rollout_routing_replay or args.use_rollout_indexer_replay
    ), "sglang-omni rollout has no routing/indexer replay; unset --use-rollout-routing-replay / --use-rollout-indexer-replay"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)
    has_multimodal_inputs = sample.multimodal_inputs and any(
        value is not None for value in sample.multimodal_inputs.values()
    )
    if has_multimodal_inputs and sample.multimodal_train_inputs is None:
        raise ValueError(
            "SGLang Omni multimodal rollout requires processor-produced " "sample.multimodal_train_inputs"
        )

    if sample.response:
        input_ids = sample.tokens
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)
        assert sampling_params["max_new_tokens"] >= 0
        if sampling_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    payload, halt_status = compute_request_payload(
        args,
        input_ids=input_ids,
        sampling_params=sampling_params,
        multimodal_inputs=sample.multimodal_inputs,
        multimodal_train_inputs=sample.multimodal_train_inputs,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    payload["output_modalities"] = ["text"]
    payload["return_omni_rollout"] = False
    # the trainer recompute cannot replay a repetition penalty (logprobs would diverge)
    payload["sampling_params"]["repetition_penalty"] = 1.0
    unknown_keys = set(payload["sampling_params"]) - _OMNI_SAMPLING_KEYS - _DETOK_ONLY_KEYS
    assert not unknown_keys, f"sampling params outside the omni /generate schema: {sorted(unknown_keys)}"
    payload["sampling_params"] = {k: v for k, v in payload["sampling_params"].items() if k in _OMNI_SAMPLING_KEYS}
    if sample.metadata:
        payload["metadata"] = sample.metadata

    output = await post(url, payload, headers=compute_routing_headers(args, sample))
    await update_sample_from_response(
        args,
        sample,
        payload=payload,
        output=output,
    )
    return GenerateFnOutput(samples=sample)
