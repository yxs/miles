"""Structured Higgs policy batching and joint-row GRPO math.

This module deliberately depends only on PyTorch.  The tensor contract is
shared by the Megatron integration and CPU unit tests, while checkpoint
conversion remains a separate concern.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

HIGGS_MODEL_FAMILY = "higgs_tts"
HIGGS_STREAM_NAME = "higgs_codes"


@dataclass(frozen=True)
class HiggsPolicyBatch:
    """One padded, unpacked BSHD batch for Higgs teacher forcing."""

    input_ids: torch.Tensor
    prior_codes: torch.Tensor
    codec_position_mask: torch.Tensor
    sequence_mask: torch.Tensor
    prediction_positions: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    old_cell_logprobs: torch.Tensor
    old_joint_logprobs: torch.Tensor
    row_mask: torch.Tensor
    advantages: torch.Tensor | None
    prompt_lengths: torch.Tensor
    action_lengths: torch.Tensor
    num_codebooks: int
    codebook_vocab_size: int


def is_higgs_policy_enabled(args: Any) -> bool:
    return getattr(args, "structured_policy_model_family", None) == HIGGS_MODEL_FAMILY


def validate_higgs_single_device_config(args: Any, *, data_parallel_size: int | None = None) -> None:
    """Reject configurations outside the first, deliberately naive backend."""

    if not is_higgs_policy_enabled(args):
        return

    required_values = {
        "train_backend": "megatron",
        "qkv_format": "bshd",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 1,
        "advantage_estimator": "grpo",
        "bf16": True,
        "fp16": False,
        "true_on_policy_mode": False,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "vocab_size": 151936,
        "padded_vocab_size": 151936,
        "group_query_attention": True,
        "num_query_groups": 8,
        "kv_channels": 128,
        "qk_layernorm": True,
        "swiglu": True,
        "add_bias_linear": False,
        "untie_embeddings_and_output_weights": False,
        "normalization": "RMSNorm",
        "use_rotary_position_embeddings": True,
        "rotary_percent": 1.0,
        "rotary_interleaved": False,
        "use_rope_scaling": False,
    }
    errors = []
    for name, expected in required_values.items():
        actual = getattr(args, name, None)
        if actual != expected:
            errors.append(f"{name}={actual!r} (expected {expected!r})")
    # Megatron resolves an unspecified expert TP size to the ordinary TP size
    # during its own validation, which is already fixed to one above.
    if getattr(args, "expert_tensor_parallel_size", None) not in (None, 1):
        errors.append(f"expert_tensor_parallel_size={args.expert_tensor_parallel_size!r} (expected None or 1)")
    if getattr(args, "num_experts", None) not in (None, 0):
        errors.append("num_experts must be unset for the dense Higgs Qwen3 backbone")
    if getattr(args, "mtp_num_layers", None) not in (None, 0):
        errors.append("mtp_num_layers must be unset for the initial Higgs policy")
    if getattr(args, "spec", None) is not None:
        errors.append("custom Megatron layer specs are not supported by the initial Higgs policy")

    false_flags = (
        "sequence_parallel",
        "use_dynamic_batch_size",
        "use_dynamic_global_batch_size",
        "allgather_cp",
        "enable_mtp_training",
        "use_critic",
        "use_tis",
        "use_opsm",
        "use_opd",
        "calculate_per_token_loss",
        "get_mismatch_metrics",
        "observe_training_entropy",
        "use_rollout_entropy",
        "use_rollout_routing_replay",
        "use_rollout_indexer_replay",
        "keep_old_actor",
        "debug_disable_optimizer",
    )
    for name in false_flags:
        if bool(getattr(args, name, False)):
            errors.append(f"{name}=True (expected False)")

    if getattr(args, "lora_rank", 0) not in (None, 0):
        errors.append("lora_rank must be 0")
    if not isinstance(getattr(args, "hf_checkpoint", None), str) or not args.hf_checkpoint:
        errors.append("hf_checkpoint must identify the concrete Higgs v3 TTS checkpoint")
    if not bool(getattr(args, "use_rollout_logprobs", False)):
        errors.append("use_rollout_logprobs must be enabled")
    if not bool(getattr(args, "compute_advantages_and_returns", False)):
        errors.append("compute_advantages_and_returns must be enabled")
    if not bool(getattr(args, "rewards_normalization", False)):
        errors.append("rewards_normalization must be enabled for grouped GRPO")
    n_samples_per_prompt = getattr(args, "n_samples_per_prompt", None)
    if (
        isinstance(n_samples_per_prompt, bool)
        or not isinstance(n_samples_per_prompt, int)
        or n_samples_per_prompt <= 1
    ):
        errors.append("n_samples_per_prompt must be greater than 1 for a nonzero grouped GRPO signal")
    kl_coef = getattr(args, "kl_coef", 0.0)
    if kl_coef is None:
        kl_coef = 0.0
    entropy_coef = getattr(args, "entropy_coef", 0.0)
    if entropy_coef is None:
        entropy_coef = 0.0
    if kl_coef != 0.0 or bool(getattr(args, "use_kl_loss", False)):
        errors.append("reference-policy KL is not implemented for the initial Higgs path")
    if entropy_coef != 0.0:
        errors.append("entropy_coef must be 0 for the initial Higgs path")
    if bool(getattr(args, "normalize_advantages", False)):
        errors.append("normalize_advantages must be disabled; GRPO rewards are normalized before batching")
    if getattr(args, "megatron_to_hf_mode", None) == "bridge":
        errors.append("megatron_to_hf_mode='bridge' has no verified Higgs checkpoint mapping")
    if getattr(args, "save_hf", None) is not None:
        errors.append(
            "save_hf is disabled because standalone Higgs HF snapshot export is not implemented; "
            "online raw weight conversion is supported"
        )
    parity_atol = getattr(args, "higgs_logprob_parity_atol", None)
    if (
        parity_atol is None
        or isinstance(parity_atol, bool)
        or not isinstance(parity_atol, (int, float))
        or parity_atol <= 0
    ):
        errors.append("higgs_logprob_parity_atol must be configured to a positive measured tolerance")
    if data_parallel_size is not None and data_parallel_size != 1:
        errors.append(f"data_parallel_size={data_parallel_size!r} (expected 1)")

    if errors:
        raise ValueError("Higgs structured policy requires the single-device Megatron profile: " + "; ".join(errors))


def _config_dict(config: Any, name: str) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    try:
        return vars(config)
    except TypeError as error:
        raise ValueError(f"Higgs {name} must be a configuration mapping or object") from error


def validate_higgs_hf_config(
    config: Any,
    *,
    num_codebooks: int,
    codebook_vocab_size: int,
) -> None:
    """Validate the concrete v3 TTS checkpoint architecture this adapter mirrors."""

    root = _config_dict(config, "root config")
    audio = _config_dict(root.get("audio_encoder_config"), "audio_encoder_config")
    text = _config_dict(root.get("text_config"), "text_config")
    expected = {
        "model_type": (root.get("model_type"), "higgs_multimodal_qwen3"),
        "audio.encoder_type": (audio.get("encoder_type"), "discrete"),
        "audio.num_codebooks": (audio.get("num_codebooks"), num_codebooks),
        "audio.vocab_size": (audio.get("vocab_size"), codebook_vocab_size),
        "audio.out_dim": (audio.get("out_dim"), 2560),
        "audio.tie_word_embeddings": (audio.get("tie_word_embeddings"), True),
        "audio.use_delay_pattern": (audio.get("use_delay_pattern"), True),
        "text.model_type": (text.get("model_type"), "qwen3"),
        "text.hidden_size": (text.get("hidden_size"), 2560),
        "text.num_hidden_layers": (text.get("num_hidden_layers"), 36),
        "text.num_attention_heads": (text.get("num_attention_heads"), 32),
        "text.num_key_value_heads": (text.get("num_key_value_heads"), 8),
        "text.head_dim": (text.get("head_dim"), 128),
        "text.intermediate_size": (text.get("intermediate_size"), 9728),
        "text.rms_norm_eps": (text.get("rms_norm_eps"), 1e-6),
        "text.vocab_size": (text.get("vocab_size"), 151936),
        "text.max_position_embeddings": (text.get("max_position_embeddings"), 32768),
        "text.hidden_act": (text.get("hidden_act"), "silu"),
        "text.tie_word_embeddings": (text.get("tie_word_embeddings"), True),
        "text.attention_dropout": (text.get("attention_dropout"), 0.0),
    }
    errors = [
        f"{name}={actual!r} (expected {wanted!r})" for name, (actual, wanted) in expected.items() if actual != wanted
    ]
    architectures = root.get("architectures")
    if architectures != ["HiggsMultimodalQwen3ForConditionalGeneration"]:
        errors.append("architectures must be ['HiggsMultimodalQwen3ForConditionalGeneration']")
    if audio.get("out_dim") != text.get("hidden_size"):
        errors.append("audio.out_dim must equal text.hidden_size")
    rope_parameters = text.get("rope_parameters")
    if not isinstance(rope_parameters, Mapping) or rope_parameters.get("rope_theta") != 1_000_000:
        errors.append("text.rope_parameters.rope_theta must be 1000000")
    dtype = text.get("dtype")
    if dtype not in ("bfloat16", torch.bfloat16):
        errors.append(f"text.dtype={dtype!r} (expected bfloat16)")
    if errors:
        raise ValueError("unsupported Higgs checkpoint configuration: " + "; ".join(errors))


def _as_prompt_tensor(prompt: Any, *, device: torch.device | str | None) -> torch.Tensor:
    prompt_tensor = torch.as_tensor(prompt, dtype=torch.long, device=device)
    if prompt_tensor.ndim != 1 or prompt_tensor.numel() == 0:
        raise ValueError("each Higgs prompt must be a nonempty one-dimensional token sequence")
    if bool((prompt_tensor < 0).any()):
        raise ValueError("Higgs prompt token IDs must be non-negative")
    return prompt_tensor


def _higgs_stream(trace: Any) -> Any:
    validate = getattr(trace, "validate", None)
    if not callable(validate):
        raise ValueError("Higgs action traces must provide strict validation")
    validate()
    if trace.model_family != HIGGS_MODEL_FAMILY:
        raise ValueError(f"expected model_family={HIGGS_MODEL_FAMILY!r}, got {trace.model_family!r}")
    if len(trace.action_streams) != 1:
        raise ValueError("the initial Higgs policy path requires exactly one action stream")
    stream = trace.action_streams[0]
    if stream.name != HIGGS_STREAM_NAME:
        raise ValueError(f"expected action stream {HIGGS_STREAM_NAME!r}, got {stream.name!r}")
    if stream.action_type != "multi_discrete" or stream.layout != "time_codebook":
        raise ValueError("Higgs actions must use multi_discrete/time_codebook layout")
    return stream


def collate_higgs_policy_batch(
    prompts: Sequence[Any],
    action_traces: Sequence[Any],
    *,
    advantages: Sequence[Any] | None = None,
    old_policy_joint_logprobs: Sequence[Any] | None = None,
    pad_token_id: int = 0,
    device: torch.device | str | None = None,
) -> HiggsPolicyBatch:
    """Collate prompt IDs and complete prior codebook rows without packing.

    For action row ``t``, the model reads the prompt plus complete rows
    ``[0, t)``.  Consequently row zero is predicted from the last prompt
    position and row ``t > 0`` from the position containing row ``t - 1``.
    Forced BOC/EOC cells remain in ``prior_codes`` even though they are masked
    out of the policy loss. Server per-cell logprobs remain in
    ``old_cell_logprobs`` for diagnostics; when supplied, the pre-update
    Megatron joint logprobs are the GRPO old-policy baseline.
    """

    if len(prompts) == 0 or len(prompts) != len(action_traces):
        raise ValueError("prompts and action_traces must have the same nonzero batch size")
    if advantages is not None and len(advantages) != len(prompts):
        raise ValueError("advantages must have one value or row vector per sample")
    if old_policy_joint_logprobs is not None and len(old_policy_joint_logprobs) != len(prompts):
        raise ValueError("old-policy joint logprobs must have one row vector per sample")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a non-negative integer")

    prompt_tensors = [_as_prompt_tensor(prompt, device=device) for prompt in prompts]
    streams = [_higgs_stream(trace) for trace in action_traces]
    num_codebooks = streams[0].shape[1]
    codebook_vocab_size = streams[0].vocab_size
    for stream in streams:
        if stream.shape[0] <= 0:
            raise ValueError("each Higgs action stream must contain at least one row")
        if stream.shape[1] != num_codebooks or stream.vocab_size != codebook_vocab_size:
            raise ValueError("all Higgs streams in a batch must share codebook shape and vocabulary")

    batch_size = len(prompts)
    prompt_lengths = torch.tensor([prompt.numel() for prompt in prompt_tensors], dtype=torch.long, device=device)
    action_lengths = torch.tensor([stream.shape[0] for stream in streams], dtype=torch.long, device=device)
    max_actions = int(action_lengths.max().item())
    # The final action is a label only; every earlier complete row is an input.
    input_lengths = prompt_lengths + action_lengths - 1
    max_input = int(input_lengths.max().item())

    input_ids = torch.full((batch_size, max_input), pad_token_id, dtype=torch.long, device=device)
    prior_codes = torch.zeros((batch_size, max_input, num_codebooks), dtype=torch.long, device=device)
    codec_position_mask = torch.zeros((batch_size, max_input), dtype=torch.bool, device=device)
    sequence_mask = torch.zeros((batch_size, max_input), dtype=torch.bool, device=device)
    prediction_positions = torch.zeros((batch_size, max_actions), dtype=torch.long, device=device)
    actions = torch.zeros((batch_size, max_actions, num_codebooks), dtype=torch.long, device=device)
    action_mask = torch.zeros((batch_size, max_actions, num_codebooks), dtype=torch.bool, device=device)
    old_cell_logprobs = torch.zeros((batch_size, max_actions, num_codebooks), dtype=torch.float32, device=device)
    advantage_rows = (
        torch.zeros((batch_size, max_actions), dtype=torch.float32, device=device) if advantages is not None else None
    )

    for batch_index, (prompt, stream) in enumerate(zip(prompt_tensors, streams, strict=True)):
        prompt_length = prompt.numel()
        action_length = stream.shape[0]
        input_length = prompt_length + action_length - 1
        input_ids[batch_index, :prompt_length] = prompt
        sequence_mask[batch_index, :input_length] = True

        stream_actions = torch.as_tensor(stream.actions, dtype=torch.long, device=device)
        stream_mask = torch.as_tensor(stream.action_mask, dtype=torch.bool, device=device)
        stream_logprobs = torch.as_tensor(stream.policy_logprobs, dtype=torch.float32, device=device)
        if action_length > 1:
            prior_slice = slice(prompt_length, prompt_length + action_length - 1)
            prior_codes[batch_index, prior_slice] = stream_actions[:-1]
            codec_position_mask[batch_index, prior_slice] = True

        actions[batch_index, :action_length] = stream_actions
        action_mask[batch_index, :action_length] = stream_mask
        old_cell_logprobs[batch_index, :action_length] = torch.where(
            stream_mask, stream_logprobs, torch.zeros_like(stream_logprobs)
        )
        prediction_positions[batch_index, :action_length] = torch.arange(
            prompt_length - 1,
            prompt_length - 1 + action_length,
            dtype=torch.long,
            device=device,
        )

        if advantage_rows is not None:
            advantage = torch.as_tensor(advantages[batch_index], dtype=torch.float32, device=device)
            if advantage.ndim == 0 or advantage.numel() == 1:
                advantage_rows[batch_index, :action_length] = advantage.reshape(())
            elif advantage.ndim == 1 and advantage.numel() == action_length:
                advantage_rows[batch_index, :action_length] = advantage
            else:
                raise ValueError("each Higgs advantage must be scalar or match its action-row count")

    row_mask = action_mask.any(dim=-1)
    old_joint_logprobs = old_cell_logprobs.sum(dim=-1)
    if old_policy_joint_logprobs is not None:
        old_joint_logprobs.zero_()
        for batch_index, (values, action_length) in enumerate(
            zip(old_policy_joint_logprobs, action_lengths.tolist(), strict=True)
        ):
            values = torch.as_tensor(values, dtype=torch.float32, device=device)
            if values.ndim != 1 or values.numel() != action_length:
                raise ValueError("each old-policy joint logprob vector must match its action-row count")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("old-policy joint logprobs must be finite")
            old_joint_logprobs[batch_index, :action_length] = values
    return HiggsPolicyBatch(
        input_ids=input_ids,
        prior_codes=prior_codes,
        codec_position_mask=codec_position_mask,
        sequence_mask=sequence_mask,
        prediction_positions=prediction_positions,
        actions=actions,
        action_mask=action_mask,
        old_cell_logprobs=old_cell_logprobs,
        old_joint_logprobs=old_joint_logprobs,
        row_mask=row_mask,
        advantages=advantage_rows,
        prompt_lengths=prompt_lengths,
        action_lengths=action_lengths,
        num_codebooks=num_codebooks,
        codebook_vocab_size=codebook_vocab_size,
    )


def build_higgs_teacher_embeddings(
    text_embeddings: torch.Tensor,
    codec_weight: torch.Tensor,
    prior_codes: torch.Tensor,
    codec_position_mask: torch.Tensor,
) -> torch.Tensor:
    """Overlay summed, channel-offset codebook embeddings on text embeddings."""

    if text_embeddings.ndim != 3:
        raise ValueError("text_embeddings must have shape [batch, sequence, hidden]")
    if prior_codes.ndim != 3 or prior_codes.shape[:2] != text_embeddings.shape[:2]:
        raise ValueError("prior_codes must have shape [batch, sequence, codebooks]")
    if codec_position_mask.shape != text_embeddings.shape[:2]:
        raise ValueError("codec_position_mask must have shape [batch, sequence]")
    num_codebooks = prior_codes.shape[-1]
    if codec_weight.ndim != 2 or codec_weight.shape[0] % num_codebooks != 0:
        raise ValueError("codec_weight rows must be divisible by the number of codebooks")
    vocab_size = codec_weight.shape[0] // num_codebooks
    active_codes = prior_codes[codec_position_mask]
    if active_codes.numel() and (bool((active_codes < 0).any()) or bool((active_codes >= vocab_size).any())):
        raise ValueError("prior codebook IDs are outside the codec vocabulary")

    offsets = torch.arange(num_codebooks, device=prior_codes.device, dtype=prior_codes.dtype) * vocab_size
    codec_embeddings = F.embedding(prior_codes + offsets, codec_weight).sum(dim=-2)
    return torch.where(codec_position_mask.unsqueeze(-1), codec_embeddings, text_embeddings)


def selected_higgs_logprobs(
    logits: torch.Tensor,
    actions: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return masked per-cell and joint-row logprobs from ``[B,L,Q,V]`` logits."""

    if logits.ndim != 4:
        raise ValueError("Higgs logits must have shape [batch, time, codebook, vocabulary]")
    if actions.shape != logits.shape[:-1] or action_mask.shape != actions.shape:
        raise ValueError("Higgs actions and masks must match logits [batch, time, codebook]")
    if actions.dtype != torch.long:
        actions = actions.long()
    if action_mask.dtype != torch.bool:
        action_mask = action_mask.bool()
    if bool((actions[action_mask] < 0).any()) or bool((actions[action_mask] >= logits.shape[-1]).any()):
        raise ValueError("sampled Higgs action is outside the model vocabulary")

    # The policy contract is defined against full-vocabulary fp32 softmax.
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    selected = logprobs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    selected = torch.where(action_mask, selected, torch.zeros_like(selected))
    if not bool(torch.isfinite(selected[action_mask]).all()):
        raise RuntimeError("Higgs model produced a non-finite sampled-action logprob")
    row_mask = action_mask.any(dim=-1)
    return selected, selected.sum(dim=-1), row_mask


def _sum_of_sample_row_means(values: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
    counts = row_mask.sum(dim=-1)
    if bool((counts == 0).any()):
        raise ValueError("every Higgs sample must contain at least one active action row")
    masked = torch.where(row_mask, values, torch.zeros_like(values))
    return (masked.sum(dim=-1) / counts.to(values.dtype)).sum()


def higgs_joint_policy_loss(
    current_joint_logprobs: torch.Tensor,
    old_joint_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    eps_clip: float,
    eps_clip_high: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply GRPO's clipped surrogate once to each joint multi-codebook row ratio."""

    if current_joint_logprobs.shape != old_joint_logprobs.shape or row_mask.shape != current_joint_logprobs.shape:
        raise ValueError("joint logprobs and row_mask must have identical [batch, time] shapes")
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1).expand_as(current_joint_logprobs)
    if advantages.shape != current_joint_logprobs.shape:
        raise ValueError("Higgs advantages must have shape [batch] or [batch, time]")
    if eps_clip < 0 or (eps_clip_high is not None and eps_clip_high < 0):
        raise ValueError("GRPO clipping thresholds must be non-negative")
    eps_clip_high = eps_clip if eps_clip_high is None else eps_clip_high

    zeros = current_joint_logprobs.new_zeros(())
    if not bool(torch.isfinite(current_joint_logprobs[row_mask]).all()):
        raise RuntimeError("current Higgs joint logprobs must be finite on active rows")
    if not bool(torch.isfinite(old_joint_logprobs[row_mask]).all()):
        raise RuntimeError("rollout Higgs joint logprobs must be finite on active rows")
    if not bool(torch.isfinite(advantages[row_mask]).all()):
        raise RuntimeError("Higgs advantages must be finite on active rows")
    log_ratio = torch.where(
        row_mask,
        current_joint_logprobs - old_joint_logprobs,
        zeros,
    )
    ratio = torch.exp(log_ratio)
    if not bool(torch.isfinite(ratio[row_mask]).all()):
        raise RuntimeError("Higgs joint importance ratio overflowed on an active row")
    clean_advantages = torch.where(row_mask, advantages, zeros)
    unclipped = ratio * clean_advantages
    clipped_ratio = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip_high)
    clipped = clipped_ratio * clean_advantages
    per_row_loss = -torch.minimum(unclipped, clipped)
    clipfrac = ((ratio < 1.0 - eps_clip) | (ratio > 1.0 + eps_clip_high)).to(current_joint_logprobs.dtype)
    loss = _sum_of_sample_row_means(per_row_loss, row_mask)
    metrics = {
        "loss": loss.detach(),
        "pg_loss": loss.detach(),
        "pg_clipfrac": _sum_of_sample_row_means(clipfrac, row_mask).detach(),
    }
    return loss, metrics


def higgs_policy_loss_from_logits(
    logits: torch.Tensor,
    batch: HiggsPolicyBatch,
    *,
    eps_clip: float,
    eps_clip_high: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    if batch.advantages is None:
        raise ValueError("Higgs training batches require advantages")
    cell_logprobs, joint_logprobs, row_mask = selected_higgs_logprobs(logits, batch.actions, batch.action_mask)
    if not torch.equal(row_mask, batch.row_mask):
        raise ValueError("model and rollout Higgs row masks disagree")
    loss, metrics = higgs_joint_policy_loss(
        joint_logprobs,
        batch.old_joint_logprobs,
        batch.advantages,
        row_mask,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
    )
    abs_diff = (joint_logprobs.detach() - batch.old_joint_logprobs).abs()
    metrics["train_old_policy_logprob_abs_diff"] = _sum_of_sample_row_means(abs_diff, row_mask).detach()
    return loss, metrics, cell_logprobs


def get_higgs_joint_log_probs(
    logits: torch.Tensor,
    *,
    higgs_batch: HiggsPolicyBatch,
    non_loss_data: bool = True,
    **_: Any,
) -> dict[str, list[torch.Tensor]]:
    if not non_loss_data:
        raise ValueError("Higgs logprob collection is only valid as non-loss data")
    _, joint_logprobs, _ = selected_higgs_logprobs(logits, higgs_batch.actions, higgs_batch.action_mask)
    return {
        "log_probs": [
            joint_logprobs[index, : int(length.item())] for index, length in enumerate(higgs_batch.action_lengths)
        ]
    }


def validate_higgs_logprob_parity(
    action_traces: Sequence[Any],
    recomputed_joint_logprobs: Sequence[torch.Tensor],
    *,
    atol: float,
) -> dict[str, float]:
    """Hard-gate training on every active joint-row rollout logprob."""

    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or atol <= 0:
        raise ValueError("Higgs parity atol must be a positive number")
    if len(action_traces) == 0 or len(action_traces) != len(recomputed_joint_logprobs):
        raise ValueError("Higgs parity inputs must have the same nonzero sample count")

    diffs = []
    for sample_index, (trace, recomputed) in enumerate(zip(action_traces, recomputed_joint_logprobs, strict=True)):
        stream = _higgs_stream(trace)
        device = recomputed.device
        action_mask = torch.as_tensor(stream.action_mask, dtype=torch.bool, device=device)
        old_cell_logprobs = torch.as_tensor(stream.policy_logprobs, dtype=torch.float32, device=device)
        row_mask = action_mask.any(dim=-1)
        expected = torch.where(action_mask, old_cell_logprobs, 0.0).sum(dim=-1)
        recomputed = recomputed.float()
        if recomputed.ndim != 1 or recomputed.numel() != stream.shape[0]:
            raise ValueError(
                f"Higgs parity sample {sample_index} recomputed shape {tuple(recomputed.shape)} "
                f"does not match {stream.shape[0]} action rows"
            )
        if not bool(torch.isfinite(recomputed[row_mask]).all()):
            raise RuntimeError(f"Higgs parity sample {sample_index} contains non-finite trainer logprobs")
        diffs.append((recomputed[row_mask] - expected[row_mask]).abs())

    all_diffs = torch.cat(diffs)
    max_abs_diff = float(all_diffs.max().item())
    mean_abs_diff = float(all_diffs.mean().item())
    if max_abs_diff > float(atol):
        raise RuntimeError(
            "Higgs trainer/rollout joint-logprob parity failed before optimizer step: "
            f"max_abs_diff={max_abs_diff:.6g}, mean_abs_diff={mean_abs_diff:.6g}, atol={float(atol):.6g}"
        )
    return {"max_abs_diff": max_abs_diff, "mean_abs_diff": mean_abs_diff}


def validate_higgs_weight_versions(
    sample_weight_versions: Sequence[Sequence[str]] | None,
    *,
    trainer_weight_version: Any,
) -> str:
    """Reject missing, resumed, mixed, or stale structured trajectories."""

    if not sample_weight_versions:
        raise ValueError("Higgs training requires one rollout weight version per sample")
    versions = []
    for sample_index, sample_versions in enumerate(sample_weight_versions):
        if not isinstance(sample_versions, (list, tuple)) or len(sample_versions) != 1:
            raise ValueError(f"Higgs sample {sample_index} must contain exactly one rollout weight version")
        version = sample_versions[0]
        if not isinstance(version, str) or not version:
            raise ValueError(f"Higgs sample {sample_index} weight version must be a nonempty string")
        versions.append(version)
    if len(set(versions)) != 1:
        raise RuntimeError(f"Higgs training batch mixes rollout weight versions: {sorted(set(versions))}")
    trainer_version = str(trainer_weight_version)
    # SGLang's untouched startup checkpoint is reported as ``default`` while
    # Miles' updater starts its monotonic version counter at zero.  This alias
    # is valid only before the first update; every later version must match
    # exactly.
    initial_version_alias = versions[0] == "default" and trainer_version == "0"
    if versions[0] != trainer_version and not initial_version_alias:
        raise RuntimeError(f"stale Higgs rollout weight version {versions[0]!r}; trainer expects {trainer_version!r}")
    return versions[0]


__all__ = [
    "HIGGS_MODEL_FAMILY",
    "HIGGS_STREAM_NAME",
    "HiggsPolicyBatch",
    "build_higgs_teacher_embeddings",
    "collate_higgs_policy_batch",
    "get_higgs_joint_log_probs",
    "higgs_joint_policy_loss",
    "higgs_policy_loss_from_logits",
    "is_higgs_policy_enabled",
    "selected_higgs_logprobs",
    "validate_higgs_hf_config",
    "validate_higgs_single_device_config",
    "validate_higgs_logprob_parity",
    "validate_higgs_weight_versions",
]
