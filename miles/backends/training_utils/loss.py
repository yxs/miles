from argparse import Namespace

import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.cp_utils import get_sum_of_sample_mean
from miles.backends.training_utils.higgs_policy import (
    higgs_policy_loss_from_logits,
    is_higgs_policy_enabled,
    validate_higgs_single_device_config,
)
from miles.backends.training_utils.loss_hub.advantages import compute_advantages, normalize_advantages
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy, get_values  # noqa: F401
from miles.backends.training_utils.loss_hub.losses import get_loss_function
from miles.backends.training_utils.loss_hub.math_utils import compute_approx_kl
from miles.backends.training_utils.loss_hub.opd import apply_opd_kl_to_advantages
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". On-policy distillation (OPD) is applied
    orthogonally on top of any estimator via `args.use_opd`. When
    `args.normalize_advantages` is True, advantages are whitened across the
    data-parallel group using masked statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    if "action_traces" in rollout_data:
        validate_higgs_single_device_config(args, data_parallel_size=get_parallel_state().intra_dp.size)
        if not is_higgs_policy_enabled(args):
            raise ValueError("structured action traces require the Higgs policy adapter")
        traces = rollout_data["action_traces"]
        rewards = rollout_data["rewards"]
        if len(traces) != len(rewards):
            raise ValueError("Higgs action traces and rewards must have identical batch size")
        device = rollout_data["tokens"][0].device
        advantages = []
        for trace, reward in zip(traces, rewards, strict=True):
            trace.validate()
            if trace.model_family != "higgs_tts" or len(trace.action_streams) != 1:
                raise ValueError("the initial Higgs path requires one higgs_tts action stream")
            row_count = trace.action_streams[0].shape[0]
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device)
            if reward_tensor.ndim != 0:
                raise ValueError("each Higgs rollout reward must be scalar")
            if not bool(torch.isfinite(reward_tensor)):
                raise ValueError("each Higgs rollout reward must be finite")
            advantages.append(reward_tensor.expand(row_count).clone())
        rollout_data["advantages"] = advantages
        rollout_data["returns"] = [advantage.clone() for advantage in advantages]
        return

    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if log_probs is None and values is None:
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    advantages, returns = compute_advantages(
        args=args,
        kl=kl,
        rewards=rewards,
        log_probs=log_probs,
        loss_masks=loss_masks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        values=values,
    )

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    if args.normalize_advantages:
        advantages = normalize_advantages(args, advantages, loss_masks, total_lengths, response_lengths, max_seq_lens)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
    apply_megatron_loss_scaling: bool = False,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    parallel_state = get_parallel_state()
    if "higgs_policy_batch" in batch:
        validate_higgs_single_device_config(args, data_parallel_size=parallel_state.intra_dp.size)
        higgs_batch = batch["higgs_policy_batch"]
        loss, log, _ = higgs_policy_loss_from_logits(
            logits,
            higgs_batch,
            eps_clip=args.eps_clip,
            eps_clip_high=args.eps_clip_high,
        )
        if apply_megatron_loss_scaling:
            assert not args.use_dynamic_global_batch_size
            loss = loss * num_microbatches / args.global_batch_size
        else:
            loss = loss / args.global_batch_size
        metric_values = [
            torch.tensor(float(higgs_batch.input_ids.shape[0]), device=logits.device),
            *(value.to(device=logits.device) for value in log.values()),
        ]
        return (
            loss,
            torch.tensor(1, device=logits.device),
            {
                "keys": list(log.keys()),
                "values": torch.stack(metric_values),
            },
        )

    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    func = get_loss_function(args)

    if args.recompute_loss_function:
        loss, log = checkpoint(
            func,
            args,
            batch,
            logits,
            sum_of_sample_mean,
        )
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Forces autograd to traverse the full graph on every rank to avoid hang.
    if parallel_state.cp.size > 1 and args.allgather_cp:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in batch)
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        if apply_megatron_loss_scaling:
            loss_parallel_size = (
                parallel_state.intra_dp.size
                if args.true_on_policy_mode and parallel_state.is_ulysses_cp
                else parallel_state.intra_dp_cp.size
            )
            loss = loss * num_microbatches / global_batch_size * loss_parallel_size
        else:
            loss = loss / global_batch_size * parallel_state.intra_dp.size
    else:
        if apply_megatron_loss_scaling:
            loss = loss * parallel_state.cp.size

    return (
        loss,
        torch.tensor(num_tokens if args.calculate_per_token_loss else 1, device=logits.device),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
