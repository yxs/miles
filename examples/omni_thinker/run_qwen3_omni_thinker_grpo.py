"""GRPO on the Qwen3-Omni-30B-A3B thinker (text MoE) with DAPO-math reward.

Off-policy until live weight-sync lands: point --sglang-router-ip/port at a standalone omni
server; the rollout half serves the frozen base while the trainer updates its own copy, with
TIS + --get-mismatch-metrics absorbing/measuring the gap.
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

OMNI_MODEL = "Qwen3-Omni-30B-A3B-Instruct"
THINKER_MODEL = "Qwen3-Omni-30B-A3B-Thinker"
MEGATRON_MODEL_TYPE = "qwen3-omni-30B-A3B-thinker"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    num_gpus_per_node: int = 8
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    omni_router_ip: str = "127.0.0.1"
    omni_router_port: int = 30000
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download Qwen/{OMNI_MODEL} --local-dir {args.model_dir}/{OMNI_MODEL}")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    U.exec_command(
        f"python {repo}/tools/extract_qwen3_omni_thinker.py "
        f"--src {args.model_dir}/{OMNI_MODEL} --dst {args.model_dir}/{THINKER_MODEL}"
    )
    U.convert_checkpoint(
        model_name=THINKER_MODEL,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{THINKER_MODEL}",
        megatron_path=args.megatron_path,
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def execute(args: ScriptArgs):
    ref_load_path = f"{args.model_dir}/{THINKER_MODEL}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{THINKER_MODEL}/ "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        "--model-name qwen3omni_moe "  # body.* broadcast naming
    )

    rollout_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.omni_thinker.generate "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type dapo "
        "--reward-key score "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
        f"--sglang-router-ip {args.omni_router_ip} "
        f"--sglang-router-port {args.omni_router_port} "
    )

    consistency_args = (
        "--use-rollout-logprobs "
        "--get-mismatch-metrics "
        "--use-tis "
        "--tis-clip 2.0 "
    )

    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{consistency_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE,
        train_script="train.py",
        megatron_path=args.megatron_path,
        extra_env_vars={
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "PYTHONPATH": f"{args.megatron_path}",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
