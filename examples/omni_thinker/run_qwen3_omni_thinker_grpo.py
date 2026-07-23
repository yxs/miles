"""GRPO on the Qwen3-Omni-30B-A3B thinker with audio-input AVQA (MCQ reward).

Topology: rollout runs on a standalone sglang-omni text server (external engines; miles
launches nothing locally); the trainer holds the extracted text backbone and injects
frozen-audio-tower embeddings at placeholder positions (--qwen3-omni-audio-encoder-path).

Weight sync: `--sync-mode skip` freezes the server (off-policy debug; TIS absorbs the gap),
`--sync-mode distributed` pushes thinker.* weights over NCCL each step (on-policy).
"""

import os
from dataclasses import dataclass
from typing import Literal


import miles.utils.external_utils.command_utils as U

OMNI_MODEL = "Qwen3-Omni-30B-A3B-Instruct"
THINKER_MODEL = "Qwen3-Omni-30B-A3B-Thinker"
MEGATRON_MODEL_TYPE = "qwen3-omni-30B-A3B-thinker"
AVQA_DATASET = "Joysw909/AVQA"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    sync_mode: Literal["skip", "distributed"] = "distributed"
    run_id: str = U.create_run_id()
    actor_tp: int = 4  # actor GPUs (TP=EP); ray also reserves omni_server_tp bundles for the external attach
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    omni_server_ip: str = "127.0.0.1"
    omni_server_port: int = 30000
    omni_server_tp: int = 4  # TP size of the external omni server (NCCL group world_size = tp + 1)
    avqa_max_samples: int = 5120
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
        num_gpus_per_node=args.actor_tp,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{THINKER_MODEL}",
        megatron_path=args.megatron_path,
    )
    avqa_dir = f"{args.data_dir}/{AVQA_DATASET.split('/')[-1]}"
    U.exec_command(f"hf download {AVQA_DATASET} --repo-type dataset --local-dir {avqa_dir}")
    U.exec_command(
        f"python {repo}/examples/omni_thinker/prepare_avqa.py "
        f"--src {avqa_dir}/train_r1aqa_line.json --dst {args.data_dir}/avqa.jsonl "
        f"--audio-root {avqa_dir} --max-samples {args.avqa_max_samples}"
    )


def execute(args: ScriptArgs):
    ref_load_path = f"{args.model_dir}/{THINKER_MODEL}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{THINKER_MODEL}/ "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        "--model-name qwen3omni_moe "  # thinker.* broadcast naming
    )

    debug_minimal = args.mode == "debug_minimal"
    rollout_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.sglang_omni.generate "
        f"--prompt-data {args.data_dir}/avqa.jsonl "
        "--input-key prompt "
        "--label-key label "
        '--multimodal-keys \'{"audio": "audios"}\' '
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type gpqa "
        f"--num-rollout {6 if debug_minimal else args.avqa_max_samples // 32} "
        f"--rollout-batch-size {8 if debug_minimal else 32} "
        f"--n-samples-per-prompt {4 if debug_minimal else 8} "
        f"--rollout-max-response-len {100 if debug_minimal else 2048} "
        "--rollout-temperature 1 "
        f"--global-batch-size {32 if debug_minimal else 256} "
        "--balance-data "
        # the standalone omni server doubles as the router: the adapter posts straight to it
        f"--sglang-router-ip {args.omni_server_ip} "
        f"--sglang-router-port {args.omni_server_port} "
        "--rollout-external "
        f"--rollout-external-engine-addrs {args.omni_server_ip}:{args.omni_server_port} "
        "--rollout-external-admin-api sglang-omni "
        "--rollout-weight-update-stages thinker "
        f"--rollout-num-gpus {args.omni_server_tp} "
        f"--rollout-num-gpus-per-engine {args.omni_server_tp} "
    )

    # frozen audio tower for placeholder-embedding injection (audio-input training)
    mm_args = f"--qwen3-omni-audio-encoder-path {args.model_dir}/{OMNI_MODEL} "

    consistency_args = "--use-rollout-logprobs " "--get-mismatch-metrics " "--use-tis " "--tis-clip 2.0 "
    if args.sync_mode == "skip":
        consistency_args += "--debug-skip-weight-update "

    perf_args = (
        f"--tensor-model-parallel-size {args.actor_tp} "
        # no --sequence-parallel: audio injection scatters full-sequence embeddings
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.actor_tp} "
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
        f"--actor-num-gpus-per-node {args.actor_tp} "
        f"--num-gpus-per-node {args.actor_tp + args.omni_server_tp} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{mm_args} "
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
        num_gpus_per_node=args.actor_tp + args.omni_server_tp,
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
    main()
