"""GRPO on the Qwen3-Omni thinker with image/video input (pseudo-Qwen3-VL bridge route).

The thinker (visual tower + text backbone) is extracted as a qwen3_vl_moe checkpoint and
trained on miles' bridge Qwen3-VL path; --qwen3-omni-vl installs the omni TM-RoPE video
override (image positions already match; see miles_plugins/models/qwen3_omni_thinker_vl.py).
Rollout runs on a standalone sglang-omni server through the processed-multimodal contract.

Data: Video-R1-260k MCQ rows scored by --rm-type gpqa. --modality video uses the CLEVRER
zip; --modality image uses the Chart zips (single-source subsets keep the download small).

v1 is off-policy (--debug-skip-weight-update): the bridge exports pseudo-VL names, and the
rename shim into the omni server's thinker.* namespace is not wired into the update path
yet (pseudo_vl_to_omni_server_name is ready + tested).
"""

import os
from dataclasses import dataclass
from typing import Literal

import miles.utils.external_utils.command_utils as U

OMNI_MODEL = "Qwen3-Omni-30B-A3B-Instruct"
THINKER_VL_MODEL = "Qwen3-Omni-30B-A3B-Thinker-VL"
MEGATRON_MODEL_TYPE = "qwen3-omni-30B-A3B-thinker"
VIDEO_R1 = "Video-R1/Video-R1-data"
_MODALITY_SOURCES = {"video": ["CLEVRER"], "image": ["Chart"]}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    modality: Literal["video", "image"] = "video"
    run_id: str = U.create_run_id()
    actor_tp: int = 4
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    omni_server_ip: str = "127.0.0.1"
    omni_server_port: int = 30000
    omni_server_tp: int = 4
    max_samples: int = 5120
    extra_args: str = ""


def prepare(args: ScriptArgs):
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command(f"hf download Qwen/{OMNI_MODEL} --local-dir {args.model_dir}/{OMNI_MODEL}")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    U.exec_command(
        f"python {repo}/tools/extract_qwen3_omni_thinker.py --variant vl "
        f"--src {args.model_dir}/{OMNI_MODEL} --dst {args.model_dir}/{THINKER_VL_MODEL}"
    )
    vr1_dir = f"{args.data_dir}/Video-R1-data"
    sources = _MODALITY_SOURCES[args.modality]
    includes = " ".join(f'--include "{src}/*" ' for src in sources)
    U.exec_command(f"hf download {VIDEO_R1} --repo-type dataset Video-R1-260k.json --local-dir {vr1_dir}")
    U.exec_command(f"hf download {VIDEO_R1} --repo-type dataset {includes} --local-dir {vr1_dir}")
    for src in sources:
        U.exec_command(f"bash -c 'cd {vr1_dir}/{src} && for z in *.zip; do unzip -n -q $z; done'")
    U.exec_command(
        f"python {repo}/examples/omni_thinker/prepare_video_r1.py "
        f"--src {vr1_dir}/Video-R1-260k.json --dst {args.data_dir}/video_r1_{args.modality}.jsonl "
        f"--media-root {vr1_dir} --data-types {args.modality} "
        f"--path-prefixes {' '.join(sources)} --require-media --max-samples {args.max_samples}"
    )


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{THINKER_VL_MODEL}/ "
        f"--load {args.model_dir}/{THINKER_VL_MODEL}/ "
        "--megatron-to-hf-mode bridge "
        "--qwen3-omni-vl "
    )

    debug_minimal = args.mode == "debug_minimal"
    multimodal_keys = '{"video": "videos", "image": "images"}'
    rollout_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.sglang_omni.generate "
        f"--prompt-data {args.data_dir}/video_r1_{args.modality}.jsonl "
        "--input-key prompt "
        "--label-key label "
        f"--multimodal-keys '{multimodal_keys}' "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type gpqa "
        f"--num-rollout {6 if debug_minimal else args.max_samples // 32} "
        f"--rollout-batch-size {8 if debug_minimal else 32} "
        f"--n-samples-per-prompt {4 if debug_minimal else 8} "
        f"--rollout-max-response-len {100 if debug_minimal else 2048} "
        "--rollout-temperature 1 "
        f"--global-batch-size {32 if debug_minimal else 256} "
        "--balance-data "
        f"--sglang-router-ip {args.omni_server_ip} "
        f"--sglang-router-port {args.omni_server_port} "
        "--rollout-external "
        f"--rollout-external-engine-addrs {args.omni_server_ip}:{args.omni_server_port} "
        "--rollout-external-admin-api sglang-omni "
        "--rollout-weight-update-stages thinker "
        f"--rollout-num-gpus {args.omni_server_tp} "
        f"--rollout-num-gpus-per-engine {args.omni_server_tp} "
    )

    consistency_args = (
        "--get-mismatch-metrics "
        "--use-tis "
        "--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        "--custom-config-path examples/train_infer_mismatch_helper/mis.yaml "
        "--debug-skip-weight-update "  # v1: pseudo-VL -> omni-server rename shim not wired yet
    )

    # mirrors the geo3k megatron/bridge tier (SP stays on: no audio injection in VL runs)
    perf_args = (
        f"--tensor-model-parallel-size {args.actor_tp} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if debug_minimal else 4096} "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        + ("" if debug_minimal else "--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl ")
        + "--entropy-coef 0.00 "
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
        + ("" if debug_minimal else "--accumulate-allreduce-grads-in-fp32 ")
        + "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_tp} "
        f"--num-gpus-per-node {args.actor_tp + args.omni_server_tp} "
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
