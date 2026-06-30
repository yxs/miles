#!/bin/bash
# Thinker text RL through the real miles FSDP trainer.
# Rollout is an EXTERNAL sglang-omni thinker server (the omni pipeline); the trainer
# does FSDP + GRPO + LoRA and talks to it via OmniGenerateFn over HTTP. Weight-sync to
# the external server is deferred (M2) -> this M1 run is off-policy.
#
# Prereq: sglang-omni thinker server already serving on $SERVER_PORT (e.g. GPU2):
#   PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 ... python examples/run_qwen3_omni_server.py \
#     --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8003
set -ex

pkill -9 -f "ray::" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=16
# Trainer GPUs: avoid GPU2 (held by the external sglang-omni server).
TRAINER_GPUS=${TRAINER_GPUS:-"0,1,3,4"}
export CUDA_VISIBLE_DEVICES=$TRAINER_GPUS
NGPU=$(echo $TRAINER_GPUS | tr ',' '\n' | wc -l)

REPO=/root/rl-omni/miles
THINKER=/root/qwen3-omni-thinker
SERVER_PORT=${SERVER_PORT:-8003}

CKPT_ARGS=(
   --hf-checkpoint $THINKER
)

ROLLOUT_ARGS=(
   --prompt-data $REPO/examples/thinker_text_rl/math_harder_msgs.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --custom-generate-function-path miles_plugins.omni.omni_generate_fn.OmniGenerateFn
   --custom-rm-path miles_plugins.omni.math_reward.compute_math_reward
   --rollout-external
   --rollout-num-gpus 0
   --rollout-external-engine-addrs "localhost:${SERVER_PORT}"
   --sglang-router-ip localhost
   --sglang-router-port ${SERVER_PORT}
   --num-rollout 4
   --rollout-batch-size 4
   --n-samples-per-prompt 4
   --rollout-max-response-len 64
   --rollout-temperature 0.8
   --global-batch-size 16
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.28
   --kl-coef 0.00
   --entropy-coef 0.00
)

LORA_ARGS=(
   --lora-rank 8
   --lora-alpha 16
   --target-modules q_proj,v_proj
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-5
   --lr-decay-style constant
)

TRAIN_BACKEND_ARGS=(
   --train-backend fsdp
   --gradient-checkpointing
   --attn-implementation sdpa
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

MISC_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node $NGPU
   # FSDP backend is gated behind --ci-test (experimental); disable CI checkers so they don't interfere
   --ci-test
   --ci-disable-kl-checker
   --ci-disable-logprobs-checker
)

ray start --head --node-ip-address 127.0.0.1 --num-gpus $NGPU --disable-usage-stats

# train.py connects to the running cluster via ray.init(address="auto"); no dashboard / job-submit needed
export PYTHONPATH=${REPO}:/root/rl-omni/sglang-omni
export HF_HUB_OFFLINE=1
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

python3 train.py \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${TRAIN_BACKEND_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${MISC_ARGS[@]}
