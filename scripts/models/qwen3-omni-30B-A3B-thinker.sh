# Megatron model args for the Qwen3-Omni-30B-A3B thinker (text MoE).
# Architecturally Qwen3-30B-A3B; differs in vocab (152064 vs 151936) and the inert dense
# ffn size: the omni thinker config stamps intermediate_size=768 (== moe_intermediate_size),
# and every layer is MoE, so ffn-hidden-size never materializes weights - it must simply
# match the HF config for hf_validate_args.

NLAYERS="${MODEL_ARGS_NUM_LAYERS:-48}"
FIRST_K_DENSE_REPLACE=0

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"


MODEL_ARGS=(
   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 32
   --num-query-groups 4
   --kv-channels 128
   --num-layers $NLAYERS
   --hidden-size 2048
   --ffn-hidden-size 768

   --normalization RMSNorm
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 1.0
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size "${MODEL_ARGS_VOCAB_SIZE:-152064}"

   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"

   # moe
   --moe-ffn-hidden-size 768
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --num-experts 128
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0
)
