#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MILES_TTS_ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
HOST="${MILES_TTS_ASR_HOST:-127.0.0.1}"
PORT="${MILES_TTS_ASR_PORT:-8080}"
GPU="${MILES_TTS_ASR_GPU:-0}"
MEM_FRACTION="${MILES_TTS_ASR_MEM_FRACTION:-0.1}"
MAX_RUNNING_REQUESTS="${MILES_TTS_ASR_MAX_RUNNING_REQUESTS:-16}"

if [[ -n "${SGLANG_OMNI_ROOT:-}" ]]; then
  export PYTHONPATH="${SGLANG_OMNI_ROOT}:${PYTHONPATH:-}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
exec python -m sglang_omni.cli serve \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
