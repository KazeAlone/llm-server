#!/usr/bin/env bash
BACKEND_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND=sglang DEFAULT_PORT=8083
source "$BACKEND_DIR/../common/runtime.sh"
setup_backend() {
  setup_common
  check_cuda
  # The maintained fork pins Torch 2.9.1; don't silently downgrade the user's 2.10.
  "$PYTHON" -c 'import torch; assert torch.__version__.split("+")[0] == "2.9.1", "此 SGLang-V100 版本要求 Torch 2.9.1；请按 sglang/README.md 单独建环境。不会替换已有 Torch。"'
  checkout https://github.com/haohervchb/sglang-V100.git "${SGLANG_REF:-d85b3e2705411df781dadf9e8fa4db52c8f07a5a}" "$STATE/source"
  "$PYTHON" "$BACKEND_DIR/prepare_install.py" "$STATE/source" "$STATE/install.sh"
  export SGLANG_V100_DEPS_DIR="$STATE/deps" SGLANG_V100_PYTHON="$PYTHON"
  select_gpus
  bash "$STATE/install.sh"
  setup_done
}
run_backend() {
  export SGLANG_MAMBA_CONV_DTYPE=float16 SGLANG_MAMBA_SSM_DTYPE=float16
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  args=(--model-path "$MODEL_DIR" --served-model-name "$MODEL_ID" --dtype float16
    --host "$HOST" --port "$PORT" --tensor-parallel-size "$TP"
    --context-length "$CTX_SIZE" --mem-fraction-static "${GPU_MEMORY_UTILIZATION:-0.80}"
    --max-running-requests "${MAX_RUNNING_REQUESTS:-1}" --chunked-prefill-size "${CHUNK_SIZE:-1024}"
    --attention-backend tilelang_fa_v100 --linear-attn-prefill-backend tilelang
    --linear-attn-decode-backend triton --disable-custom-all-reduce
    --enable-metrics --reasoning-parser qwen3 --tool-call-parser qwen3_coder)
  [[ $VISION == 0 ]] || args+=(--enable-multimodal --mm-attention-backend triton_attn)
  # API key passed inside Python, avoiding ps-visible --api-key.
  export LLM_SERVER_API_KEY="$API_KEY" MEDIA_PATH VISION
  exec "$PYTHON" "$BACKEND_DIR/launch.py" "${args[@]}" "${EXTRA_ARGS[@]}"
}
dispatch "$@"
