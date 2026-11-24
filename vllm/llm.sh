#!/usr/bin/env bash
BACKEND_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND=vllm DEFAULT_PORT=8082
source "$BACKEND_DIR/../common/runtime.sh"
setup_backend() {
  setup_common
  check_cuda
  printf 'transformers==5.8.1\n' >> "$PIP_CONSTRAINT"
  checkout https://github.com/vllm-project/vllm.git "${VLLM_REF:-b1388b1fbf5aaef47937fabe98931211684666a6}" "$STATE/source"
  (
    cd "$STATE/source"
    "$PYTHON" use_existing_torch.py --prefix
    pip_install -r requirements/build.txt
    pip_install --no-build-isolation -e .
  )
  "$PYTHON" -c 'import vllm; print("vLLM", vllm.__version__)'
  setup_done
}
run_backend() {
  export VLLM_API_KEY="$API_KEY"
  args=(--model "$MODEL_DIR" --served-model-name "$MODEL_ID" --dtype float16
    --host "$HOST" --port "$PORT" --tensor-parallel-size "$TP"
    --max-model-len "$CTX_SIZE" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --max-num-seqs "${MAX_RUNNING_REQUESTS:-1}" --attention-backend TRITON_ATTN
    --disable-custom-all-reduce --enable-chunked-prefill --max-num-batched-tokens "${CHUNK_SIZE:-1024}"
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder)
  [[ -z $MEDIA_PATH ]] || args+=(--allowed-local-media-path "$MEDIA_PATH")
  if [[ $VISION == 0 ]]; then
    args+=(--limit-mm-per-prompt '{"image":0,"video":0}' --skip-mm-profiling)
  else
    args+=(--mm-encoder-attn-backend TORCH_SDPA)
  fi
  if [[ ${ENFORCE_EAGER:-0} == 1 ]]; then args+=(--enforce-eager); fi
  exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${args[@]}" "${EXTRA_ARGS[@]}"
}
dispatch "$@"
