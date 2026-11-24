#!/usr/bin/env bash
BACKEND_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND=transformers DEFAULT_PORT=8081
source "$BACKEND_DIR/../common/runtime.sh"
setup_backend() {
  setup_common
  pip_install -r "$BACKEND_DIR/requirements.txt"
  "$PYTHON" -c 'from transformers import AutoProcessor, AutoModelForImageTextToText; import fastapi, uvicorn, accelerate'
  setup_done
}
run_backend() {
  export MODEL_DIR MODEL_ID HOST PORT API_KEY CTX_SIZE VISION MEDIA_PATH
  export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa} GPU_MEMORY_GIB=${GPU_MEMORY_GIB:-28}
  exec "$PYTHON" "$BACKEND_DIR/server.py"
}
dispatch "$@"
