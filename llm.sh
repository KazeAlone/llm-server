#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  # .env 是本机用户维护的 Shell 配置。
  source "$ROOT/.env"
fi
MODEL_DIR=${MODEL_DIR:-}
GPU_IDS=${GPU_IDS:-0,1,2,3}
CTX_SIZE=${CTX_SIZE:-32768}
PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
PRECISION=${PRECISION:-f16}
VISION=${VISION:-0}
FLASH_ATTN=${FLASH_ATTN:-auto}
BUILD_JOBS=${BUILD_JOBS:-8}
PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHON_ENV=${PYTHON_ENV:-auto}
STATE="$ROOT/.runtime"
SOURCE="$STATE/llama.cpp"
BINARY="$SOURCE/build-volta/bin/llama-server"
MODEL="$STATE/models/Qwen3.8-27B-$PRECISION.gguf"
PROJECTOR="$STATE/models/Qwen3.8-27B-mmproj-F16.gguf"
SESSION="llm-$(printf '%s' "$ROOT" | sha256sum | cut -c1-12)"

die() { echo "错误：$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "缺少命令 $1，请按 README 安装依赖。"; }

check_python() {
  "$1" -c 'import sys; print("Python", sys.version.split()[0], "解释器", sys.executable); sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "转换依赖要求 Python >=3.10。Conda 模式请激活新版 Python 环境；venv 模式请设置 PYTHON_BIN 并备份旧 .runtime/venv 后重新 setup。"
}

select_python() {
  local mode=$PYTHON_ENV
  if [[ $mode == auto ]]; then
    if [[ -n ${CONDA_PREFIX:-} ]]; then mode=conda; else mode=venv; fi
  fi
  case $mode in
    conda)
      [[ -n ${CONDA_PREFIX:-} && -x "$CONDA_PREFIX/bin/python" ]] \
        || die "请先 conda activate 环境名，再执行命令；或设置 PYTHON_ENV=venv。"
      CONVERT_PYTHON="$CONDA_PREFIX/bin/python"
      ;;
    venv) CONVERT_PYTHON="$STATE/venv/bin/python" ;;
    *) die "PYTHON_ENV 只能是 auto、conda 或 venv" ;;
  esac
  SELECTED_ENV=$mode
  echo "转换环境 $SELECTED_ENV，解释器 $CONVERT_PYTHON"
}

if [[ -n ${CUDA_HOME:-} ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
[[ $PRECISION == f16 || $PRECISION == q8_0 ]] || die "PRECISION 只能是 f16 或 q8_0"
[[ $VISION == 0 || $VISION == 1 ]] || die "VISION 只能是 0 或 1"

# flock 的父进程持锁，子进程关闭锁描述符，避免 tmux 服务继承锁。
case ${1:-help} in
  setup|convert|deploy|start|stop)
    if [[ ${LLM_DEPLOY_LOCKED:-0} != 1 ]]; then
      need flock
      mkdir -p "$STATE"
      exec flock --nonblock --close --conflict-exit-code 75 "$STATE/deploy.lock" \
        env LLM_DEPLOY_LOCKED=1 bash "$ROOT/llm.sh" "$@"
    fi
    ;;
esac

help_text() {
  cat <<'EOF'
用法 bash llm.sh <命令>
  doctor   显示 GPU、拓扑、CUDA、内存及磁盘
  deploy   编译后端、转换已有权重、后台启动（复用已有结果）
  setup    编译 sm_70 后端并建立转换环境
  convert  将 MODEL_DIR 转换为 GGUF，VISION=1 时同时转换视觉模块
  run      在当前终端运行服务，Ctrl+C 停止
  start    在 tmux 中后台启动，加载过程查看 logs
  status   检查后台会话、GPU 占用、HTTP 健康状态
  logs     持续查看日志，Ctrl+C 退出查看
  stop     向本项目的 tmux 会话发送 Ctrl+C
配置 .env 从 .env.example 复制，至少填写 MODEL_DIR。
EOF
}

doctor() {
  need nvidia-smi
  nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.free --format=csv
  nvidia-smi topo -m
  if command -v nvcc >/dev/null; then nvcc --version; else echo "未找到 nvcc，需要 CUDA Toolkit 12.x。"; fi
  select_python
  if [[ -x "$CONVERT_PYTHON" ]]; then
    "$CONVERT_PYTHON" --version
    "$CONVERT_PYTHON" -m pip --version
  else
    echo "转换环境尚未创建，setup 将使用 $PYTHON_BIN 创建。"
  fi
  free -h
  df -h "$ROOT"
}

setup() {
  select_python
  need git; need cmake; need nvcc
  if [[ $SELECTED_ENV == venv ]]; then
    need "$PYTHON_BIN"
    check_python "$PYTHON_BIN"
    if [[ ! -x "$CONVERT_PYTHON" ]]; then
      mkdir -p "$STATE"
      "$PYTHON_BIN" -m venv "$STATE/venv"
    fi
  fi
  check_python "$CONVERT_PYTHON"
  nvcc --version | grep -q 'release 12\.' || die "V100 编译请使用 CUDA Toolkit 12.x，建议 12.8。"
  mkdir -p "$STATE/models"
  if [[ ! -d "$SOURCE/.git" ]]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$SOURCE"
    if [[ -n ${LLAMA_REF:-} ]]; then git -C "$SOURCE" checkout --detach "$LLAMA_REF"; fi
  fi
  git -C "$SOURCE" rev-parse HEAD > "$STATE/llama-cpp-commit.txt"
  cmake -S "$SOURCE" -B "$SOURCE/build-volta" \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
  cmake --build "$SOURCE/build-volta" --target llama-server -j "$BUILD_JOBS"
  "$CONVERT_PYTHON" -m pip install --upgrade pip
  "$CONVERT_PYTHON" "$ROOT/install_conversion_deps.py" \
    "$SOURCE/requirements/requirements-convert_hf_to_gguf.txt"
  "$CONVERT_PYTHON" "$SOURCE/convert_hf_to_gguf.py" --help >/dev/null
  "$BINARY" --version
}

convert() {
  select_python
  [[ -d "$MODEL_DIR" ]] || die "请在 .env 填写有效 MODEL_DIR。"
  [[ -x "$CONVERT_PYTHON" ]] || die "请先执行 setup。"
  check_python "$CONVERT_PYTHON"
  "$CONVERT_PYTHON" - "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"]
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit(f"模型目录缺少文件 {missing}")
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
missing = [name for name in shards if not (root / name).is_file()]
if missing:
    raise SystemExit(f"缺少权重分片 {missing}")
print(f"已检查 {len(shards)} 个权重分片。")
PY
  # 临时文件仅在转换成功后改名，重试不会复用中断产物。
  if [[ ! -s "$MODEL" ]]; then
    "$CONVERT_PYTHON" "$SOURCE/convert_hf_to_gguf.py" "$MODEL_DIR" \
      --outtype "$PRECISION" --outfile "$MODEL.partial"
    mv -- "$MODEL.partial" "$MODEL"
  else
    echo "复用 $MODEL"
  fi
  if [[ $VISION == 1 && ! -s "$PROJECTOR" ]]; then
    "$CONVERT_PYTHON" "$SOURCE/convert_hf_to_gguf.py" "$MODEL_DIR" \
      --mmproj --outtype f16 --outfile "$PROJECTOR.partial"
    mv -- "$PROJECTOR.partial" "$PROJECTOR"
  fi
}

run() {
  need nvidia-smi; need python3
  [[ -x "$BINARY" ]] || die "请先执行 setup。"
  [[ -s "$MODEL" ]] || die "请先执行 convert。"
  # 用 nvidia-smi 的 index 查 UUID，避免 CUDA 枚举顺序与物理编号混淆。
  export CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=$(python3 - "$GPU_IDS" <<'PY'
import csv
import subprocess
import sys

output = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
)
lookup = {}
for row in csv.reader(output.splitlines()):
    index, uuid = (item.strip() for item in row)
    lookup[index] = uuid
    lookup[uuid] = uuid
selected = [item.strip() for item in sys.argv[1].split(",")]
if any(item not in lookup for item in selected):
    raise SystemExit("GPU_IDS 包含不存在的编号或 UUID，请执行 doctor 检查。")
uuids = [lookup[item] for item in selected]
if len(set(uuids)) != len(uuids):
    raise SystemExit("GPU_IDS 包含重复显卡。")
print(",".join(uuids))
PY
)
  echo "选定 GPU $CUDA_VISIBLE_DEVICES"
  local split
  split=$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{for(i=1;i<=NF;i++) printf "%s1",i==1?"":","}')
  local args=(
    --model "$MODEL" --alias qwen3.8-27b
    --n-gpu-layers all --split-mode layer --tensor-split "$split"
    --ctx-size "$CTX_SIZE" --parallel 1 --batch-size 512 --ubatch-size 128
    --cache-type-k f16 --cache-type-v f16 --flash-attn "$FLASH_ATTN"
    --jinja --temp 1.0 --top-p 0.95 --top-k 20 --metrics
    --host "$HOST" --port "$PORT"
  )
  if [[ $VISION == 1 ]]; then
    [[ -s "$PROJECTOR" ]] || die "缺少视觉模块，请执行 convert。"
    args+=(--mmproj "$PROJECTOR")
  fi
  exec "$BINARY" "${args[@]}"
}

start() {
  need tmux
  mkdir -p "$STATE"
  if tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "后台会话已存在，请执行 status 或 logs。"
    return
  fi
  # tmux 使用独立进程；日志在每次启动时重建。
  tmux new-session -d -s "$SESSION" "bash $(printf '%q' "$ROOT/llm.sh") run > $(printf '%q' "$STATE/server.log") 2>&1"
  echo "已提交后台启动。模型加载可能需要数分钟，执行 bash llm.sh logs 查看。"
  echo "服务地址 http://$HOST:$PORT，API 路径 /v1，监控路径 /metrics"
}

status() {
  if command -v tmux >/dev/null && tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "后台会话存在。"
  else
    echo "未发现后台会话，若启动失败请查看 logs。"
  fi
  need curl
  local health_host=$HOST
  [[ $health_host != 0.0.0.0 ]] || health_host=127.0.0.1
  if curl --fail --silent --show-error --max-time 5 "http://$health_host:$PORT/health"; then
    echo; echo "HTTP 健康检查通过。"
  else
    echo "服务可能正在加载或已退出，请检查日志。"
  fi
  if command -v nvidia-smi >/dev/null; then nvidia-smi; fi
}

case ${1:-help} in
  help|--help|-h) help_text ;;
  doctor) doctor ;;
  setup) setup ;;
  convert) convert ;;
  deploy) need tmux; setup; convert; start ;;
  run) run ;;
  start) start ;;
  status) status ;;
  logs) [[ -f "$STATE/server.log" ]] || die "暂无后台日志。"; tail -n 100 -F "$STATE/server.log" ;;
  stop)
    need tmux
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
      tmux send-keys -t "=$SESSION" C-c
      echo "已发送停止信号，请通过 status 确认退出。"
    else
      echo "没有后台会话。"
    fi
    ;;
  *) help_text; exit 1 ;;
esac
