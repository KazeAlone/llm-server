#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$BACKEND_DIR/.." && pwd)"
# 保留历史模型、编译产物、tmux 会话及根目录配置。
ENV_FILE="$ROOT/.env"
[[ ! -f "$BACKEND_DIR/.env" ]] || ENV_FILE="$BACKEND_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  # .env 是本机用户维护的 Shell 配置。
  source "$ENV_FILE"
fi
if [[ ${1:-} == ui ]]; then
  HOST=${HOST:-127.0.0.1}; PORT=${PORT:-8080}
  [[ $HOST != 0.0.0.0 ]] || HOST=127.0.0.1
  export API_KEY=${API_KEY:-${LLAMA_API_KEY:-}}
  exec "${CONDA_PREFIX:+$CONDA_PREFIX/bin/}python3" "$ROOT/common/console.py" --upstream "http://$HOST:$PORT" --host "${UI_HOST:-127.0.0.1}" --port "${UI_PORT:-$(( PORT + 100 ))}"
fi
MODEL_DIR=${MODEL_DIR:-}
GPU_IDS=${GPU_IDS:-0,1,2,3}
CTX_SIZE=${CTX_SIZE:-32768}
PORT=${PORT:-8080}
HOST=${HOST:-127.0.0.1}
API_KEY=${API_KEY:-${LLAMA_API_KEY:-}}
PRECISION=${PRECISION:-f16}
VISION=${VISION:-0}
MEDIA_PATH=${MEDIA_PATH-/}
FLASH_ATTN=${FLASH_ATTN:-auto}
BATCH_SIZE=${BATCH_SIZE:-512}
UBATCH_SIZE=${UBATCH_SIZE:-128}
REASONING_EFFORT=${REASONING_EFFORT:-}
BUILD_JOBS=${BUILD_JOBS:-8}
PYTHON_BIN=${PYTHON_BIN:-python3}
PYTHON_ENV=${PYTHON_ENV:-conda}
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
case $REASONING_EFFORT in
  ''|low|medium|xhigh) ;;
  *) die "REASONING_EFFORT 只能为空、low、medium 或 xhigh" ;;
esac
[[ $BATCH_SIZE =~ ^[1-9][0-9]*$ && $UBATCH_SIZE =~ ^[1-9][0-9]*$ ]] \
  || die "BATCH_SIZE 和 UBATCH_SIZE 必须为正整数"
(( UBATCH_SIZE <= BATCH_SIZE )) || die "UBATCH_SIZE 不能大于 BATCH_SIZE"

# flock 的父进程持锁，子进程关闭锁描述符，避免 tmux 服务继承锁。
case ${1:-help} in
  setup|convert|deploy|start|stop|restart)
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
  diagnose 显示性能配置、GPU 状态及最近推理耗时
  deploy   编译后端、转换已有权重、后台启动（复用已有结果）
  setup    编译 sm_70 后端并建立转换环境
  convert  将 MODEL_DIR 转换为 GGUF，VISION=1 时同时转换视觉模块
  run      在当前终端运行服务，Ctrl+C 停止
  start    在 tmux 中后台启动，加载过程查看 logs
  restart  停止旧服务并按当前 .env 配置启动
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
    --ctx-size "$CTX_SIZE" --parallel 1 --batch-size "$BATCH_SIZE" --ubatch-size "$UBATCH_SIZE"
    --cache-type-k f16 --cache-type-v f16 --flash-attn "$FLASH_ATTN"
    --jinja --temp 1.0 --top-p 0.95 --top-k 20 --metrics
    --host "$HOST" --port "$PORT"
  )
  if [[ -n $REASONING_EFFORT ]]; then
    args+=(--chat-template-kwargs "{\"reasoning_effort\":\"$REASONING_EFFORT\"}")
  fi
  if [[ $VISION == 1 ]]; then
    [[ -s "$PROJECTOR" ]] || die "缺少视觉模块，请执行 convert。"
    args+=(--mmproj "$PROJECTOR")
  fi
  if [[ -n $MEDIA_PATH ]]; then
    [[ -d "$MEDIA_PATH" ]] || die "MEDIA_PATH 目录不存在：$MEDIA_PATH"
    args+=(--media-path "$MEDIA_PATH")
  fi
  # llama-server 原生读取该环境变量，避免把密钥放入命令行参数。
  export LLAMA_API_KEY="$API_KEY"
  exec "$BINARY" "${args[@]}"
}

diagnose() {
  echo "配置 PRECISION=$PRECISION GPU_IDS=$GPU_IDS CTX_SIZE=$CTX_SIZE"
  echo "批次 BATCH_SIZE=$BATCH_SIZE UBATCH_SIZE=$UBATCH_SIZE FLASH_ATTN=$FLASH_ATTN"
  echo "思考 REASONING_EFFORT=${REASONING_EFFORT:-模型默认}，并发槽位 1"
  if [[ -f "$STATE/llama-cpp-commit.txt" ]]; then cat "$STATE/llama-cpp-commit.txt"; fi
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu --format=csv
    nvidia-smi topo -m
  fi
  if [[ -f "$STATE/server.log" ]]; then
    echo "最近加载与耗时记录"
    { grep -E 'prompt eval time|eval time|offloaded|flash_attn|CUDA.*buffer|KV.*buffer' "$STATE/server.log" || true; } | tail -n 60
  else
    echo "暂无后台日志，请先通过 start 启动并完成一次请求。"
  fi
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

stop() {
  need tmux
  if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
    echo "没有后台会话。"
    return
  fi
  # send-keys 接受 pane 目标。先从本项目会话查询稳定的 pane ID。
  local panes pane
  panes=$(tmux list-panes -s -t "$SESSION" -F '#{pane_id}')
  pane=${panes%%$'\n'*}
  [[ -n $pane ]] || die "会话内没有可用 pane，请执行 tmux list-panes -a 检查。"
  if ! tmux send-keys -t "$pane" C-c; then
    if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
      echo "服务已退出。"
      return
    fi
    die "停止信号发送失败，请查看 tmux 会话。"
  fi
  local attempt dead
  for ((attempt=0; attempt<30; attempt++)); do
    if ! tmux has-session -t "=$SESSION" 2>/dev/null; then
      echo "服务已停止。"
      return
    fi
    dead=$(tmux display-message -p -t "$pane" '#{pane_dead}' 2>/dev/null || true)
    if [[ $dead == 1 ]]; then
      # 兼容用户配置 remain-on-exit 的情况，仅移除已退出的 pane。
      tmux kill-pane -t "$pane"
      echo "服务已停止。"
      return
    fi
    sleep 1
  done
  die "等待服务退出超时，未启动新服务。请查看 logs 和 tmux 会话。"
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
  diagnose) diagnose ;;
  setup) setup ;;
  convert) convert ;;
  deploy) need tmux; setup; convert; start ;;
  run) run ;;
  start) start ;;
  restart) stop; start ;;
  status) status ;;
  logs) [[ -f "$STATE/server.log" ]] || die "暂无后台日志。"; tail -n 100 -F "$STATE/server.log" ;;
  stop) stop ;;
  *) help_text; exit 1 ;;
esac
