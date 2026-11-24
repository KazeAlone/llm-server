# Source after setting BACKEND_DIR, BACKEND and DEFAULT_PORT.
set -euo pipefail
REPO="$(cd -- "$BACKEND_DIR/.." && pwd)"
[[ ! -f "$BACKEND_DIR/.env" ]] || source "$BACKEND_DIR/.env"
STATE="$BACKEND_DIR/.runtime"
SESSION="llm-$BACKEND-$(printf '%s' "$BACKEND_DIR" | sha256sum | cut -c1-10)"
MODEL_DIR=${MODEL_DIR:-}
MODEL_ID=${MODEL_ID:-qwen3.8-27b}
GPU_IDS=${GPU_IDS:-0,1,2,3}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-$DEFAULT_PORT}
API_KEY=${API_KEY:-}
CTX_SIZE=${CTX_SIZE:-8192}
VISION=${VISION:-1}
MEDIA_PATH=${MEDIA_PATH-/}
BUILD_JOBS=${BUILD_JOBS:-4}
EXTRA_ARGS=(${EXTRA_ARGS+"${EXTRA_ARGS[@]}"})
[[ -z ${CUDA_HOME:-} ]] || export PATH="$CUDA_HOME/bin:$PATH" LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
die() { echo "错误：$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "缺少 $1"; }
python_select() {
  if [[ -n ${CONDA_PREFIX:-} && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -f "$STATE/python-path" ]]; then
    PYTHON=$(cat "$STATE/python-path")
  else die "先 conda activate 环境名，再执行 setup。"; fi
  [[ -x "$PYTHON" ]] || die "Conda 解释器不存在，请重新 setup"
  export PATH="$(dirname "$PYTHON"):$PATH"
}
validate() {
  [[ $PORT =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || die "PORT 必须为 1–65535"
  [[ $CTX_SIZE =~ ^[1-9][0-9]*$ ]] || die "CTX_SIZE 必须为正整数"
  [[ $VISION == 0 || $VISION == 1 ]] || die "VISION 必须为 0 或 1"
  [[ -d "$MODEL_DIR" && -f "$MODEL_DIR/config.json" ]] || die "MODEL_DIR 必须是原始 HF 模型目录，含 config.json"
  "$PYTHON" "$REPO/common/validate_model.py" "$MODEL_DIR"
  [[ -z $MEDIA_PATH || -d $MEDIA_PATH ]] || die "MEDIA_PATH 目录不存在"
}
select_gpus() {
  export CUDA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES=$("$PYTHON" "$REPO/common/gpus.py" "$GPU_IDS")
  IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
  TP=${#GPU_ARRAY[@]}
}
setup_common() {
  [[ -n ${CONDA_PREFIX:-} ]] || die "setup 必须先激活 Conda 环境"
  python_select
  "$PYTHON" -c 'import sys; assert sys.version_info >= (3,11), "需要 Python >=3.11，推荐3.12"'
  mkdir -p "$STATE"
  "$PYTHON" "$REPO/common/deps.py" lock "$STATE/constraints.txt"
  # Absolute path inherited by every pip subprocess; incompatible dependencies fail.
  export PIP_CONSTRAINT="$STATE/constraints.txt"
  export TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS="$BUILD_JOBS" CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
  printf '%s\n' "$PYTHON" > "$STATE/python-path.pending"
}
pip_install() { "$PYTHON" "$REPO/common/deps.py" "$@"; }
setup_done() {
  "$PYTHON" -m pip check
  mv "$STATE/python-path.pending" "$STATE/python-path"
}
checkout() {
  local url=$1 rev=$2 target=$3
  need git
  local wanted current
  if [[ ! -d "$target/.git" ]]; then
    git init "$target"
    git -C "$target" remote add origin "$url"
  fi
  if wanted=$(git -C "$target" rev-parse --verify "$rev^{commit}" 2>/dev/null); then :; else
    git -C "$target" fetch --depth 1 origin "$rev"
    wanted=$(git -C "$target" rev-parse FETCH_HEAD)
  fi
  if current=$(git -C "$target" rev-parse --verify HEAD 2>/dev/null); then
    [[ $current == "$wanted" ]] || die "源码版本不匹配，请保留旧目录并改名后重新 setup：$target"
  else
    git -C "$target" checkout --detach "$wanted"
  fi
  git -C "$target" rev-parse HEAD > "$STATE/source-commit.txt"
}
check_cuda() {
  need nvcc
  nvcc --version | grep -Eq 'release 12\.(6|7|8),' || die "此源码构建需要 CUDA Toolkit 12.6–12.8；CUDA_HOME 指向实际 Toolkit"
}
start() {
  need tmux
  python_select
  [[ -f "$STATE/python-path" ]] || die "先 setup"
  [[ "$PYTHON" == "$(cat "$STATE/python-path")" ]] || die "当前 Conda 与 setup 环境不同，请激活原环境或重新 setup"
  validate
  if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "已经启动：$SESSION"; return; fi
  # tmux server may predate conda activation; pin this session's interpreter explicitly.
  local command
  printf -v command 'env CONDA_PREFIX=%q PATH=%q CUDA_HOME=%q LD_LIBRARY_PATH=%q bash %q run > %q 2>&1' "$(dirname "$(dirname "$PYTHON")")" "$PATH" "${CUDA_HOME:-}" "${LD_LIBRARY_PATH:-}" "$BACKEND_DIR/llm.sh" "$STATE/server.log"
  tmux new-session -d -s "$SESSION" "$command"
  echo "已启动 tmux: $SESSION；模型加载期间请看 bash llm.sh logs"
}
stop() {
  need tmux
  if ! tmux has-session -t "=$SESSION" 2>/dev/null; then echo "服务未运行"; return; fi
  local pane panes
  panes=$(tmux list-panes -s -t "=$SESSION" -F '#{pane_id}')
  pane=${panes%%$'\n'*}
  [[ -n $pane ]] || die "未找到服务 pane"
  tmux send-keys -t "$pane" C-c
  for ((i=0; i<30; i++)); do
    tmux has-session -t "=$SESSION" 2>/dev/null || { echo "已停止"; return; }
    if [[ $(tmux display-message -p -t "$pane" '#{pane_dead}' 2>/dev/null || true) == 1 ]]; then
      tmux kill-session -t "=$SESSION"; return
    fi
    sleep 1
  done
  die "30 秒内未退出；请检查日志及 nvidia-smi，未强杀服务"
}
status() {
  need tmux
  if tmux has-session -t "=$SESSION" 2>/dev/null; then echo "tmux 会话运行中"; else echo "服务未运行"; return 1; fi
  local address=$HOST
  [[ $address != 0.0.0.0 ]] || address=127.0.0.1
  # Key supplied through stdin config, never command-line argv.
  printf 'header = "Authorization: Bearer %s"\n' "$API_KEY" | curl --config - --fail --silent --show-error --max-time 5 "http://$address:$PORT/health"
}
doctor() {
  echo "后端 $BACKEND / 配置 $BACKEND_DIR/.env / GPU_IDS=$GPU_IDS / port=$PORT"
  nvidia-smi -L
  nvidia-smi topo -m
  nvcc --version || true
  python_select
  "$PYTHON" -c 'import sys, torch; print(sys.executable); print("Torch",torch.__version__,"CUDA",torch.version.cuda,"arch",torch.cuda.get_arch_list())'
}
dispatch() {
  local action=${1:-help}
  case $action in
    setup|deploy|start|stop|restart)
      need flock; mkdir -p "$STATE"
      if [[ ${LLM_BACKEND_LOCKED:-} != "$BACKEND_DIR" ]]; then
        exec flock --nonblock --close --conflict-exit-code 75 "$STATE/deploy.lock" env LLM_BACKEND_LOCKED="$BACKEND_DIR" bash "$BACKEND_DIR/llm.sh" "$@"
      fi ;;
  esac
  case $action in
    setup) setup_backend ;;
    deploy) setup_backend; start ;;
    run) python_select; validate; select_gpus; run_backend ;;
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    ui)
      python_select
      address=$HOST; [[ $address != 0.0.0.0 ]] || address=127.0.0.1
      export API_KEY
      exec "$PYTHON" "$REPO/common/console.py" --upstream "http://$address:$PORT" --host "${UI_HOST:-127.0.0.1}" --port "${UI_PORT:-$(( PORT + 100 ))}"
      ;;
    doctor) doctor ;;
    diagnose) doctor; [[ ! -f "$STATE/server.log" ]] || tail -n 70 "$STATE/server.log" ;;
    logs) tail -n 100 -F "$STATE/server.log" ;;
    *) echo "用法 bash llm.sh {doctor|setup|deploy|run|start|stop|restart|status|logs|diagnose|ui}" ;;
  esac
}
