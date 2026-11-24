#!/usr/bin/env bash
# 兼容历史入口，原 .env/.runtime/tmux 会话继续使用。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/llama.cpp/llm.sh" "$@"
