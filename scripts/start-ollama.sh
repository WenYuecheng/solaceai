#!/usr/bin/env bash
# 探测并启动本地 Ollama 服务；已有健康实例时保持幂等并直接退出。
set -euo pipefail

DEFAULT_OLLAMA_BIN="$(command -v ollama || true)"
# 兼容未把 Ollama CLI 加入 PATH 的 macOS 应用安装。
if [ -z "$DEFAULT_OLLAMA_BIN" ] && [ -x "/Applications/Ollama.app/Contents/Resources/ollama" ]; then
  DEFAULT_OLLAMA_BIN="/Applications/Ollama.app/Contents/Resources/ollama"
fi
OLLAMA_BIN="${OLLAMA_BIN:-$DEFAULT_OLLAMA_BIN}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Cannot find Ollama."
  echo "Install Ollama or set OLLAMA_BIN to the ollama executable path."
  exit 1
fi

# 标签接口成功即可证明服务已可用，无需重复启动守护进程。
if curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already running at $OLLAMA_BASE_URL"
  exit 0
fi

echo "Starting Ollama at $OLLAMA_HOST ..."
# 使用 exec 保留正确的信号传播和退出状态。
exec env OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" serve
