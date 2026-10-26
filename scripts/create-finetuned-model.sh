#!/usr/bin/env bash
# 将微调后的 GGUF 权重注册为本地 Ollama 模型（历史 Qwen2.5 基线方案；Qwen3 双 LoRA 方案请使用 experiments/qwen3_bonus 内的 local_stack 启动）。
set -euo pipefail

# 所有路径均可通过环境变量覆盖，默认值适用于仓库内置模型布局。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${FINETUNED_MODEL_NAME:-solaceai-qwen2.5-7b-ft:latest}"
MODEL_DIR="${FINETUNED_MODEL_DIR:-$ROOT_DIR/models/solaceai-qwen2.5-7b-ft}"
GGUF_FILE="${FINETUNED_MODEL_FILE:-solaceai-qwen2.5-7b-ft-q4_k_m.gguf}"
UPSTREAM_GGUF="${UPSTREAM_GGUF:-}"

DEFAULT_OLLAMA_BIN="$(command -v ollama || true)"
# macOS 应用安装可能不会把 Ollama 加入 PATH，因此补充标准应用路径探测。
if [ -z "$DEFAULT_OLLAMA_BIN" ] && [ -x "/Applications/Ollama.app/Contents/Resources/ollama" ]; then
  DEFAULT_OLLAMA_BIN="/Applications/Ollama.app/Contents/Resources/ollama"
fi
OLLAMA_BIN="${OLLAMA_BIN:-$DEFAULT_OLLAMA_BIN}"

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Cannot find Ollama."
  echo "Install Ollama or set OLLAMA_BIN to the ollama executable path."
  exit 1
fi

mkdir -p "$MODEL_DIR"

# 权重不存在时可链接上游文件，避免复制体积很大的 GGUF；否则给出可操作错误。
if [ ! -f "$MODEL_DIR/$GGUF_FILE" ]; then
  if [ -n "$UPSTREAM_GGUF" ] && [ -f "$UPSTREAM_GGUF" ]; then
    echo "Linking GGUF from upstream SolaceAI project..."
    ln -sf "$UPSTREAM_GGUF" "$MODEL_DIR/$GGUF_FILE"
  else
    echo "Missing GGUF model file:"
    echo "  $MODEL_DIR/$GGUF_FILE"
    echo
    echo "Put the model file there, or set UPSTREAM_GGUF to an existing GGUF path."
    exit 1
  fi
fi

# Modelfile 负责绑定权重和运行参数，create 会将其注册到本机 Ollama。
"$OLLAMA_BIN" create "$MODEL_NAME" -f "$MODEL_DIR/Modelfile"

echo "Created $MODEL_NAME"
echo "Run SolaceAI Python with: AI_PROVIDER=ollama ./scripts/run-dev.sh"
