#!/usr/bin/env bash
# 启动本地开发服务器，并在前端源码更新时自动刷新静态构建产物。
set -euo pipefail

cd "$(dirname "$0")/.."
# 仅在静态入口缺失或前端入口较新时构建，缩短重复启动时间。
if [[ ! -f packages/core/app/static/index.html || frontend/index.html -nt packages/core/app/static/index.html ]]; then
  (cd frontend && pnpm build)
fi
PROFILE="${1:-offline}"
case "$PROFILE" in online|offline) ;; *) echo "Usage: $0 [online|offline]" >&2; exit 2 ;; esac
# An accepted local LoRA deployment owns the offline gateway and app together.
if [[ "$PROFILE" == offline && -f experiments/qwen3_bonus/artifacts/local-deployment/acceptance.json ]]; then
  exec "${APP_PYTHON:-.venv/bin/python}" -m experiments.qwen3_bonus.local_stack run
fi
exec "${APP_PYTHON:-.venv/bin/python}" -m "apps.$PROFILE.main"
