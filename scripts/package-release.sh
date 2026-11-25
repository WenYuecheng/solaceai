#!/usr/bin/env bash
# 生成不含运行数据、凭据、缓存和大型模型权重的可部署源码包。
set -euo pipefail

# 使用时间戳隔离暂存目录和输出文件，便于保留多版发布包。
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="solaceai"
STAMP="$(date +%Y%m%d-%H%M%S)"
DIST_DIR="$ROOT_DIR/dist"
STAGE_DIR="$DIST_DIR/stage-$STAMP"
ARCHIVE="$DIST_DIR/${PROJECT_NAME}-app-$STAMP.tar.gz"

mkdir -p "$DIST_DIR"
# 暂存目录名称由当前时间戳精确解析后再删除，避免触及其他发布产物。
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR/$PROJECT_NAME"

# 同步白名单范围内的项目文件；以下排除项同时控制包体积和敏感信息泄漏风险。
rsync -a "$ROOT_DIR/" "$STAGE_DIR/$PROJECT_NAME/" \
  --exclude '.app-compat-pending-removal/' \
  --exclude '.venv-online/' \
  --exclude '.venv-offline/' \
  --exclude 'models/docling-layout-heron/' \
  --exclude 'experiments/' \
  --exclude 'qwen3_bonus/' \
  --exclude '**/*.egg-info/' \
  --exclude '**/build/' \
  --exclude '.git/' \
  --exclude '.github-release/' \
  --exclude 'archive/' \
  --exclude '毕设/' \
  --exclude '求职面试/' \
  --exclude 'resume/' \
  --exclude '.venv/' \
  --exclude '.venv-rag/' \
  --exclude '.venv-eval/' \
  --exclude '.deploy-private/' \
  --exclude 'models/bge-m3/' \
  --exclude 'models/bge-reranker-large/' \
  --exclude 'frontend/node_modules/' \
  --exclude 'frontend/dist/' \
  --exclude '.uploads/' \
  --exclude '.idea/' \
  --exclude '.vscode/' \
  --exclude '.env' \
  --include '.env.example' \
  --exclude '.env.*' \
  --exclude '.env.local' \
  --exclude '.env.production' \
  --exclude '.env.development' \
  --exclude '.env.test' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'target/' \
  --exclude 'dist/' \
  --exclude 'deploy/.site-next/' \
  --exclude 'deploy/.site-previous/' \
  --exclude 'data/' \
  --exclude 'output/' \
  --exclude 'tmp/' \
  --exclude '.DS_Store' \
  --exclude '*.log' \
  --exclude '*.pem' \
  --exclude '*.key' \
  --exclude '*secret*' \
  --exclude '*token*.env' \
  --exclude '*token*.json' \
  --exclude '*token*.txt' \
  --exclude '*.db' \
  --exclude '*.sqlite' \
  --exclude '*.sqlite3' \
  --exclude '*.xlsx' \
  --exclude '*.gguf' \
  --exclude '*.gguf.zip' \
  --exclude '*.zip' \
  --exclude '*.tar.gz'

(
  # 固定归档所有者并移除 macOS 扩展属性，使不同机器生成的包结构一致。
  cd "$STAGE_DIR"
  COPYFILE_DISABLE=1 tar \
    --no-xattrs \
    --uid 0 --gid 0 \
    --uname root --gname root \
    -czf "$ARCHIVE" "$PROJECT_NAME"
)

rm -rf "$STAGE_DIR"

# 用一个必须存在的前端构建文件做发布包完整性哨兵检查。
tar -tzf "$ARCHIVE" | grep -q "^$PROJECT_NAME/frontend/src/styles/tokens.css$"

echo "Created release package:"
echo "$ARCHIVE"
echo "Model GGUF is intentionally excluded. Send the model zip separately."
du -sh "$ARCHIVE"
