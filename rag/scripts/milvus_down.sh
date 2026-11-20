#!/usr/bin/env bash
# 停止知识库用的 Milvus 容器。向量数据保留在 rag/milvus/volumes/，下次启动仍在。
# 用法：bash rag/scripts/milvus_down.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_ARGS=()
if [ -f "$ROOT/.env" ]; then
  ENV_ARGS=(--env-file "$ROOT/.env")
fi

docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$ROOT/milvus/docker-compose.yml" down
