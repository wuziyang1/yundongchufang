#!/usr/bin/env bash
# 启动知识库用的 Milvus 向量数据库（单容器，无 MinIO），并等待它就绪。
# 用法：bash rag/scripts/milvus_up.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_ARGS=()
if [ -f "$ROOT/.env" ]; then
  ENV_ARGS=(--env-file "$ROOT/.env")
fi

docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$ROOT/milvus/docker-compose.yml" up -d

echo "等待 Milvus 就绪..."
for _ in $(seq 1 60); do
  status="$(docker inspect -f '{{.State.Health.Status}}' yundongchufang-milvus 2>/dev/null || echo "missing")"
  if [ "$status" = "healthy" ]; then
    echo "Milvus 已就绪。"
    docker compose ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} -f "$ROOT/milvus/docker-compose.yml" ps
    exit 0
  fi
  sleep 5
done

echo "Milvus 在 5 分钟内未就绪，请查看日志：docker logs yundongchufang-milvus" >&2
exit 1
