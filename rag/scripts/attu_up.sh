#!/usr/bin/env bash
# 启动 Attu（Milvus 可视化管理界面），并连接本项目的 Milvus。
# 注意：Attu 大版本需与 Milvus 对齐——本项目 Milvus 为 v2.5.16，用 attu v2.5.x；
#       attu v3.0+ 只支持 Milvus 2.6/3.x，连不上 2.5。
# 启动后浏览器打开 http://localhost:3000 ，连接地址填 host.docker.internal:19530，数据库选 yundongchufang。
# 用法：bash rag/scripts/attu_up.sh
set -euo pipefail

if docker ps --format '{{.Names}}' | grep -qx attu; then
  echo "Attu 已在运行：http://localhost:3000"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx attu; then
  docker start attu
else
  docker run -d --name attu -p 3000:3000 \
    -e MILVUS_URL=host.docker.internal:19530 \
    -v attu-data:/data \
    zilliz/attu:v2.5.12
fi

echo "Attu 启动中：http://localhost:3000 （连 host.docker.internal:19530，库名 yundongchufang）"
