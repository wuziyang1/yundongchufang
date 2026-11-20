#!/usr/bin/env bash
# 停止并删除 Attu 容器（attu-data 卷保留，下次启动连接记录仍在）。
# 用法：bash rag/scripts/attu_down.sh
set -euo pipefail

docker rm -f attu
echo "Attu 已停止。"
