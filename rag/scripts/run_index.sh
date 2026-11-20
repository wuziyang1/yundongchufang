#!/usr/bin/env bash
# 知识库离线入库：扫描 rag/knowledge 下的 PDF → 解析/OCR → 切片 → 向量化 → 写入 Milvus。
# 参数原样传给 indexer.main，例如：
#   bash rag/scripts/run_index.sh                        # backfill（默认）：只处理新增或有变化的文件
#   bash rag/scripts/run_index.sh --mode full-refresh    # 全部重新入库
#   bash rag/scripts/run_index.sh --mode rebuild         # 清空集合后全量入库
#   bash rag/scripts/run_index.sh --mode dry-run         # 只解析切片，预览写到 rag/cache/preview/
#   bash rag/scripts/run_index.sh --mode stats
#   bash rag/scripts/run_index.sh --mode query --text "青少年每天应该运动多久"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec uv run --frozen python -u -m indexer.main "$@"
