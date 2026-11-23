"""重排：给「问题 — 候选切片」逐对打相关性分，用来判断检索回来的内容是否真的回答了问题。

向量相似度只能说明「话题相近」，不能说明「有答案」：知识库里没有踝关节扭伤的内容时，
讲热身的段落和讲跳跃练习的段落照样排在前面，相似度还不低。重排模型把问题和段落放在一起读，
分数的区分度好得多，可以设阈值把「没有针对性资料」识别出来。

接口是智谱 /rerank 的格式（Cohere / Jina 风格，多数重排服务兼容）：
    POST {base}/rerank  {"model", "query", "documents", "top_n", "return_raw_scores": true}
    → {"results": [{"index": 0, "relevance_score": 18.4}, ...]}
取原始分（return_raw_scores）：归一化后的分数在相关段落上全部贴近 1，分不出高低。

重排服务不可用时返回 None，由上层退回向量相似度把关；失败后 5 分钟内不再重试。
"""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# 单条候选送去重排的最大字符数：切片上限约 600 折算字数，英文切片约 1800 字符
_DOC_MAX_CHARS = 2000
_TIMEOUT = 10.0
_COOLDOWN = 300.0

_lock = threading.Lock()
_disabled_until = 0.0
_last_error: str | None = None


def available() -> bool:
    return (
        settings.kb_rerank_enabled
        and bool(settings.rerank_api_key)
        and time.monotonic() >= _disabled_until
    )


def problem() -> str | None:
    """重排当前不可用的原因；None 表示可用。"""
    if not settings.kb_rerank_enabled:
        return "已关闭（KB_RERANK_ENABLED=false）"
    if not settings.rerank_api_key:
        return "未配置密钥（RERANK_API_KEY / EMBEDDING_API_KEY / ZHIPU_API_KEY）"
    if time.monotonic() < _disabled_until:
        return _last_error or "重排服务暂不可用"
    return None


@lru_cache(maxsize=1)
def _http() -> httpx.Client:
    return httpx.Client(timeout=_TIMEOUT)


def scores(question: str, documents: list[str]) -> list[float] | None:
    """返回与 documents 一一对应的相关性原始分；重排不可用时返回 None。"""
    global _disabled_until, _last_error
    if not documents:
        return []
    if not available():
        return None
    try:
        response = _http().post(
            settings.rerank_base_url.rstrip("/") + "/rerank",
            headers={"Authorization": f"Bearer {settings.rerank_api_key}"},
            json={
                "model": settings.rerank_model,
                "query": question,
                "documents": [document[:_DOC_MAX_CHARS] for document in documents],
                "top_n": len(documents),
                "return_documents": False,
                "return_raw_scores": True,
            },
        )
        response.raise_for_status()
        results = response.json()["results"]
        ordered = [0.0] * len(documents)
        seen = 0
        for item in results:
            ordered[int(item["index"])] = float(item["relevance_score"])
            seen += 1
        if seen != len(documents):
            raise ValueError(f"返回 {seen} 条结果，与请求的 {len(documents)} 条不符")
        return ordered
    except Exception as exc:  # noqa: BLE001 —— 重排只是把关手段，失败退回向量相似度
        message = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__
        with _lock:
            # 一份处方的各个主题并发重排，同时失败时只记一次日志
            first = time.monotonic() >= _disabled_until
            _disabled_until = time.monotonic() + _COOLDOWN
            _last_error = f"重排请求失败：{message}"
        if first:
            logger.warning("%s；%d 秒内退回向量相似度把关", _last_error, int(_COOLDOWN))
        return None
