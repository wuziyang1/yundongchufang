"""文本向量化：调用 OpenAI 兼容的 embeddings 接口（默认智谱 embedding-3）。"""

from __future__ import annotations

import time

from openai import OpenAI

from .config import Settings
from .logger import get_logger

logger = get_logger()

_MAX_ATTEMPTS = 4


class EmbeddingTool:
    def __init__(self, settings: Settings):
        if not settings.embedding_api_key:
            raise ValueError(
                "未配置嵌入密钥：请在 rag/.env 填 EMBEDDING_API_KEY，或在 backend/.env 填 ZHIPU_API_KEY"
            )
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._send_dimensions = settings.embedding_send_dimensions
        self._batch_size = settings.embedding_batch_size
        self._client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            timeout=60,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """按批向量化，返回与 texts 一一对应的向量。任何一批最终失败都会抛出异常。"""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        kwargs = {"model": self.model, "input": batch}
        if self._send_dimensions:
            kwargs["dimensions"] = self.dimensions

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._client.embeddings.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 —— 限流、网络抖动都值得重试
                if attempt == _MAX_ATTEMPTS:
                    raise
                wait = 2**attempt
                logger.warning("嵌入请求失败（%s），%d 秒后重试第 %d 次", exc, wait, attempt + 1)
                time.sleep(wait)

        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in ordered]
        if len(vectors) != len(batch):
            raise RuntimeError(f"嵌入接口返回 {len(vectors)} 条向量，与请求的 {len(batch)} 条不符")
        for vector in vectors:
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"嵌入向量是 {len(vector)} 维，与配置的 EMBEDDING_DIMENSIONS={self.dimensions} 不符"
                )
        return vectors
