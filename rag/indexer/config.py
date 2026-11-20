"""离线入库的运行配置。

读取顺序：rag/.env 优先；其中没有的项再从 backend/.env 补（主要是复用 ZHIPU_API_KEY），
这样嵌入密钥只需在一处维护。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

RAG_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = RAG_DIR.parent

load_dotenv(RAG_DIR / ".env")
load_dotenv(PROJECT_ROOT / "backend" / ".env")


def _env(name: str, default: str = "") -> str:
    """空字符串视同未配置，回落到默认值。"""
    return os.getenv(name, "").strip() or default


@dataclass(frozen=True)
class Settings:
    knowledge_dir: Path = Path(_env("RAG_KNOWLEDGE_DIR") or RAG_DIR / "knowledge").expanduser()
    cache_dir: Path = RAG_DIR / "cache"

    milvus_uri: str = f"http://{_env('MILVUS_HOST', '127.0.0.1')}:{_env('MILVUS_PORT', '19530')}"
    milvus_user: str = _env("MILVUS_USER")
    milvus_password: str = _env("MILVUS_PASSWORD")
    milvus_database: str = _env("MILVUS_DATABASE", "yundongchufang")
    milvus_collection: str = _env("MILVUS_COLLECTION", "knowledge_chunks")
    milvus_index_name: str = _env("MILVUS_INDEX_NAME", "content_vector_hnsw")

    # 必须与在线检索时给查询文本做嵌入的模型、维度一致，否则向量不可比
    embedding_api_key: str = _env("EMBEDDING_API_KEY") or _env("ZHIPU_API_KEY")
    embedding_base_url: str = _env("EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    embedding_model: str = _env("EMBEDDING_MODEL", "embedding-3")
    embedding_dimensions: int = int(_env("EMBEDDING_DIMENSIONS", "1024"))
    # 智谱 embedding-3 默认输出 2048 维，需显式传 dimensions；维度固定的模型（如 bge-m3）传了会报错，改为 false
    embedding_send_dimensions: bool = _env("EMBEDDING_SEND_DIMENSIONS", "true").lower() == "true"
    embedding_batch_size: int = int(_env("EMBEDDING_BATCH_SIZE", "16"))

    # 切片长度按「折算字数」计：汉字 1 个算 1，其余字符约 3 个算 1
    chunk_size: int = int(_env("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "80"))

    ocr_enabled: bool = _env("ENABLE_PDF_OCR", "true").lower() == "true"
    ocr_dpi: int = int(_env("OCR_DPI", "200"))


settings = Settings()
