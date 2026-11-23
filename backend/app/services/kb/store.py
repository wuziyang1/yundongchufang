"""知识库（Milvus）访问层：连接与结构检查、查询向量化、向量检索、关键词检索、标量查询。

知识库由项目根目录的 rag/ 离线入库。这里不关心业务，只提供四种取数方式，都是同步函数，
由上层放到线程里调用并统一加超时：

- dense_search    向量相似度（COSINE），找语义相近的切片；
- keyword_search  Milvus 内置 BM25，找字面包含查询词的切片（「哮喘」「肥胖」「仰卧起坐」这类词
                  光靠向量相似度召回不稳）；
- query           按条件取数（评分表、动作条目），不涉及相似度；
- info            库里有哪些文档、各多少切片，以及知识库指纹（文档内容变了指纹就变）。

Milvus 连不上时进入 30 秒冷却，冷却期内直接报不可用，避免每个请求都卡在连接超时上。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from functools import lru_cache
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

VECTOR_FIELD = "content_vector"
SPARSE_FIELD = "content_sparse"
# 在线检索依赖的字段；集合缺少它们说明是旧结构，需要用 rag 的 rebuild 模式重建
_REQUIRED_FIELDS = ("doc_kind", "stages", "meta", "search_text", SPARSE_FIELD)

CHUNK_FIELDS = [
    "doc_title",
    "doc_kind",
    "section",
    "page_start",
    "page_end",
    "chunk_type",
    "content",
]

_CONNECT_TIMEOUT = 3.0
_RETRY_COOLDOWN = 30.0
_INFO_TTL = 60.0


class KnowledgeBaseUnavailable(RuntimeError):
    """知识库不可用：未配置、Milvus 连不上、集合不存在或结构不符。"""


# ---------------------------------------------------------------------------
# 连接
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_client: Any = None
_unavailable_until = 0.0
_last_error: str | None = None
_info_cache: tuple[float, dict[str, Any]] | None = None
_query_cache: dict[str, list[float]] = {}


def config_problem() -> str | None:
    """配置层面不能检索的原因；None 表示可以尝试连接。"""
    if not settings.embedding_api_key:
        return "未配置嵌入密钥（EMBEDDING_API_KEY / ZHIPU_API_KEY）"
    try:
        import pymilvus  # noqa: F401
    except ImportError:
        return "未安装 pymilvus"
    return None


def rpc_options() -> dict[str, int]:
    """pymilvus 的 RPC 失败默认重试 75 次（约 3.5 分钟），且只有整数 timeout 才会限制重试时长。

    Milvus 中途停掉时，检索线程会被这些重试一直占着；这里限定为整数秒超时、最多重试 2 次。
    """
    return {"timeout": max(1, int(settings.kb_timeout // 2)), "retry_times": 2}


def _short(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text[:200]


def client() -> Any:
    """返回可用的 MilvusClient；连不上时进入冷却期，冷却期内直接报不可用。"""
    global _client, _unavailable_until, _last_error
    problem = config_problem()
    if problem is not None:
        raise KnowledgeBaseUnavailable(problem)
    with _lock:
        if _client is not None:
            return _client
        if time.monotonic() < _unavailable_until:
            raise KnowledgeBaseUnavailable(_last_error or "Milvus 不可用")

        from pymilvus import MilvusClient

        auth: dict[str, str] = {}
        if settings.milvus_user:
            auth = {"user": settings.milvus_user, "password": settings.milvus_password}
        try:
            connected = MilvusClient(
                uri=settings.milvus_uri,
                db_name=settings.milvus_database,
                timeout=min(_CONNECT_TIMEOUT, settings.kb_timeout),
                **auth,
            )
            _check_collection(connected)
        except KnowledgeBaseUnavailable as exc:
            _unavailable_until = time.monotonic() + _RETRY_COOLDOWN
            _last_error = str(exc)
            raise
        except Exception as exc:  # noqa: BLE001 —— 任何连接问题都按不可用处理
            _unavailable_until = time.monotonic() + _RETRY_COOLDOWN
            _last_error = f"连不上 Milvus（{settings.milvus_uri}）：{_short(exc)}。请先启动：bash rag/scripts/milvus_up.sh"
            raise KnowledgeBaseUnavailable(_last_error) from exc

        _client, _last_error = connected, None
        return connected


def _check_collection(connected: Any) -> None:
    """确认集合存在、是当前结构、向量维度和嵌入模型与在线配置一致。"""
    collection = settings.milvus_collection
    rpc = rpc_options()
    if not connected.has_collection(collection, **rpc):
        raise KnowledgeBaseUnavailable(
            f"数据库 {settings.milvus_database} 中没有集合 {collection}，请先入库：bash rag/scripts/run_index.sh"
        )
    fields = {field["name"]: field for field in connected.describe_collection(collection, **rpc)["fields"]}
    missing = [name for name in _REQUIRED_FIELDS if name not in fields]
    if missing:
        raise KnowledgeBaseUnavailable(
            f"集合 {collection} 是旧结构（缺少 {'、'.join(missing)}），"
            "请重建：bash rag/scripts/run_index.sh --mode rebuild"
        )
    dim = int(fields[VECTOR_FIELD]["params"]["dim"])
    if dim != settings.embedding_dimensions:
        raise KnowledgeBaseUnavailable(
            f"集合向量是 {dim} 维，与 EMBEDDING_DIMENSIONS={settings.embedding_dimensions} 不一致"
        )
    connected.load_collection(collection, **rpc)
    sample = connected.query(
        collection, filter="chunk_index == 0", output_fields=["embedding_model"], limit=1, **rpc
    )
    if not sample:
        raise KnowledgeBaseUnavailable(f"集合 {collection} 是空的，请先入库：bash rag/scripts/run_index.sh")
    model = sample[0]["embedding_model"]
    if model and model != settings.embedding_model:
        raise KnowledgeBaseUnavailable(
            f"知识库用 {model} 入库，与当前 EMBEDDING_MODEL={settings.embedding_model} 不一致，向量不可比"
        )


def _reset(error: str) -> None:
    """检索出错：丢掉客户端并进入冷却期，冷却期过后重新连接。"""
    global _client, _last_error, _info_cache, _unavailable_until
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # noqa: BLE001
                pass
        _client = None
        _last_error = error
        _info_cache = None
        _unavailable_until = time.monotonic() + _RETRY_COOLDOWN


def _call(action: str, func: Any) -> Any:
    try:
        return func()
    except KnowledgeBaseUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 —— 连接可能已失效（Milvus 停止或重启）
        message = f"知识库{action}失败：{_short(exc)}"
        _reset(message)
        raise KnowledgeBaseUnavailable(message) from exc


# ---------------------------------------------------------------------------
# 查询向量化
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _embed_client():
    from openai import OpenAI

    return OpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        timeout=settings.kb_timeout,
        max_retries=1,
    )


_EMBED_BATCH = 16


def embed(texts: list[str]) -> list[list[float]]:
    """查询文本向量化：与入库一致，embedding-3 必须显式下发 dimensions 才是 1024 维。带进程内缓存。"""
    missing = list(dict.fromkeys(text for text in texts if text not in _query_cache))
    if missing and len(_query_cache) > 2048:
        _query_cache.clear()
        missing = list(dict.fromkeys(texts))
    for start in range(0, len(missing), _EMBED_BATCH):
        batch = missing[start : start + _EMBED_BATCH]
        try:
            response = _embed_client().embeddings.create(
                model=settings.embedding_model,
                input=batch,
                dimensions=settings.embedding_dimensions,
            )
        except Exception as exc:  # noqa: BLE001
            raise KnowledgeBaseUnavailable(f"查询向量化失败：{_short(exc)}") from exc
        for item in sorted(response.data, key=lambda d: d.index):
            _query_cache[batch[item.index]] = item.embedding
    return [_query_cache[text] for text in texts]


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------


def quote(value: str) -> str:
    """把字符串写进过滤表达式：转义反斜杠与双引号。"""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _flatten(hits: Any) -> list[dict[str, Any]]:
    return [{"id": hit["id"], "score": float(hit["distance"]), **hit["entity"]} for hit in hits]


def dense_search(
    texts: list[str],
    *,
    filter: str = "",
    limit: int = 10,
    output_fields: list[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """向量相似度检索：每条查询返回一组命中，score 为 COSINE 相似度。"""
    if not texts:
        return []
    connected = client()
    vectors = embed(texts)
    results = _call(
        "向量检索",
        lambda: connected.search(
            settings.milvus_collection,
            data=vectors,
            anns_field=VECTOR_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit * 2)}},
            filter=filter,
            limit=limit,
            output_fields=output_fields or CHUNK_FIELDS,
            **rpc_options(),
        ),
    )
    return [_flatten(hits) for hits in results]


def keyword_search(
    texts: list[str],
    *,
    filter: str = "",
    limit: int = 10,
    output_fields: list[str] | None = None,
) -> list[list[dict[str, Any]]]:
    """关键词检索（BM25）：查询文本由 Milvus 按入库时的分词器切词，score 为 BM25 得分。"""
    if not texts:
        return []
    connected = client()
    results = _call(
        "关键词检索",
        lambda: connected.search(
            settings.milvus_collection,
            data=texts,
            anns_field=SPARSE_FIELD,
            search_params={"metric_type": "BM25"},
            filter=filter,
            limit=limit,
            output_fields=output_fields or CHUNK_FIELDS,
            **rpc_options(),
        ),
    )
    return [_flatten(hits) for hits in results]


def query(filter: str, output_fields: list[str]) -> list[dict[str, Any]]:
    """按条件取出全部符合的切片（评分表、动作条目等），不涉及相似度。"""
    connected = client()

    def run() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        iterator = connected.query_iterator(
            settings.milvus_collection,
            filter=filter,
            output_fields=output_fields,
            batch_size=500,
            **rpc_options(),
        )
        try:
            while batch := iterator.next():
                rows.extend(batch)
        finally:
            iterator.close()
        return rows

    return _call("查询", run)


# ---------------------------------------------------------------------------
# 库的概况
# ---------------------------------------------------------------------------


def info() -> dict[str, Any]:
    """库里的文档清单与知识库指纹，结果缓存 60 秒。

    指纹由各文档的内容 MD5 算出：增删或替换任何一份资料，指纹都会变。它随处方一起留档，
    用来回答「这份处方是基于哪一版知识库生成的」；按指纹缓存的评分表、动作目录也据此失效。
    """
    global _info_cache
    now = time.monotonic()
    if _info_cache is not None and now - _info_cache[0] < _INFO_TTL:
        return _info_cache[1]

    rows = query("", ["doc_title", "doc_kind", "doc_md5", "stages", "chunk_type"])
    documents: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = documents.setdefault(
            row["doc_title"],
            {
                "title": row["doc_title"],
                "kind": row.get("doc_kind", ""),
                "stages": list(row.get("stages") or []),
                "md5": row["doc_md5"],
                "chunks": 0,
                "exercises": 0,
            },
        )
        item["chunks"] += 1
        if row["chunk_type"] == "exercise":
            item["exercises"] += 1
    ordered = [documents[title] for title in sorted(documents)]
    digest = hashlib.md5("|".join(f"{doc['title']}:{doc['md5']}" for doc in ordered).encode()).hexdigest()
    result = {
        "collection": settings.milvus_collection,
        "fingerprint": digest[:12],
        "chunk_count": len(rows),
        "exercise_count": sum(doc["exercises"] for doc in ordered),
        "embedding_model": settings.embedding_model,
        "documents": ordered,
    }
    _info_cache = (now, result)
    return result


def status() -> dict[str, Any]:
    """/meta 用的知识库状态。连不上时给出原因，不抛异常。"""
    base = {
        "milvus_uri": settings.milvus_uri,
        "database": settings.milvus_database,
        "collection": settings.milvus_collection,
    }
    try:
        return {**base, "ready": True, "problem": None, **info()}
    except KnowledgeBaseUnavailable as exc:
        return {**base, "ready": False, "problem": str(exc)}
