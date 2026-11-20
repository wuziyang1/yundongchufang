"""权威资料检索：从 Milvus 知识库检索与学生情况相关的原文摘录，补充进处方 prompt。

知识库由项目根目录的 rag/ 离线入库：国家学生体质健康标准、中国儿童青少年身体活动指南、
高中体育与健康课程标准、青少年健康体能教学与训练指导手册、NSCA / ACSM 青少年抗阻训练文献，
解析切片后连同出处（文档标题、章节路径、页码）存在 Milvus 集合里。

定位是补充，不替代现有逻辑：知识卡片（knowledge.json）与动作库的结构化字段仍承担禁忌过滤、
动作 ID 校验和前端展示；这里检索到的原文以「权威资料摘录」注入处方 prompt，每条带出处
「《文档标题》章节 第 N 页」，供模型确定训练量、强度与注意事项时参考。

检索：按学生情况拆成几条短查询（总体推荐量、每项慢病/伤病/症状各一条、抗阻训练剂量），
一次嵌入、一次向量检索；合并后按相似度阈值过滤，每条健康状况查询保底一条，同一文档条数设上限。

降级：开关关闭、mock 模式、无嵌入密钥、未安装 pymilvus、Milvus 未启动、检索超时或出错，
一律返回空列表并记一条 warning，绝不拖垮处方生成主链路。Milvus 连不上时 30 秒内不再重试，
避免每次请求都卡在连接超时上。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from functools import lru_cache
from typing import Any

from app.config import settings
from app.services.rag import _record_grade, _school_stage, _to_int

logger = logging.getLogger(__name__)

_VECTOR_FIELD = "content_vector"
_OUTPUT_FIELDS = [
    "doc_title",
    "section",
    "page_start",
    "page_end",
    "chunk_type",
    "content",
    "embedding_model",
]
# 每条查询先多取一些，合并去重后再按阈值、单文档上限截断
_PER_QUERY_LIMIT = 8
_MAX_QUERIES = 6
# 挑选时各类查询的先后与贡献上限：健康状况最优先；抗阻剂量的英文查询与英文文献相似度普遍偏高，只取 1 条
_QUERY_PRIORITY = {"health": 0, "general": 1, "dosage": 2}
_QUERY_CAP = {"health": 2, "general": 2, "dosage": 1}
# 抗阻训练的组数、次数、强度主要来自英文文献（NSCA / ACSM），用英文查询召回更准；
# 不带性别年龄：带上「girls」会把讲青春期激素差异的段落排到讲组数次数的段落前面
_DOSAGE_QUERY = (
    "resistance training guidelines for children and adolescents: "
    "number of sets and repetitions, load, rest, frequency per week"
)
# 摘录正文的长度上限（折算字数：汉字算 1，其余字符约 3 个算 1），超出按句截断
_EXCERPT_MAX_UNITS = 360
_CONNECT_TIMEOUT = 3.0
_RETRY_COOLDOWN = 30.0
_STATUS_TTL = 30.0


class KnowledgeBaseUnavailable(RuntimeError):
    """Milvus 连不上或集合不可用。"""


# ---------------------------------------------------------------------------
# 查询构造
# ---------------------------------------------------------------------------


def _injury_text(injury: Any) -> str:
    if not isinstance(injury, dict):
        return str(injury)
    return f"{injury.get('site', '')}{injury.get('type', '')}"


def build_queries(record: dict[str, Any]) -> list[tuple[str, str]]:
    """把学生情况拆成几条短查询：(类别, 查询文本)。

    一条长查询里的伤病描述容易被年龄、学段等通用信息稀释，所以每项健康状况单独一条，
    写成「<状况> 青少年运动注意事项」：句式统一、状况词占主导，知识库里没有相关内容时
    相似度明显偏低，会被阈值挡掉（实测相关约 0.78-0.80，不相关 0.63-0.71）。
    """
    exam = record.get("physical_exam") or {}
    health = record.get("health_profile") or {}
    age = _to_int(exam.get("NL"))
    sex = str(exam.get("XB") or "").strip()
    stage = _school_stage(_record_grade(record), age) or ""
    who = f"{stage}{sex}生" + (f"（{age}岁）" if age else "")

    conditions = [
        *(str(item) for item in health.get("chronic_conditions") or []),
        *(_injury_text(item) for item in health.get("injury_history") or []),
        *(str(item) for item in health.get("recent_symptoms") or []),
    ]
    spine = str(exam.get("JZ") or "").strip()
    if spine and "未见异常" not in spine:
        conditions.append(f"脊柱筛查{spine}")
    health_queries = [
        ("health", f"{condition} 青少年运动注意事项") for condition in conditions if condition.strip()
    ]

    return [
        ("general", f"{who}每天身体活动时间、运动强度、每周频率与运动类型的推荐"),
        # 健康状况很多时截断，总体与剂量两条始终保留
        *health_queries[: _MAX_QUERIES - 2],
        ("dosage", _DOSAGE_QUERY),
    ]


# ---------------------------------------------------------------------------
# 连接与嵌入（同步，在线程里调用）
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_client: Any = None
_collection_info: dict[str, Any] = {}
_unavailable_until = 0.0
_last_error: str | None = None
_status_cache: tuple[float, dict[str, Any]] | None = None
_query_cache: dict[str, list[float]] = {}


def _skip_reason() -> str | None:
    """不检索的原因；None 表示可以检索。"""
    if not settings.kb_enabled:
        return "已关闭（KB_RETRIEVAL_ENABLED=false）"
    if settings.llm_provider == "mock":
        return "mock 模式"
    if not settings.embedding_api_key:
        return "未配置嵌入密钥（EMBEDDING_API_KEY / ZHIPU_API_KEY）"
    try:
        import pymilvus  # noqa: F401
    except ImportError:
        return "未安装 pymilvus"
    return None


def _connect() -> Any:
    """返回可用的 MilvusClient；连不上时进入冷却期，冷却期内直接报不可用。"""
    global _client, _collection_info, _unavailable_until, _last_error
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
            client = MilvusClient(
                uri=settings.milvus_uri,
                db_name=settings.milvus_database,
                timeout=min(_CONNECT_TIMEOUT, settings.kb_timeout),
                **auth,
            )
            info = _check_collection(client)
        except Exception as exc:  # noqa: BLE001 —— 任何连接问题都按不可用处理
            _unavailable_until = time.monotonic() + _RETRY_COOLDOWN
            _last_error = f"{settings.milvus_uri} 不可用：{_short(exc)}"
            raise KnowledgeBaseUnavailable(_last_error) from exc

        _client, _collection_info, _last_error = client, info, None
        return client


def _rpc_options() -> dict[str, int]:
    """pymilvus 的 RPC 失败默认重试 75 次（约 3.5 分钟），且只有整数 timeout 才会限制重试时长。

    Milvus 中途停掉时，检索线程会被这些重试一直占着；这里限定为整数秒超时（总超时的一半，
    留出查询嵌入的时间）、最多重试 2 次。
    """
    return {"timeout": max(1, int(settings.kb_timeout // 2)), "retry_times": 2}


def _check_collection(client: Any) -> dict[str, Any]:
    """确认集合存在、向量维度与当前嵌入配置一致，并取切片数与入库所用的嵌入模型。"""
    collection = settings.milvus_collection
    rpc = _rpc_options()
    if not client.has_collection(collection, **rpc):
        raise KnowledgeBaseUnavailable(
            f"数据库 {settings.milvus_database} 中没有集合 {collection}，请先运行 rag/scripts/run_index.sh 入库"
        )
    dim = None
    for field in client.describe_collection(collection, **rpc)["fields"]:
        if field["name"] == _VECTOR_FIELD:
            dim = int(field["params"]["dim"])
    if dim != settings.embedding_dimensions:
        raise KnowledgeBaseUnavailable(
            f"集合向量是 {dim} 维，与 EMBEDDING_DIMENSIONS={settings.embedding_dimensions} 不一致"
        )
    client.load_collection(collection, **rpc)
    count = client.query(collection, filter="", output_fields=["count(*)"], **rpc)[0]["count(*)"]
    sample = client.query(
        collection, filter="chunk_index == 0", output_fields=["embedding_model"], limit=1, **rpc
    )
    model = sample[0]["embedding_model"] if sample else ""
    if model and model != settings.embedding_model:
        raise KnowledgeBaseUnavailable(
            f"知识库用 {model} 入库，与当前 EMBEDDING_MODEL={settings.embedding_model} 不一致，向量不可比"
        )
    return {"chunk_count": int(count), "embedding_model": model, "dimensions": dim}


def _reset_client(error: str) -> None:
    """检索出错：丢掉客户端并进入冷却期，冷却期内的请求直接跳过，不再各自等一遍超时。"""
    global _client, _last_error, _status_cache, _unavailable_until
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # noqa: BLE001
                pass
        _client = None
        _last_error = error
        _status_cache = None
        _unavailable_until = time.monotonic() + _RETRY_COOLDOWN


@lru_cache(maxsize=1)
def _embed_client():
    from openai import OpenAI

    return OpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        timeout=settings.kb_timeout,
        max_retries=0,
    )


def _embed(texts: list[str]) -> list[list[float]]:
    """查询文本向量化：与入库一致，embedding-3 必须显式下发 dimensions 才是 1024 维。"""
    missing = [text for text in texts if text not in _query_cache]
    if missing:
        response = _embed_client().embeddings.create(
            model=settings.embedding_model,
            input=missing,
            dimensions=settings.embedding_dimensions,
        )
        if len(_query_cache) > 256:
            _query_cache.clear()
        for item in sorted(response.data, key=lambda d: d.index):
            _query_cache[missing[item.index]] = item.embedding
    return [_query_cache[text] for text in texts]


def _short(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text[:160]


# ---------------------------------------------------------------------------
# 检索与挑选
# ---------------------------------------------------------------------------


def _units(text: str) -> int:
    han = sum(1 for ch in text if "一" <= ch <= "鿿")
    return han + (len(text) - han + 2) // 3


def _trim(text: str, limit: int = _EXCERPT_MAX_UNITS) -> str:
    """超长摘录按句截断，结尾加省略号。"""
    text = " ".join(text.split())
    if _units(text) <= limit:
        return text
    cut, used = 0, 0
    for index, ch in enumerate(text):
        used += 3 if "一" <= ch <= "鿿" else 1
        if used > limit * 3:
            break
        cut = index + 1
    head = text[:cut]
    for mark in ("。", "；", "！", "？", ". ", "; "):
        pos = head.rfind(mark)
        if pos >= len(head) * 0.6:
            return head[: pos + len(mark)].rstrip() + "…"
    return head.rstrip() + "…"


def _pages(entity: dict[str, Any]) -> str:
    start, end = entity.get("page_start"), entity.get("page_end")
    return f"{start}" if not end or end == start else f"{start}-{end}"


def format_source(excerpt: dict[str, Any]) -> str:
    """出处：《文档标题》章节 第 N 页。"""
    section = f" {excerpt['section']}" if excerpt.get("section") else ""
    return f"《{excerpt['doc_title']}》{section} 第 {excerpt['pages']} 页"


def _select(
    queries: list[tuple[str, str]],
    results: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """阈值过滤后挑选摘录：

    1. 每条查询先保底一条（健康状况查询优先，其次总体推荐、抗阻剂量），保证每项伤病都有机会被覆盖；
    2. 剩余名额按同样的先后补满 top-k（同类内按相似度），每条查询最多贡献 _QUERY_CAP 条；
    3. 同一文档不超过 KB_MAX_PER_DOC 条，同一切片被多条查询命中时只取一次。
    """
    threshold = settings.kb_score_threshold
    best: dict[Any, dict[str, Any]] = {}  # 切片 id → 摘录（取各查询中的最高分）
    per_query: list[list[dict[str, Any]]] = []
    for hits in results:
        ranked = []
        for hit in hits:
            score = float(hit["distance"])
            if score < threshold:
                continue
            entry = best.setdefault(hit["id"], {"score": score, "entity": hit["entity"]})
            entry["score"] = max(entry["score"], score)
            ranked.append(entry)
        per_query.append(ranked)

    picked: list[dict[str, Any]] = []
    per_doc: dict[str, int] = {}
    taken_by: list[int] = [0] * len(queries)

    def take(entry: dict[str, Any], query: int) -> bool:
        doc = entry["entity"].get("doc_title", "")
        if (
            len(picked) >= settings.kb_top_k
            or any(entry is item for item in picked)
            or per_doc.get(doc, 0) >= settings.kb_max_per_doc
            or taken_by[query] >= _QUERY_CAP[queries[query][0]]
        ):
            return False
        picked.append(entry)
        per_doc[doc] = per_doc.get(doc, 0) + 1
        taken_by[query] += 1
        return True

    order = sorted(range(len(queries)), key=lambda i: _QUERY_PRIORITY[queries[i][0]])
    for query in order:
        next((entry for entry in per_query[query] if take(entry, query)), None)
    # 补位同样先健康状况、再总体推荐、最后剂量；同类内按相似度
    pool = [(_QUERY_PRIORITY[queries[q][0]], -entry["score"], q, entry) for q in order for entry in per_query[q]]
    for _priority, _score, query, entry in sorted(pool, key=lambda item: item[:2]):
        take(entry, query)

    picked.sort(key=lambda e: e["score"], reverse=True)
    excerpts = []
    for entry in picked:
        entity = entry["entity"]
        excerpt = {
            "doc_title": entity.get("doc_title", ""),
            "section": entity.get("section", ""),
            "pages": _pages(entity),
            "chunk_type": entity.get("chunk_type", "text"),
            "content": _trim(entity.get("content", "")),
            "score": round(entry["score"], 3),
        }
        excerpt["source"] = format_source(excerpt)
        excerpts.append(excerpt)
    return excerpts


def _search(queries: list[tuple[str, str]]) -> list[dict[str, Any]]:
    client = _connect()
    vectors = _embed([text for _kind, text in queries])
    try:
        results = client.search(
            settings.milvus_collection,
            data=vectors,
            anns_field=_VECTOR_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=_PER_QUERY_LIMIT,
            output_fields=_OUTPUT_FIELDS,
            **_rpc_options(),
        )
    except Exception as exc:
        # 连接可能已失效（Milvus 停止或重启）：丢掉客户端，冷却期过后重新连接
        _reset_client(f"检索失败：{_short(exc)}")
        raise
    return _select(queries, [list(hits) for hits in results])


async def retrieve_excerpts(record: dict[str, Any]) -> list[dict[str, Any]]:
    """处方生成：检索与该学生相关的权威资料摘录。任何问题都返回空列表并记 warning。"""
    reason = _skip_reason()
    if reason is not None:
        log = logger.info if not settings.kb_enabled else logger.warning
        log("权威资料检索跳过：%s", reason)
        return []

    queries = build_queries(record)
    started = time.monotonic()
    try:
        excerpts = await asyncio.wait_for(asyncio.to_thread(_search, queries), timeout=settings.kb_timeout)
    except asyncio.TimeoutError:
        logger.warning("权威资料检索超时（>%g 秒），本次跳过", settings.kb_timeout)
        return []
    except KnowledgeBaseUnavailable as exc:
        logger.warning("权威资料检索跳过：%s", exc)
        return []
    except Exception as exc:  # noqa: BLE001 —— 检索失败绝不能拖垮生成主链路
        logger.warning("权威资料检索失败，本次跳过：%s", _short(exc))
        return []

    logger.info(
        "权威资料检索：%d 条查询 → %d 条摘录，耗时 %.1fs",
        len(queries), len(excerpts), time.monotonic() - started,
    )
    return excerpts


# ---------------------------------------------------------------------------
# /meta 状态
# ---------------------------------------------------------------------------


def status() -> dict[str, Any]:
    """/meta 用的知识库检索状态。连通性探测结果缓存 30 秒，Milvus 不可用时最多等连接超时一次。"""
    global _status_cache
    base = {
        "enabled": settings.kb_enabled,
        "milvus_uri": settings.milvus_uri,
        "database": settings.milvus_database,
        "collection": settings.milvus_collection,
        "top_k": settings.kb_top_k,
        "score_threshold": settings.kb_score_threshold,
        "max_per_doc": settings.kb_max_per_doc,
    }
    reason = _skip_reason()
    if reason is not None:
        return {**base, "ready": False, "skip_reason": reason}

    now = time.monotonic()
    if _status_cache is not None and now - _status_cache[0] < _STATUS_TTL:
        return {**base, **_status_cache[1]}
    try:
        _connect()
        probe = {"ready": True, "skip_reason": None, **_collection_info}
    except KnowledgeBaseUnavailable as exc:
        probe = {"ready": False, "skip_reason": str(exc)}
    _status_cache = (now, probe)
    return {**base, **probe}
