"""RAG 检索服务：把动作库与知识卡片做成外挂知识库，按学生情况检索子集注入 prompt。

仅服务运动处方生成阶段（档案理解阶段不使用 RAG，画像全部由模型直接输出）。

分工：
- 语料向量由 scripts/build_index.py 离线构建，落盘 data/index/embeddings.json；
- 运行时把"该学生的情况"压成一句查询文本，调一次嵌入接口（带缓存），
  再做 结构化过滤/保底配额 + 向量相似度 + 字符 bigram 词面分 的混合排序；
- 任何一步不可用（索引缺失、无密钥、网络失败、mock 模式）都降级为纯词面检索，
  保证离线环境全链路可用。

安全边界不在本模块：动作的禁忌标签/年龄/器械/场地硬过滤仍由
prescription_service.filter_candidates 在代码层完成，检索只负责"在安全集合内挑最相关的"。
"""

from __future__ import annotations

import json
import logging
import math
import string
from functools import lru_cache
from hashlib import sha256
from typing import Any

from app.config import DATA_DIR, settings
from app.services.data_loader import exercise_library, knowledge_cards

logger = logging.getLogger(__name__)

INDEX_PATH = DATA_DIR / "index" / "embeddings.json"

# 向量分与词面分的混合权重；词面分用于奖励"哮喘""1000米"这类必须精确命中的词
_VEC_WEIGHT = 0.75
_LEX_WEIGHT = 0.25

# ---------------------------------------------------------------------------
# 语料文本：检索与建索引共用同一份文本表示，保证离线向量与在线打分一致
# ---------------------------------------------------------------------------


def knowledge_doc_text(card: dict[str, Any]) -> str:
    scope = "、".join(card.get("age_scope") or [])
    return f"{card['title']}。领域：{card.get('domain', '')}。适用：{scope}。{card.get('content', '')}"


def exercise_doc_text(exercise: dict[str, Any]) -> str:
    parts = [
        exercise.get("name", ""),
        f"类别：{exercise.get('category', '')}·{exercise.get('subcategory', '')}",
        "目标肌群：" + "、".join(exercise.get("muscle_groups") or []),
        f"强度{exercise.get('intensity_level', '')}，冲击{exercise.get('impact_level', '')}",
        "器械：" + "、".join(exercise.get("equipment") or ["无"]),
        "要点：" + "；".join(exercise.get("key_points") or []),
    ]
    return "。".join(part for part in parts if part)


def corpus_docs() -> dict[str, list[tuple[str, str]]]:
    """建索引脚本使用：两套语料的 (id, 文本) 列表。"""
    return {
        "knowledge": [(card["id"], knowledge_doc_text(card)) for card in knowledge_cards()],
        "exercises": [
            (item["id"], exercise_doc_text(item))
            for item in exercise_library()["exercises"]
        ],
    }


def doc_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 向量索引加载与查询嵌入
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_index() -> dict[str, Any] | None:
    """读入离线索引并逐条校验内容 hash；源数据已改动的条目丢弃其向量。"""
    if not INDEX_PATH.is_file():
        logger.warning("未找到嵌入索引 %s，检索降级为词面匹配。可运行 scripts/build_index.py 构建。", INDEX_PATH)
        return None
    try:
        with INDEX_PATH.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("嵌入索引读取失败（%s），检索降级为词面匹配。", exc)
        return None

    current = {name: dict(docs) for name, docs in corpus_docs().items()}
    collections: dict[str, dict[str, list[float]]] = {}
    stale = 0
    for name, items in (raw.get("collections") or {}).items():
        vectors: dict[str, list[float]] = {}
        docs = current.get(name, {})
        for item in items:
            text = docs.get(item.get("id", ""))
            if text is None or doc_hash(text) != item.get("hash"):
                stale += 1
                continue
            vectors[item["id"]] = item["vector"]
        collections[name] = vectors
    if stale:
        logger.warning(
            "嵌入索引中有 %d 条与当前数据不一致（数据已更新），这些条目退回词面打分。"
            "建议重新运行 scripts/build_index.py。",
            stale,
        )
    return {
        "model": raw.get("model", settings.embedding_model),
        "dimensions": raw.get("dimensions", settings.embedding_dimensions),
        "collections": collections,
    }


@lru_cache(maxsize=1)
def _embed_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
        timeout=settings.embedding_timeout,
    )


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


_query_cache: dict[str, list[float]] = {}


async def _embed_query(text: str) -> list[float] | None:
    """查询文本 → 单位向量。索引缺失、无密钥或调用失败时返回 None（降级词面）。"""
    index = _load_index()
    if index is None or not settings.embedding_api_key:
        return None
    cached = _query_cache.get(text)
    if cached is not None:
        return cached
    try:
        response = await _embed_client().embeddings.create(
            model=index["model"],
            input=text[:3000],
            dimensions=index["dimensions"],
        )
        vector = normalize(response.data[0].embedding)
    except Exception as exc:  # noqa: BLE001 —— 检索失败绝不能拖垮生成主链路
        logger.warning("查询嵌入失败（%s），本次检索降级为词面匹配。", exc)
        return None
    if len(_query_cache) >= 128:
        _query_cache.clear()
    _query_cache[text] = vector
    return vector


# ---------------------------------------------------------------------------
# 词面打分：中文字符 bigram 集合余弦，零依赖、离线可用
# ---------------------------------------------------------------------------

_PUNCT = set(string.punctuation + string.whitespace + "，。、；：！？（）《》【】“”‘’·—…")


def _bigrams(text: str) -> set[str]:
    chars = [ch for ch in text if ch not in _PUNCT]
    if len(chars) < 2:
        return set(chars)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def _lexical_score(query_bigrams: set[str], doc_bigrams: set[str]) -> float:
    if not query_bigrams or not doc_bigrams:
        return 0.0
    overlap = len(query_bigrams & doc_bigrams)
    return overlap / math.sqrt(len(query_bigrams) * len(doc_bigrams))


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


async def _score_docs(
    collection: str,
    docs: list[tuple[str, str]],
    query_text: str,
) -> dict[str, float]:
    """混合打分。缺向量的条目用词面分顶替向量分，保证同一尺度可比。"""
    query_bg = _bigrams(query_text)
    lexical = {doc_id: _lexical_score(query_bg, _bigrams(text)) for doc_id, text in docs}

    query_vec = await _embed_query(query_text)
    index = _load_index()
    if query_vec is None or index is None:
        return lexical

    vectors = index["collections"].get(collection, {})
    scores: dict[str, float] = {}
    for doc_id, _text in docs:
        lex = lexical[doc_id]
        vec = vectors.get(doc_id)
        cos = _dot(query_vec, vec) if vec is not None else lex
        scores[doc_id] = _VEC_WEIGHT * cos + _LEX_WEIGHT * lex
    return scores


# ---------------------------------------------------------------------------
# 学段推断与查询文本构造
# ---------------------------------------------------------------------------


def _school_stage(grade: str, age: int | None) -> str | None:
    """归一化为知识卡片 age_scope 的四档：小学低年级/小学高年级/初中/高中。"""
    grade = grade or ""
    if "高" in grade and "小" not in grade:
        return "高中"
    if "初" in grade:
        return "初中"
    if "小" in grade or "年级" in grade:
        if any(token in grade for token in ("一", "二", "1", "2")) and "三" not in grade:
            return "小学低年级"
        return "小学高年级"
    if age is None:
        return None
    if age <= 7:
        return "小学低年级"
    if age <= 11:
        return "小学高年级"
    if age <= 14:
        return "初中"
    return "高中"


def _stage_filter(cards: list[dict[str, Any]], stage: str | None) -> list[dict[str, Any]]:
    if stage is None:
        return cards
    kept = [
        card
        for card in cards
        if not card.get("age_scope") or "通用" in card["age_scope"] or stage in card["age_scope"]
    ]
    return kept or cards


def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _record_grade(record: dict[str, Any]) -> str:
    exam = record.get("physical_exam") or {}
    fitness = record.get("fitness_test") or {}
    return str(exam.get("SCHOOL_GRADE") or fitness.get("NJ") or "")


def _prescription_query_text(record: dict[str, Any]) -> str:
    """处方阶段的查询：学段/性别 + 健康状况 + 体检异常。"""
    pieces: list[str] = []
    exam = record.get("physical_exam") or {}
    pieces.append(f"{_record_grade(record)} {exam.get('XB', '')}生 {exam.get('NL', '')}岁")

    health = record.get("health_profile") or {}
    pieces.extend(str(item) for item in health.get("chronic_conditions") or [])
    pieces.extend(f"用药 {item}" for item in health.get("medications") or [])
    pieces.extend(str(item) for item in health.get("recent_symptoms") or [])
    for injury in health.get("injury_history") or []:
        if isinstance(injury, dict):
            pieces.append(
                f"{injury.get('site', '')}{injury.get('type', '')}（{injury.get('status', '')}）"
            )
        else:
            pieces.append(str(injury))

    spine = str(exam.get("JZ") or "")
    if spine and "未见异常" not in spine:
        pieces.append(f"脊柱 {spine}")
    vision = str(exam.get("JSLPJ") or "")
    if vision and vision not in ("正常", "良好"):
        pieces.append(f"视力 {vision}")

    return "；".join(str(piece) for piece in pieces if str(piece).strip())


# ---------------------------------------------------------------------------
# 对外检索接口
# ---------------------------------------------------------------------------


# 处方输出结构固定含 FITT-VP、强度控制、安全等板块，这些领域保底入选，
# 其余名额按相似度竞争（体测专项等自然浮现）
_PRESCRIPTION_DOMAIN_QUOTA = {
    "处方框架": 1,
    "强度控制": 2,
    "训练结构": 1,
    "安全": 1,
}


def _pick_with_quota(
    ranked: list[dict[str, Any]],
    quota: dict[str, int],
    top_k: int,
) -> list[dict[str, Any]]:
    """先按领域配额保底（域内取相似度最高者），再按总分补满。"""
    picked: list[dict[str, Any]] = []
    picked_ids: set[str] = set()
    for domain, count in quota.items():
        taken = sum(1 for item in picked if item.get("domain") == domain)
        for card in [item for item in ranked if item.get("domain") == domain][: max(count - taken, 0)]:
            if card["id"] not in picked_ids:
                picked.append(card)
                picked_ids.add(card["id"])
    for card in ranked:
        if len(picked) >= top_k:
            break
        if card["id"] not in picked_ids:
            picked.append(card)
            picked_ids.add(card["id"])
    picked.sort(key=lambda card: card["id"])
    return picked


def _health_quota(has_chronic: bool, has_injury: bool) -> dict[str, int]:
    """长查询里伤病描述易被短板/目标淹没，带伤病档案时相关领域强制占位。"""
    quota: dict[str, int] = {}
    if has_chronic:
        quota["特殊情况"] = 1
    if has_injury:
        quota["损伤与恢复"] = 1
    return quota


async def retrieve_knowledge_for_prescription(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """处方生成：领域保底配额 + 混合排序，返回 top-k 知识卡片。"""
    cards = knowledge_cards()
    if not settings.rag_enabled:
        return cards

    exam = record.get("physical_exam") or {}
    stage = _school_stage(_record_grade(record), _to_int(exam.get("NL")))
    pool = _stage_filter(cards, stage)

    top_k = settings.rag_knowledge_top_k
    if len(pool) <= top_k:
        return pool

    docs = [(card["id"], knowledge_doc_text(card)) for card in pool]
    scores = await _score_docs("knowledge", docs, _prescription_query_text(record))
    ranked = sorted(pool, key=lambda card: scores.get(card["id"], 0.0), reverse=True)

    health = record.get("health_profile") or {}
    quota = dict(_PRESCRIPTION_DOMAIN_QUOTA)
    quota.update(
        _health_quota(
            has_chronic=bool(health.get("chronic_conditions") or health.get("medications")),
            has_injury=bool(health.get("injury_history")),
        )
    )
    # 学生不再填目标：体测专项卡随弱项成绩的语义相似度竞争入选
    return _pick_with_quota(ranked, quota, top_k)


# 热身与放松是每个训练日的必备块，全部保留；主训练五类各保底 2 个
_ALWAYS_KEEP_CATEGORIES = ("热身", "放松")
_MAIN_CATEGORY_FLOOR = 2
_CATEGORY_ORDER = {
    name: order
    for order, name in enumerate(("热身", "有氧", "抗阻", "核心", "柔韧", "速度灵敏", "放松"))
}


async def rank_exercises(
    candidates: list[dict[str, Any]],
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """在硬过滤（年龄/校园默认器材）后的候选内，按相关度排序并截断到配置上限。

    保底规则保证模型拿到的子集仍然可编排出完整一周计划：
    热身/放松全保留，主训练五类各至少 2 个（若候选中存在）。
    """
    limit = settings.rag_exercise_limit
    if not settings.rag_enabled or len(candidates) <= limit:
        return candidates

    docs = [(item["id"], exercise_doc_text(item)) for item in candidates]
    scores = await _score_docs("exercises", docs, _prescription_query_text(record))

    def final_score(exercise: dict[str, Any]) -> float:
        return scores.get(exercise["id"], 0.0)

    ranked = sorted(candidates, key=final_score, reverse=True)

    picked: list[dict[str, Any]] = []
    picked_ids: set[str] = set()

    def take(exercise: dict[str, Any]) -> None:
        if exercise["id"] not in picked_ids:
            picked.append(exercise)
            picked_ids.add(exercise["id"])

    for exercise in ranked:
        if exercise.get("category") in _ALWAYS_KEEP_CATEGORIES:
            take(exercise)
    for category in ("有氧", "抗阻", "核心", "柔韧", "速度灵敏"):
        matched = [item for item in ranked if item.get("category") == category]
        for exercise in matched[:_MAIN_CATEGORY_FLOOR]:
            take(exercise)
    for exercise in ranked:
        if len(picked) >= limit:
            break
        take(exercise)

    picked.sort(
        key=lambda item: (
            _CATEGORY_ORDER.get(item.get("category", ""), 99),
            -final_score(item),
        )
    )
    return picked


def status() -> dict[str, Any]:
    """/meta 用的 RAG 状态摘要（含 Milvus 权威资料知识库的检索状态）。"""
    from app.services import literature  # literature 引用了本模块的学段推断，延迟导入避免循环

    index = _load_index() if INDEX_PATH.is_file() else None
    indexed = (
        {name: len(vectors) for name, vectors in index["collections"].items()} if index else {}
    )
    return {
        "enabled": settings.rag_enabled,
        "index_ready": bool(index),
        "embedding_model": index["model"] if index else settings.embedding_model,
        "knowledge_top_k": settings.rag_knowledge_top_k,
        "exercise_limit": settings.rag_exercise_limit,
        "indexed_counts": indexed,
        "knowledge_base": literature.status(),
    }
