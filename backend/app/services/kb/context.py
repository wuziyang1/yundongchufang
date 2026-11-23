"""一次处方生成所需的全部知识，全部取自知识库（Milvus）。

build() 依次做四件事：
1. 体质评定：把国标评分表取出来，代码查表得到每项体测成绩的得分与等级（S 条目）；
2. 检索主题：固定主题（活动量、强度、类型、抗阻剂量、进阶、课次结构、安全、环境）
   加上按学生情况追加的主题（体重管理、得分最低的几项短板、每项健康状况）；
3. 权威资料摘录：每个主题分别检索并做相关性把关（E 条目），查不到的记为未覆盖；
4. 可选动作：按学段、器材、单人可做筛出候选集，并为每项短板检索最相关的动作。

知识库不可用（Milvus 未启动、集合是旧结构、嵌入失败、超时）时抛 KnowledgeBaseUnavailable，
由处方服务决定如何处理 —— 处方必须有依据，不会悄悄退回「只靠模型记忆」。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.services.kb import evidence as ev
from app.services.kb import exercises as ex
from app.services.kb import rerank, standards, store
from app.services.kb.store import KnowledgeBaseUnavailable

logger = logging.getLogger(__name__)

STAGES = ("小学低年级", "小学高年级", "初中", "高中")

# 只对得分最低的几项短板做专项检索
_WEAK_ITEMS = 3
_HEALTH_CONDITIONS = 6
_NORMAL_VISION = {"", "正常", "良好", "视力正常"}


@dataclass
class Fact:
    """体质评定条目：代码查表得到的确定结论，模型可按编号引用。"""

    id: str  # S1、S2…
    item: standards.ItemScore


@dataclass
class Evidence:
    """权威资料摘录条目：模型可按编号引用。"""

    id: str  # E1、E2…
    hit: ev.Hit
    topics: list[str] = field(default_factory=list)  # 因哪些检索主题被选中


@dataclass
class Topic:
    """一个检索主题的结果：覆盖与否，以及对应的摘录编号。"""

    key: str
    label: str
    covered: bool
    evidence_ids: list[str]
    top_score: float | None
    gap_note: str


@dataclass
class Recommendation:
    """针对一项体测短板检索到的动作。"""

    item: standards.ItemScore
    exercises: list[ex.Exercise]


@dataclass
class KnowledgeContext:
    info: dict[str, Any]
    stage: str | None
    assessment: standards.Assessment
    facts: list[Fact] = field(default_factory=list)
    topics: list[Topic] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    candidates: list[ex.Exercise] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    retrieval_mode: str = ""  # rerank / dense
    elapsed: float = 0.0

    @property
    def available(self) -> bool:
        return bool(self.info)

    @property
    def fingerprint(self) -> str:
        return str(self.info.get("fingerprint", ""))

    @property
    def evidence_ids(self) -> set[str]:
        return {item.id for item in self.evidence}

    @property
    def fact_ids(self) -> set[str]:
        return {item.id for item in self.facts}

    @property
    def gaps(self) -> list[Topic]:
        return [topic for topic in self.topics if not topic.covered]

    @property
    def candidate_index(self) -> dict[str, ex.Exercise]:
        return {exercise.code: exercise for exercise in self.candidates}


def offline() -> KnowledgeContext:
    """没有知识库时的空上下文，仅供 mock 模式离线联调页面。"""
    return KnowledgeContext(info={}, stage=None, assessment=standards.Assessment(sex="", grade="", grade_inferred=False))


# ---------------------------------------------------------------------------
# 学生情况 → 学段、健康状况
# ---------------------------------------------------------------------------


def school_stage(grade: str, age: int | None) -> str | None:
    """归一化为知识库的四个学段。grade 是评分表的列名（一年级…高三）。"""
    if grade in standards.PRIMARY:
        return "小学低年级" if grade in standards.PRIMARY[:2] else "小学高年级"
    if grade in standards.SECONDARY:
        return "初中" if grade in standards.SECONDARY[:3] else "高中"
    if age is None:
        return None
    if age <= 8:
        return "小学低年级"
    if age <= 12:
        return "小学高年级"
    if age <= 15:
        return "初中"
    return "高中"


def health_conditions(record: dict[str, Any]) -> list[str]:
    """档案里需要在运动中留意的情况：慢病、伤病史、近期症状、脊柱与视力筛查异常。"""
    exam = record.get("physical_exam") or {}
    health = record.get("health_profile") or {}
    conditions: list[str] = [str(item).strip() for item in health.get("chronic_conditions") or []]
    for injury in health.get("injury_history") or []:
        if isinstance(injury, dict):
            conditions.append(f"{injury.get('site', '')}{injury.get('type', '')}".strip() or str(injury.get("note", "")).strip())
        else:
            conditions.append(str(injury).strip())
    conditions.extend(str(item).strip() for item in health.get("recent_symptoms") or [])
    spine = str(exam.get("JZ") or "").strip()
    if spine and "未见异常" not in spine:
        conditions.append(f"脊柱筛查{spine}")
    vision = str(exam.get("JSLPJ") or "").strip()
    if vision not in _NORMAL_VISION:
        conditions.append(vision)
    return list(dict.fromkeys(item for item in conditions if item))[:_HEALTH_CONDITIONS]


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------


def build(record: dict[str, Any]) -> KnowledgeContext:
    """同步执行，由 load() 放到线程里并加超时。"""
    started = time.monotonic()
    info = store.info()

    assessment = standards.assess(record)
    exam = record.get("physical_exam") or {}
    stage = school_stage(assessment.grade, standards.to_int(exam.get("NL")))
    facts = [Fact(id=f"S{number}", item=item) for number, item in enumerate(assessment.items, 1)]

    weak = assessment.weak_items[:_WEAK_ITEMS]
    needs = ev.base_needs()
    if assessment.bmi is not None and (need := ev.weight_need(assessment.bmi.level)) is not None:
        needs.append(need)
    needs.extend(ev.weak_item_need(item.key, item.name, item.focus or item.name, item.score) for item in weak)
    needs.extend(ev.health_need(index, text) for index, text in enumerate(health_conditions(record)))

    results = ev.retrieve(needs, stage)

    evidence: list[Evidence] = []
    by_chunk: dict[int, Evidence] = {}
    topics: list[Topic] = []
    for result in results:
        ids: list[str] = []
        for hit in result.hits:
            entry = by_chunk.get(hit.chunk_id)
            if entry is None:
                entry = Evidence(id=f"E{len(evidence) + 1}", hit=hit)
                by_chunk[hit.chunk_id] = entry
                evidence.append(entry)
            entry.topics.append(result.need.label)
            ids.append(entry.id)
        topics.append(
            Topic(
                key=result.need.key,
                label=result.need.label,
                covered=result.covered,
                evidence_ids=ids,
                top_score=result.top_score,
                gap_note=result.need.gap_note,
            )
        )
    modes = {result.mode for result in results if result.mode}
    mode = "dense" if "dense" in modes else "rerank" if modes else ""

    pool = ex.candidates(stage)
    picked = ex.recommend(
        [f"发展{item.focus or item.name}的练习，用于提高{item.name}成绩" for item in weak], pool
    )
    recommendations = [
        Recommendation(item=item, exercises=[exercise for exercise, _score in row])
        for item, row in zip(weak, picked)
    ]
    candidates = ex.trim(pool, [exercise.code for rec in recommendations for exercise in rec.exercises])

    context = KnowledgeContext(
        info=info,
        stage=stage,
        assessment=assessment,
        facts=facts,
        topics=topics,
        evidence=evidence,
        candidates=candidates,
        recommendations=recommendations,
        retrieval_mode=mode,
        elapsed=time.monotonic() - started,
    )
    logger.info(
        "知识库检索：%d 个主题（未覆盖 %d）→ %d 条摘录，评定 %d 项，候选动作 %d 个，把关方式 %s，耗时 %.1fs",
        len(topics), len(context.gaps), len(evidence), len(facts), len(candidates), mode or "—", context.elapsed,
    )
    if mode == "dense":
        logger.warning("重排不可用（%s），本次用向量相似度把关，识别「未覆盖」的能力变弱", rerank.problem())
    return context


async def load(record: dict[str, Any]) -> KnowledgeContext:
    """取一次处方生成所需的知识；知识库不可用或超时抛 KnowledgeBaseUnavailable。"""
    try:
        return await asyncio.wait_for(asyncio.to_thread(build, record), timeout=settings.kb_timeout)
    except asyncio.TimeoutError as exc:
        raise KnowledgeBaseUnavailable(f"知识库检索超时（>{settings.kb_timeout:g} 秒）") from exc
