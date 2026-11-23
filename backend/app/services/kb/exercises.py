"""可选动作：处方里的动作只能从知识库训练手册的练习条目中选。

rag 入库时把手册里的每个「练习N：名称」整理成一条动作条目（编号、名称、素质类别、器材、
是否需要同伴，以及按原书小标题拆开的功能、起始姿势、练习步骤、指导提示、安全事项）。
这里负责：

- 取出全部动作条目（按知识库指纹缓存）；
- 按学段、器材、能否单人完成筛出候选集 —— 代码硬过滤，模型只在候选集里选；
- 针对体测短板，在候选集里用向量检索找出最相关的动作，作为排课时的优先选择；
- 给 prompt 用的一行摘要，给前端用的完整详情（带出处页码）。

动作条目里没有的日常身体活动（快步走、慢跑、跳绳等）不在这里，由处方服务按
「权威资料摘录里是否提到」单独核实。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.services.kb import store

# prompt 与前端按这个顺序分组展示
CATEGORY_ORDER = ("心肺耐力", "肌肉耐力", "力量", "速度", "灵敏", "协调", "平衡", "柔韧", "综合")

# 原书里的用量说明：带「次 / 秒 / 分钟 / 组 / 遍」或三位数以上距离的句子
_SENTENCE_RE = re.compile(r"[^。；;]+[。；;]?")
_DOSAGE_RE = re.compile(
    r"\d+\s*(?:[~～\-—至]\s*\d+)?\s*(?:次|秒|分钟|组|遍)|\d{3,}\s*(?:[~～\-—至]\s*\d+)?\s*米"
)
_ENUMERATOR_RE = re.compile(r"^\d+\s*[.．、]\s*")
_DOSAGE_MAX_CHARS = 70
_PURPOSE_MAX_CHARS = 60

_RECOMMEND_PER_ITEM = 6
_RECOMMEND_POOL = 80


@dataclass
class Exercise:
    code: str
    name: str
    category: str
    group: str  # 单人 / 多人
    equipment: list[str]
    fields: dict[str, str]
    content: str  # 条目原文；个别条目没有分小标题（如「长距离慢跑」），说明都在这里
    doc_title: str
    section: str
    page_start: int
    page_end: int
    stages: list[str] = field(default_factory=list)

    @property
    def purpose(self) -> str:
        return self.fields.get("purpose", "")

    @property
    def body(self) -> str:
        """没有分小标题的条目的原文（压成一行）；分了小标题的返回空串。"""
        if any(key in self.fields for key in ("purpose", "steps", "start")):
            return ""
        return " ".join(self.content.split())

    @property
    def source(self) -> str:
        pages = f"{self.page_start}" if self.page_end == self.page_start else f"{self.page_start}-{self.page_end}"
        chapter = " > ".join(self.section.split(" > ")[:-1])
        return f"《{self.doc_title}》{chapter} 第 {pages} 页"

    @property
    def dosage_hint(self) -> str:
        """原书对该练习用量的说明（指导提示优先，其次练习步骤、原文），没有则为空。"""
        for text in (self.fields.get("tips", ""), self.fields.get("steps", ""), self.body):
            picked = [
                _ENUMERATOR_RE.sub("", sentence.strip())
                for sentence in _SENTENCE_RE.findall(text)
                if _DOSAGE_RE.search(sentence)
            ]
            if picked:
                hint = "".join(picked[:2])
                if len(hint) > _DOSAGE_MAX_CHARS:
                    hint = picked[0]
                return hint if len(hint) <= _DOSAGE_MAX_CHARS else hint[:_DOSAGE_MAX_CHARS] + "…"
        return ""

    def prompt_line(self) -> str:
        equipment = "、".join(self.equipment) or "无"
        parts = [f"[{self.code}] {self.name}", f"器材 {equipment}"]
        summary = self.purpose or self.body
        if summary:
            parts.append(summary if len(summary) <= _PURPOSE_MAX_CHARS else summary[:_PURPOSE_MAX_CHARS] + "…")
        if hint := self.dosage_hint:
            parts.append(f"原书用量：{hint}")
        return "｜".join(parts)

    def detail(self) -> dict[str, Any]:
        """回填进处方结果、供前端展示的完整信息。"""
        return {
            "id": self.code,
            "name": self.name,
            "category": self.category,
            "equipment": self.equipment,
            "purpose": self.purpose,
            "start": self.fields.get("start", ""),
            "steps": self.fields.get("steps", "") or self.body,
            "tips": self.fields.get("tips", ""),
            "safety": self.fields.get("safety", ""),
            "variants": self.fields.get("variants", ""),
            "dosage_hint": self.dosage_hint,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_cache: tuple[str, dict[str, Exercise]] | None = None


def catalog() -> dict[str, Exercise]:
    """知识库里的全部动作条目：{编号: 动作}，按原书顺序。按知识库指纹缓存。"""
    global _cache
    fingerprint = store.info()["fingerprint"]
    with _lock:
        if _cache is not None and _cache[0] == fingerprint:
            return _cache[1]

    rows = store.query(
        'chunk_type == "exercise"',
        ["doc_title", "section", "page_start", "page_end", "chunk_index", "stages", "meta", "content"],
    )
    result: dict[str, Exercise] = {}
    for row in sorted(rows, key=lambda item: (item["doc_title"], item["chunk_index"])):
        meta = row.get("meta") or {}
        code = str(meta.get("code") or "")
        if not code:
            continue
        result[code] = Exercise(
            code=code,
            name=str(meta.get("name") or code),
            category=str(meta.get("category") or "综合"),
            group=str(meta.get("group") or "单人"),
            equipment=list(meta.get("equipment") or []),
            fields=dict(meta.get("fields") or {}),
            content=str(row.get("content") or ""),
            doc_title=row["doc_title"],
            section=row["section"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            stages=list(row.get("stages") or []),
        )
    with _lock:
        _cache = (fingerprint, result)
    return result


def candidates(stage: str | None) -> list[Exercise]:
    """候选集：适用于该学段、一个人能完成、所需器材都在可得器材清单内的动作。"""
    allowed = set(settings.kb_equipment)
    return [
        exercise
        for exercise in catalog().values()
        if exercise.group == "单人"
        and all(item in allowed for item in exercise.equipment)
        and (stage is None or not exercise.stages or stage in exercise.stages)
    ]


def recommend(queries: list[str], pool: list[Exercise]) -> list[list[tuple[Exercise, float]]]:
    """针对每条查询（一个体测短板），在候选集里按向量相似度找出最相关的动作。"""
    if not queries or not pool:
        return [[] for _ in queries]
    allowed = {exercise.code: exercise for exercise in pool}
    results = store.dense_search(
        queries,
        filter='chunk_type == "exercise"',
        limit=_RECOMMEND_POOL,
        output_fields=["meta"],
    )
    picked: list[list[tuple[Exercise, float]]] = []
    for hits in results:
        row: list[tuple[Exercise, float]] = []
        for hit in hits:
            exercise = allowed.get(str((hit.get("meta") or {}).get("code") or ""))
            if exercise is not None:
                row.append((exercise, hit["score"]))
            if len(row) >= _RECOMMEND_PER_ITEM:
                break
        picked.append(row)
    return picked


def trim(pool: list[Exercise], keep_first: list[str]) -> list[Exercise]:
    """候选集超过上限时截取：先留针对短板检索到的动作，其余按素质类别轮流取，保证各类都有。"""
    limit = settings.kb_exercise_limit
    if len(pool) <= limit:
        return pool
    by_code = {exercise.code: exercise for exercise in pool}
    picked: dict[str, Exercise] = {code: by_code[code] for code in keep_first if code in by_code}
    groups: dict[str, list[Exercise]] = {}
    for exercise in pool:
        if exercise.code not in picked:
            groups.setdefault(exercise.category, []).append(exercise)
    while len(picked) < limit and any(groups.values()):
        for category in list(groups):
            if groups[category] and len(picked) < limit:
                exercise = groups[category].pop(0)
                picked[exercise.code] = exercise
    return [exercise for exercise in pool if exercise.code in picked]
