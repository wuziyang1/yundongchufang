"""权威资料摘录：按「处方要做的每个决定」分别到知识库检索依据。

为什么不是一句话检索一次：把学生情况压成一句话去比相似度，回来的总是同一段「每天运动 60 分钟」，
训练强度怎么定、抗阻练几组、超重怎么练、某项伤病要注意什么，这些决定各自需要的依据一条也没有。
所以这里把处方拆成若干「检索主题」（Need），每个主题单独检索、单独把关：

1. 两路召回：向量相似度（语义相近）+ BM25 关键词（字面命中），取并集。知识库里有英文文献，
   抗阻、进阶、安全等主题另带一条英文查询；重排的问题也附上英文说法，否则中文问题对英文段落打分偏低。
2. 学段过滤：只取适用于该学生学段的资料（入库时每份文档登记了适用学段），评分表和动作条目
   不参与这里的检索，它们各有专门的取法。
3. 相关性把关：候选交给重排模型逐条打分，低于阈值的丢弃。一个主题的候选全部被丢弃，
   说明知识库里没有针对性资料 —— 如实记为「未覆盖」，交给模型时明确告知，而不是塞几段
   沾边的内容充数，也不让模型凭记忆补一个看似有出处的结论。

这里只定义「要查什么」，不包含任何专业结论；结论全部来自检索到的原文。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.services.kb import rerank, store

logger = logging.getLogger(__name__)

_DENSE_LIMIT = 12
_KEYWORD_LIMIT = 8
# 每个主题送去重排的候选上限：每条向量查询取前 10，关键词取前 8
_DENSE_POOL = 10
_RERANK_WORKERS = 6
# 摘录正文的字符上限（切片本身约 600 折算字数，英文约 1800 字符）
_EXCERPT_MAX_CHARS = 2200
# 从严把关的主题在阈值上再加的余量（重排原始分 / 向量相似度）
_STRICT_RERANK_MARGIN = 1.0
_STRICT_DENSE_MARGIN = 0.03


@dataclass
class Need:
    """一个检索主题：处方里需要依据的一个决定。"""

    key: str
    label: str  # 展示用
    question: str  # 交给重排模型的问题
    dense: list[str]  # 向量查询（可多条，中英文各一条时跨语言召回更稳）
    keywords: str  # BM25 查询
    max_items: int = 2
    # 从严把关：针对个人的主题（某项短板怎么练、某种健康状况要注意什么）必须是明确对题的资料，
    # 把沾边的内容当成依据比承认没有依据更危险，阈值再抬高一档
    strict: bool = False
    # 未覆盖时写进处方的说明
    gap_note: str = "知识库中未检索到针对性资料，相关安排按保守原则给出，须专业人员复核"


@dataclass
class Hit:
    chunk_id: int
    score: float  # 把关所用的分数：重排原始分，或降级时的向量相似度
    doc_title: str
    doc_kind: str
    section: str
    page_start: int
    page_end: int
    chunk_type: str
    content: str

    @property
    def pages(self) -> str:
        return f"{self.page_start}" if self.page_end == self.page_start else f"{self.page_start}-{self.page_end}"

    @property
    def source(self) -> str:
        section = f" {self.section}" if self.section else ""
        return f"《{self.doc_title}》{section} 第 {self.pages} 页"


@dataclass
class NeedResult:
    need: Need
    hits: list[Hit] = field(default_factory=list)
    top_score: float | None = None  # 候选中的最高分（未达阈值时也记录，便于标定阈值）
    mode: str = ""  # rerank / dense

    @property
    def covered(self) -> bool:
        return bool(self.hits)


# ---------------------------------------------------------------------------
# 每份处方都要查的主题
# ---------------------------------------------------------------------------


def base_needs() -> list[Need]:
    return [
        Need(
            key="volume",
            label="每日活动量与每周频率",
            question="儿童青少年每天应进行多少身体活动，每周几天",
            dense=["儿童青少年每天应进行多少身体活动：每日活动时间、每周频率的推荐量"],
            keywords="每天 身体活动 推荐量 每周 频率 累计",
        ),
        Need(
            key="intensity",
            label="运动强度的评估与控制",
            question="如何评估和控制儿童青少年身体活动的强度，心率和主观用力等级怎么用",
            dense=[
                "如何评估和控制儿童青少年身体活动强度：心率、最大心率百分比、RPE 主观用力等级",
                "低强度、中等强度、高强度身体活动的定义与身体表现",
            ],
            keywords="强度 心率 最大心率 RPE 中等强度 高强度",
            max_items=3,
        ),
        Need(
            key="load",
            label="运动负荷与适宜心率范围",
            question="儿童青少年体育锻炼时适宜的运动负荷和心率范围是多少",
            dense=["中小学生体育锻炼的运动负荷要求：适宜的心率范围、平均心率与练习密度"],
            keywords="运动负荷 靶心率 心率 次/分 练习密度",
        ),
        Need(
            key="types",
            label="运动类型的搭配",
            question="儿童青少年应进行哪些类型的身体活动，有氧、抗阻、柔韧、协调等如何搭配",
            dense=["儿童青少年应进行哪些类型的身体活动：有氧运动、抗阻训练、柔韧、协调、平衡如何搭配"],
            keywords="有氧 抗阻 柔韧 协调 平衡 类型",
        ),
        Need(
            key="activities",
            label="常见的有氧、抗阻运动项目",
            question="常见的有氧运动、无氧运动、抗阻运动分别有哪些具体项目",
            dense=["常见的有氧运动项目、无氧运动项目、抗阻运动项目有哪些"],
            keywords="常见 有氧运动 无氧运动 抗阻 项目 步行 慢跑",
        ),
        Need(
            key="activity_intensity",
            label="常见身体活动的强度",
            question="儿童青少年常见的身体活动项目有哪些，各自的强度（代谢当量）是多少",
            dense=["儿童青少年常见身体活动对应的强度（代谢当量 MET）"],
            keywords="常见 身体活动 代谢当量 MET 步行 跑步",
            max_items=1,
        ),
        Need(
            key="resistance",
            label="抗阻训练的组数、次数与频率",
            question=(
                "青少年抗阻训练每次做几组、每组几次、每周练几次"
                "（how many sets and repetitions, how many times per week for youth resistance training）"
            ),
            dense=[
                "青少年抗阻训练的组数、每组次数、每周频率与负荷建议",
                "youth resistance training recommendations: number of sets and repetitions, "
                "training frequency per week, load",
            ],
            keywords="sets repetitions times per week nonconsecutive 组 次数 抗阻",
            max_items=3,
        ),
        Need(
            key="progression",
            label="进阶与负荷递增",
            question=(
                "儿童青少年的运动量和运动强度应如何循序渐进地增加"
                "（how to gradually progress training load and volume for youth）"
            ),
            dense=[
                "如何循序渐进地增加运动量和运动强度：进阶与负荷递增原则",
                "how to progress youth training: gradually increasing load, sets and volume",
            ],
            keywords="循序渐进 逐渐增加 渐进 progression gradually increase",
        ),
        Need(
            key="session",
            label="单次锻炼的结构与各部分时长",
            question="一次锻炼的热身、主体、放松各部分安排多长时间、做什么内容",
            dense=[
                "一次锻炼课的结构与时间安排：热身准备部分、基本部分、整理放松部分各多长时间、做什么",
                "warm-up and cool-down for youth training sessions",
            ],
            keywords="准备部分 热身 基本部分 结束部分 放松 拉伸 warm-up cool-down",
            max_items=3,
        ),
        Need(
            key="safety",
            label="运动安全与损伤预防",
            question="儿童青少年进行体育锻炼时如何预防运动伤害、保证安全",
            dense=[
                "青少年体育锻炼的安全注意事项与运动伤害预防",
                "safety, supervision and injury prevention in youth training",
            ],
            keywords="安全 伤害 预防 防护 supervision safety injury",
        ),
        Need(
            key="environment",
            label="户外活动的环境条件",
            question="不同空气质量、气候条件下如何安排儿童青少年的户外身体活动",
            dense=["不同空气质量、气候环境条件下如何指导儿童青少年开展户外身体活动"],
            keywords="空气质量 污染 户外 气候 环境",
            max_items=1,
        ),
    ]


# ---------------------------------------------------------------------------
# 按学生情况追加的主题
# ---------------------------------------------------------------------------


def weight_need(level: str) -> Need | None:
    """BMI 评定不是「正常」时追加体重管理主题；level 取国标评分表里的等级词。"""
    if level in ("超重", "肥胖"):
        return Need(
            key="weight",
            label=f"体重管理（BMI 评定：{level}）",
            question="超重或肥胖的儿童青少年应如何进行体育锻炼",
            dense=[
                "超重、肥胖儿童青少年的身体活动建议与运动干预效果",
                "overweight or obese youth: exercise and strength training recommendations",
            ],
            keywords="超重 肥胖 体重 脂肪 overweight obese",
        )
    if level == "低体重":
        return Need(
            key="weight",
            label="体重管理（BMI 评定：低体重）",
            question="体重偏低的儿童青少年应如何进行体育锻炼",
            dense=["体重偏低、消瘦的儿童青少年的身体活动与力量练习建议"],
            keywords="低体重 消瘦 体重 营养",
        )
    return None


def weak_item_need(key: str, name: str, focus: str, score: int) -> Need:
    return Need(
        key=f"weak:{key}",
        label=f"短板：{name}（{score} 分）",
        question=f"如何提高{name}成绩，怎样发展{focus}",
        dense=[f"如何提高{name}成绩：发展{focus}的训练方法"],
        keywords=f"{name} {focus}",
        max_items=1,
        strict=True,
        gap_note="知识库中没有该项目专项提升方法的文字资料，训练安排以目录中检索到的相关动作为依据，须专业人员复核",
    )


def health_need(index: int, condition: str) -> Need:
    return Need(
        key=f"health:{index}",
        label=f"健康状况：{condition}",
        question=f"学生有“{condition}”的情况，进行体育锻炼时应注意什么",
        dense=[f"{condition} 儿童青少年运动注意事项"],
        keywords=condition,
        strict=True,
        gap_note="知识库中未检索到针对该情况的运动注意事项，相关安排按保守原则给出，须专业人员复核",
    )


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------


def _filter(stage: str | None) -> str:
    """正文与普通表格；评分表（lookup）和动作条目（exercise）不参与摘录检索。"""
    expression = 'chunk_type in ["text", "table"]'
    if stage:
        expression += f" and array_contains(stages, {store.quote(stage)})"
    return expression


def _to_hit(row: dict[str, Any], score: float) -> Hit:
    content = " ".join(str(row.get("content") or "").split())
    return Hit(
        chunk_id=row["id"],
        score=score,
        doc_title=row.get("doc_title", ""),
        doc_kind=row.get("doc_kind", ""),
        section=row.get("section", ""),
        page_start=row.get("page_start", 0),
        page_end=row.get("page_end", 0),
        chunk_type=row.get("chunk_type", "text"),
        content=content[:_EXCERPT_MAX_CHARS],
    )


def _judge(need: Need, pool: dict[int, dict[str, Any]]) -> NeedResult:
    """对一个主题的候选做相关性把关，挑出最多 max_items 条。"""
    result = NeedResult(need=need)
    rows = list(pool.values())
    if not rows:
        return result

    reranked = rerank.scores(need.question, [str(row.get("content") or "") for row in rows])
    if reranked is not None:
        result.mode = "rerank"
        scored = sorted(zip(rows, reranked), key=lambda pair: pair[1], reverse=True)
        threshold = settings.kb_rerank_min_score + (_STRICT_RERANK_MARGIN if need.strict else 0.0)
    else:
        # 降级：只有向量相似度可用；仅被关键词召回、没有相似度的候选不参与
        result.mode = "dense"
        scored = sorted(
            ((row, row["dense_score"]) for row in rows if row.get("dense_score") is not None),
            key=lambda pair: pair[1],
            reverse=True,
        )
        threshold = settings.kb_dense_min_score + (_STRICT_DENSE_MARGIN if need.strict else 0.0)

    if scored:
        result.top_score = round(scored[0][1], 3)
    result.hits = [
        _to_hit(row, round(score, 3)) for row, score in scored[: need.max_items] if score >= threshold
    ]
    return result


def retrieve(needs: list[Need], stage: str | None) -> list[NeedResult]:
    """为每个主题检索并把关。顺序与 needs 一致。"""
    if not needs:
        return []
    expression = _filter(stage)

    dense_queries = [query for need in needs for query in need.dense]
    dense_results = store.dense_search(dense_queries, filter=expression, limit=_DENSE_LIMIT)
    keyword_results = store.keyword_search(
        [need.keywords for need in needs], filter=expression, limit=_KEYWORD_LIMIT
    )

    pools: list[dict[int, dict[str, Any]]] = []
    cursor = 0
    for need, keyword_hits in zip(needs, keyword_results):
        pool: dict[int, dict[str, Any]] = {}
        for _query in need.dense:
            for hit in dense_results[cursor][:_DENSE_POOL]:
                row = pool.setdefault(hit["id"], {**hit, "dense_score": hit["score"]})
                row["dense_score"] = max(row["dense_score"], hit["score"])
            cursor += 1
        for hit in keyword_hits:
            pool.setdefault(hit["id"], {**hit, "dense_score": None})
        pools.append(pool)

    with ThreadPoolExecutor(max_workers=_RERANK_WORKERS) as executor:
        return list(executor.map(_judge, needs, pools))
