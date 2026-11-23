"""体质评定：把国标查表结果组织成画像要用的结构。

查表（单项得分与等级、附加分、学年总分与综合等级）在 kb/standards.py，数据全部取自知识库。
这里只做两件事：给每个得分项编号（S1、S2…，与处方里的编号一致），以及把项目归到体质六维里，
给出每一维的得分与等级。

体质六维是画像的展示分组，每一维由国标的若干测试项目支撑：
    身体形态 — 体重指数；心肺耐力 — 肺活量 + 耐力跑（小学 50米×8 往返跑，中学 800 / 1000 米跑）；
    肌肉力量与耐力 — 引体向上或一分钟仰卧起坐；速度与灵敏 — 50米跑；
    爆发力 — 立定跳远（中学）或一分钟跳绳（小学）；柔韧性 — 坐位体前屈。

维度等级不由模型判断：
- 一维只有一个项目：直接用该项目的得分与等级。身体形态只有体重指数，等级词是国标体重指数评分表里的
  正常 / 低体重 / 超重 / 肥胖；
- 一维有多个项目（心肺耐力）：按国标权重对各项得分加权平均，再按国标的等级划分定级；
- 没有任何成绩：该年级的必测项目里本来就没有这一维的项目时记「本学段不测」，否则记「数据不足」。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.services.kb import standards, store
from app.services.kb.context import Fact
from app.services.kb.store import KnowledgeBaseUnavailable

NO_DATA = "数据不足"
# 该年级的必测项目里没有这一维的项目（如小学一二年级不测仰卧起坐）
NOT_TESTED = "本学段不测"

# 维度名 → 支撑它的体测项目（档案字段名；BMI 由身高体重算出）
DIMENSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("身体形态", ("BMI",)),
    ("心肺耐力", ("FHLCJ", "WSMWFPCJ", "BBMCJ", "YQMCJ")),
    ("肌肉力量与耐力", ("YTXSCJ", "YWQZCJ")),
    ("速度与灵敏", ("WSMCJ",)),
    ("爆发力", ("LDTYCJ", "YFZTSCJ")),
    ("柔韧性", ("ZWTQQCJ",)),
)
DIMENSION_NAMES = tuple(name for name, _keys in DIMENSIONS)


@dataclass
class Dimension:
    name: str
    facts: list[Fact] = field(default_factory=list)
    missing: list[standards.MissingItem] = field(default_factory=list)
    score: float | None = None
    level: str = NO_DATA
    weights: list[float] = field(default_factory=list)  # 多个项目加权平均时各项的权重，与 facts 对应


@dataclass
class Grading:
    info: dict[str, Any]
    assessment: standards.Assessment
    facts: list[Fact] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.info)

    @property
    def fingerprint(self) -> str:
        return str(self.info.get("fingerprint", ""))

    def not_weak_reason(self, dimension: Dimension) -> str | None:
        """这个维度按查表结果不该列为短板的原因；可以列为短板时返回 None。

        不列为短板：没有成绩的；等级在最高两档（优秀、良好）的；得分达到最高一档起始分的
        （体重指数「正常」是 100 分，等级词不在那套等级里，靠得分判断）。
        """
        if not dimension.facts:
            return f"没有成绩（{dimension.level}）"
        scale = self.assessment.scale
        good_levels = [level for _minimum, level in scale[:2]]
        top_line = scale[0][0] if scale else None
        if dimension.level in good_levels or (
            top_line is not None and dimension.score is not None and dimension.score >= top_line
        ):
            return f"评定为「{dimension.level}」"
        return None

    @property
    def weak_candidates(self) -> list[str]:
        """按查表结果可以列为短板的维度。"""
        return [dimension.name for dimension in self.dimensions if self.not_weak_reason(dimension) is None]

    @property
    def notes(self) -> list[str]:
        """评定里需要说明的数据问题：同一指标两处数值不同、有成绩但没能评分的项目。

        必测却没有得分的项目已经列在各维度与总分的说明里，这里不重复。
        """
        missing = {item.name for item in self.assessment.missing}
        skipped = [text for text in self.assessment.skipped if text.split("：", 1)[0] not in missing]
        return [*self.assessment.notes, *skipped]


def _dimension(name: str, keys: tuple[str, ...], facts: list[Fact], assessment: standards.Assessment) -> Dimension:
    picked = [fact for fact in facts if fact.item.key in keys]
    dimension = Dimension(
        name=name,
        facts=picked,
        missing=[item for item in assessment.missing if item.key in keys],
    )
    if len(picked) == 1:
        dimension.score = float(picked[0].item.score)
        dimension.level = picked[0].item.level
    elif picked:
        # 权重表里没有的项目按平均处理（权重表缺失时全部等权）
        weights = [assessment.weights.get(fact.item.name) for fact in picked]
        if any(weight is None for weight in weights):
            weights = [1.0] * len(picked)
        dimension.weights = [float(weight) for weight in weights]
        total = sum(fact.item.score * weight for fact, weight in zip(picked, dimension.weights))
        dimension.score = round(total / sum(dimension.weights), 1)
        dimension.level = assessment.level_of(dimension.score) or NO_DATA
    elif assessment.weights and not any(standards.ITEM_NAMES.get(key) in assessment.weights for key in keys):
        dimension.level = NOT_TESTED
    return dimension


def build(record: dict[str, Any]) -> Grading:
    """同步执行，由 load() 放到线程里并加超时。"""
    info = store.info()
    assessment = standards.assess(record)
    facts = [Fact(id=f"S{number}", item=item) for number, item in enumerate(assessment.items, 1)]
    return Grading(
        info=info,
        assessment=assessment,
        facts=facts,
        dimensions=[_dimension(name, keys, facts, assessment) for name, keys in DIMENSIONS],
    )


def offline(record: dict[str, Any]) -> Grading:
    """没有知识库时的空评定，仅供 mock 模式离线联调页面：六维全部「数据不足」。"""
    assessment = standards.Assessment(sex="", grade="", grade_inferred=False)
    assessment.height_cm, assessment.weight_kg = standards.body_size(record, assessment.notes)
    return Grading(
        info={},
        assessment=assessment,
        dimensions=[Dimension(name=name) for name in DIMENSION_NAMES],
    )


async def load(record: dict[str, Any]) -> Grading:
    """取一份档案的体质评定；知识库不可用或超时抛 KnowledgeBaseUnavailable。"""
    try:
        return await asyncio.wait_for(asyncio.to_thread(build, record), timeout=settings.kb_timeout)
    except asyncio.TimeoutError as exc:
        raise KnowledgeBaseUnavailable(f"知识库查表超时（>{settings.kb_timeout:g} 秒）") from exc
