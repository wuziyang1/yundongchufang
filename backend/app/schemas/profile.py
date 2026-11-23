"""体质画像输出契约（档案理解能力的产物）。

得分、等级、学年总分由代码按知识库里的《国家学生体质健康标准》查表得到（assessment 里的 S 条目），
模型只写解读文字：各维度的说明、综合评价的文字、优势与短板、健康关注点。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import AssessmentItem, KnowledgeBaseInfo, Versions

__all__ = ["ProfileResult", "EmergencyBlock", "Versions"]


class BasicEvaluation(BaseModel):
    model_config = ConfigDict(extra="allow")

    grade: str = ""
    age: int | None = None
    gender: str = ""
    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None
    bmi_evaluation: str = ""  # 国标体重指数评分表的等级词：正常 / 低体重 / 超重 / 肥胖
    bmi_ref: str = ""  # 对应的 S 条目编号


class DimensionIndicator(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    raw: str = ""
    unit: str = ""
    level: str = ""
    score: int | None = None
    ref: str = ""  # 对应的 S 条目编号；旧版留档没有


class FitnessDimension(BaseModel):
    model_config = ConfigDict(extra="allow")

    dimension: str
    indicators: list[DimensionIndicator] = Field(default_factory=list)
    level: str = ""
    # 维度得分：一个项目时就是它的得分，多个项目时按国标权重加权平均
    score: float | None = None
    missing: list[str] = Field(default_factory=list)  # 该维度必测但没有成绩的项目
    basis: str = ""  # 模型写的解读


class Weakness(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int | None = None
    dimension: str = ""
    reason: str = ""


class OverallEvaluation(BaseModel):
    model_config = ConfigDict(extra="allow")

    level: str = ""  # 综合等级，按学年总分评定；必测项目没测全时为「数据不足」
    score: float | None = None  # 学年总分 = 标准分 + 附加分
    standard_score: float | None = None
    bonus: int = 0
    score_basis: str = ""  # 总分的算式，或不能计算的原因
    score_source: str = ""  # 权重表出处
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[Weakness] = Field(default_factory=list)


class HealthFocus(BaseModel):
    model_config = ConfigDict(extra="allow")

    item: str = ""
    type: str = ""
    impact: str = ""
    attention: str = ""


class ProfileResult(BaseModel):
    """画像接口的完整返回体。"""

    model_config = ConfigDict(extra="allow")

    student_id: str = ""
    name: str = ""
    generated_at: str = ""
    versions: Versions = Field(default_factory=Versions)
    basic: BasicEvaluation = Field(default_factory=BasicEvaluation)
    fitness_dimensions: list[FitnessDimension] = Field(default_factory=list)
    overall: OverallEvaluation = Field(default_factory=OverallEvaluation)
    health_focus: list[HealthFocus] = Field(default_factory=list)
    # 以下由代码填写：国标查表结果与所用知识库
    assessment: list[AssessmentItem] = Field(default_factory=list)
    assessment_notes: list[str] = Field(default_factory=list)  # 数据问题：缺测、同一指标两处数值不同
    knowledge_base: KnowledgeBaseInfo = Field(default_factory=KnowledgeBaseInfo)
    disclaimer: str = ""


class EmergencyBlock(BaseModel):
    """急症熔断返回体，由代码规则触发，不经过模型。"""

    model_config = ConfigDict(extra="allow")

    blocked: bool = True
    level: str = "高"
    title: str = ""
    message: str = ""
    matched_keywords: list[str] = Field(default_factory=list)
    matched_text: list[str] = Field(default_factory=list)
    immediate_advice: list[str] = Field(default_factory=list)
    allowed_activity: list[str] = Field(default_factory=list)
    next_step: str = ""
    disclaimer: str = ""
