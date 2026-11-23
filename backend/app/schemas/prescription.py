"""运动处方输出契约（一人一策能力的产物）。

处方里每个带专业判断的条目都有 evidence_ids，指向本份处方 assessment（S 条目，国标查表结果）
或 evidence（E 条目，知识库摘录）里的编号；编号由代码核对，随处方一起留档，可逐条回溯出处。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import AssessmentItem, KnowledgeBaseInfo, KnowledgeDocument, KnowledgeTopic, Versions

__all__ = ["PrescriptionResult", "AssessmentItem", "KnowledgeBaseInfo", "KnowledgeDocument", "KnowledgeTopic"]


class FittItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    value: str = ""
    detail: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class FittVp(BaseModel):
    model_config = ConfigDict(extra="allow")

    frequency: FittItem = Field(default_factory=FittItem)
    intensity: FittItem = Field(default_factory=FittItem)
    time: FittItem = Field(default_factory=FittItem)
    type: FittItem = Field(default_factory=FittItem)
    volume: FittItem = Field(default_factory=FittItem)
    progression: FittItem = Field(default_factory=FittItem)


class PlanItem(BaseModel):
    """周计划里的一个条目：目录动作（exercise_id 来自知识库动作条目）或常规活动（exercise_id 为空）。"""

    model_config = ConfigDict(extra="allow")

    exercise_id: str = ""
    name: str = ""
    dosage: str = ""
    notes: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    # 由代码填写。常规活动的名称没有出现在任何一条摘录里时为 False，前端标注「待核实」
    verified: bool = True


class PlanBlock(BaseModel):
    model_config = ConfigDict(extra="allow")

    phase: str = ""
    minutes: int | None = None
    items: list[PlanItem] = Field(default_factory=list)


class DayPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    day: str = ""
    theme: str = ""
    total_minutes: int | None = None
    is_training_day: bool = True
    rationale: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    blocks: list[PlanBlock] = Field(default_factory=list)


class ExerciseAlternative(BaseModel):
    """仅旧版留档使用。"""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    why: str = ""


class ExerciseDetail(BaseModel):
    """被处方引用的动作的完整信息，由代码从知识库动作条目回填，不由模型生成。

    key_points / common_errors / impact_level / venue / alternatives 是旧版留档里的字段，
    保留只为旧处方仍能载入展示。
    """

    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    category: str = ""
    equipment: list[str] = Field(default_factory=list)
    purpose: str = ""
    start: str = ""
    steps: str = ""
    tips: str = ""
    safety: str = ""
    variants: str = ""
    dosage_hint: str = ""
    source: str = ""
    impact_level: str = ""
    venue: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)
    alternatives: list[ExerciseAlternative] = Field(default_factory=list)


class TargetHrZone(BaseModel):
    model_config = ConfigDict(extra="allow")

    lower: int | None = None
    upper: int | None = None
    formula_note: str = ""


class IntensityControl(BaseModel):
    model_config = ConfigDict(extra="allow")

    target_hr_zone: TargetHrZone = Field(default_factory=TargetHrZone)
    rpe_range: str = ""
    rpe_scale: str = ""
    talk_test: str = ""
    stop_signals: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class RiskNote(BaseModel):
    model_config = ConfigDict(extra="allow")

    level: str = ""
    item: str = ""
    reason: str = ""
    action: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class Restriction(BaseModel):
    model_config = ConfigDict(extra="allow")

    target: str = ""
    restriction: str = ""
    reason: str = ""
    source: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """权威资料摘录条目（E 条目）：本次检索到并交给模型的一段知识库原文。"""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    source: str = ""  # 《文档》章节 第 N 页
    doc_title: str = ""
    doc_kind: str = ""
    section: str = ""
    pages: str = ""
    topics: list[str] = Field(default_factory=list)  # 因哪些检索主题被选中
    content: str = ""
    score: float | None = None  # 相关性分数（重排原始分；降级时为向量相似度）
    cited_by: list[str] = Field(default_factory=list)  # 处方里引用它的位置；为空表示检索到但未被引用


class KnowledgeGap(BaseModel):
    """知识库未覆盖的问题：检索不到针对性资料，相关安排没有资料依据。"""

    model_config = ConfigDict(extra="allow")

    topic: str = ""
    note: str = ""


class PrescriptionResult(BaseModel):
    """处方接口的完整返回体。"""

    model_config = ConfigDict(extra="allow")

    student_id: str = ""
    name: str = ""
    generated_at: str = ""
    versions: Versions = Field(default_factory=Versions)
    goals: list[str] = Field(default_factory=list)
    fitt_vp: FittVp = Field(default_factory=FittVp)
    weekly_plan: list[DayPlan] = Field(default_factory=list)
    exercise_details: list[ExerciseDetail] = Field(default_factory=list)
    intensity_control: IntensityControl = Field(default_factory=IntensityControl)
    risk_notes: list[RiskNote] = Field(default_factory=list)
    forbidden_or_limited: list[Restriction] = Field(default_factory=list)
    unknown_exercise_ids: list[str] = Field(default_factory=list)
    # 以下由代码填写：依据链路
    assessment: list[AssessmentItem] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    knowledge_gaps: list[KnowledgeGap] = Field(default_factory=list)
    knowledge_base: KnowledgeBaseInfo = Field(default_factory=KnowledgeBaseInfo)
    disclaimer: str = ""
