"""档案理解服务：原始档案 → 体质画像。

体测成绩的得分与等级、体质六维的等级、学年总分与综合等级，由代码按知识库里的
《国家学生体质健康标准》查表得到（与处方用的是同一份查表结果）；模型只写解读文字：
各维度的说明、综合评价、优势与短板排序、健康关注点。模型输出里不含等级字段，
等级由代码填进结果，所以画像与处方对同一个学生的评定不会不一致。

知识库不可用时不生成画像（mock 模式离线联调除外）：等级必须来自查表，不退回让模型估计。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError

from app.llm.provider import LLMError, extract_json, get_provider
from app.prompts import profile as prompt
from app.schemas.profile import ProfileResult
from app.schemas.record import HealthRecord
from app.services import profile_grading
from app.services.common import current_versions, now_iso
from app.services.data_loader import clear_emergency, save_profile_result
from app.services.kb import standards
from app.services.kb.store import KnowledgeBaseUnavailable
from app.services.profile_grading import DIMENSION_NAMES, NO_DATA, Grading

DISCLAIMER = "本画像由 AI 基于所提供档案生成，仅供参考，不作为医疗诊断依据。"

# 短板最多列几条：画像页按处方优先级展示，列太多等于没有重点
MAX_WEAKNESSES = 3


# ---------------------------------------------------------------------------
# 模型输出的整理与校验
# ---------------------------------------------------------------------------


def _dimension_notes(payload: dict[str, Any]) -> dict[str, str]:
    """模型写的各维度解读：{维度名: 文字}。容忍写成对象或列表两种形式。"""
    raw = payload.get("dimension_notes")
    notes: dict[str, str] = {}
    if isinstance(raw, dict):
        for name, value in raw.items():
            text = value.get("basis") if isinstance(value, dict) else value
            notes[str(name).strip()] = str(text or "").strip()
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                notes[str(item.get("dimension") or "").strip()] = str(item.get("basis") or "").strip()
    return notes


def _weaknesses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}
    return [item for item in overall.get("weaknesses") or [] if isinstance(item, dict)]


def validate_payload(payload: dict[str, Any], grading: Grading) -> list[str]:
    """返回问题清单；为空表示通过。不通过则带问题清单重生成一次。"""
    problems: list[str] = []
    if not isinstance(payload.get("overall"), dict):
        problems.append("缺少 overall 对象")

    missing = [name for name in DIMENSION_NAMES if not _dimension_notes(payload).get(name)]
    if missing:
        problems.append(f"dimension_notes 缺少以下维度的解读：{'、'.join(missing)}")

    dimensions = {dimension.name: dimension for dimension in grading.dimensions}
    for item in _weaknesses(payload):
        name = str(item.get("dimension") or "").strip()
        if name not in dimensions:
            problems.append(f"weaknesses 的 dimension「{name}」不是体质六维的名称")
        elif grading.available and (reason := grading.not_weak_reason(dimensions[name])):
            problems.append(f"「{name}」{reason}，不应列为短板")
    if len(_weaknesses(payload)) > MAX_WEAKNESSES:
        problems.append(f"weaknesses 最多 {MAX_WEAKNESSES} 条，只保留最需要优先改善的")
    return problems


# ---------------------------------------------------------------------------
# 组装结果：等级、得分、总分由代码填写
# ---------------------------------------------------------------------------


def _basic(record: HealthRecord, grading: Grading) -> dict[str, Any]:
    assessment = grading.assessment
    exam, fitness = record.physical_exam, record.fitness_test
    bmi = next((fact for fact in grading.facts if fact.item.key == "BMI"), None)
    height, weight = assessment.height_cm, assessment.weight_kg
    value = bmi.item.value if bmi else (round(weight / (height / 100) ** 2, 1) if height and weight else None)
    return {
        "grade": (exam.SCHOOL_GRADE or fitness.NJ or assessment.grade).strip(),
        "age": standards.to_int(exam.NL),
        "gender": (exam.XB or fitness.XB).strip(),
        "height_cm": height,
        "weight_kg": weight,
        "bmi": value,
        "bmi_evaluation": bmi.item.level if bmi else "",
        "bmi_ref": bmi.id if bmi else "",
    }


def _assemble(payload: dict[str, Any], *, record: HealthRecord, grading: Grading) -> ProfileResult:
    assessment = grading.assessment
    notes = _dimension_notes(payload)
    overall = payload.get("overall") if isinstance(payload.get("overall"), dict) else {}

    # 模型列的短板里，查表结果不支持的（维度名不对、等级良好以上、没有成绩）和超出条数上限的去掉，再重排序号
    dimensions = {dimension.name: dimension for dimension in grading.dimensions}
    weaknesses = []
    for item in _weaknesses(payload):
        name = str(item.get("dimension") or "").strip()
        supported = name in dimensions and not (grading.available and grading.not_weak_reason(dimensions[name]))
        if supported and len(weaknesses) < MAX_WEAKNESSES:
            weaknesses.append({"rank": len(weaknesses) + 1, "dimension": name, "reason": str(item.get("reason") or "")})

    result = {
        "student_id": record.student_id,
        "name": record.name,
        "generated_at": now_iso(),
        "versions": current_versions(grading.fingerprint).model_dump(),
        "basic": _basic(record, grading),
        "fitness_dimensions": [
            {
                "dimension": dimension.name,
                "level": dimension.level,
                "score": dimension.score,
                "indicators": [
                    {
                        "name": fact.item.name,
                        "raw": fact.item.raw,
                        "unit": fact.item.unit,
                        "level": fact.item.level,
                        "score": fact.item.score,
                        "ref": fact.id,
                    }
                    for fact in dimension.facts
                ],
                "missing": [item.text for item in dimension.missing],
                "basis": notes.get(dimension.name, ""),
            }
            for dimension in grading.dimensions
        ],
        "overall": {
            "level": assessment.level or NO_DATA,
            "score": assessment.total,
            "standard_score": assessment.standard_score,
            "bonus": assessment.bonus,
            "score_basis": assessment.total_basis or "；".join(assessment.skipped),
            "score_source": assessment.total_source,
            "summary": str(overall.get("summary") or ""),
            "strengths": [str(item) for item in overall.get("strengths") or [] if str(item).strip()],
            "weaknesses": weaknesses,
        },
        "health_focus": payload.get("health_focus") or [],
        "assessment": [fact.to_dict() for fact in grading.facts],
        "assessment_notes": grading.notes,
        "knowledge_base": {
            "available": grading.available,
            "fingerprint": grading.fingerprint,
            "collection": grading.info.get("collection", ""),
            "embedding_model": grading.info.get("embedding_model", ""),
            "documents": [
                {"title": doc["title"], "kind": doc["kind"], "md5": doc["md5"]}
                for doc in grading.info.get("documents", [])
            ],
        },
        "disclaimer": DISCLAIMER,
    }
    return ProfileResult.model_validate(result)


def _grading_summary(grading: Grading) -> str:
    assessment = grading.assessment
    if assessment.total is not None:
        total = f"学年总分 {assessment.total:g}（{assessment.level}）"
    elif assessment.missing:
        total = f"必测项目缺 {len(assessment.missing)} 项，未计算学年总分"
    else:
        total = "未计算学年总分"
    return f"国标查表完成：{len(grading.facts)} 项得分，{total}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def stream_profile(record: HealthRecord) -> AsyncIterator[tuple[str, Any]]:
    """流式生成画像。

    产出 ("status", 说明) / ("thinking", 片段) / ("token", 片段) 供前端实时展示，
    ("retry", 原因) 表示触发了一次重生成，最后产出 ("result", ProfileResult) 或 ("error", 说明)。
    """
    provider = get_provider()
    record_view = record.to_prompt_dict()

    yield ("status", "正在按国标评分表查表")
    try:
        grading = await profile_grading.load(record_view)
    except KnowledgeBaseUnavailable as exc:
        if provider.name != "mock":
            yield ("error", f"知识库不可用，无法按国标评分表评定体质：{exc}")
            return
        # mock 模式只为离线联调页面：没有知识库也放行，六维全部「数据不足」
        grading = profile_grading.offline(record_view)
    if grading.available:
        yield ("status", _grading_summary(grading))

    user_prompt = prompt.build_user_prompt(record_view, grading)
    last_error = ""

    for attempt in range(2):
        if attempt == 1:
            yield ("retry", last_error)
            user_prompt = prompt.build_repair_prompt(record_view, grading, last_error)

        buffer: list[str] = []
        try:
            async for kind, chunk in provider.stream(prompt.SYSTEM, user_prompt):
                if kind == "thinking":
                    yield ("thinking", chunk)
                    continue
                buffer.append(chunk)
                yield ("token", chunk)
            payload = extract_json("".join(buffer))
        except LLMError as exc:
            last_error = str(exc)
            continue

        problems = validate_payload(payload, grading)
        if problems and attempt == 0:
            last_error = "\n".join(problems)
            continue

        try:
            result = _assemble(payload, record=record, grading=grading)
        except ValidationError as exc:
            last_error = exc.json(indent=2)
            continue

        try:
            # mock 占位结果不落盘，避免污染 data/profiles 的真实画像留档
            if provider.name != "mock":
                save_profile_result(result.model_dump())
            # 生成成功即解除熔断：恢复正常留档展示
            clear_emergency(record.student_id)
        except OSError:
            # 落盘只是旁路留档，磁盘问题不应中断画像返回
            pass

        yield ("result", result)
        return

    yield ("error", f"画像生成两次均未通过校验：{last_error}")


async def analyze_profile(record: HealthRecord) -> ProfileResult:
    """非流式入口：消费 stream_profile，供同步端点与脚本使用。"""
    error = "画像生成失败"
    async for kind, payload in stream_profile(record):
        if kind == "result":
            return payload
        if kind == "error":
            error = str(payload)
    raise LLMError(error)
