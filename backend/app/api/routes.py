"""API 端点。

每一层单独成端点，便于分层调试与后续单独评测。
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.llm.provider import LLMError
from app.schemas.record import HealthRecord
from app.services import prescription_service, profile_service
from app.services.common import current_versions
from app.services.data_loader import (
    all_students,
    find_student,
    load_saved_result,
    record_emergency,
    save_student,
    saved_result_ids,
)
from app.services.kb import exercises as kb_exercises
from app.services.kb import rerank, store
from app.services.kb.store import KnowledgeBaseUnavailable
from app.services.safety import check_emergency

router = APIRouter(prefix="/api/v1")


class StudentSummary(BaseModel):
    student_id: str
    name: str
    school: str
    grade: str
    gender: str
    age: str
    height_cm: str
    weight_kg: str
    highlights: list[str]


class ProfileRequest(BaseModel):
    record: HealthRecord


class PrescriptionRequestBody(BaseModel):
    record: HealthRecord


def _highlights(student: dict[str, Any]) -> list[str]:
    profile = student.get("health_profile") or {}
    notes: list[str] = []
    notes.extend(profile.get("chronic_conditions") or [])
    for injury in profile.get("injury_history") or []:
        notes.append(f"{injury.get('site', '')}{injury.get('type', '')}（{injury.get('status', '')}）")
    notes.extend(profile.get("recent_symptoms") or [])
    return notes[:3]


def _knowledge_base_status() -> dict[str, Any]:
    """知识库（Milvus）当前状态：是否可用、指纹、文档清单、相关性把关方式。"""
    status = store.status()
    status["rerank"] = {
        "model": settings.rerank_model,
        "problem": rerank.problem(),
        "min_score": settings.kb_rerank_min_score,
    }
    return status


@router.get("/meta")
def get_meta() -> dict[str, Any]:
    """前端启动时拉取的元信息：版本、模型、知识库状态。"""
    return {
        "llm_provider": settings.llm_provider,
        "model": settings.active_model,
        "versions": current_versions().model_dump(),
        "student_count": len(all_students()),
        "knowledge_base": _knowledge_base_status(),
    }


@router.get("/students", response_model=list[StudentSummary])
def list_students() -> list[StudentSummary]:
    summaries: list[StudentSummary] = []
    for student in all_students():
        exam = student.get("physical_exam") or {}
        summaries.append(
            StudentSummary(
                student_id=student["student_id"],
                name=student.get("name", ""),
                school=student.get("school", ""),
                grade=exam.get("SCHOOL_GRADE", ""),
                gender=exam.get("XB", ""),
                age=exam.get("NL", ""),
                height_cm=exam.get("SG", ""),
                weight_kg=exam.get("TZ", ""),
                highlights=_highlights(student),
            )
        )
    return summaries


@router.post("/students")
def create_student(record: HealthRecord) -> dict[str, Any]:
    """把前端录入的新档案追加进 mock_students.json，返回带服务端学号的完整档案。

    保存后该学生即出现在 /students 列表中，可直接走画像与处方生成流程。
    """
    return save_student(record.model_dump())


@router.get("/students/{student_id}")
def get_student(student_id: str) -> dict[str, Any]:
    student = find_student(student_id)
    if student is None:
        raise HTTPException(status_code=404, detail=f"未找到档案 {student_id}")
    return student


@router.get("/knowledge-base")
def get_knowledge_base() -> dict[str, Any]:
    """知识库（Milvus）状态与文档清单。处方的评分表、权威资料摘录、动作都来自这里。"""
    return _knowledge_base_status()


@router.get("/exercises")
def list_exercises() -> dict[str, Any]:
    """知识库里的全部动作条目（训练手册的练习），附可选器材清单。"""
    try:
        catalog = kb_exercises.catalog()
    except KnowledgeBaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"知识库不可用：{exc}") from exc
    return {
        "source": "knowledge_base",
        "fingerprint": store.info()["fingerprint"],
        "available_equipment": settings.kb_equipment,
        "count": len(catalog),
        "exercises": [
            {**exercise.detail(), "group": exercise.group} for exercise in catalog.values()
        ],
    }


@router.get("/exercises/{exercise_id}")
def get_exercise(exercise_id: str) -> dict[str, Any]:
    try:
        exercise = kb_exercises.catalog().get(exercise_id)
    except KnowledgeBaseUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"知识库不可用：{exc}") from exc
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"未找到动作 {exercise_id}")
    return {**exercise.detail(), "group": exercise.group}


@router.get("/pregenerated")
def list_saved_results() -> dict[str, Any]:
    """有留档结果可直接载入的学生清单（profiles ∪ prescriptions）。"""
    items = []
    for student_id in saved_result_ids():
        data = load_saved_result(student_id)
        if data is None:
            continue
        items.append(
            {
                "student_id": student_id,
                "name": data.get("name", ""),
                "generated_at": data.get("generated_at", ""),
                "has_prescription": data.get("prescription") is not None,
            }
        )
    return {"count": len(items), "items": items}


@router.get("/pregenerated/{student_id}")
def get_saved_result(student_id: str) -> dict[str, Any]:
    """读取留档的画像与处方（由 profiles/ 与 prescriptions/ 拼装），不调用模型。"""
    data = load_saved_result(student_id)
    if data is None:
        raise HTTPException(
            status_code=404, detail=f"{student_id} 还没有生成过结果，请点「生成处方」实时生成"
        )
    return data


@router.post("/profile:analyze")
async def analyze_profile(body: ProfileRequest) -> dict[str, Any]:
    """健康档案理解：国标查表 + 体质画像。"""
    record_dict = body.record.model_dump()

    emergency = check_emergency(record_dict)
    if emergency is not None:
        record_emergency(
            body.record.student_id, body.record.name, emergency.model_dump()
        )
        return {"emergency": emergency.model_dump(), "profile": None}

    try:
        profile = await profile_service.analyze_profile(body.record)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"emergency": None, "profile": profile.model_dump()}


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/profile:analyze:stream")
async def analyze_profile_stream(body: ProfileRequest) -> StreamingResponse:
    """健康档案理解，以 SSE 流式返回模型输出过程。

    事件类型：
      status     进度说明（开始生成、国标查表结果）
      token      模型原始输出片段（phase=profile）
      retry      触发了一次重生成
      emergency  急症熔断
      result     最终画像 JSON
      error      生成失败（含知识库不可用）
      done       流结束
    """

    async def event_stream():
        record_dict = body.record.model_dump()

        emergency = check_emergency(record_dict)
        if emergency is not None:
            record_emergency(
                body.record.student_id, body.record.name, emergency.model_dump()
            )
            yield _sse("emergency", emergency.model_dump())
            yield _sse("done", {"ok": False})
            return

        yield _sse("status", {"message": "正在生成体质画像", "phase": "profile"})
        try:
            async for kind, payload in profile_service.stream_profile(body.record):
                if kind == "status":
                    yield _sse("status", {"message": payload, "phase": "profile"})
                elif kind == "thinking":
                    yield _sse("thinking", {"text": payload, "phase": "profile"})
                elif kind == "token":
                    yield _sse("token", {"text": payload, "phase": "profile"})
                elif kind == "retry":
                    yield _sse("retry", {"problems": payload, "phase": "profile"})
                elif kind == "result":
                    yield _sse("result", payload.model_dump())
                    yield _sse("done", {"ok": True})
                    return
                elif kind == "error":
                    yield _sse("error", {"message": payload})
                    yield _sse("done", {"ok": False})
                    return
        except LLMError as exc:
            yield _sse("error", {"message": str(exc)})

        yield _sse("done", {"ok": False})

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.post("/prescription:generate")
async def generate_prescription(body: PrescriptionRequestBody) -> StreamingResponse:
    """一人一策运动处方，以 SSE 流式返回。

    事件类型：
      status  进度说明（开始生成、知识库检索结果）
      token   模型原始输出片段
      retry   触发了一次重生成，data 为问题清单
      result  最终处方 JSON
      error   生成失败（含知识库不可用）
      done    流结束
    """

    async def event_stream():
        record_dict = body.record.model_dump()

        emergency = check_emergency(record_dict)
        if emergency is not None:
            record_emergency(
                body.record.student_id, body.record.name, emergency.model_dump()
            )
            yield _sse("emergency", emergency.model_dump())
            yield _sse("done", {"ok": False})
            return

        yield _sse("status", {"message": "正在生成运动处方", "phase": "prescription"})
        try:
            async for kind, payload in prescription_service.stream_prescription(body.record):
                if kind == "status":
                    yield _sse("status", {"message": payload, "phase": "prescription"})
                elif kind == "thinking":
                    yield _sse("thinking", {"text": payload, "phase": "prescription"})
                elif kind == "token":
                    yield _sse("token", {"text": payload, "phase": "prescription"})
                elif kind == "retry":
                    yield _sse("retry", {"problems": payload, "phase": "prescription"})
                elif kind == "result":
                    yield _sse("result", payload.model_dump())
                    yield _sse("done", {"ok": True})
                    return
                elif kind == "error":
                    yield _sse("error", {"message": payload})
                    yield _sse("done", {"ok": False})
                    return
        except LLMError as exc:
            yield _sse("error", {"message": str(exc)})

        yield _sse("done", {"ok": False})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
