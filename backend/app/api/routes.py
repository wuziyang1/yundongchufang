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
from app.services import prescription_service, profile_service, rag
from app.services.common import current_versions
from app.services.data_loader import (
    all_students,
    record_emergency,
    exercise_index,
    exercise_library,
    find_student,
    knowledge_base,
    knowledge_cards,
    load_saved_result,
    save_student,
    saved_result_ids,
)
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


@router.get("/meta")
def get_meta() -> dict[str, Any]:
    """前端启动时拉取的元信息：版本、模型、资产规模。"""
    return {
        "llm_provider": settings.llm_provider,
        "model": settings.active_model,
        "versions": current_versions().model_dump(),
        "exercise_count": len(exercise_library()["exercises"]),
        "knowledge_card_count": len(knowledge_cards()),
        "student_count": len(all_students()),
        "contraindication_tags": exercise_library()["contraindication_tag_vocabulary"],
        "rag": rag.status(),
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


@router.get("/exercises")
def list_exercises() -> dict[str, Any]:
    return exercise_library()


@router.get("/exercises/{exercise_id}")
def get_exercise(exercise_id: str) -> dict[str, Any]:
    exercise = exercise_index().get(exercise_id)
    if exercise is None:
        raise HTTPException(status_code=404, detail=f"未找到动作 {exercise_id}")
    return exercise


@router.get("/knowledge")
def list_knowledge() -> dict[str, Any]:
    return knowledge_base()


@router.get("/knowledge/{card_id}")
def get_knowledge(card_id: str) -> dict[str, Any]:
    for card in knowledge_cards():
        if card["id"] == card_id:
            return card
    raise HTTPException(status_code=404, detail=f"未找到知识卡片 {card_id}")


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
    """健康档案理解：归一化 + 体质画像 + 安全评估。"""
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
      status     进度说明
      token      模型原始输出片段（phase=profile）
      retry      触发了一次重生成
      emergency  急症熔断
      result     最终画像 JSON
      error      生成失败
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
                if kind == "thinking":
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
      status  进度说明
      token   模型原始输出片段
      retry   触发了一次重生成，data 为问题清单
      result  最终处方 JSON
      error   生成失败
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
                if kind == "thinking":
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
