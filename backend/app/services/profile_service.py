"""档案理解服务：原始档案 → 体质画像。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError

from app.llm.provider import LLMError, extract_json, get_provider
from app.prompts import profile as prompt
from app.schemas.profile import ProfileResult
from app.schemas.record import HealthRecord
from app.services.common import current_versions, now_iso
from app.services.data_loader import clear_emergency, save_profile_result

DISCLAIMER = "本画像由 AI 基于所提供档案生成，仅供参考，不作为医疗诊断依据。"


async def stream_profile(record: HealthRecord) -> AsyncIterator[tuple[str, Any]]:
    """流式生成画像。

    产出 ("thinking", 片段) / ("token", 片段) 供前端实时展示模型输出，
    ("retry", 原因) 表示触发了一次重生成，最后产出 ("result", ProfileResult)
    或 ("error", 说明)。
    """
    provider = get_provider()
    record_view = record.to_prompt_dict()

    user_prompt = prompt.build_user_prompt(record_view)
    last_error = ""

    for attempt in range(2):
        if attempt == 1:
            yield ("retry", last_error)
            user_prompt = prompt.build_repair_prompt(record_view, last_error)

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

        payload.update(
            {
                "student_id": record.student_id,
                "name": record.name,
                "generated_at": now_iso(),
                "versions": current_versions().model_dump(),
                "disclaimer": DISCLAIMER,
            }
        )

        try:
            result = ProfileResult.model_validate(payload)
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

    yield ("error", f"画像生成两次均未通过结构校验：{last_error}")


async def analyze_profile(record: HealthRecord) -> ProfileResult:
    """非流式入口：消费 stream_profile，供同步端点与脚本使用。"""
    error = "画像生成失败"
    async for kind, payload in stream_profile(record):
        if kind == "result":
            return payload
        if kind == "error":
            error = str(payload)
    raise LLMError(error)
