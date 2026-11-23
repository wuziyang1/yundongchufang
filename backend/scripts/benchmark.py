"""耗时拆解基准：把一次画像/处方调用的时间花在哪里量出来。

用法： uv run python -u scripts/benchmark.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import AsyncAnthropic  # noqa: E402

from app.config import settings  # noqa: E402
from app.prompts import prescription as rx_prompt  # noqa: E402
from app.prompts import profile as pf_prompt  # noqa: E402
from app.schemas.record import HealthRecord  # noqa: E402
from app.services.data_loader import find_student  # noqa: E402
from app.services.kb import context as kb_context  # noqa: E402
from app.services.prescription_service import parse_age  # noqa: E402

STUDENT_ID = "BJ2025090314"

# 要对比的 (模型, 是否开思考)。关思考是最大的延迟杠杆，优先保证这两格有数。
MATRIX = [
    ("glm-5.3", True),
    ("glm-5.3", False),
]


async def build_profile_prompt() -> tuple[str, str]:
    record = HealthRecord.model_validate(find_student(STUDENT_ID))
    record_view = record.to_prompt_dict()
    return pf_prompt.SYSTEM, pf_prompt.build_user_prompt(record_view)


async def build_prescription_prompt() -> tuple[str, str]:
    """处方 prompt 由原始档案与知识库材料构造，与线上链路一致（需要 Milvus 已启动）。"""
    record = HealthRecord.model_validate(find_student(STUDENT_ID))
    record_view = record.to_prompt_dict()
    t0 = time.time()
    context = await kb_context.load(record_view)
    print(f"知识库检索 {time.time() - t0:.1f}s：{len(context.evidence)} 条摘录，"
          f"{len(context.candidates)} 个候选动作，{len(context.gaps)} 个主题未覆盖")
    return rx_prompt.SYSTEM, rx_prompt.build_user_prompt(
        record=record_view,
        age=parse_age(record),
        context=context,
    )


async def run(client: AsyncAnthropic, model: str, system: str, user: str,
              thinking: bool, label: str) -> None:
    kwargs: dict = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": settings.max_tokens,
    }
    if not thinking:
        kwargs["thinking"] = {"type": "disabled"}
        kwargs["extra_body"] = {"temperature": settings.temperature}

    first_token_at = None
    text_chars = 0
    think_chars = 0
    t0 = time.time()
    try:
        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type != "content_block_delta":
                    continue
                kind = getattr(event.delta, "type", "")
                if kind == "thinking_delta":
                    if first_token_at is None:
                        first_token_at = time.time()
                    think_chars += len(getattr(event.delta, "thinking", "") or "")
                elif kind == "text_delta":
                    if first_token_at is None:
                        first_token_at = time.time()
                    text_chars += len(event.delta.text)
            final = await stream.get_final_message()
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:34} 失败: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        return

    total = time.time() - t0
    ttft = (first_token_at - t0) if first_token_at else total
    usage = final.usage
    out = usage.output_tokens
    think_est = round(out * think_chars / max(think_chars + text_chars, 1))
    speed = out / total if total else 0

    print(
        f"  {label:34} {total:6.1f}s  首字 {ttft:5.1f}s  "
        f"入 {usage.input_tokens:6}  出 {out:6}"
        f"（思考约 {think_est:5} / 正文约 {out - think_est:5}）"
        f"  {speed:5.1f} tok/s  stop={final.stop_reason}",
        flush=True,
    )


async def main() -> None:
    client = AsyncAnthropic(
        base_url=settings.anthropic_base_url,
        api_key=settings.anthropic_api_key,
        timeout=settings.timeout,
    )

    pf_sys, pf_user = await build_profile_prompt()
    rx_sys, rx_user = await build_prescription_prompt()
    print(f"画像 prompt 约 {len(pf_sys) + len(pf_user)} 字符；"
          f"处方 prompt 约 {len(rx_sys) + len(rx_user)} 字符\n")

    for stage, (s, u) in (("画像", (pf_sys, pf_user)), ("处方", (rx_sys, rx_user))):
        print(f"=== {stage} ===")
        for model, thinking in MATRIX:
            tag = f"{model} {'思考开' if thinking else '思考关'}"
            await run(client, model, s, u, thinking, tag)
        print()


if __name__ == "__main__":
    asyncio.run(main())
