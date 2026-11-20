"""权威资料知识库（Milvus）检索自检：不调用大模型，打印检索查询、命中的摘录与注入处方 prompt 的片段。

用法：
    uv run python -u scripts/check_kb.py                              # 默认：哮喘+踝伤、膝伤恢复期、无伤病
    uv run python -u scripts/check_kb.py BJ2025140258 BJ2025080412    # 指定学号

Milvus 未启动、没有嵌入密钥或 mock 模式时，检索会降级为空，这里会打印跳过原因。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prompts import prescription as prompt  # noqa: E402
from app.schemas.record import HealthRecord  # noqa: E402
from app.services import literature  # noqa: E402
from app.services.data_loader import find_student  # noqa: E402

DEFAULT_STUDENTS = ["BJ2025090314", "BJ2025110267", "BJ2025050137"]


async def check(student_id: str) -> None:
    raw = find_student(student_id)
    if raw is None:
        print(f"找不到学生 {student_id}")
        return
    record = HealthRecord.model_validate(raw).to_prompt_dict()
    health = record.get("health_profile") or {}
    print("=" * 100)
    print(f"{student_id} {raw.get('name', '')}  慢病={health.get('chronic_conditions')}  "
          f"伤病={[i.get('site', '') + i.get('type', '') for i in health.get('injury_history') or []]}  "
          f"症状={health.get('recent_symptoms')}")
    print("查询：")
    for kind, text in literature.build_queries(record):
        print(f"  [{kind}] {text}")

    excerpts = await literature.retrieve_excerpts(record)
    print(f"命中 {len(excerpts)} 条：")
    for item in excerpts:
        print(f"  {item['score']:.3f}  {item['source']}")

    user_prompt = prompt.build_user_prompt(
        record=record, age=None, candidates=[], knowledge_cards=[], excerpts=excerpts
    )
    start = user_prompt.find("权威资料摘录")
    end = user_prompt.find("可用动作候选集")
    print("\n注入 prompt 的片段：")
    print(user_prompt[start:end].strip() if start >= 0 else "（无摘录，prompt 与未接入知识库时一致）")


async def main(student_ids: list[str]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("知识库状态：", json.dumps(literature.status(), ensure_ascii=False))
    for student_id in student_ids:
        await check(student_id)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or DEFAULT_STUDENTS))
