"""知识库（Milvus）自检：不调用大模型，打印一份处方会从知识库拿到哪些材料。

用法：
    uv run python -u scripts/check_kb.py                         # 默认三位学生，逐个打印明细
    uv run python -u scripts/check_kb.py BJ2025200148            # 指定学号
    uv run python -u scripts/check_kb.py BJ2025200148 --prompt   # 再打印完整的处方 prompt
    uv run python -u scripts/check_kb.py --all                   # 全部学生的主题覆盖表（标定阈值用）

明细包括：知识库状态、国标查表结果、每个检索主题的最高相关性分数与命中的摘录、
知识库未覆盖的问题、候选动作数量与针对短板检索到的动作、prompt 长度。

标定相关性阈值：换了重排服务或嵌入模型后，跑 --all，看「✓ 覆盖」与「✗ 未覆盖」两组的
最高分各落在什么范围，把 KB_RERANK_MIN_SCORE（或 KB_DENSE_MIN_SCORE）设在两组之间。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.prompts import prescription as prompt  # noqa: E402
from app.schemas.record import HealthRecord  # noqa: E402
from app.services.data_loader import all_students, find_student  # noqa: E402
from app.services.kb import context as kb_context  # noqa: E402
from app.services.kb import rerank, store  # noqa: E402
from app.services.kb.store import KnowledgeBaseUnavailable  # noqa: E402
from app.services.prescription_service import parse_age  # noqa: E402

# 李沐阳（肥胖 + 哮喘 + 踝伤）、刘星月（膝痛恢复期）、彭语嫣（肥胖 + 小腿酸痛）
DEFAULT_STUDENTS = ["BJ2025090314", "BJ2025110267", "BJ2025200148"]


def _threshold_note(mode: str) -> str:
    if mode == "dense":
        return f"向量相似度 ≥ {settings.kb_dense_min_score:g}（重排不可用：{rerank.problem()}）"
    return f"重排原始分 ≥ {settings.kb_rerank_min_score:g}，健康状况类主题再加 1"


def check(student_id: str, *, show_prompt: bool) -> None:
    raw = find_student(student_id)
    if raw is None:
        print(f"找不到学生 {student_id}")
        return
    record = HealthRecord.model_validate(raw)
    view = record.to_prompt_dict()
    context = kb_context.build(view)

    print("=" * 100)
    print(f"{student_id} {raw.get('name', '')}　学段：{context.stage}　检索耗时 {context.elapsed:.1f}s")

    print("\n— 体质评定（国标评分表查表）—")
    for fact in context.facts:
        item = fact.item
        print(f"  [{fact.id}] {item.name}：{item.raw} → {item.score} 分「{item.level}」  {item.source}")
    for reason in context.assessment.skipped:
        print(f"  未评分：{reason}")
    weak = context.assessment.weak_items
    print("  短板：" + ("、".join(f"{item.name} {item.score} 分" for item in weak) or "无"))

    print(f"\n— 检索主题（把关：{_threshold_note(context.retrieval_mode)}）—")
    for topic in context.topics:
        mark = "✓" if topic.covered else "✗"
        print(f"  {mark} {topic.label}　最高分 {topic.top_score}　{'、'.join(topic.evidence_ids) or '未覆盖'}")

    print(f"\n— 权威资料摘录（{len(context.evidence)} 条）—")
    for entry in context.evidence:
        print(f"  [{entry.id}] {entry.hit.score}  {entry.hit.source}")
        print(f"        {entry.hit.content[:110]}…")

    print(f"\n— 可选动作：候选 {len(context.candidates)} 个（可得器材：{'、'.join(settings.kb_equipment)}）—")
    counts: dict[str, int] = {}
    for exercise in context.candidates:
        counts[exercise.category] = counts.get(exercise.category, 0) + 1
    print("  " + "、".join(f"{name} {count}" for name, count in counts.items()))
    for rec in context.recommendations:
        names = "、".join(f"{exercise.code} {exercise.name}" for exercise in rec.exercises)
        print(f"  针对「{rec.item.name}」：{names}")

    user_prompt = prompt.build_user_prompt(record=view, age=parse_age(record), context=context)
    print(f"\nprompt 长度：system {len(prompt.SYSTEM)} 字符 + user {len(user_prompt)} 字符")
    if show_prompt:
        print("\n" + "-" * 100 + "\n" + user_prompt)


def coverage_table() -> None:
    """全部学生的主题覆盖情况：每个主题在多少份档案里出现、覆盖了多少次、最高分的范围。"""
    rows: dict[str, dict[str, list[float]]] = {}
    for student in all_students():
        view = HealthRecord.model_validate(student).to_prompt_dict()
        context = kb_context.build(view)
        gaps = "、".join(topic.label for topic in context.gaps) or "无"
        print(f"{student['student_id']} {student.get('name', ''):<4} 摘录 {len(context.evidence):>2} 条　未覆盖：{gaps}")
        for topic in context.topics:
            # 短板、健康状况按类别汇总，固定主题按名称汇总
            name = topic.label.split("：")[0] if "：" in topic.label else topic.label.split("（")[0]
            bucket = rows.setdefault(name, {"covered": [], "missed": []})
            if topic.top_score is not None:
                bucket["covered" if topic.covered else "missed"].append(topic.top_score)

    def span(values: list[float]) -> str:
        return f"{min(values):.2f}–{max(values):.2f}（{len(values)} 次）" if values else "—"

    print("\n主题　　　　　　　　　　　　　　✓ 覆盖时的最高分　　　　　✗ 未覆盖时的最高分")
    for name, bucket in rows.items():
        print(f"  {name:<22} {span(bucket['covered']):<26} {span(bucket['missed'])}")


def main(args: list[str]) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    status = store.status()
    summary = {key: value for key, value in status.items() if key != "documents"}
    print("知识库状态：", json.dumps(summary, ensure_ascii=False))
    if not status["ready"]:
        return 1
    for doc in status["documents"]:
        print(f"  ［{doc['kind'] or '未登记类别'}｜{'/'.join(doc['stages'])}］{doc['title']}："
              f"{doc['chunks']} 个切片" + (f"，其中动作条目 {doc['exercises']} 个" if doc["exercises"] else ""))
    print(f"相关性把关：{rerank.problem() and f'向量相似度（重排不可用：{rerank.problem()}）' or f'重排模型 {settings.rerank_model}'}")

    flags = {arg for arg in args if arg.startswith("--")}
    ids = [arg for arg in args if not arg.startswith("--")]
    try:
        if "--all" in flags:
            coverage_table()
        else:
            for student_id in ids or DEFAULT_STUDENTS:
                check(student_id, show_prompt="--prompt" in flags)
    except KnowledgeBaseUnavailable as exc:
        print(f"知识库不可用：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
