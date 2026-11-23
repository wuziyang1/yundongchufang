"""处方生成服务：原始健康档案 → 一人一策运动处方。

专业依据全部来自知识库（Milvus）：体测成绩由代码按国标评分表查表，频率、强度、时间等决定
依据按主题检索到的权威资料摘录，动作只能从知识库训练手册的练习条目里选。模型负责理解档案、
取舍与排期，并在每个决定上写明依据的条目编号；代码负责核对：

- 动作编号必须在候选集内，周计划完整，训练量不超上限；
- 引用的编号必须真实存在，FITT-VP 六项每项至少引用一条摘录；
  以上任一条不满足，带问题清单重生成一次；
- 目录里没有的常规活动（快步走、慢跑等），名称要能在摘录原文里找到，找不到的标记「待核实」；
- 检索不到针对性资料的问题，原样记入 knowledge_gaps，随处方交给审核人。

知识库不可用时不生成处方（mock 模式离线联调除外）：处方必须有依据，不退回只靠模型记忆。
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

from pydantic import ValidationError

from app.llm.provider import LLMError, LLMTruncatedError, extract_json, get_provider
from app.prompts import prescription as prompt
from app.schemas.prescription import PrescriptionResult, RiskNote
from app.schemas.record import HealthRecord
from app.services.common import current_versions, now_iso
from app.services.data_loader import clear_emergency, save_prescription_result
from app.services.kb import context as kb_context
from app.services.kb.context import KnowledgeContext
from app.services.kb.store import KnowledgeBaseUnavailable

DISCLAIMER = (
    "本运动处方由 AI 生成，仅供参考，不作为医疗诊断依据。"
    "执行中出现不适请立即停止并咨询专业人员。"
)

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 训练量内置上限：学生不填写可用时间，超限触发一次重生成
MAX_TRAINING_DAYS = 6
MAX_SESSION_MINUTES = 90
MAX_SESSION_MINUTES_UNDER_12 = 60

FITT_LABELS = {
    "frequency": "运动频率",
    "intensity": "运动强度",
    "time": "运动时间",
    "type": "运动类型",
    "volume": "周运动总量",
    "progression": "运动进阶",
}

# NL 为空时按年级估算年龄，用于单次时长上限与 prompt 提示
_GRADE_AGE = {
    "初一": 13, "初二": 14, "初三": 15,
    "七年级": 13, "八年级": 14, "九年级": 15,
    "高一": 16, "高二": 17, "高三": 18,
    "一年级": 7, "二年级": 8, "三年级": 9,
    "四年级": 10, "五年级": 11, "六年级": 12,
}

_ID_RE = re.compile(r"[ES]\d+")
_SPACE_RE = re.compile(r"\s+")
# 常规活动名称里的连接词与修饰性后缀：核对时拆开、去掉，只拿活动名本身去摘录里找
_ACTIVITY_SPLIT_RE = re.compile(r"[+＋、，,/／·]|[与和及或]|交替|结合")
_ACTIVITY_SUFFIX_RE = re.compile(r"(?:练习|训练|运动|活动|放松|热身|整理|恢复)$")
_PARENTHESES_RE = re.compile(r"[（(][^）)]*[）)]")


def parse_age(record: HealthRecord) -> int | None:
    """从原始档案解析年龄：NL 字段优先，空值按年级估算。"""
    try:
        return int(float((record.physical_exam.NL or "").strip()))
    except ValueError:
        pass
    grade = (record.physical_exam.SCHOOL_GRADE or record.fitness_test.NJ or "").strip()
    for token, age in _GRADE_AGE.items():
        if token in grade:
            return age
    return None


# ---------------------------------------------------------------------------
# 模型输出的遍历与归一
# ---------------------------------------------------------------------------


def _ids(value: Any) -> list[str]:
    """把模型写的引用编号归一成 ["E2", "S3"]：容忍 "[E2]"、"E2、E3"、小写等写法。"""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    found: list[str] = []
    for item in value:
        for token in _ID_RE.findall(str(item).upper()):
            if token not in found:
                found.append(token)
    return found


def _plan_items(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """周计划里的全部条目：(星期, 条目)。"""
    items: list[tuple[str, dict[str, Any]]] = []
    for day in payload.get("weekly_plan") or []:
        if not isinstance(day, dict):
            continue
        for block in day.get("blocks") or []:
            for item in (block.get("items") if isinstance(block, dict) else None) or []:
                if isinstance(item, dict):
                    items.append((str(day.get("day", "")), item))
    return items


def _citing_elements(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """处方里所有可以带引用的位置：(位置说明, 元素)。位置说明用于「被哪里引用」的回溯。"""
    elements: list[tuple[str, dict[str, Any]]] = []
    fitt = payload.get("fitt_vp") if isinstance(payload.get("fitt_vp"), dict) else {}
    for key, label in FITT_LABELS.items():
        if isinstance(fitt.get(key), dict):
            elements.append((label, fitt[key]))
    if isinstance(payload.get("intensity_control"), dict):
        elements.append(("强度监测", payload["intensity_control"]))
    for day in payload.get("weekly_plan") or []:
        if isinstance(day, dict):
            elements.append((f"{day.get('day', '')}安排", day))
    for day_name, item in _plan_items(payload):
        elements.append((f"{day_name}·{item.get('name', '')}", item))
    for note in payload.get("risk_notes") or []:
        if isinstance(note, dict):
            elements.append((f"风险提示：{note.get('item', '')}", note))
    for item in payload.get("forbidden_or_limited") or []:
        if isinstance(item, dict):
            elements.append((f"运动禁忌：{item.get('target', '')}", item))
    return elements


def _used_exercise_ids(payload: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for _day, item in _plan_items(payload):
        exercise_id = str(item.get("exercise_id") or "").strip()
        if exercise_id:
            ids.append(exercise_id)
    return ids


# ---------------------------------------------------------------------------
# 校验：不通过则带问题清单重生成一次
# ---------------------------------------------------------------------------


def validate_payload(
    payload: dict[str, Any],
    *,
    context: KnowledgeContext,
    age: int | None = None,
) -> list[str]:
    """返回问题清单；为空表示通过。"""
    problems: list[str] = []

    if context.available:
        candidates = context.candidate_index
        off_list = sorted({item for item in _used_exercise_ids(payload) if item not in candidates})
        if off_list:
            problems.append(
                f"以下 exercise_id 不在可用动作目录中：{'、'.join(off_list)}"
                "（目录动作只能用目录里的编号；目录里没有的常规活动 exercise_id 留空）"
            )

        valid_ids = context.evidence_ids | context.fact_ids
        cited = [token for _where, element in _citing_elements(payload) for token in _ids(element.get("evidence_ids"))]
        unknown = sorted({token for token in cited if token not in valid_ids}, key=lambda t: (t[0], int(t[1:])))
        if unknown:
            problems.append(f"evidence_ids 引用了本次没有提供的编号：{'、'.join(unknown)}")

        if context.evidence:
            fitt = payload.get("fitt_vp") if isinstance(payload.get("fitt_vp"), dict) else {}
            missing = [
                label
                for key, label in FITT_LABELS.items()
                if not any(
                    token in context.evidence_ids
                    for token in _ids((fitt.get(key) or {}).get("evidence_ids") if isinstance(fitt.get(key), dict) else None)
                )
            ]
            if missing:
                problems.append(
                    f"fitt_vp 的以下各项没有引用权威资料摘录（evidence_ids 至少写一个 E 条目）：{'、'.join(missing)}"
                )

    days = [str(day.get("day", "")) for day in payload.get("weekly_plan") or [] if isinstance(day, dict)]
    missing_days = [day for day in WEEKDAYS if day not in days]
    if missing_days:
        problems.append(f"weekly_plan 缺少以下日期：{'、'.join(missing_days)}")

    training_days = sum(
        1 for day in payload.get("weekly_plan") or [] if isinstance(day, dict) and day.get("is_training_day")
    )
    if training_days > MAX_TRAINING_DAYS:
        problems.append(f"训练日 {training_days} 天超过了青少年每周上限 {MAX_TRAINING_DAYS} 天")

    max_minutes = MAX_SESSION_MINUTES_UNDER_12 if age is not None and age < 12 else MAX_SESSION_MINUTES
    for day in payload.get("weekly_plan") or []:
        if not isinstance(day, dict):
            continue
        minutes = day.get("total_minutes")
        if isinstance(minutes, int) and minutes > max_minutes and day.get("is_training_day"):
            problems.append(f"{day.get('day', '')} 安排了 {minutes} 分钟，超过单次上限 {max_minutes} 分钟")

    return problems


# ---------------------------------------------------------------------------
# 核对与回填：把依据链路补全进结果
# ---------------------------------------------------------------------------


def _compact(text: str) -> str:
    return _SPACE_RE.sub("", text)


def _activity_terms(name: str) -> list[str]:
    """「快步走与慢跑交替」→ ["快步走", "慢跑"]；「慢跑热身」→ ["慢跑"]；括号里的补充说明不算。"""
    terms: list[str] = []
    for part in _ACTIVITY_SPLIT_RE.split(_PARENTHESES_RE.sub("", _compact(name))):
        part = part.strip("：:")
        stripped = _ACTIVITY_SUFFIX_RE.sub("", part)
        term = stripped if len(stripped) >= 2 else part
        if len(term) >= 2:
            terms.append(term)
    return terms


def _resolve_items(payload: dict[str, Any], context: KnowledgeContext) -> list[str]:
    """逐条核对周计划条目，返回没有在摘录里找到依据的常规活动名称。

    - 没写编号但名称与目录动作完全一致：补上编号，按目录动作处理；
    - 常规活动：名称（拆掉连接词与修饰后缀）要能在某条摘录原文里找到，找到就把该摘录
      记到条目的 evidence_ids 上；找不到的标记 verified = False。
    """
    by_name = {exercise.name: exercise.code for exercise in context.candidates}
    texts = {entry.id: _compact(entry.hit.content) for entry in context.evidence}
    unverified: list[str] = []

    for _day, item in _plan_items(payload):
        item["evidence_ids"] = [token for token in _ids(item.get("evidence_ids")) if token in texts or token in context.fact_ids]
        exercise_id = str(item.get("exercise_id") or "").strip()
        name = str(item.get("name") or "").strip()
        if not exercise_id and name in by_name:
            exercise_id = by_name[name]
        item["exercise_id"] = exercise_id
        item["verified"] = True
        if exercise_id or not context.available:
            continue

        terms = _activity_terms(name)
        supporting: list[str] = []
        for term in terms:
            hit = next(
                (
                    evidence_id
                    for evidence_id in [*item["evidence_ids"], *texts]
                    if evidence_id in texts and term in texts[evidence_id]
                ),
                None,
            )
            if hit is None:
                supporting = []
                break
            supporting.append(hit)
        if terms and supporting:
            for evidence_id in supporting:
                if evidence_id not in item["evidence_ids"]:
                    item["evidence_ids"].append(evidence_id)
        else:
            item["verified"] = False
            if name and name not in unverified:
                unverified.append(name)
    return unverified


def _knowledge_fields(payload: dict[str, Any], context: KnowledgeContext) -> dict[str, Any]:
    """由代码填写的依据链路：评定条目、摘录条目（含被引用位置）、未覆盖问题、知识库状态。"""
    valid_ids = context.evidence_ids | context.fact_ids
    cited_by: dict[str, list[str]] = {}
    for where, element in _citing_elements(payload):
        element["evidence_ids"] = [token for token in _ids(element.get("evidence_ids")) if token in valid_ids]
        for token in element["evidence_ids"]:
            if where not in cited_by.setdefault(token, []):
                cited_by[token].append(where)

    return {
        "assessment": [
            {
                "id": fact.id,
                "key": fact.item.key,
                "name": fact.item.name,
                "raw": fact.item.raw,
                "unit": fact.item.unit,
                "score": fact.item.score,
                "level": fact.item.level,
                "basis": fact.item.basis,
                "source": fact.item.source,
            }
            for fact in context.facts
        ],
        "evidence": [
            {
                "id": entry.id,
                "source": entry.hit.source,
                "doc_title": entry.hit.doc_title,
                "doc_kind": entry.hit.doc_kind,
                "section": entry.hit.section,
                "pages": entry.hit.pages,
                "topics": entry.topics,
                "content": entry.hit.content,
                "score": entry.hit.score,
                "cited_by": cited_by.get(entry.id, []),
            }
            for entry in context.evidence
        ],
        "knowledge_gaps": [{"topic": topic.label, "note": topic.gap_note} for topic in context.gaps],
        "knowledge_base": {
            "available": context.available,
            "fingerprint": context.fingerprint,
            "collection": context.info.get("collection", ""),
            "embedding_model": context.info.get("embedding_model", ""),
            "retrieval_mode": context.retrieval_mode,
            "stage": context.stage or "",
            "documents": [
                {"title": doc["title"], "kind": doc["kind"], "md5": doc["md5"]}
                for doc in context.info.get("documents", [])
            ],
            "topics": [
                {
                    "label": topic.label,
                    "covered": topic.covered,
                    "evidence_ids": topic.evidence_ids,
                    "top_score": topic.top_score,
                }
                for topic in context.topics
            ],
            "candidate_exercises": len(context.candidates),
        },
    }


def _assemble(
    payload: dict[str, Any],
    *,
    record: HealthRecord,
    context: KnowledgeContext,
) -> PrescriptionResult:
    unverified = _resolve_items(payload, context)

    candidates = context.candidate_index
    details: list[dict[str, Any]] = []
    unknown: list[str] = []
    for exercise_id in dict.fromkeys(_used_exercise_ids(payload)):
        exercise = candidates.get(exercise_id)
        if exercise is None:
            unknown.append(exercise_id)
        else:
            details.append(exercise.detail())

    payload.update(_knowledge_fields(payload, context))
    payload.update(
        {
            "student_id": record.student_id,
            "name": record.name,
            "generated_at": now_iso(),
            "versions": current_versions(context.fingerprint).model_dump(),
            "goals": payload.get("goals") or [],
            "exercise_details": details,
            "unknown_exercise_ids": unknown,
            "disclaimer": DISCLAIMER,
        }
    )
    result = PrescriptionResult.model_validate(payload)

    if unverified:
        result.risk_notes.append(
            RiskNote(
                level="低",
                item="自动校验提示",
                reason="以下常规活动未在知识库摘录中找到：" + "、".join(unverified),
                action="这些活动没有资料依据，建议由专业人员复核后再执行",
            )
        )
    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _retrieval_summary(context: KnowledgeContext) -> str:
    gaps = len(context.gaps)
    mode = "" if context.retrieval_mode != "dense" else "（重排不可用，已降级为向量相似度把关）"
    return (
        f"知识库检索完成：{len(context.topics)} 个主题，命中 {len(context.evidence)} 条权威资料摘录"
        f"{f'，{gaps} 个主题知识库未覆盖' if gaps else ''}；"
        f"国标评分 {len(context.facts)} 项，可选动作 {len(context.candidates)} 个{mode}"
    )


async def stream_prescription(
    record: HealthRecord,
) -> AsyncIterator[tuple[str, Any]]:
    """流式生成处方。

    产出 ("status", 说明) / ("thinking", 片段) / ("token", 片段) 供前端展示进度，
    ("retry", 原因) 表示触发了一次重生成，最后产出 ("result", PrescriptionResult) 或 ("error", 说明)。
    """
    provider = get_provider()
    record_view = record.to_prompt_dict()
    age = parse_age(record)

    yield ("status", "正在检索知识库")
    try:
        context = await kb_context.load(record_view)
    except KnowledgeBaseUnavailable as exc:
        if provider.name != "mock":
            yield ("error", f"知识库不可用，无法生成有依据的处方：{exc}")
            return
        # mock 模式只为离线联调页面：没有知识库也放行，结果里不带任何依据
        context = kb_context.offline()
    if context.available:
        yield ("status", _retrieval_summary(context))

    base_prompt = prompt.build_user_prompt(record=record_view, age=age, context=context)

    user_prompt = base_prompt
    last_problems: list[str] = []
    truncated = False

    for attempt in range(2):
        if attempt == 1:
            yield ("retry", "；".join(last_problems))
            # 截断与"内容不合规"是两类问题：截断要压缩表述，不合规要带问题清单重做
            user_prompt = (
                prompt.build_concise_prompt(base_prompt)
                if truncated
                else prompt.build_repair_prompt(base_prompt, "\n".join(last_problems))
            )

        buffer: list[str] = []
        truncated = False
        try:
            async for kind, chunk in provider.stream(prompt.SYSTEM, user_prompt):
                if kind == "thinking":
                    yield ("thinking", chunk)
                    continue
                buffer.append(chunk)
                yield ("token", chunk)
        except LLMTruncatedError as exc:
            truncated = True
            last_problems = [str(exc)]
            if attempt == 0:
                continue
            yield ("error", f"处方生成失败：{exc}")
            return

        raw = "".join(buffer)
        try:
            payload = extract_json(raw)
        except LLMTruncatedError as exc:
            # 上游未报 max_tokens，但 JSON 没写完：同样按"太长"处理，重试要求精简
            truncated = True
            last_problems = [str(exc)]
            if attempt == 0:
                continue
            yield ("error", f"处方生成失败：{exc}")
            return
        except LLMError as exc:
            last_problems = [str(exc)]
            continue

        problems = validate_payload(payload, context=context, age=age)
        if problems and attempt == 0:
            last_problems = problems
            continue

        try:
            result = _assemble(payload, record=record, context=context)
        except ValidationError as exc:
            last_problems = [exc.json()]
            if attempt == 0:
                continue
            yield ("error", f"处方结构校验失败：{exc.errors()[:3]}")
            return

        if problems:
            result.risk_notes.append(
                RiskNote(
                    level="低",
                    item="自动校验提示",
                    reason="重生成后仍存在以下问题：" + "；".join(problems),
                    action="建议由专业人员复核相关条目",
                )
            )

        try:
            # mock 占位结果不落盘，避免覆盖 data/prescriptions 里真实模型生成的处方
            if provider.name != "mock":
                save_prescription_result(result.model_dump())
            # 生成成功即解除熔断：恢复正常留档展示
            clear_emergency(record.student_id)
        except OSError:
            # 落盘只是旁路留档，磁盘问题不应中断处方返回
            pass

        yield ("result", result)
        return

    yield ("error", "处方生成两次均未通过校验：" + "；".join(last_problems))
