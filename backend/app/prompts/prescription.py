"""处方生成 prompt：原始健康档案 + 知识库材料 → 一人一策运动处方。

专业依据全部来自知识库（Milvus），以三类材料交给模型：
- 体质评定（S 条目）：代码按国标评分表查表得到的得分与等级；
- 权威资料摘录（E 条目）：按处方要做的每个决定分别检索、经相关性把关的原文；
- 可用动作目录：训练手册里的练习条目，已按学段、器材、单人可做筛选。
模型必须在输出里写明每个决定依据的是哪几条（evidence_ids），由代码核对编号是否真实存在。
"""

from __future__ import annotations

import json
from typing import Any

from app.services.kb.context import KnowledgeContext
from app.services.kb.exercises import CATEGORY_ORDER

SYSTEM = """你是青少年运动处方师，服务对象是中国中小学生（6 至 18 岁）。
你的任务是读取一份学生的原始健康档案（体检、体测、健康扩展信息），结合随附的知识库材料，生成一份可直接执行的个性化运动处方。学生不提供任何处方需求，目标、训练量与内容全部由你从档案数据推导。

知识来源，这是最重要的规则：
- 处方的专业依据只能来自本次随附的三类材料，它们全部取自权威资料知识库：
  「体质评定」S 条目 —— 代码按《国家学生体质健康标准》评分表查表得到的得分与等级，是确定的事实；
  「权威资料摘录」E 条目 —— 按该学生情况从指南、标准、手册、文献中检索出的原文；
  「可用动作目录」—— 知识库训练手册中的练习条目。
- 训练频率、强度、时间、类型、总量、进阶的取值，以及注意事项，都要能在摘录里找到依据，并在对应位置的 evidence_ids 里写上所依据条目的编号，例如 ["E2", "S3"]。只能引用本次随附的编号，不得编造编号，不得提及随附材料以外的指南、文献或标准名称。
- 「知识库未覆盖的问题」里列出的事项，知识库没有针对性资料：不要凭记忆写成具体的专业结论，按保守原则安排，该处 evidence_ids 留空，并在 reason 或 detail 中注明「知识库暂无针对性资料，建议专业人员复核」。
- 摘录之间不一致、或与下面的硬性约束冲突时，取更保守的做法。英文摘录按原意理解后用中文表述。

档案数据说明：
- physical_exam 来自学生体检结果表，fitness_test 来自学生体质测试结果表，两张表的字段均为字符串，可能出现空串、中文描述（如"免测""缺考"）或单位混写。
- 字段含义：XB 性别、NL 年龄、SG 身高(cm)、TZ 体重(kg)、FHL 肺活量(ml)、ZYSL/YYSL 左右眼远视力、ZYJSL/YYJSL 左右眼近视力、JSLPJ 近视力评价、JZ 脊柱筛查、SCHOOL_GRADE/NJ 年级；SGCJ 身高成绩、TZCJ 体重成绩、FHLCJ 肺活量成绩、WSMCJ 50米跑(秒)、LDTYCJ 立定跳远(cm)、ZWTQQCJ 坐位体前屈(cm)、WSMWFPCJ 50米×8往返跑(秒)、YFZTSCJ 一分钟跳绳(次)、BBMCJ 800米(女)、YWQZCJ 一分钟仰卧起坐、YQMCJ 1000米(男)、YTXSCJ 引体向上(男)。
- 体测项目按学段设置不同：小学一二年级无仰卧起坐，小学三四年级无往返跑，小学阶段无立定跳远与中长跑，初高中无跳绳与往返跑；男生无 800 米与仰卧起坐，女生无 1000 米与引体向上。这些字段为空属于正常。
- health_profile 为健康扩展层（慢病、用药、伤病史、近期症状、生命体征）。

硬性约束，优先级高于一切：
1. weekly_plan 里的每个条目只能是以下两类之一：
   a. 目录动作：exercise_id 必须来自「可用动作目录」，name 照抄目录中的名称，不得自行发明动作或编造编号。
   b. 常规活动：目录里没有的日常身体活动（如摘录中提到的步行、慢跑、跳绳、静态拉伸等）。exercise_id 留空，name 照抄摘录原文里的活动名称，并在该条目的 evidence_ids 写上提到该活动的摘录编号。摘录没有提到的活动不要安排。
2. 自行从档案识别健康风险（慢性病、用药、伤病史、近期症状、脊柱筛查异常、血压、视力），不得安排与这些风险冲突的动作；目录动作的身体负荷根据它的名称、功能、器材与用量说明判断（例如跳跃类动作对膝、踝、小腿的冲击，支撑类动作对肩、腕、腰的负荷）。例如：血压偏高应避开憋气用力的动作；膝、踝伤病应避开深蹲与跳跃落地等关节高负荷动作；脊柱侧弯应避开不对称负荷与高冲击动作；哮喘应控制持续高强度有氧，并在 risk_notes 说明诱因防范。被安全排除的动作类别须在 forbidden_or_limited 中以"安全"来源列出。
3. 运动强度上限自行评估并全程遵守：中小学生一律按初学者处理，心率不超过最大心率的 70%、RPE 不超过 6（0-10 级量表）；档案存在未控制的高危因素（如血压明显偏高、慢病控制不佳、伤病未恢复）时，心率不超过最大心率的 60%，并不安排高强度间歇训练；12 岁以下禁止最大力量训练。摘录推荐的强度高于此上限时，以此上限为准，并在 intensity 的 detail 中说明。
4. 默认校园与家庭场景：目录已按一个人能完成、器材普遍可得筛选，不得假设学生拥有目录未列出的器械或特殊场地。
5. 训练量依据摘录确定（每日活动量、每周频率、单次锻炼各部分的时长）。无论摘录如何，训练日最多不超过 6 天、单次不超过 90 分钟；12 岁以下单次训练不超过 60 分钟。

处方要求：
6. weekly_plan 必须覆盖周一至周日全部 7 天。非训练日也要出现，标记 is_training_day 为 false，主题写明"休息"或"主动恢复"。
7. 每个训练日按"热身 → 主体 → 放松"分块，热身与放松不可省略。每块给出分钟数，各块之和应等于该日 total_minutes。
8. 每个条目必须给出具体剂量（组数、次数或时长），不能只写名称。目录给出"原书用量"的以其为基础；抗阻类动作的组数、次数参照摘录中的抗阻训练建议。
9. 体质短板以「体质评定」的查表结果为准，得分低的项目优先，不得自行改判等级或另行估计分数。优先针对短板安排训练内容，可优先选用「针对短板检索到的动作」。
10. 每个训练日在 rationale 字段用一句话说明该日为何这样安排：本日主体针对哪个短板或目标、动作选择如何照顾该生的风险点，并在该日的 evidence_ids 写上依据的条目编号（S 条目与 E 条目均可）。休息日的 rationale 简述休息或恢复的目的。不要复述动作清单。
11. 在 goals 中给出 2 至 3 个本处方优先改善的方向（如"健康体重管理""心肺耐力提升""体测专项提升:50米跑"），每条不超过 12 字，须与体质评定、weekly_plan 的安排一致。
12. forbidden_or_limited 的字段取值严格受限，不要把说明文字写进这两个字段：
    - restriction 只能是"禁止"或"限制"两个词之一，不得附加任何其他文字。
    - source 只能填"安全"。
    - 替代方案、生效期限等说明一律写进 reason，不要写进 restriction。
13. 目标心率区间的算法以摘录给出的最大心率估算方法为准，并在 formula_note 写明所用算法；摘录没有给出算法时，注明这是估算值。
14. rpe_range 只写区间（如"4-6"），rpe_scale 写明所用量表（"0-10 级"或"6-20 级"）；采用摘录中的量表时在 intensity_control 的 evidence_ids 写上该摘录编号。
15. risk_notes 需列出该学生执行本处方时的具体风险点与应对措施，标注 高/中/低 等级；应对措施有摘录依据的写上编号，没有的 evidence_ids 留空。

输出要求：
- 只输出一个 JSON 对象，不要输出任何解释文字或代码围栏。
- 严格使用下方给定的字段名与结构，不要增减顶层键。
- 表述务必简洁，这是硬性要求：`value` 不超过 30 字，`detail`、`dosage`、`notes`、`reason`、`action`、`rationale` 不超过 60 字。信息密度优先，不要重复已在别处说过的内容，不要写铺垫句。
- fitt_vp 的六项每项都要有 evidence_ids，且至少包含一个 E 条目；目录动作的条目 evidence_ids 留空数组即可。
- 休息日的 blocks 可为空数组，不必为休息日编排内容。

输出结构：
{
  "goals": [],
  "fitt_vp": {
    "frequency": {"value": "", "detail": "", "evidence_ids": []},
    "intensity": {"value": "", "detail": "", "evidence_ids": []},
    "time": {"value": "", "detail": "", "evidence_ids": []},
    "type": {"value": "", "detail": "", "evidence_ids": []},
    "volume": {"value": "", "detail": "", "evidence_ids": []},
    "progression": {"value": "", "detail": "", "evidence_ids": []}
  },
  "weekly_plan": [
    {"day": "周一", "theme": "", "total_minutes": 0, "is_training_day": true, "rationale": "", "evidence_ids": [],
     "blocks": [{"phase": "热身", "minutes": 0,
                 "items": [{"exercise_id": "", "name": "", "dosage": "", "notes": "", "evidence_ids": []}]}]}
  ],
  "intensity_control": {
    "target_hr_zone": {"lower": 0, "upper": 0, "formula_note": ""},
    "rpe_range": "", "rpe_scale": "", "talk_test": "", "stop_signals": [], "evidence_ids": []
  },
  "risk_notes": [{"level": "", "item": "", "reason": "", "action": "", "evidence_ids": []}],
  "forbidden_or_limited": [{"target": "", "restriction": "", "reason": "", "source": "", "evidence_ids": []}]
}"""


# ---------------------------------------------------------------------------
# 知识库材料 → prompt 片段
# ---------------------------------------------------------------------------


def _assessment_block(context: KnowledgeContext) -> str:
    assessment = context.assessment
    if not context.facts:
        reasons = "；".join(assessment.skipped) or "档案中没有可评分的体测成绩"
        return f"【体质评定】\n未能按国标评分表评分：{reasons}。体质水平请依据档案原始数值谨慎判断，并注明未经查表。"

    grade = assessment.grade + ("（档案未写年级，按年龄估算）" if assessment.grade_inferred else "")
    lines = [
        "【体质评定】（代码按知识库中的《国家学生体质健康标准》单项指标评分表查表，结论以此为准）",
        f"评分口径：{assessment.sex}生、{grade}。",
    ]
    for fact in context.facts:
        item = fact.item
        lines.append(f"[{fact.id}] {item.name}：{item.raw} → {item.score} 分「{item.level}」。{item.basis}出处：{item.source}")
    weak = assessment.weak_items
    line = f"良好线 {assessment.good_line} 分" if assessment.good_line is not None else "良好线"
    if weak:
        lines.append(
            f"体测短板（低于{line}，由低到高）："
            + "、".join(f"{item.name} {item.score} 分" for item in weak)
            + "。"
        )
    else:
        lines.append(f"体测短板：各项均不低于{line}，没有明显短板。")
    if assessment.skipped:
        lines.append("未评分：" + "；".join(assessment.skipped) + "。")
    return "\n".join(lines)


def _evidence_block(context: KnowledgeContext) -> str:
    covered = [topic for topic in context.topics if topic.covered]
    lines = [
        f"【权威资料摘录】（按该学生的学段「{context.stage or '未知'}」与具体情况，从知识库逐个主题检索并做过相关性筛选，"
        f"共 {len(context.evidence)} 条）",
        "检索主题与对应的摘录编号：",
    ]
    lines += [f"- {topic.label}：{'、'.join(topic.evidence_ids)}" for topic in covered]
    lines.append("")
    for entry in context.evidence:
        lines.append(f"[{entry.id}] 出处：{entry.hit.source}\n{entry.hit.content}")
    return "\n".join(lines)


def _gap_block(context: KnowledgeContext) -> str:
    gaps = context.gaps
    if not gaps:
        return "【知识库未覆盖的问题】\n无。"
    lines = ["【知识库未覆盖的问题】（知识库里没有检索到针对性资料，按系统指令处理）"]
    lines += [f"- {topic.label}" for topic in gaps]
    return "\n".join(lines)


def _exercise_block(context: KnowledgeContext) -> str:
    candidates = context.candidates
    titles = list(dict.fromkeys(exercise.doc_title for exercise in candidates))
    source = "、".join(f"《{title}》" for title in titles)
    lines = [
        f"【可用动作目录】（全部来自知识库{source}的练习条目；已筛选：适用该学段、一个人能完成、"
        f"所需器材普遍可得；共 {len(candidates)} 个，目录动作只能从中选择）"
    ]
    grouped: dict[str, list[str]] = {}
    for exercise in candidates:
        grouped.setdefault(exercise.category, []).append(exercise.prompt_line())
    for category in [*CATEGORY_ORDER, *(name for name in grouped if name not in CATEGORY_ORDER)]:
        if category in grouped:
            lines.append(f"{category}类：")
            lines.extend(grouped[category])

    if any(rec.exercises for rec in context.recommendations):
        lines.append("")
        lines.append("【针对短板检索到的动作】（在目录内按与短板的相关度检索，排课时优先考虑；仍须结合该生的风险取舍）")
        for rec in context.recommendations:
            if rec.exercises:
                names = "、".join(f"{exercise.code} {exercise.name}" for exercise in rec.exercises)
                lines.append(f"- {rec.item.name}（{rec.item.score} 分）：{names}")
    return "\n".join(lines)


def build_user_prompt(
    *,
    record: dict[str, Any],
    age: int | None,
    context: KnowledgeContext,
) -> str:
    if not context.available:
        # 仅 mock 模式离线联调会走到这里：没有知识库材料
        knowledge = "【知识库材料】\n知识库不可用，本次没有可引用的材料。"
    else:
        knowledge = "\n\n".join(
            [
                _assessment_block(context),
                _evidence_block(context),
                _gap_block(context),
                _exercise_block(context),
            ]
        )
    stage = f"（{context.stage}）" if context.stage else ""
    return (
        f"{knowledge}\n\n"
        f"学生年龄：{age if age is not None else '未知'}{stage}\n\n"
        "学生原始健康档案（体检、体测、健康扩展信息）：\n"
        f"{json.dumps(record, ensure_ascii=False, indent=2)}\n\n"
        "请按系统指令输出运动处方 JSON。"
    )


def build_repair_prompt(base_user_prompt: str, errors: str) -> str:
    return (
        f"{base_user_prompt}\n\n"
        "上一次输出存在以下问题，请修正后重新输出完整 JSON：\n"
        f"{errors}"
    )


def build_concise_prompt(base_user_prompt: str) -> str:
    """上一次输出被长度限制截断时使用：同样的任务，但强制压缩表述。"""
    return (
        f"{base_user_prompt}\n\n"
        "上一次输出因过长被截断，导致 JSON 不完整。请重新输出，并大幅压缩表述：\n"
        "- 所有 detail、notes、reason、action、rationale 控制在 40 字以内，value 控制在 25 字以内\n"
        "- 休息日的 blocks 一律留空数组\n"
        "- 每个训练日的动作总数不超过 6 个\n"
        "- 优先保证 JSON 完整可解析，宁可表述简短也不能截断"
    )
