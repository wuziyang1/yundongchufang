"""档案理解 prompt：原始健康档案 + 国标查表结果 → 体质画像的解读文字。

体测成绩的得分、等级、六维等级、学年总分与综合等级由代码按知识库里的
《国家学生体质健康标准》查表得到，作为确定的事实交给模型；模型只负责解读：
各维度一句话说明、综合评价、优势与短板排序、健康关注点。
"""

from __future__ import annotations

import json
from typing import Any

from app.services.profile_grading import NO_DATA, Dimension, Grading

SYSTEM = """你是青少年运动健康领域的体质评估专家，服务对象是中国中小学生（6 至 18 岁）。
你的任务是读取一份学生健康档案和随附的「体质评定」，写出体质画像的解读文字。

体质评定，这是最重要的规则：
- 「体质评定」由代码按知识库中的《国家学生体质健康标准》查表得到：每项体测成绩的得分与等级（S 条目）、体质六维的等级、学年总分与综合等级，都是确定的事实。
- 你不需要、也不允许再判断等级：不得改判，不得另行估计分数或界值，不得使用评定里没有的等级说法。文字里提到等级、得分、界值时，照抄评定里的写法（例如体重指数评定为「肥胖」就写肥胖，不要写成超重或偏胖）。
- 评定里标为「数据不足」的维度、没有计算出学年总分的情况，如实说明缺哪些项目，不要自行补一个估计；标为「本学段不测」的维度说明该年级不测这一项即可。

数据说明：
- physical_exam 来自学生体检结果表，fitness_test 来自学生体质测试结果表，两张表的字段均为字符串，可能出现空串、中文描述（如"免测""缺考"）或单位混写。
- 字段含义：XB 性别、NL 年龄、SG 身高(cm)、TZ 体重(kg)、FHL 肺活量(ml)、ZYSL/YYSL 左右眼远视力、ZYJSL/YYJSL 左右眼近视力、JSLPJ 近视力评价、JZ 脊柱筛查、SCHOOL_GRADE/NJ 年级；SGCJ 身高成绩、TZCJ 体重成绩、FHLCJ 肺活量成绩、WSMCJ 50米跑(秒)、LDTYCJ 立定跳远(cm)、ZWTQQCJ 坐位体前屈(cm)、WSMWFPCJ 50米×8往返跑(秒)、YFZTSCJ 一分钟跳绳(次)、BBMCJ 800米(女)、YWQZCJ 一分钟仰卧起坐、YQMCJ 1000米(男)、YTXSCJ 引体向上(男)。
- health_profile 为健康扩展层（慢病、用药、伤病史、近期症状、生命体征）。

写作要求：
1. dimension_notes：体质六维每维一条，名称与顺序照抄评定。basis 用一两句话解读该维度的查表结果对这名学生意味着什么：处在什么水平、离上一档还差多少、与档案里的症状或伤病有没有关联。差距的数字直接用评定里写好的（"达到多少可得多少分（差多少）""达到多少进入上一等级（差多少）"），不要自己做加减计算，评定里没有给出的界值不要写。不要整段复述界值表。数据不足的维度写明缺哪项、建议补测。
2. overall.summary：两到四句，概括综合等级与学年总分、主要优势与问题、运动中需要留意的健康情况。「评定说明」里列出的数据问题（缺测、同一指标两处数值不同等）要如实提及。
3. overall.strengths：来自评定中等级为优秀、良好的项目（体重指数为正常也算）以及档案里的积极信息，每条一句；没有就留空数组，不要硬凑。
4. overall.weaknesses：最多 3 条，只列最需要优先改善的，按"对学生当前健康影响最大"排序并给出 rank，而不是单纯按分数从低到高。dimension 只能从评定里「可列为短板的维度」中选（等级为优秀、良好或体重指数正常的维度，以及数据不足的维度，不在其中）。reason 引用查表结果（原始成绩、得分、等级）。没有可列的维度就留空数组。
5. health_focus：覆盖档案中所有需要在运动中留意的健康项：慢性病、用药、伤病史、近期症状、脊柱筛查异常、视力情况。
6. 不做医疗诊断，不推测未在档案中出现的疾病，不给出用药建议。

输出要求：
- 只输出一个 JSON 对象，不要输出任何解释文字或代码围栏。
- 严格使用下方给定的字段名与结构，不要增减顶层键，不要输出等级、得分字段。
- basis、reason 不超过 80 字，strengths 每条不超过 40 字。

输出结构：
{
  "dimension_notes": [{"dimension": "身体形态", "basis": ""}],
  "overall": {"summary": "",
              "strengths": [],
              "weaknesses": [{"rank": 1, "dimension": "", "reason": ""}]},
  "health_focus": [{"item": "", "type": "", "impact": "", "attention": ""}]
}"""


def _dimension_line(dimension: Dimension) -> str:
    if not dimension.facts:
        missing = "、".join(item.text for item in dimension.missing)
        return f"- {dimension.name}：{dimension.level}" + (f"（缺少 {missing}）" if missing else "")
    items = "、".join(f"{fact.item.name} {fact.item.score} 分「{fact.item.level}」[{fact.id}]" for fact in dimension.facts)
    line = f"- {dimension.name}：{dimension.level}（{dimension.score:g} 分）— {items}"
    if len(dimension.facts) > 1:
        ratio = ":".join(f"{weight:g}" for weight in dimension.weights)
        line += f"，按国标权重 {ratio} 加权平均"
    if dimension.missing:
        line += f"；另缺 {'、'.join(item.text for item in dimension.missing)}"
    return line


def assessment_block(grading: Grading) -> str:
    """交给模型的「体质评定」：S 条目、六维等级、学年总分、评定说明。"""
    if not grading.available:
        # 仅 mock 模式离线联调会走到这里
        return "【体质评定】\n知识库不可用，本次没有查表结果，体质六维均为数据不足。"

    assessment = grading.assessment
    if not grading.facts:
        reasons = "；".join(assessment.skipped) or "档案中没有可评分的体测成绩"
        return f"【体质评定】\n未能按国标评分表评分：{reasons}。体质六维均为数据不足。"

    grade = assessment.grade + ("（档案未写年级，按年龄估算）" if assessment.grade_inferred else "")
    lines = [
        "【体质评定】（代码按知识库中的《国家学生体质健康标准》查表，结论以此为准）",
        f"评分口径：{assessment.sex}生、{grade}。",
        "单项得分：",
    ]
    lines += [
        f"[{fact.id}] {fact.item.name}：{fact.item.raw} → {fact.item.score} 分「{fact.item.level}」。{fact.item.basis}"
        for fact in grading.facts
    ]
    lines.append("体质六维：")
    lines += [_dimension_line(dimension) for dimension in grading.dimensions]
    lines.append("可列为短板的维度：" + ("、".join(grading.weak_candidates) or "无（各维度等级均在良好及以上，或没有成绩）") + "。")
    if assessment.total is not None:
        lines.append(f"学年总分：{assessment.total:g} 分，综合等级「{assessment.level}」。{assessment.total_basis}")
    else:
        lines.append(f"学年总分：未计算，综合等级为{NO_DATA}。{assessment.total_basis}")
    if grading.notes:
        lines.append("评定说明：" + "；".join(grading.notes) + "。")
    return "\n".join(lines)


def build_user_prompt(record: dict[str, Any], grading: Grading) -> str:
    return (
        f"{assessment_block(grading)}\n\n"
        "待评估的学生健康档案：\n"
        f"{json.dumps(record, ensure_ascii=False, indent=2)}\n\n"
        "请按系统指令输出体质画像的解读文字 JSON。"
    )


def build_repair_prompt(record: dict[str, Any], grading: Grading, errors: str) -> str:
    return (
        f"{build_user_prompt(record, grading)}\n\n"
        "上一次输出存在以下问题，请修正后重新输出完整 JSON：\n"
        f"{errors}"
    )
