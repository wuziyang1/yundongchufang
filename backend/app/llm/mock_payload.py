"""离线联调用的占位返回。

仅在 LLM_PROVIDER=mock 时使用，用于前端在没有模型密钥的情况下打通页面渲染。
所有文案带 [MOCK] 前缀，避免被误认为真实模型输出。
"""

from __future__ import annotations

import re
from typing import Any

_MOCK = "[MOCK]"


def _is_prescription_task(system: str) -> bool:
    return "运动处方" in system and "weekly_plan" in system


def build_mock_payload(system: str, user: str) -> dict[str, Any]:
    if _is_prescription_task(system):
        return _mock_prescription(user)
    return _mock_profile()


def _mock_profile() -> dict[str, Any]:
    return {
        "basic": {
            "grade": "初二",
            "age": 14,
            "gender": "男",
            "height_cm": 168.5,
            "weight_kg": 72.3,
            "bmi": 25.5,
            "bmi_evaluation": f"{_MOCK} 肥胖",
        },
        "fitness_dimensions": [
            {
                "dimension": "身体形态",
                "indicators": [{"name": "BMI", "raw": "25.5", "level": "较差"}],
                "level": "较差",
                "basis": f"{_MOCK} 离线占位",
            },
            {
                "dimension": "心肺耐力",
                "indicators": [{"name": "1000米", "raw": "4分32秒", "level": "及格"}],
                "level": "及格",
                "basis": f"{_MOCK} 离线占位",
            },
            {
                "dimension": "肌肉力量与耐力",
                "indicators": [{"name": "引体向上", "raw": "3", "level": "及格"}],
                "level": "及格",
                "basis": f"{_MOCK} 离线占位",
            },
            {
                "dimension": "速度与灵敏",
                "indicators": [{"name": "50米", "raw": "8.9", "level": "及格"}],
                "level": "及格",
                "basis": f"{_MOCK} 离线占位",
            },
            {
                "dimension": "爆发力",
                "indicators": [{"name": "立定跳远", "raw": "185", "level": "及格"}],
                "level": "及格",
                "basis": f"{_MOCK} 离线占位",
            },
            {
                "dimension": "柔韧性",
                "indicators": [{"name": "坐位体前屈", "raw": "4.2", "level": "较差"}],
                "level": "较差",
                "basis": f"{_MOCK} 离线占位",
            },
        ],
        "overall": {
            "level": "中等偏下",
            "summary": f"{_MOCK} 这是离线占位画像，请配置真实模型后重新生成",
            "strengths": [f"{_MOCK} 身高发育良好"],
            "weaknesses": [
                {"rank": 1, "dimension": "身体形态", "reason": f"{_MOCK} BMI 属肥胖，优先改善"},
                {"rank": 2, "dimension": "柔韧性", "reason": f"{_MOCK} 低于及格线"},
                {"rank": 3, "dimension": "心肺耐力", "reason": f"{_MOCK} 提升空间最大"},
            ],
        },
        "health_focus": [
            {
                "item": f"{_MOCK} 哮喘",
                "type": "慢性病",
                "impact": "冷空气高强度有氧可能诱发",
                "attention": "延长热身，低温转室内",
            }
        ],
    }


# 处方 prompt 里知识库材料的写法（见 app/prompts/prescription.py）
_EXERCISE_LINE_RE = re.compile(r"^\[([A-Z]{1,4}\d{3}[A-Z]?)\] ([^｜\n]+)")
_CATEGORY_LINE_RE = re.compile(r"^(\S+)类：$")
_TOPIC_LINE_RE = re.compile(r"^- (.+?)：((?:E\d+、?)+)$")


def _parse_materials(user: str) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[str]]]:
    """从处方 prompt 里读出可用动作（按素质类别）与各检索主题的摘录编号。"""
    exercises: dict[str, list[tuple[str, str]]] = {}
    topics: dict[str, list[str]] = {}
    category = ""
    for line in user.splitlines():
        if match := _CATEGORY_LINE_RE.match(line):
            category = match.group(1)
        elif match := _EXERCISE_LINE_RE.match(line):
            exercises.setdefault(category, []).append((match.group(1), match.group(2).strip()))
        elif match := _TOPIC_LINE_RE.match(line):
            topics[match.group(1)] = match.group(2).split("、")
    return exercises, topics


def _mock_prescription(user: str) -> dict[str, Any]:
    """占位处方。prompt 里带有知识库材料时，动作编号与引用编号都取自材料，

    这样 mock 模式下也能走通「校验编号 → 回填动作详情 → 生成依据链路」的完整流程；
    没有知识库（纯离线）时只排常规活动，不带任何编号。
    """
    exercises, topics = _parse_materials(user)
    used: dict[str, int] = {}

    def pick(category: str, dosage: str) -> list[dict[str, Any]]:
        pool = exercises.get(category) or []
        if not pool:
            return []
        index = used.get(category, 0)
        used[category] = index + 1
        code, name = pool[index % len(pool)]
        return [{"exercise_id": code, "name": name, "dosage": dosage, "notes": "", "evidence_ids": []}]

    def cite(*labels: str) -> list[str]:
        return [ids[0] for label in labels if (ids := topics.get(label))]

    def activity(name: str, dosage: str, *labels: str) -> dict[str, Any]:
        return {"exercise_id": "", "name": name, "dosage": dosage, "notes": "", "evidence_ids": cite(*labels)}

    session = "单次锻炼的结构与各部分时长"
    menu = "常见的有氧、抗阻运动项目"

    def training_day(day: str, theme: str, main: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
        return {
            "day": day,
            "theme": f"{_MOCK} {theme}",
            "total_minutes": 45,
            "is_training_day": True,
            "rationale": f"{_MOCK} 占位示例：说明本日针对的短板与依据",
            "evidence_ids": cite(*labels),
            "blocks": [
                {"phase": "热身", "minutes": 8,
                 "items": [activity("慢跑", "3 分钟", session), *pick("柔韧", "每侧 20 秒 × 2")]},
                {"phase": "主体", "minutes": 30, "items": main},
                {"phase": "放松", "minutes": 7,
                 "items": [*pick("柔韧", "每侧 30 秒 × 2"), activity("静态拉伸", "5 分钟", session)]},
            ],
        }

    def rest_day(day: str, theme: str) -> dict[str, Any]:
        return {"day": day, "theme": f"{_MOCK} {theme}", "total_minutes": 0, "is_training_day": False,
                "rationale": f"{_MOCK} 训练间隔日，安排休息恢复", "evidence_ids": [], "blocks": []}

    volume, intensity = "每日活动量与每周频率", "运动强度的评估与控制"
    resistance = "抗阻训练的组数、次数与频率"
    return {
        "goals": [f"{_MOCK} 健康体重管理", "心肺耐力提升", "体测专项提升:50米跑"],
        "fitt_vp": {
            "frequency": {"value": "每周 4 次", "detail": f"{_MOCK} 有氧 2 次 + 力量 2 次",
                          "evidence_ids": cite(volume, resistance)},
            "intensity": {"value": "最大心率 60%-70%", "detail": f"{_MOCK} 以谈话测试辅助控制",
                          "evidence_ids": cite(intensity)},
            "time": {"value": "单次 45 分钟", "detail": f"{_MOCK} 热身 8 + 主体 30 + 放松 7",
                     "evidence_ids": cite(session)},
            "type": {"value": "有氧为主，配合力量与柔韧", "detail": f"{_MOCK} 离线占位",
                     "evidence_ids": cite("运动类型的搭配")},
            "volume": {"value": "每周 180 分钟", "detail": f"{_MOCK} 离线占位", "evidence_ids": cite(volume)},
            "progression": {"value": "每 2 周递增", "detail": f"{_MOCK} 先加时长再加强度",
                            "evidence_ids": cite("进阶与负荷递增")},
        },
        "weekly_plan": [
            training_day("周一", "力量与核心",
                         [*pick("力量", "10 次 × 2 组"), *pick("力量", "30 秒 × 2 组"), *pick("肌肉耐力", "10 次 × 2 组")]
                         or [activity("俯卧撑", "8 次 × 2 组", menu)],
                         (resistance,)),
            rest_day("周二", "休息"),
            training_day("周三", "有氧耐力", [activity("快步走", "20 分钟", menu), activity("慢跑", "10 分钟", menu)],
                         (volume,)),
            rest_day("周四", "主动恢复"),
            training_day("周五", "速度与灵敏",
                         [*pick("速度", "10 秒 × 4 组"), *pick("灵敏", "4 组"), *pick("平衡", "每侧 20 秒 × 2")]
                         or [activity("跑步", "15 分钟", menu)],
                         (session,)),
            training_day("周六", "趣味有氧", [activity("跳绳", "1 分钟 × 5 组", menu),
                                         activity("深呼吸调整", "2 分钟")], (volume,)),
            rest_day("周日", "休息"),
        ],
        "intensity_control": {
            "target_hr_zone": {"lower": 120, "upper": 140, "formula_note": f"{_MOCK} 离线占位估算"},
            "rpe_range": "11-12",
            "rpe_scale": "6-20 级",
            "talk_test": f"{_MOCK} 能说完整句子但不能唱歌",
            "stop_signals": ["胸闷胸痛", "呼吸困难", "头晕", "关节锐痛"],
            "evidence_ids": cite(intensity),
        },
        "risk_notes": [
            {"level": "中", "item": f"{_MOCK} 离线占位风险项", "reason": "占位", "action": "配置真实模型后重新生成",
             "evidence_ids": cite("运动安全与损伤预防")}
        ],
        "forbidden_or_limited": [
            {"target": f"{_MOCK} 憋气类动作", "restriction": "禁止", "reason": "占位", "source": "安全",
             "evidence_ids": []}
        ],
    }
