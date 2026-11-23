"""国标评分：按《国家学生体质健康标准》的单项指标评分表给体测成绩打分。

评分表存在 Milvus 里（rag 入库时把表格逐行展开成「表头：值；表头：值」）。这类内容不能靠相似度
检索：检索回来的往往是别的年级、别的性别的行，模型自己读表还会读错。所以这里按条件把评分表
整张取出、还原成表格，再用代码按「性别 × 年级 × 成绩」查出得分与等级，结果作为确定的事实交给模型。

查表用到的界值全部来自知识库，代码里不写任何分数线；代码只知道三件事：
- 档案里的字段对应哪张表（表题里的项目名）；
- 合并在一张表里的两个项目各适用哪些年级（表注原文：「小学三年级～六年级：一分钟仰卧起坐；
  初中、高中、大学：引体向上」；耐力跑同理，小学是 50米×8 往返跑，初中起是 800 / 1000 米跑）；
- 成绩是越大越好还是越小越好，由表内 100 分与 60 分的界值谁大谁小自行判断。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

from app.services.kb import store

# 入库时表格行的写法：「等级：优秀；单项得分：100；一年级：13.5~18.1；…」
_TITLE_RE = re.compile(r"^表\s*\d+\s*-\s*\d+\s*(男生|女生)(.+?)单项评分表(?:（单位：(.+?)）)?\s*$")
_LEVEL_KEY = "等级"
_SCORE_KEY = "单项得分"

PRIMARY = ("一年级", "二年级", "三年级", "四年级", "五年级", "六年级")
SECONDARY = ("初一", "初二", "初三", "高一", "高二", "高三")
GRADES = PRIMARY + SECONDARY

_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
# 年级缺失时按年龄估算（与入学年龄 6-7 岁对应）
_AGE_TO_GRADE = dict(zip(range(7, 19), GRADES)) | {6: "一年级"}


@dataclass(frozen=True)
class _Item:
    key: str  # 档案字段名；BMI 由身高体重算出
    name: str
    table: str  # 表题里的项目名
    unit: str
    # 该项目反映的身体素质，只用来组织检索语句（「发展速度素质的练习」），不参与评分
    focus: str = ""
    sexes: tuple[str, ...] = ("男", "女")
    grades: tuple[str, ...] | None = None  # 合并表里该项目适用的年级；None 表示表里有哪列就算哪列
    is_time: bool = False


_ITEMS = (
    _Item("BMI", "体重指数（BMI）", "体重指数", "千克/米²"),
    _Item("FHLCJ", "肺活量", "肺活量", "毫升", focus="心肺耐力"),
    _Item("WSMCJ", "50米跑", "50米跑", "秒", focus="速度素质"),
    _Item("ZWTQQCJ", "坐位体前屈", "坐位体前屈", "厘米", focus="柔韧性"),
    _Item("YFZTSCJ", "一分钟跳绳", "一分钟跳绳", "次", focus="下肢力量、协调与耐力"),
    _Item("LDTYCJ", "立定跳远", "立定跳远", "厘米", focus="下肢爆发力"),
    _Item("YWQZCJ", "一分钟仰卧起坐", "仰卧起坐", "次", focus="腰腹核心力量与肌肉耐力", sexes=("女",)),
    _Item(
        "YWQZCJ", "一分钟仰卧起坐", "仰卧起坐", "次",
        focus="腰腹核心力量与肌肉耐力", sexes=("男",), grades=PRIMARY,
    ),
    _Item("YTXSCJ", "引体向上", "引体向上", "次", focus="上肢与肩背力量", sexes=("男",), grades=SECONDARY),
    _Item("WSMWFPCJ", "50米×8往返跑", "耐力跑", "秒", focus="心肺耐力", grades=PRIMARY, is_time=True),
    _Item("BBMCJ", "800米跑", "耐力跑", "秒", focus="心肺耐力", sexes=("女",), grades=SECONDARY, is_time=True),
    _Item("YQMCJ", "1000米跑", "耐力跑", "秒", focus="心肺耐力", sexes=("男",), grades=SECONDARY, is_time=True),
)


@dataclass
class ScoreTable:
    title: str
    sex: str
    subject: str
    doc_title: str
    section: str
    page_start: int
    page_end: int
    rows: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ItemScore:
    key: str
    name: str
    raw: str  # 档案原值
    value: float
    unit: str
    score: int
    level: str
    basis: str  # 查表依据：该年级该性别的界值原文
    table: str
    source: str  # 《文档》章节 第 N 页
    focus: str = ""  # 该项目反映的身体素质（检索用）


@dataclass
class Assessment:
    sex: str
    grade: str
    grade_inferred: bool
    items: list[ItemScore] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # 没能评分的项目及原因

    @property
    def bmi(self) -> ItemScore | None:
        return next((item for item in self.items if item.key == "BMI"), None)

    @property
    def weak_items(self) -> list[ItemScore]:
        """体测短板：得分低于良好线（80 分）的项目，分数低的在前。BMI 单独处理，不算在内。"""
        scored = [item for item in self.items if item.key != "BMI" and item.score < 80]
        return sorted(scored, key=lambda item: item.score)


# ---------------------------------------------------------------------------
# 评分表：从知识库取出并还原
# ---------------------------------------------------------------------------

_tables_lock = threading.Lock()
_tables_cache: tuple[str, list[ScoreTable]] | None = None


def _parse_row(line: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for part in line.split("；"):
        key, sep, value = part.partition("：")
        if sep and key.strip():
            row[key.strip()] = value.strip()
    return row


def load_tables() -> list[ScoreTable]:
    """取出知识库里全部「单项评分表」并按表题合并（一张表入库时会切成几块）。按知识库指纹缓存。"""
    global _tables_cache
    fingerprint = store.info()["fingerprint"]
    with _tables_lock:
        if _tables_cache is not None and _tables_cache[0] == fingerprint:
            return _tables_cache[1]

    # 入库时由 doc_rules.toml 的 lookup_tables 标记为「查表数据」的表格
    chunks = store.query(
        'chunk_type == "lookup"',
        ["doc_title", "section", "page_start", "page_end", "chunk_index", "content"],
    )
    tables: dict[tuple[str, str], ScoreTable] = {}
    for chunk in sorted(chunks, key=lambda item: (item["doc_title"], item["chunk_index"])):
        title, _, body = chunk["content"].partition("\n")
        match = _TITLE_RE.match(title.strip())
        if match is None:
            continue
        table = tables.setdefault(
            (chunk["doc_title"], title.strip()),
            ScoreTable(
                title=title.strip(),
                sex=match.group(1)[0],
                subject=match.group(2),
                doc_title=chunk["doc_title"],
                section=chunk["section"],
                page_start=chunk["page_start"],
                page_end=chunk["page_end"],
            ),
        )
        table.page_start = min(table.page_start, chunk["page_start"])
        table.page_end = max(table.page_end, chunk["page_end"])
        for line in body.split("\n"):
            row = _parse_row(line)
            if _SCORE_KEY in row and row not in table.rows:
                table.rows.append(row)

    result = list(tables.values())
    with _tables_lock:
        _tables_cache = (fingerprint, result)
    return result


# ---------------------------------------------------------------------------
# 档案字段解析
# ---------------------------------------------------------------------------


def grade_label(text: str, age: int | None) -> tuple[str | None, bool]:
    """把「小学六年级」「初二」「八年级」「高中一年级」归一成评分表的列名；返回 (年级, 是否按年龄估算)。"""
    text = (text or "").strip()
    if text:
        if match := re.search(r"高(?:中)?\s*([一二三123])", text):
            return SECONDARY[3 + _digit(match.group(1)) - 1], False
        if match := re.search(r"初(?:中)?\s*([一二三123])", text):
            return SECONDARY[_digit(match.group(1)) - 1], False
        if match := re.search(r"([一二三四五六七八九1-9])\s*年级", text):
            number = _digit(match.group(1))
            if number <= 6:
                return PRIMARY[number - 1], False
            return SECONDARY[number - 7], False
    if age is not None and age in _AGE_TO_GRADE:
        return _AGE_TO_GRADE[age], True
    return None, False


def _digit(token: str) -> int:
    return _CN_DIGITS.get(token) or int(token)


def _number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group()) if match else None


def _seconds(text: str) -> float | None:
    """「4分32秒」「4'32"」「4′32″」「4:32」→ 272；纯数字按秒算。"""
    text = (text or "").strip()
    match = re.search(r"(\d+)\s*(?:分钟|分|'|′|’|:|：)\s*(\d+(?:\.\d+)?)?", text)
    if match:
        return int(match.group(1)) * 60 + float(match.group(2) or 0)
    return _number(text)


def to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 查表
# ---------------------------------------------------------------------------


def _source(table: ScoreTable) -> str:
    pages = f"{table.page_start}" if table.page_end == table.page_start else f"{table.page_start}-{table.page_end}"
    return f"《{table.doc_title}》{table.section} 第 {pages} 页"


def _in_range(value: float, cell: str) -> bool:
    """BMI 表的界值写法：「14.2~20.8」「≤14.1」「≥23.7」。"""
    cell = cell.replace("～", "~").replace(" ", "")
    if "~" in cell:
        low, high = (_number(part) for part in cell.split("~", 1))
        return low is not None and high is not None and low <= value <= high
    bound = _number(cell)
    if bound is None:
        return False
    if cell.startswith(("≤", "<")):
        return value <= bound
    if cell.startswith(("≥", ">")):
        return value >= bound
    return value == bound


def _score_bmi(table: ScoreTable, grade: str, value: float, raw: str, item: _Item) -> ItemScore | None:
    rows = [row for row in table.rows if row.get(grade)]
    if not rows:
        return None
    value = round(value, 1)  # 标准的界值到一位小数
    hit = next((row for row in rows if _in_range(value, row[grade])), None)
    if hit is None:
        return None
    cells = "；".join(f"{row.get(_LEVEL_KEY, '')} {row[grade]}（{row[_SCORE_KEY]} 分）" for row in rows)
    return ItemScore(
        key=item.key,
        name=item.name,
        raw=raw,
        value=value,
        unit=item.unit,
        score=int(float(hit[_SCORE_KEY])),
        level=hit.get(_LEVEL_KEY, ""),
        basis=f"{table.sex}生{grade}：{cells}。本人 {value}，判为「{hit.get(_LEVEL_KEY, '')}」。",
        table=table.title,
        source=_source(table),
        focus=item.focus,
    )


def _score_threshold(table: ScoreTable, grade: str, value: float, raw: str, item: _Item) -> ItemScore | None:
    parse = _seconds if item.is_time else _number
    ladder: list[tuple[int, str, float, str]] = []  # (得分, 等级, 界值, 界值原文)
    for row in table.rows:
        bound = parse(row.get(grade, ""))
        if bound is not None:
            ladder.append((int(float(row[_SCORE_KEY])), row.get(_LEVEL_KEY, ""), bound, row[grade]))
    if len(ladder) < 2:
        return None
    ladder.sort(reverse=True)
    lower_better = ladder[0][2] < ladder[-1][2]

    def reached(bound: float) -> bool:
        return value <= bound if lower_better else value >= bound

    hit = next((step for step in ladder if reached(step[2])), None)
    score, level = (hit[0], hit[1]) if hit else (0, ladder[-1][1])

    sign = "≤" if lower_better else "≥"
    marks = [step for step in ladder if step[0] in (90, 80, 60)]
    lines = "、".join(f"{step[1]}（{step[0]} 分）{sign}{step[3]}" for step in marks)
    basis = f"{table.sex}生{grade}：{lines}。本人 {raw}，得 {score} 分，等级「{level}」"
    better = [step for step in ladder if step[0] > score]
    if better:
        nxt = better[-1]
        basis += f"；达到 {nxt[3]} 可得 {nxt[0]} 分"
    return ItemScore(
        key=item.key,
        name=item.name,
        raw=raw,
        value=value,
        unit=item.unit,
        score=score,
        level=level,
        basis=basis + "。",
        table=table.title,
        source=_source(table),
        focus=item.focus,
    )


def assess(record: dict[str, Any]) -> Assessment:
    """给一份档案的体测成绩逐项查表评分。评分表取不到时抛 KnowledgeBaseUnavailable。"""
    exam = record.get("physical_exam") or {}
    fitness = record.get("fitness_test") or {}
    sex = str(exam.get("XB") or fitness.get("XB") or "").strip()[:1]
    age = to_int(exam.get("NL"))
    grade, inferred = grade_label(str(exam.get("SCHOOL_GRADE") or fitness.get("NJ") or ""), age)
    result = Assessment(sex=sex, grade=grade or "", grade_inferred=inferred)
    if sex not in ("男", "女") or grade is None:
        result.skipped.append("档案缺少性别或年级，无法按国标评分表评分")
        return result

    tables = [table for table in load_tables() if table.sex == sex]
    if not tables:
        raise store.KnowledgeBaseUnavailable("知识库里没有国家学生体质健康标准的单项评分表，无法给体测成绩评分")

    for item in _ITEMS:
        if sex not in item.sexes:
            continue
        if item.key == "BMI":
            height = _number(str(exam.get("SG") or fitness.get("SGCJ") or ""))
            weight = _number(str(exam.get("TZ") or fitness.get("TZCJ") or ""))
            if not height or not weight:
                result.skipped.append("体重指数（BMI）：缺少身高或体重")
                continue
            value: float | None = weight / (height / 100) ** 2
            raw = f"{value:.1f}"
        else:
            raw = str(fitness.get(item.key) or "").strip()
            if item.key == "FHLCJ" and not raw:
                raw = str(exam.get("FHL") or "").strip()
            if not raw:
                continue
            value = (_seconds if item.is_time else _number)(raw)
            if value is None:
                result.skipped.append(f"{item.name}：成绩「{raw}」不是数值")
                continue
        if item.grades is not None and grade not in item.grades:
            result.skipped.append(f"{item.name}：{grade}不测该项目，成绩「{raw}」未评分")
            continue

        table = next((t for t in tables if item.table in t.subject), None)
        scorer = _score_bmi if item.key == "BMI" else _score_threshold
        scored = scorer(table, grade, value, raw, item) if table else None
        if scored is None:
            result.skipped.append(f"{item.name}：评分表里没有{sex}生{grade}的界值")
            continue
        result.items.append(scored)
    return result
