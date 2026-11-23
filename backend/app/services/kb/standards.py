"""国标评分：按《国家学生体质健康标准》的单项指标评分表给体测成绩打分。

评分表存在 Milvus 里（rag 入库时把表格逐行展开成「表头：值；表头：值」）。这类内容不能靠相似度
检索：检索回来的往往是别的年级、别的性别的行，模型自己读表还会读错。所以这里按条件把评分表
整张取出、还原成表格，再用代码按「性别 × 年级 × 成绩」查出得分与等级，结果作为确定的事实交给模型。

查表用到的界值全部来自知识库，代码里不写任何分数线；代码只知道三件事：
- 档案里的字段对应哪张表（表题里的项目名）；
- 合并在一张表里的两个项目各适用哪些年级（表注原文：「小学三年级～六年级：一分钟仰卧起坐；
  初中、高中、大学：引体向上」；耐力跑同理，小学是 50米×8 往返跑，初中起是 800 / 1000 米跑）；
- 成绩是越大越好还是越小越好，由表内 100 分与 60 分的界值谁大谁小自行判断。

除了单项得分，这里还按标准算出学年总分与综合等级，用到的数据同样取自知识库：
- 各年级的必测项目与权重：「单项指标与权重」表；
- 附加分：「加分指标评分表」，成绩超过单项 100 分界值后按超出的量加分；
- 等级划分：由单项评分表里每个等级的最低得分得出（优秀、良好、及格各自从多少分起），
  与标准条文「90.0 分及以上为优秀，80.0～89.9 分为良好，60.0～79.9 分为及格」一致。
学年总分 = 标准分（各必测项目得分 × 权重之和）+ 附加分。必测项目没有测全时不计算总分。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

from app.services.kb import store

# 入库时表格行的写法：「等级：优秀；单项得分：100；一年级：13.5~18.1；…」
_TITLE_RE = re.compile(r"^表\s*\d+\s*-\s*\d+\s*(男生|女生)(.+?)单项评分表(?:（单位：(.+?)）)?\s*$")
# 加分指标评分表：「表2-1 男生一分钟跳绳评分表（单位：次）」，行是「加分：20；一年级：40；…」
_BONUS_TITLE_RE = re.compile(r"^表\s*\d+\s*-\s*\d+\s*(男生|女生)(.+?)评分表(?:（单位：(.+?)）)?\s*$")
_LEVEL_KEY = "等级"
_SCORE_KEY = "单项得分"
_BONUS_KEY = "加分"
# 单项指标与权重表的三列
_SCOPE_KEY = "测试对象"
_ITEM_KEY = "单项指标"
_WEIGHT_KEY = "权重（%）"
_SEX_SUFFIX_RE = re.compile(r"^(.+?)（(男|女)）$")

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


# 档案字段名 → 项目名（与权重表、评分表里的写法一致）
ITEM_NAMES: dict[str, str] = {item.key: item.name for item in _ITEMS}


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
class WeightRow:
    """单项指标与权重表的一行：哪些年级、哪个性别、哪个项目、权重多少。"""

    grades: tuple[str, ...]
    sexes: tuple[str, ...]
    name: str
    weight: float


@dataclass
class Standard:
    """知识库里的国标查表数据。"""

    tables: list[ScoreTable] = field(default_factory=list)  # 单项指标评分表
    bonus_tables: list[ScoreTable] = field(default_factory=list)  # 加分指标评分表
    weights: list[WeightRow] = field(default_factory=list)
    weights_source: str = ""


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
    # 100 分的界值与成绩方向，算附加分用
    full_mark: float | None = None
    lower_better: bool = False
    bonus: int = 0  # 附加分
    bonus_basis: str = ""


@dataclass
class MissingItem:
    """按该年级的权重表必测、但没有得分的项目。"""

    key: str
    name: str
    raw: str  # 档案里的原值，如「免测」；没填为空

    @property
    def text(self) -> str:
        return f"{self.name}（{self.raw or '未测'}）"


@dataclass
class Assessment:
    sex: str
    grade: str
    grade_inferred: bool
    items: list[ItemScore] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # 没能评分的项目及原因
    height_cm: float | None = None
    weight_kg: float | None = None
    notes: list[str] = field(default_factory=list)  # 数据问题：同一指标有两个不同的数值等
    # 学年总分：必测项目与权重、等级划分都取自知识库
    weights: dict[str, float] = field(default_factory=dict)  # 该性别该年级的必测项目 → 权重（%）
    missing: list[MissingItem] = field(default_factory=list)
    scale: list[tuple[int, str]] = field(default_factory=list)  # [(最低得分, 等级)]，高分在前
    standard_score: float | None = None
    bonus: int = 0
    total: float | None = None
    level: str = ""  # 综合等级；必测项目没测全时为空
    total_basis: str = ""
    total_source: str = ""

    @property
    def bmi(self) -> ItemScore | None:
        return next((item for item in self.items if item.key == "BMI"), None)

    @property
    def good_line(self) -> int | None:
        """第二高等级（良好）的起始分；评分表里只有一个等级时为空。"""
        return self.scale[1][0] if len(self.scale) >= 2 else None

    def level_of(self, score: float) -> str:
        """一个得分落在哪个等级：取起始分不高于它的最高等级，低于所有起始分的归最低等级。"""
        for minimum, level in self.scale:
            if score >= minimum:
                return level
        return self.scale[-1][1] if self.scale else ""

    @property
    def weak_items(self) -> list[ItemScore]:
        """体测短板：得分低于良好线的项目，分数低的在前。BMI 单独处理，不算在内。"""
        line = self.good_line
        if line is None:
            return []
        scored = [item for item in self.items if item.key != "BMI" and item.score < line]
        return sorted(scored, key=lambda item: item.score)


# ---------------------------------------------------------------------------
# 评分表：从知识库取出并还原
# ---------------------------------------------------------------------------

_standard_lock = threading.Lock()
_standard_cache: tuple[str, Standard] | None = None


def _parse_row(line: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for part in line.split("；"):
        key, sep, value = part.partition("：")
        if sep and key.strip():
            row[key.strip()] = value.strip()
    return row


def _scope_grades(scope: str) -> tuple[str, ...]:
    """权重表的「测试对象」→ 评分表的年级列名。

    「小学一年级至大学四年级」→ 全部；「小学五、六年级」→ 五年级、六年级；
    「初中、高中、大学各年级」→ 初一至高三。
    """
    if "至" in scope:
        return GRADES
    grades: list[str] = []
    if "小学" in scope:
        segment = scope.split("小学", 1)[1].split("年级", 1)[0]
        grades += [PRIMARY[_CN_DIGITS[ch] - 1] for ch in segment if _CN_DIGITS.get(ch, 9) <= len(PRIMARY)]
    if "初中" in scope:
        grades += SECONDARY[:3]
    if "高中" in scope:
        grades += SECONDARY[3:]
    return tuple(grades)


def _weight_rows(row: dict[str, str]) -> list[WeightRow]:
    """「引体向上（男）/1分钟仰卧起坐（女）」拆成两行；项目名里的「1分钟」统一写成「一分钟」。"""
    weight = _number(row.get(_WEIGHT_KEY, ""))
    grades = _scope_grades(row.get(_SCOPE_KEY, ""))
    if weight is None or not grades:
        return []
    rows: list[WeightRow] = []
    for part in row.get(_ITEM_KEY, "").split("/"):
        name = part.strip().replace("1分钟", "一分钟")
        sexes: tuple[str, ...] = ("男", "女")
        if match := _SEX_SUFFIX_RE.match(name):
            name, sexes = match.group(1).strip(), (match.group(2),)
        if name:
            rows.append(WeightRow(grades=grades, sexes=sexes, name=name, weight=weight))
    return rows


def load_standard() -> Standard:
    """取出知识库里的单项评分表、加分指标评分表与权重表。按知识库指纹缓存。

    评分表入库时由 doc_rules.toml 的 lookup_tables 标为「查表数据」；权重表是同一份文档里的普通表格，
    靠列名（测试对象、单项指标、权重）认出来。一张表入库时会切成几块，按表题合并。
    """
    global _standard_cache
    fingerprint = store.info()["fingerprint"]
    with _standard_lock:
        if _standard_cache is not None and _standard_cache[0] == fingerprint:
            return _standard_cache[1]

    chunks = store.query(
        'chunk_type in ["lookup", "table"]',
        ["doc_title", "section", "page_start", "page_end", "chunk_index", "chunk_type", "content"],
    )
    chunks.sort(key=lambda item: (item["doc_title"], item["chunk_index"]))

    def collect(pattern: re.Pattern, row_key: str) -> list[ScoreTable]:
        tables: dict[tuple[str, str], ScoreTable] = {}
        for chunk in chunks:
            if chunk["chunk_type"] != "lookup":
                continue
            title, _, body = chunk["content"].partition("\n")
            match = pattern.match(title.strip())
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
                if row_key in row and row not in table.rows:
                    table.rows.append(row)
        return [table for table in tables.values() if table.rows]

    standard = Standard(tables=collect(_TITLE_RE, _SCORE_KEY))
    scored_titles = {table.title for table in standard.tables}
    standard.bonus_tables = [
        table for table in collect(_BONUS_TITLE_RE, _BONUS_KEY) if table.title not in scored_titles
    ]

    documents = {table.doc_title for table in standard.tables}
    for chunk in chunks:
        if chunk["doc_title"] not in documents:
            continue
        rows = [row for line in chunk["content"].split("\n") for row in _weight_rows(_parse_row(line))]
        if rows:
            standard.weights.extend(rows)
            pages = (
                f"{chunk['page_start']}"
                if chunk["page_end"] == chunk["page_start"]
                else f"{chunk['page_start']}-{chunk['page_end']}"
            )
            standard.weights_source = f"《{chunk['doc_title']}》{chunk['section']} 第 {pages} 页"

    with _standard_lock:
        _standard_cache = (fingerprint, standard)
    return standard


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

    # 离下一档、上一等级还差多少由代码算好写进依据，模型解读时照抄，不自己做加减
    def gap(bound: float) -> str:
        return f"{round(abs(bound - value), 1):g} {'秒' if item.is_time else item.unit}"

    better = [step for step in ladder if step[0] > score]
    if better:
        nxt = better[-1]
        upper = next((step for step in reversed(better) if step[1] != level), None)
        if upper is not None and upper[0] == nxt[0]:
            basis += f"；达到 {nxt[3]} 可得 {nxt[0]} 分、进入「{upper[1]}」（差 {gap(nxt[2])}）"
        else:
            basis += f"；达到 {nxt[3]} 可得 {nxt[0]} 分（差 {gap(nxt[2])}）"
            if upper is not None:
                basis += f"，达到 {upper[3]} 进入「{upper[1]}」（差 {gap(upper[2])}）"
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
        full_mark=ladder[0][2],
        lower_better=lower_better,
    )


def _pick(first: str, second: str, label: str, names: tuple[str, str], notes: list[str]) -> tuple[float | None, str]:
    """同一指标在体检表与体质测试表各有一个数值：取第一个，两者不同时记一条说明。返回 (数值, 原文)。"""
    a, b = _number(first), _number(second)
    if a is not None and b is not None and abs(a - b) > 1e-6:
        notes.append(f"{label}有两个不同的数值：{names[0]} {first.strip()}、{names[1]} {second.strip()}，评分按{names[0]}数据")
    if a is not None:
        return a, first.strip()
    return b, second.strip()


_SOURCES = ("体检", "体质测试")


def body_size(record: dict[str, Any], notes: list[str]) -> tuple[float | None, float | None]:
    """身高(cm)、体重(kg)：体检表优先，没有再取体质测试表；两处数值不同时在 notes 里记一条。"""
    exam = record.get("physical_exam") or {}
    fitness = record.get("fitness_test") or {}
    height, _ = _pick(str(exam.get("SG") or ""), str(fitness.get("SGCJ") or ""), "身高", _SOURCES, notes)
    weight, _ = _pick(str(exam.get("TZ") or ""), str(fitness.get("TZCJ") or ""), "体重", _SOURCES, notes)
    return height, weight


def _bonus(table: ScoreTable, grade: str, scored: ItemScore) -> None:
    """附加分：成绩超过单项 100 分界值后，按超出的量查加分表。"""
    if scored.full_mark is None:
        return
    over = scored.full_mark - scored.value if scored.lower_better else scored.value - scored.full_mark
    if over <= 0:
        return
    steps: list[tuple[int, float, str]] = []  # (加分, 需要超出的量, 原文)
    for row in table.rows:
        need = _number(row.get(grade, ""))
        points = _number(row.get(_BONUS_KEY, ""))
        if need is not None and points is not None:
            steps.append((int(points), abs(need), row[grade]))
    hit = max((step for step in steps if over >= step[1]), default=None)
    if hit is None:
        return
    scored.bonus = hit[0]
    scored.bonus_basis = (
        f"{scored.name}成绩 {scored.raw} 超出 100 分界值 {over:g} {'秒' if scored.lower_better else scored.unit}，"
        f"按《{table.title}》加 {hit[0]} 分"
    )


def _level_scale(tables: list[ScoreTable]) -> list[tuple[int, str]]:
    """等级划分：每个等级在评分表里的最低得分。体重指数表用的是另一套等级词，不参与。"""
    lowest: dict[str, int] = {}
    for table in tables:
        if "体重指数" in table.subject:
            continue
        for row in table.rows:
            level, score = row.get(_LEVEL_KEY, ""), _number(row.get(_SCORE_KEY, ""))
            if level and score is not None:
                lowest[level] = min(lowest.get(level, 10**6), int(score))
    return sorted(((score, level) for level, score in lowest.items()), reverse=True)


def _total(result: Assessment, standard: Standard) -> None:
    """学年总分 = 标准分（必测项目得分 × 权重之和）+ 附加分；必测项目没测全时不计算。"""
    if not result.weights:
        result.total_basis = "知识库里没有找到该年级的单项指标与权重，无法计算学年总分。"
        return
    result.total_source = standard.weights_source
    if result.missing:
        result.total_basis = (
            f"{result.sex}生{result.grade}的必测项目没有测全，缺少：{'、'.join(item.text for item in result.missing)}。"
            "标准分由全部必测项目的得分与权重相乘求和，缺项时不计算学年总分与综合等级。"
        )
        return
    scored = {item.name: item for item in result.items}
    parts = [f"{name} {scored[name].score}×{weight:g}%" for name, weight in result.weights.items()]
    result.standard_score = round(
        sum(scored[name].score * weight for name, weight in result.weights.items()) / 100, 1
    )
    result.bonus = sum(item.bonus for item in result.items)
    result.total = round(result.standard_score + result.bonus, 1)
    result.level = result.level_of(result.total)
    bonus_notes = "；".join(item.bonus_basis for item in result.items if item.bonus)
    scale = "、".join(f"{level} {minimum} 分及以上" for minimum, level in result.scale[:-1])
    result.total_basis = (
        f"标准分 = 各必测项目得分 × 权重之和：{' + '.join(parts)} = {result.standard_score:g}。"
        f"附加分 {result.bonus}{f'（{bonus_notes}）' if bonus_notes else ''}。"
        f"学年总分 {result.total:g}，综合等级「{result.level}」（{scale}）。"
    )


def assess(record: dict[str, Any]) -> Assessment:
    """给一份档案的体测成绩逐项查表评分，并按标准计算学年总分。评分表取不到时抛 KnowledgeBaseUnavailable。"""
    exam = record.get("physical_exam") or {}
    fitness = record.get("fitness_test") or {}
    sex = str(exam.get("XB") or fitness.get("XB") or "").strip()[:1]
    age = to_int(exam.get("NL"))
    grade, inferred = grade_label(str(exam.get("SCHOOL_GRADE") or fitness.get("NJ") or ""), age)
    result = Assessment(sex=sex, grade=grade or "", grade_inferred=inferred)

    result.height_cm, result.weight_kg = body_size(record, result.notes)
    _, lung_raw = _pick(
        str(fitness.get("FHLCJ") or ""), str(exam.get("FHL") or ""), "肺活量", _SOURCES[::-1], result.notes
    )

    if sex not in ("男", "女") or grade is None:
        result.skipped.append("档案缺少性别或年级，无法按国标评分表评分")
        return result

    standard = load_standard()
    tables = [table for table in standard.tables if table.sex == sex]
    if not tables:
        raise store.KnowledgeBaseUnavailable("知识库里没有国家学生体质健康标准的单项评分表，无法给体测成绩评分")
    result.scale = _level_scale(tables)
    result.weights = {
        row.name: row.weight for row in standard.weights if sex in row.sexes and grade in row.grades
    }

    raws: dict[str, str] = {}
    for item in _ITEMS:
        if sex not in item.sexes:
            continue
        if item.key == "BMI":
            if not result.height_cm or not result.weight_kg:
                result.skipped.append("体重指数（BMI）：缺少身高或体重")
                continue
            value: float | None = result.weight_kg / (result.height_cm / 100) ** 2
            raw = f"{value:.1f}"
        else:
            raw = lung_raw if item.key == "FHLCJ" else str(fitness.get(item.key) or "").strip()
            raws.setdefault(item.name, raw)
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
        bonus_table = next(
            (t for t in standard.bonus_tables if t.sex == sex and t.subject == item.name), None
        )
        if bonus_table is not None:
            _bonus(bonus_table, grade, scored)
        result.items.append(scored)

    scored_names = {item.name for item in result.items}
    keys = {name: key for key, name in ITEM_NAMES.items()}
    result.missing = [
        MissingItem(key=keys.get(name, ""), name=name, raw=raws.get(name, ""))
        for name in result.weights
        if name not in scored_names
    ]
    _total(result, standard)
    return result
