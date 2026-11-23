"""动作条目：把训练手册里的「练习N：名称」整理成结构化信息，随切片写入 Milvus 的 meta 字段。

处方生成时只能从知识库里选动作，所以动作条目要带上代码能直接用的信息：编号、名称、
素质类别、器材、是否需要同伴，以及按原书小标题拆开的各段文字。全部从原文抽取，不做改写：

- 类别：优先看所在章节（「一、力量练习」「（二）增加心肺耐力的练习」），章节没有写明素质的
  （教学组织篇、器械使用篇）看「练习功能 / 目标」里最先提到的素质；
- 器材：「场地器材 / 练习器材」是并列的器材清单，逐项取出并去掉数量和场地描述；器械使用篇的练习
  用所在小节的器材名（「一、敏捷梯」）；「场地布置」是一句话的描述（「平坦空地，每组4个栏为1组」），
  不能按标点拆，只在其中找前两处已经出现过的器材名；
- 是否需要同伴：双人、团队章节，或起始姿势、练习步骤里写了「2人」「辅助练习者」「面对面」等。
  「所有人面朝内围成一个圆圈」是课堂队形，动作本身一个人能做，不算。

只在 doc_rules.toml 给文档配置了 exercise_prefix 时启用。
"""

from __future__ import annotations

import re

# 「练习12：平板支撑」；扫描件偶尔识别丢了冒号（「练习145叠罗汉」）
ENTRY_RE = re.compile(r"^练习\s*(\d{1,4})\s*[：:]?\s*(.*)$")

CATEGORIES = ("力量", "速度", "肌肉耐力", "心肺耐力", "灵敏", "协调", "平衡", "柔韧")
DEFAULT_CATEGORY = "综合"

# 原书小标题 → 统一的段名
_FIELD_LABELS = {
    "purpose": ("练习功能", "功能", "目标", "目的"),
    "equipment": ("场地器材", "练习器材", "器材"),
    "layout": ("场地布置",),
    "start": ("起始姿势", "起始动作"),
    "steps": ("练习步骤", "方法如下", "方法"),
    "tips": ("指导提示", "练习指导", "教学提示", "练习提示"),
    "safety": ("安全事项",),
    "variants": ("拓展练习", "扩展练习", "练习拓展", "晋级提高", "晋级"),
}
_LABEL_TO_FIELD = {label: name for name, labels in _FIELD_LABELS.items() for label in labels}
_LABEL_ALTERNATION = "|".join(sorted(_LABEL_TO_FIELD, key=len, reverse=True))
# 小标题出现在行首，或紧跟在上一句的句号后面（「…协作能力。练习器材：无。」）
_LABEL_RE = re.compile(rf"(?:^|(?<=[。；\n]))[ \t]*({_LABEL_ALTERNATION})[ \t]*[：:]", re.MULTILINE)
# 标题行里混进了正文（「练习219:正(后)向跨杆目标：发展学生协调性、力量。方法：」）
_INLINE_LABEL_RE = re.compile(rf"({_LABEL_ALTERNATION})[ \t]*[：:]")
# 扫描件把「目标：」「方法：」的第一个字识别丢了，只在行首认
_DAMAGED_LABEL_RE = re.compile(r"^[ \t]*(标|法)[ \t]*[：:]", re.MULTILINE)
_DAMAGED_LABELS = {"标": "目标", "法": "方法"}

# 章节标题里写明的素质
_SECTION_CATEGORY = (
    (re.compile(r"心肺耐力"), "心肺耐力"),
    (re.compile(r"肌肉耐力"), "肌肉耐力"),
    (re.compile(r"力量练习"), "力量"),
    (re.compile(r"速度练习"), "速度"),
    (re.compile(r"灵敏练习"), "灵敏"),
    (re.compile(r"协调练习"), "协调"),
    (re.compile(r"平衡练习"), "平衡"),
    (re.compile(r"柔韧练习"), "柔韧"),
)
# 「练习功能 / 目标」里提到的素质，取最先出现的
_PURPOSE_CATEGORY = (
    (re.compile(r"心肺|有氧"), "心肺耐力"),
    (re.compile(r"耐力"), "肌肉耐力"),
    (re.compile(r"力量|爆发力|弹跳"), "力量"),
    (re.compile(r"速度|步频|加速"), "速度"),
    (re.compile(r"灵敏|敏捷|变向|反应"), "灵敏"),
    (re.compile(r"协调|节奏"), "协调"),
    (re.compile(r"平衡|稳定"), "平衡"),
    (re.compile(r"柔韧|拉伸|伸展"), "柔韧"),
)

_CJK_GAP_RE = re.compile(r"(?<=[^\x00-\x7f])\s+|\s+(?=[^\x00-\x7f])")
_NUMBERING_RE = re.compile(r"^(?:[一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[）)]|第[一二三四五六七八九十\d]+[篇章部分])\s*")
_EQUIPMENT_PART_RE = re.compile(r"器械|器材")
_EQUIPMENT_SPLIT_RE = re.compile(r"[、，,；;。]|(?<=[一-鿿])[和及与](?=[一-鿿])")
_QUANTITY_HEAD_RE = re.compile(r"^[\d~～\-—至约]+\s*[个块条副根只组张套对]?\s*")
_QUANTITY_TAIL_RE = re.compile(r"\s*[\d~～\-—至约]+\s*[个块条副根只组张套对米]?.*$")
# 场地描述不是器材
_PLACE_RE = re.compile(r"场地|空地|地面|操场|教室|跑道|平地|室内|室外|草地|地上|地板")
_NO_EQUIPMENT = {"", "无", "徒手", "不需要", "无器材"}
_EQUIPMENT_MAX_LEN = 8

_GROUP_SECTION_RE = re.compile(r"双人|团队|多人|集体")
_GROUP_TEXT_RE = re.compile(
    r"[2-9两二三四五六七八九十多]\s*(?:人|名)(?!\s*次)|辅助练习者|同伴|搭档|队友|对方|面对面"
    r"|分成.{0,6}组|小组|接力"
)


def is_entry(headings: tuple[str, ...]) -> bool:
    """章节路径的最后一级是不是「练习N：…」。"""
    return bool(headings) and ENTRY_RE.match(headings[-1]) is not None


def split_heading(heading: str) -> tuple[int, str, str]:
    """「练习219:正(后)向跨杆目标：…」→ (219, "正(后)向跨杆", "目标：…")；第三项是混进标题行的正文。"""
    match = ENTRY_RE.match(heading)
    number, rest = int(match.group(1)), match.group(2).strip()
    inline = _INLINE_LABEL_RE.search(rest)
    spill = ""
    if inline and inline.start() > 0:
        rest, spill = rest[: inline.start()], rest[inline.start() :]
    return number, rest.strip(" ：:") or f"练习{number}", spill


def split_fields(content: str) -> dict[str, str]:
    """按原书小标题把正文拆成各段；同一段名出现多次时接在一起。"""
    content = _DAMAGED_LABEL_RE.sub(lambda m: f"{_DAMAGED_LABELS[m.group(1)]}：", content)
    matches = list(_LABEL_RE.finditer(content))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        # 扫描件按视觉行折断，行与行之间不留空格（「注\n意力」→「注意力」）；西文之间保留一个空格
        value = _CJK_GAP_RE.sub("", " ".join(content[match.end() : end].split()))
        if not value:
            continue
        name = _LABEL_TO_FIELD[match.group(1)]
        fields[name] = f"{fields[name]} {value}" if name in fields else value
    return fields


def _category(headings: tuple[str, ...], purpose: str) -> str:
    for heading in headings[:-1]:
        for pattern, category in _SECTION_CATEGORY:
            if pattern.search(heading):
                return category
    first: tuple[int, str] | None = None
    for pattern, category in _PURPOSE_CATEGORY:
        match = pattern.search(purpose)
        if match and (first is None or match.start() < first[0]):
            first = (match.start(), category)
    return first[1] if first else DEFAULT_CATEGORY


def _clean_equipment(raw: str) -> str:
    """去掉数量（「4~6个小栏架」「敏捷梯2~4副」）；场地描述和整句说明不算器材，返回空串。"""
    name = _QUANTITY_TAIL_RE.sub("", _QUANTITY_HEAD_RE.sub("", raw.strip())).strip(" ：:()（）")
    if name in _NO_EQUIPMENT or len(name) > _EQUIPMENT_MAX_LEN or _PLACE_RE.search(name):
        return ""
    return name


def _section_equipment(headings: tuple[str, ...]) -> str:
    """器械使用篇：小节标题就是器材名（「一、敏捷梯」）；「十四、组合练习」不是器材。"""
    if len(headings) < 3 or not _EQUIPMENT_PART_RE.search(headings[0]):
        return ""
    section = _NUMBERING_RE.sub("", headings[1]).strip()
    return "" if section.endswith("练习") else section


def _listed_equipment(fields: dict[str, str]) -> list[str]:
    names = (_clean_equipment(raw) for raw in _EQUIPMENT_SPLIT_RE.split(fields.get("equipment", "")))
    return [name for name in names if name]


def equipment_vocabulary(entries: list[tuple[tuple[str, ...], str]]) -> set[str]:
    """全书出现过的器材名：器材清单里列出的，加上器械使用篇各小节的名称。"""
    vocabulary: set[str] = set()
    for headings, content in entries:
        vocabulary.update(_listed_equipment(split_fields(content)))
        if section := _section_equipment(headings):
            vocabulary.add(section)
    return {name for name in vocabulary if len(name) >= 2}


def _equipment(headings: tuple[str, ...], fields: dict[str, str], vocabulary: set[str]) -> list[str]:
    items: list[str] = []
    for name in (_section_equipment(headings), *_listed_equipment(fields)):
        if name and name not in items:
            items.append(name)
    layout = fields.get("layout", "")
    found = [name for name in vocabulary if name in layout]
    # 「重力球」命中时不再单算其中的「球」
    for name in sorted(found, key=len, reverse=True):
        if not any(name != other and name in other for other in items) and name not in items:
            items.append(name)
    return items


def build_meta(headings: tuple[str, ...], content: str, prefix: str, vocabulary: set[str]) -> dict:
    """由章节路径与正文生成动作条目的结构化信息。"""
    number, name, _spill = split_heading(headings[-1])
    fields = split_fields(content)
    in_group_section = any(_GROUP_SECTION_RE.search(heading) for heading in headings[:-1])
    how = " ".join(fields.get(key, "") for key in ("start", "steps"))
    return {
        "code": f"{prefix}{number:03d}",
        "no": number,
        "name": name,
        "category": _category(headings, fields.get("purpose", "")),
        "group": "多人" if in_group_section or _GROUP_TEXT_RE.search(how) else "单人",
        "equipment": _equipment(headings, fields, vocabulary),
        "fields": fields,
    }
