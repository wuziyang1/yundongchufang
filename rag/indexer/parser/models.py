"""解析结果的数据结构。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Item:
    """页面上的一块内容：文字层的一个文本块、OCR 识别出的一行，或一张表格。坐标单位为 pt。"""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    # 单行行高。多行文本块的整体高度偏大，判断缩进与段间距时要用单行行高
    line_height: float = 0.0
    # 不为 None 表示这是一张表格（首行为表头），此时 text 存表题
    rows: list[list[str]] | None = None
    # 小字号文本（页脚注释、作者单位等），排到本页正文之后，避免插进段落中间
    minor: bool = False

    @property
    def is_table(self) -> bool:
        return self.rows is not None

    @property
    def unit_height(self) -> float:
        return self.line_height or (self.y1 - self.y0)


@dataclass
class Segment:
    """文档按阅读顺序拆出的一段内容：一段正文、一个标题，或一张表格。"""

    kind: str  # "text" 或 "table"
    page: int  # 页码，从 1 开始
    method: str  # "text"（PDF 自带文字层）或 "ocr"（扫描页识别）
    text: str = ""  # 正文 / 标题文字 / 表题
    rows: list[list[str]] = field(default_factory=list)  # 表格内容，首行为表头
    heading_level: int = 0  # 大于 0 表示这是章节标题，数值越小层级越高


@dataclass
class ParsedDocument:
    doc_name: str
    doc_path_name: str  # 相对知识库目录的路径，是文档在向量库中的唯一标识
    doc_type: str
    doc_md5: str
    doc_length: int
    page_count: int
    ocr_pages: int
    segments: list[Segment]
    # 文档身份，来自 doc_rules.toml：类别、适用学段、动作编号前缀（非空表示「练习N：」按动作条目入库）
    kind: str = ""
    stages: list[str] = field(default_factory=list)
    exercise_prefix: str = ""
    # 表题命中其一的表格是查表数据（评分表），切片类型记为 lookup
    lookup_tables: list[re.Pattern] = field(default_factory=list)
