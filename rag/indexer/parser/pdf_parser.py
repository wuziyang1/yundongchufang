"""PDF 解析。

- 有文字层的页：按单词取字，再按「视觉行」聚合（上标引用、错位的标点都能回到原位）；
  用 PyMuPDF 的表格识别定位表格，单元格内容同样按视觉行取字；只有横线的三线表、
  带外框的期刊表格另按「横线 + 表头列位置」切分行列；
- 扫描页（几乎没有文字、主要是图片）：本地 OCR 识别，丢弃插图里识别出的零散字母数字；
- 去掉封面、版权页、编委会名单等前置页，期刊首页的标题、作者、单位区和作者脚注（基金、
  通讯作者邮箱）；去掉页眉、页脚和页码，小字号的脚注排到本页正文之后，按阅读顺序
  （支持期刊双栏排版）把折行拼回段落，识别章节标题，丢弃参考文献列表；
- 个别文档的特殊规则（跳过页、额外的标题写法、不入库的附录、字形修正）见 doc_rules.toml。
"""

from __future__ import annotations

import hashlib
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from ..config import Settings
from ..logger import get_logger
from .doc_rules import DocRuleBook, DocRules
from .models import Item, ParsedDocument, Segment
from .ocr import OcrEngine
from .text_utils import clean_text, is_han, is_page_number, merge_lines

logger = get_logger()

_NO_RULES = DocRules()

# 文字层不足这么多字、且图片覆盖页面达到这一比例时，按扫描页走 OCR
_SCANNED_MAX_TEXT = 50
_SCANNED_MIN_IMAGE_COVERAGE = 0.3

# 页眉页脚区域：页面顶部 / 底部 max(10% 页高, 62pt) 以内；去掉数字后在至少这么多页上重复出现即判为页眉页脚
_MARGIN_RATIO = 0.10
_MARGIN_MIN = 62
_MIN_REPEATS = 3

# 行高低于正文行高这一比例的视为小字号（脚注、作者单位）
_MINOR_RATIO = 0.85

# 非空单元格占比低于此值的「表格」多为误检（期刊分隔线）或空白表单，按普通文本处理
_MIN_TABLE_FILL = 0.4
# 表题：位于表格上方这么近（pt）以内、不超过这么多字、且含「表」字的短行
_CAPTION_GAP = 36
_CAPTION_MAX_LEN = 120

_HAS_CONTENT_RE = re.compile(r"[0-9A-Za-z一-鿿]")
# 目录条目：短行，以「／页码」或「……」「……页码」「……(页码)」结尾，如「（二）基本理念／2」「八、来回奔跑测试……(202)」
_TOC_ENTRY_RE = re.compile(r"(?:[／/]\s*\d{1,3}|(?:…+|\.{3,}|·{3,})\s*[(（]?\s*\d{0,3}\s*[)）]?)$")
_REFERENCE_TITLES = {"references", "reference", "参考文献"}

_HEADING_MAX_LEN = 40
_HEADING_PATTERNS = [
    (1, re.compile(r"^第[一二三四五六七八九十百零\d]+[篇章部分]")),
    (1, re.compile(r"^附\s*录\s*[\d一二三四五六七八九十]*(?:\s|$)")),
    (1, re.compile(r"^前\s*言$")),
    (2, re.compile(r"^[一二三四五六七八九十]+、")),
    (3, re.compile(r"^[（(][一二三四五六七八九十]+[）)]")),
    (3, re.compile(r"^问题\s*\d+")),
    (4, re.compile(r"^练习\s*\d+\s*[：:]")),
]

# ---- 前置页：封面、版权页（ISBN、CIP、定价、印刷信息）、编委会名单 ----
_FRONT_MATTER_PAGES = 12
_FRONT_MATTER_RE = re.compile(
    r"ISBN|CIP|定\s*价|印\s*张|开\s*本|印\s*刷|版\s*次|责任编辑|责任校对|封面设计|版式设计|策划编辑"
    r"|出版发行|邮\s*编|编\s*委\s*会|主\s*编|编\s*委"
)
_COVER_PAGES = 3
_COVER_MAX_TEXT = 200

# ---- 期刊首页：标题、作者、单位区，以及作者脚注（基金项目、通讯作者邮箱）----
_ARTICLE_SIGNAL_RE = re.compile(
    r"DOI\s*[:：]|E-?mail|通讯作者|作者单位|基金项目|收稿日期|Address correspondence|Corresponding author",
    re.IGNORECASE,
)
_BODY_START_RE = re.compile(r"^(?:abstract|摘\s*要|introduction|引\s*言|前\s*言)$", re.IGNORECASE)
# 脚注分隔线：页面下部一条孤立的短横线
_FOOTNOTE_RULE_MIN_Y = 0.75
_FOOTNOTE_SMALL_RATIO = 0.95

# ---- 噪声行 ----
_RUNNING_HEADER_RE = re.compile(r"^│.*│$")  # 课程标准奇数页的「│ 前言 │」
_OCR_NOISE_MAX_LEN = 30  # 扫描页上不含汉字的短行多是插图里的字母、数字、箭头

# ---- 只有横线的表格（三线表、期刊表格）----
_RULE_MIN_WIDTH = 40
_RULE_HEADER_GAP = 60  # 顶线与表头下横线的最大间距
_TABLE_CAPTION_RE = re.compile(r"^(?:续?表|TABLE|Table)\s*[\dA-Z]")
_BARE_NUMBER_RE = re.compile(r"^\d{1,2}$")
_CAPTION_MIN_LEN = 6
_CONTINUED_RE = re.compile(r"^续\s*表")
_SENTENCE_END_RE = re.compile(r"[。！？；!?;]")


def heading_level(text: str, *, allow_caps: bool = True, rules: DocRules = _NO_RULES) -> int:
    """章节标题的层级（1 最高）；不是标题返回 0。"""
    if len(text) > _HEADING_MAX_LEN:
        return 0
    if level := rules.heading_level(text):
        return level
    for level, pattern in _HEADING_PATTERNS:
        if pattern.match(text):
            return level
    # 英文文献的章节标题多为全大写短行，如 INTRODUCTION。扫描页不用这条：插图里的噪点常被识别成大写字母
    if allow_caps and text.isascii() and text.isupper():
        letters = sum(ch.isalpha() for ch in text)
        if letters >= 4 and letters >= 0.6 * len(text.replace(" ", "")):
            return 2
    return 0


@dataclass
class _Page:
    number: int
    method: str  # text / ocr
    width: float
    height: float
    items: list[Item]


@dataclass
class _Table:
    rect: pymupdf.Rect
    rows: list[list[str]]


class PDFParser:
    def __init__(self, settings: Settings):
        self._ocr = OcrEngine(settings) if settings.ocr_enabled else None
        self._rule_book = DocRuleBook(settings.doc_rules_path)

    def parse(self, path: Path, base_dir: Path) -> ParsedDocument:
        data = path.read_bytes()
        doc_md5 = hashlib.md5(data).hexdigest()
        doc_path_name = path.resolve().relative_to(base_dir.resolve()).as_posix()
        rules = self._rule_book.for_document(doc_path_name)
        pages: list[_Page] = []
        skipped_scans = 0

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
            for page in doc:
                number = page.number + 1
                if number in rules.skip_pages:
                    continue
                if _is_scanned(page):
                    if self._ocr is None:
                        skipped_scans += 1
                        continue
                    items, method = self._ocr.recognize(page, doc_md5), "ocr"
                else:
                    items, method = _text_layer_items(page), "text"
                pages.append(_Page(number, method, page.rect.width, page.rect.height, items))
                if number % 25 == 0 and number < page_count:
                    logger.info("    已解析 %d/%d 页", number, page_count)

        if skipped_scans:
            logger.warning(
                "    有 %d 页是扫描件，但 OCR 已关闭（ENABLE_PDF_OCR=false），这些页被跳过", skipped_scans
            )

        front = [page.number for page in pages if _is_front_matter(page, rules)]
        if front:
            logger.info("    跳过前置页（封面、版权页、编委会等）：第 %s 页", "、".join(map(str, front)))
            pages = [page for page in pages if page.number not in front]
        for page in pages:
            _drop_noise_items(page, rules)

        _strip_page_furniture(pages)
        for page in pages:
            _attach_captions(page)
        segments = [segment for page in pages for segment in _page_segments(page, rules)]
        segments = _drop_references(segments)
        segments = _adopt_table_captions(segments)
        segments = _drop_sections(segments, rules)
        if rules.replacements or rules.drop_lines:
            segments = _apply_line_rules(segments, rules)
        return ParsedDocument(
            doc_name=path.name,
            doc_path_name=doc_path_name,
            doc_type=path.suffix.lstrip(".").lower(),
            doc_md5=doc_md5,
            doc_length=len(data),
            page_count=page_count,
            ocr_pages=sum(1 for page in pages if page.method == "ocr"),
            segments=segments,
        )


# ---------------------------------------------------------------------------
# 页面类型判断
# ---------------------------------------------------------------------------


def _is_scanned(page: pymupdf.Page) -> bool:
    if len("".join(page.get_text().split())) >= _SCANNED_MAX_TEXT:
        return False
    page_area = abs(page.rect) or 1.0
    covered = sum(abs(pymupdf.Rect(info["bbox"]) & page.rect) for info in page.get_image_info())
    return covered / page_area >= _SCANNED_MIN_IMAGE_COVERAGE


def _is_front_matter(page: _Page, rules: DocRules) -> bool:
    """书籍开头的封面、扉页、版权页（ISBN、CIP、定价、印刷信息）、编委会名单。

    判断依据：出现两种以上出版信息标记、且几乎没有成句的正文；或者开头几页里字很少、没有句号的封面页。
    页上有章节标题时不算（如「前言」页）。
    """
    if page.number > _FRONT_MATTER_PAGES:
        return False
    texts = [item.text for item in page.items if not item.is_table]
    allow_caps = page.method == "text"
    if any(heading_level(text, allow_caps=allow_caps, rules=rules) for text in texts):
        return False
    text = "".join("".join(texts).split())
    sentences = len(re.findall(r"[。！？]", text))
    markers = {re.sub(r"\s+", "", m.group(0)) for m in _FRONT_MATTER_RE.finditer(" ".join(texts))}
    if len(markers) >= 2 and sentences <= 2:
        return True
    return page.number <= _COVER_PAGES and len(text) < _COVER_MAX_TEXT and sentences == 0


def _drop_noise_items(page: _Page, rules: DocRules) -> None:
    kept = []
    for item in page.items:
        if not item.is_table:
            if _RUNNING_HEADER_RE.match(item.text) or rules.drops_line(item.text):
                continue
            # 扫描页插图里的「TEEEE」「A→ D → C」「20m」之类：不含汉字的短行
            if (
                page.method == "ocr"
                and len(item.text) <= _OCR_NOISE_MAX_LEN
                and not any(is_han(ch) for ch in item.text)
            ):
                continue
        kept.append(item)
    page.items = kept


# ---------------------------------------------------------------------------
# 视觉行：把同一行上相邻的文字片段（含上标、错位标点）按从左到右拼起来
# ---------------------------------------------------------------------------


def _on_same_line(a: Item, b: Item) -> bool:
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    return overlap >= 0.5 * min(a.y1 - a.y0, b.y1 - b.y0)


def _sort_lines(items: list[Item]) -> list[Item]:
    """先按视觉行自上而下、行内再从左到右排序。

    直接按 y0 排序时，同一行里字体不同的片段（如粗体的「问题1」和正文）会因为顶边略有高低而颠倒。
    """
    result: list[Item] = []
    line: list[Item] = []
    for item in sorted(items, key=lambda i: (i.y0 + i.y1) / 2):
        if line and not _on_same_line(item, line[0]):
            result += sorted(line, key=lambda i: i.x0)
            line = []
        line.append(item)
    return result + sorted(line, key=lambda i: i.x0)


def _same_row(a: Item, b: Item) -> bool:
    height_a, height_b = a.y1 - a.y0, b.y1 - b.y0
    overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    if overlap < 0.5 * min(height_a, height_b):
        return False
    gap = max(a.x0, b.x0) - min(a.x1, b.x1)
    return gap <= 0.6 * max(height_a, height_b)


def _visual_rows(pieces: list[Item]) -> list[Item]:
    rows: list[list[Item]] = []
    for piece in sorted(pieces, key=lambda p: ((p.y0 + p.y1) / 2, p.x0)):
        target = next(
            (row for row in reversed(rows[-8:]) if any(_same_row(piece, member) for member in row)),
            None,
        )
        if target is None:
            rows.append([piece])
        else:
            target.append(piece)

    items: list[Item] = []
    for row in rows:
        row.sort(key=lambda p: p.x0)
        text = clean_text(merge_lines([p.text for p in row]))
        if not text:
            continue
        items.append(
            Item(
                text,
                min(p.x0 for p in row),
                min(p.y0 for p in row),
                max(p.x1 for p in row),
                max(p.y1 for p in row),
                line_height=max(p.y1 - p.y0 for p in row),
            )
        )
    return _sort_lines(items)


def _line_fragments(words: list[tuple]) -> list[Item]:
    """PyMuPDF 的单词按 (块号, 行号) 归回原始文本行。"""
    groups: dict[tuple[int, int], list[tuple]] = {}
    for word in words:
        if word[4].strip():
            groups.setdefault((word[5], word[6]), []).append(word)
    fragments = []
    for group in groups.values():
        group.sort(key=lambda w: w[7])
        fragments.append(
            Item(
                " ".join(w[4] for w in group),
                min(w[0] for w in group),
                min(w[1] for w in group),
                max(w[2] for w in group),
                max(w[3] for w in group),
            )
        )
    return fragments


def _center_inside(item: Item, rect: pymupdf.Rect, tolerance: float = 1.0) -> bool:
    cx, cy = (item.x0 + item.x1) / 2, (item.y0 + item.y1) / 2
    return (
        rect.x0 - tolerance <= cx <= rect.x1 + tolerance
        and rect.y0 - tolerance <= cy <= rect.y1 + tolerance
    )


# ---------------------------------------------------------------------------
# 文字层：正文与表格
# ---------------------------------------------------------------------------


def _text_layer_items(page: pymupdf.Page) -> list[Item]:
    words = page.get_text("words")
    rules, shades = _page_rules(page)
    tables = _extract_tables(page, words)
    tables += _ruled_tables(page, words, rules, shades, tables)

    fragments = [
        fragment
        for fragment in _line_fragments(words)
        if not any(_center_inside(fragment, table.rect) for table in tables)
    ]
    rows = [row for row in _visual_rows(fragments) if _HAS_CONTENT_RE.search(row.text)]
    _mark_minor(rows)
    _strip_article_front(page, rows, rules)
    # 表题等去掉页眉页脚之后再取（表格在页面顶部时，页眉会被误当成表题的一部分），见 _attach_captions
    rows += [Item("", *table.rect, rows=table.rows) for table in tables]
    return rows


def _attach_captions(page: _Page) -> None:
    for table in [item for item in page.items if item.is_table and not item.text]:
        table.text = _take_caption(page.items, pymupdf.Rect(table.x0, table.y0, table.x1, table.y1))


def _body_height(rows: list[Item]) -> float | None:
    sized = [row.line_height for row in rows if len(row.text) >= 5]
    return statistics.median(sized) if len(sized) >= 5 else None


def _mark_minor(rows: list[Item]) -> None:
    body = _body_height(rows)
    if body is None:
        return
    for row in rows:
        if row.line_height < _MINOR_RATIO * body:
            row.minor = True


def _strip_article_front(page: pymupdf.Page, rows: list[Item], rules: list[pymupdf.Rect]) -> None:
    """期刊论文的首页：去掉正文之前的标题、作者、单位区，以及页面下部脚注区里的作者信息。

    - 脚注区：页面下部一条孤立短横线以下、字号小于正文的行。含基金项目、作者单位、通讯作者、
      邮箱等作者信息时整块丢弃，否则只标为小字号（排到本页正文之后）；
    - 标题区：仅首页、且页面上有 DOI / 通讯作者 / Address correspondence 等论文标志时，
      「摘要 / ABSTRACT / 前言」所在行以上的内容全部丢弃。
    """
    height = page.rect.height
    body = _body_height(rows)
    for rule in rules:
        if rule.y0 < _FOOTNOTE_RULE_MIN_Y * height or rule.width > 0.5 * page.rect.width:
            continue
        if any(other is not rule and abs(other.y0 - rule.y0) < 30 for other in rules):
            continue  # 表格里的横线成组出现，脚注分隔线是孤立的
        notes = [
            row
            for row in rows
            if row.y0 >= rule.y0 - 1
            and (body is None or row.line_height < _FOOTNOTE_SMALL_RATIO * body)
        ]
        if not notes:
            continue
        if _ARTICLE_SIGNAL_RE.search(" ".join(row.text for row in notes)):
            rows[:] = [row for row in rows if all(row is not note for note in notes)]
        else:
            for note in notes:
                note.minor = True

    if page.number != 0 or not _ARTICLE_SIGNAL_RE.search(page.get_text()):
        return
    start = next((row for row in rows if _BODY_START_RE.match(re.sub(r"\s+", "", row.text))), None)
    if start is not None:
        rows[:] = [row for row in rows if row.y1 > start.y0 + 1]


def _extract_tables(page: pymupdf.Page, words: list[tuple]) -> list[_Table]:
    try:
        found = page.find_tables().tables
    except Exception as exc:  # noqa: BLE001 —— 表格识别失败时整页按普通文本处理
        logger.warning("    第 %d 页表格识别失败（%s），按普通文本处理", page.number + 1, exc)
        return []

    word_items = [Item(w[4], w[0], w[1], w[2], w[3]) for w in words if w[4].strip()]
    tables: list[_Table] = []
    for table in found:
        geometry = [list(row.cells) for row in table.rows]
        if len(geometry) < 2 or table.col_count < 2:
            continue
        raw = [
            [None if cell is None else _cell_text(word_items, cell) for cell in cells]
            for cells in geometry
        ]
        cells = [cell for row in raw for cell in row]
        filled = sum(1 for cell in cells if cell)
        if not cells or filled / len(cells) < _MIN_TABLE_FILL:
            continue
        rows = _fill_merged_cells(geometry, raw, table.col_count)
        tables.append(_Table(pymupdf.Rect(table.bbox), _unfold_repeated_columns(rows)))
    return tables


def _cell_text(word_items: list[Item], bbox: tuple) -> str:
    rect = pymupdf.Rect(bbox)
    inside = [word for word in word_items if _center_inside(word, rect)]
    return clean_text(merge_lines([row.text for row in _visual_rows(inside)]))


def _fill_merged_cells(
    geometry: list[list[tuple | None]],
    raw: list[list[str | None]],
    column_count: int,
) -> list[list[str]]:
    """合并单元格在识别结果里是 None：横向合并取左侧单元格的值，纵向合并取上方单元格的值。

    「优秀」占 3 行时，下面两行也补成「优秀」，逐行展开后每一行都能独立读懂。
    """
    # 每一列的右边界：该列真实单元格右边界的最小值
    column_right: list[float | None] = []
    for col in range(column_count):
        rights = [cells[col][2] for cells in geometry if col < len(cells) and cells[col] is not None]
        column_right.append(min(rights) if rights else None)

    grid: list[list[str]] = []
    for cells, row in zip(geometry, raw):
        out: list[str] = []
        for col in range(column_count):
            value = row[col] if col < len(row) else None
            if value is not None:
                out.append(value)
                continue
            left = _covering_left(cells, col, column_right[col])
            if left is not None:
                out.append(out[left])
            elif grid:
                out.append(grid[-1][col])
            else:
                out.append(out[-1] if out else "")
        grid.append(out)
    return grid


def _covering_left(cells: list, col: int, right: float | None) -> int | None:
    """同一行左侧最近的真实单元格若横跨到第 col 列，返回它的列号。"""
    if right is None:
        return None
    for k in range(col - 1, -1, -1):
        if k < len(cells) and cells[k] is not None:
            return k if cells[k][2] >= right - 1 else None
    return None


def _unfold_repeated_columns(rows: list[list[str]]) -> list[list[str]]:
    """表头左右重复的「双栏表」（身体活动内容 | MET | 身体活动内容 | MET）拆成一列一组，逐行展开时不串行。"""
    if not rows:
        return rows
    header = rows[0]
    for parts in (2, 3):
        if len(header) % parts or len(header) < 2 * parts:
            continue
        width = len(header) // parts
        first = header[:width]
        if not any(first) or any(header[i * width : (i + 1) * width] != first for i in range(parts)):
            continue
        body = [
            row[i * width : (i + 1) * width]
            for i in range(parts)
            for row in rows[1:]
            if any(row[i * width : (i + 1) * width])
        ]
        return [first, *body]
    return rows


# ---------------------------------------------------------------------------
# 只有横线的表格：三线表、期刊里带外框的表格
#
# PyMuPDF 的表格识别依赖单元格边框，三线表（顶线、表头下横线、底线，中间没有竖线）认不出来，
# 文字会被当成正文、行内配对关系错乱。这里用「同宽的三条横线」定位表格：表头区的文字给出各列的
# 大致位置，列与列的分界取表体文字最少跨过的竖线；行按内部横线或底纹色块切分，没有时按视觉行切分，
# 首列为空的行视为上一行的续行。
# ---------------------------------------------------------------------------


def _page_rules(page: pymupdf.Page) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """页面上的水平横线（同一高度上相接的线段合并为一条），以及底纹色块。"""
    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001 —— 取不到矢量图形时按没有横线处理
        return [], []

    segments: list[pymupdf.Rect] = []
    shades: list[pymupdf.Rect] = []
    for drawing in drawings:
        for item in drawing.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) <= 1 and abs(p1.x - p2.x) >= _RULE_MIN_WIDTH / 2:
                    y = (p1.y + p2.y) / 2
                    segments.append(pymupdf.Rect(min(p1.x, p2.x), y, max(p1.x, p2.x), y))
            elif item[0] == "re":
                rect = pymupdf.Rect(item[1])
                if rect.height <= 2 and rect.width >= _RULE_MIN_WIDTH / 2:
                    y = (rect.y0 + rect.y1) / 2
                    segments.append(pymupdf.Rect(rect.x0, y, rect.x1, y))
                elif drawing.get("fill") is not None and rect.height >= 4 and rect.width >= _RULE_MIN_WIDTH:
                    shades.append(rect)

    merged: list[pymupdf.Rect] = []
    for segment in sorted(segments, key=lambda r: (round(r.y0), r.x0)):
        last = merged[-1] if merged else None
        if last is not None and abs(last.y0 - segment.y0) <= 1.5 and segment.x0 <= last.x1 + 3:
            last.x1 = max(last.x1, segment.x1)
        else:
            merged.append(pymupdf.Rect(segment))
    return [rule for rule in merged if rule.width >= _RULE_MIN_WIDTH], shades


def _same_span(a: pymupdf.Rect, b: pymupdf.Rect) -> bool:
    return abs(a.x0 - b.x0) <= 6 and abs(a.x1 - b.x1) <= 6


def _ruled_tables(
    page: pymupdf.Page,
    words: list[tuple],
    rules: list[pymupdf.Rect],
    shades: list[pymupdf.Rect],
    existing: list[_Table],
) -> list[_Table]:
    """同宽的一组横线（≥3 条，前两条相距不远：顶线 + 表头下横线）围出一张表；中间的横线都是行分隔。

    同宽横线之间夹着表题（「表2…」「TABLE 3.」）时，说明是上下排列的两张表，从这里断开。
    """
    word_items = [Item(w[4], w[0], w[1], w[2], w[3]) for w in words if w[4].strip()]
    groups: list[list[pymupdf.Rect]] = []
    for rule in sorted(rules, key=lambda r: r.y0):
        group = next((g for g in groups if _same_span(g[0], rule)), None)
        if group is None:
            groups.append([rule])
        else:
            group.append(rule)

    runs: list[list[pymupdf.Rect]] = []
    for group in groups:
        run = [group[0]]
        for rule in group[1:]:
            between = [
                w
                for w in word_items
                if run[-1].y0 < (w.y0 + w.y1) / 2 < rule.y0 and w.x1 > rule.x0 and w.x0 < rule.x1
            ]
            if any(_TABLE_CAPTION_RE.match(line[0].text) for line in _group_lines(between)):
                runs.append(run)
                run = []
            run.append(rule)
        runs.append(run)

    found: list[_Table] = []
    for run in runs:
        if len(run) < 3 or run[1].y0 - run[0].y0 > _RULE_HEADER_GAP:
            continue
        top, header_y, bottom = run[0], run[1].y0, run[-1]
        rect = pymupdf.Rect(top.x0, top.y0, top.x1, bottom.y0)
        if any((rect & table.rect).get_area() > 0 for table in [*existing, *found]):
            continue
        cuts = [
            rule.y0
            for rule in rules
            if header_y + 3 < rule.y0 < rect.y1 - 3 and rule.x0 < rect.x1 and rule.x1 > rect.x0
        ]
        cuts += [
            y
            for shade in shades
            if shade.x0 < rect.x1 and shade.x1 > rect.x0
            for y in (shade.y0, shade.y1)
            if header_y + 3 < y < rect.y1 - 3
        ]
        grid = _parse_ruled_table(word_items, rect, header_y, cuts)
        if grid is None:
            continue
        found.append(_Table(rect, grid))
        logger.debug("    第 %d 页识别出无竖线表格：%d 行 × %d 列", page.number + 1, len(grid), len(grid[0]))
    return found


def _group_lines(items: list[Item]) -> list[list[Item]]:
    """按视觉行分组（不管水平间距），每组内从左到右。"""
    lines: list[list[Item]] = []
    for item in sorted(items, key=lambda i: (i.y0 + i.y1) / 2):
        if lines and _on_same_line(item, lines[-1][0]):
            lines[-1].append(item)
        else:
            lines.append([item])
    return [sorted(line, key=lambda i: i.x0) for line in lines]


def _header_cells(words: list[Item]) -> list[Item]:
    """表头文字按水平位置聚成单元格（同一格里的多个词、折成两行的表头都归为一格）。"""
    cells: list[Item] = []
    for word in sorted(words, key=lambda w: w.x0):
        last = cells[-1] if cells else None
        height = word.y1 - word.y0
        if last is not None and word.x0 - last.x1 <= 0.6 * height:
            last.x1 = max(last.x1, word.x1)
            last.y0, last.y1 = min(last.y0, word.y0), max(last.y1, word.y1)
        else:
            cells.append(Item(word.text, word.x0, word.y0, word.x1, word.y1))
    return cells


def _table_column_bounds(cells: list[Item], body: list[Item]) -> list[float]:
    def crossing(x: float) -> int:
        return sum(1 for w in body if w.x0 < x - 0.5 and w.x1 > x + 0.5)

    def best(lo: float, hi: float, need_left: bool = False) -> float | None:
        xs = [lo + 0.5 * k for k in range(int((hi - lo) / 0.5) + 1)]
        if need_left:
            xs = [x for x in xs if any(w.x1 <= x for w in body)]
        if not xs:
            return None
        least = min(crossing(x) for x in xs)
        runs: list[list[float]] = []
        for x in xs:
            if crossing(x) != least:
                continue
            if runs and abs(x - runs[-1][-1] - 0.5) < 1e-6:
                runs[-1].append(x)
            else:
                runs.append([x])
        run = max(runs, key=len)
        return (run[0] + run[-1]) / 2

    bounds: list[float] = []
    # 首列是行标题、表头上没有文字（期刊表格常见）：多数行从表头第一格的左边开始
    starts = [line[0].x0 for line in _group_lines(body)]
    if starts and statistics.median(starts) < cells[0].x0 - 15:
        bound = best(min(starts) + 1, cells[0].x0, need_left=True)
        if bound is not None:
            bounds.append(bound)
    for left, right in zip(cells, cells[1:]):
        bound = best(left.x1, right.x0)
        if bound is not None:
            bounds.append(bound)
    return bounds


def _join_cell(words: list[Item]) -> str:
    return clean_text(merge_lines([row.text for row in _visual_rows(words)]))


def _parse_ruled_table(
    word_items: list[Item], rect: pymupdf.Rect, header_y: float, cuts: list[float]
) -> list[list[str]] | None:
    def within(word: Item, top: float, bottom: float) -> bool:
        cx, cy = (word.x0 + word.x1) / 2, (word.y0 + word.y1) / 2
        return rect.x0 - 2 <= cx <= rect.x1 + 2 and top <= cy <= bottom

    head = [w for w in word_items if within(w, rect.y0, header_y)]
    body = [w for w in word_items if within(w, header_y, rect.y1)]
    if not head or not body:
        return None
    cells = _header_cells(head)
    bounds = _table_column_bounds(cells, body)
    if not bounds:
        return None
    columns = len(bounds) + 1

    def column(word: Item) -> int:
        center = (word.x0 + word.x1) / 2
        return sum(1 for bound in bounds if center > bound)

    def split(words: list[Item]) -> list[str]:
        grouped: list[list[Item]] = [[] for _ in range(columns)]
        for word in words:
            grouped[column(word)].append(word)
        return [_join_cell(group) for group in grouped]

    header = split(head)
    rows: list[list[str]] = []
    edges = sorted(cuts)
    if edges:
        # 行由内部横线或底纹色块分隔：每条带状区域一行
        borders = [header_y, *[y for k, y in enumerate(edges) if k == 0 or y - edges[k - 1] > 3], rect.y1]
        for top, bottom in zip(borders, borders[1:]):
            band = [w for w in body if top <= (w.y0 + w.y1) / 2 <= bottom]
            if band:
                rows.append(split(band))
    else:
        for line in _group_lines(body):
            texts = split(line)
            if rows and not texts[0]:
                rows[-1] = [merge_lines([a, b]) for a, b in zip(rows[-1], texts)]
            else:
                rows.append(texts)
    rows = [row for row in rows if any(row)]
    if not rows:
        return None
    return _unfold_repeated_columns([header, *rows])


def _take_caption(items: list[Item], table_rect: pymupdf.Rect) -> str:
    """从表格正上方的短行里取出表题（如「表1-1 男生体重指数（BMI）单项评分表」），并从正文中移除。

    表号所在的行排在最前（「表1」+「常见儿童青少年不同身体活动与相应的代谢当量」）。
    """
    near = [
        item
        for item in items
        if not item.is_table
        and item.y0 >= table_rect.y0 - _CAPTION_GAP
        and item.y1 <= table_rect.y0 + 2
        and item.x1 > table_rect.x0
        and item.x0 < table_rect.x1
        and len(item.text) <= _CAPTION_MAX_LEN
    ]
    near.sort(key=lambda i: (not _TABLE_CAPTION_RE.match(i.text), round(i.y0), i.x0))
    caption = merge_lines([item.text for item in near])
    if not re.search(r"表|^table", caption, re.IGNORECASE):
        return ""
    items[:] = [item for item in items if all(item is not taken for taken in near)]
    return caption


# ---------------------------------------------------------------------------
# 页眉页脚与页码
# ---------------------------------------------------------------------------


def _margin_key(item: Item, page: _Page) -> str | None:
    # 表题（「表2-1 …」「续表」）出现在页面顶部时也不是页眉，只是各页的表题去掉数字后可能雷同
    if item.is_table or _TABLE_CAPTION_RE.match(item.text):
        return None
    margin = max(page.height * _MARGIN_RATIO, _MARGIN_MIN)
    if item.y1 <= margin:
        zone = "top"
    elif item.y0 >= page.height - margin:
        zone = "bottom"
    else:
        return None
    return zone + ":" + re.sub(r"[\d\s]+", "", item.text)


def _strip_page_furniture(pages: list[_Page]) -> None:
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({key for item in page.items if (key := _margin_key(item, page))})

    for page in pages:
        kept = []
        for item in page.items:
            key = _margin_key(item, page)
            if key is not None and (
                is_page_number(item.text) or key.endswith(":") or counts[key] >= _MIN_REPEATS
            ):
                continue
            kept.append(item)
        page.items = kept


# ---------------------------------------------------------------------------
# 阅读顺序与段落
# ---------------------------------------------------------------------------


def _find_gutters(items: list[Item], width: float) -> list[float]:
    """找出各栏之间的空白分隔线（支持双栏、三栏，以及左窄右宽的不对称分栏）；单栏页返回空列表。

    分隔线：几乎没有文字跨过（跨过的多是通栏的标题、作者信息），且两侧各有至少 15% 的文字。
    """
    text_items = [item for item in items if not item.is_table]
    total = sum(len(item.text) for item in text_items)
    if len(text_items) < 6 or not total:
        return []

    def crossing(x: float) -> int:
        return sum(len(i.text) for i in text_items if i.x0 < x - 2 and i.x1 > x + 2)

    def qualifies(x: float) -> bool:
        left = sum(len(i.text) for i in text_items if i.x1 <= x + 2)
        right = sum(len(i.text) for i in text_items if i.x0 >= x - 2)
        return left >= 0.15 * total and right >= 0.15 * total and crossing(x) <= 0.2 * total

    # 连续满足条件的 x 归为一条空白带，每条带取跨过文字最少、最靠带中间的位置
    gutters: list[float] = []
    strip: list[int] = []
    for x in [*range(int(width * 0.15), int(width * 0.85) + 1, 2), None]:
        if x is not None and qualifies(x):
            strip.append(x)
            continue
        if strip:
            center = (strip[0] + strip[-1]) / 2
            gutters.append(float(min(strip, key=lambda s: (crossing(s), abs(s - center)))))
            strip = []
    return gutters


def _reading_order(items: list[Item], width: float) -> list[tuple[Item, str]]:
    """返回 (内容块, 所在栏)。栏：C0/C1/… 从左到右各栏 / F 通栏或单栏 / M 小字号脚注。

    分栏页按「通栏块分隔出的区段」排序：每个区段从左到右逐栏读完。脚注统一排在本页最后。
    """
    main = [item for item in items if not item.minor]
    minor = _sort_lines([item for item in items if item.minor])
    gutters = _find_gutters(main, width)

    if not gutters:
        ordered = [(item, "F") for item in _sort_lines(main)]
        return ordered + [(item, "M") for item in minor]

    def column(item: Item) -> int | None:
        if any(item.x0 < g - 2 and item.x1 > g + 2 for g in gutters):
            return None  # 跨栏
        return sum(1 for g in gutters if item.x0 >= g - 2)

    ordered: list[tuple[Item, str]] = []
    band: dict[int, list[Item]] = {}

    def close_band() -> None:
        for index in sorted(band):
            ordered.extend((item, f"C{index}") for item in _sort_lines(band[index]))
        band.clear()

    for item in sorted(main, key=lambda i: (i.y0, i.x0)):
        col = column(item)
        if col is None:
            close_band()
            ordered.append((item, "F"))
        else:
            band.setdefault(col, []).append(item)
    close_band()
    return ordered + [(item, "M") for item in minor]


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _column_bounds(ordered: list[tuple[Item, str]]) -> dict[str, tuple[float, float]]:
    """每一栏正文的左右边界，用来判断首行缩进与「上一行没写满」。"""
    columns: dict[str, list[Item]] = {}
    for item, col in ordered:
        if not item.is_table:
            columns.setdefault(col, []).append(item)
    return {
        col: (_quantile([i.x0 for i in group], 0.1), _quantile([i.x1 for i in group], 0.8))
        for col, group in columns.items()
    }


def _new_paragraph(
    prev: tuple[Item, str],
    current: tuple[Item, str],
    bounds: dict[str, tuple[float, float]],
    line_height: float,
) -> bool:
    prev_item, prev_col = prev
    item, col = current
    if col != prev_col:
        return True
    left, right = bounds.get(col, (item.x0, item.x1))
    if item.y0 - prev_item.y1 > line_height:  # 段间空白
        return True
    if item.y1 < prev_item.y0:  # 回到了上方，说明换了栏
        return True
    if item.x0 - left > 1.2 * line_height:  # 首行缩进
        return True
    if right - prev_item.x1 > 2 * line_height:  # 上一行没写满
        return True
    return False


def _page_segments(page: _Page, rules: DocRules = _NO_RULES) -> list[Segment]:
    ordered = _reading_order(page.items, page.width)
    heights = [item.unit_height for item, col in ordered if not item.is_table and col != "M"]
    line_height = statistics.median(heights) if heights else 12.0
    bounds = _column_bounds(ordered)
    allow_caps = page.method == "text"

    def level_of(text: str) -> int:
        return heading_level(text, allow_caps=allow_caps, rules=rules)

    segments: list[Segment] = []
    paragraph: list[Item] = []
    prev: tuple[Item, str] | None = None

    def flush() -> None:
        text = clean_text(merge_lines([item.text for item in paragraph]))
        paragraph.clear()
        if text:
            segments.append(Segment("text", page.number, page.method, text=text))

    index = 0
    while index < len(ordered):
        item, col = ordered[index]
        index += 1

        if item.is_table:
            flush()
            segments.append(
                Segment("table", page.number, page.method, text=item.text, rows=item.rows or [])
            )
            prev = None
            continue

        if len(item.text) <= 40 and _TOC_ENTRY_RE.search(item.text):
            continue  # 目录页条目，正文里都有，留着只会干扰标题识别

        level = level_of(item.text) if col != "M" else 0
        # 编号与标题字之间留了较宽的空白（「1」「前言」），视觉行没合并：拼起来再判断一次
        if not level and col != "M" and _BARE_NUMBER_RE.match(item.text) and index < len(ordered):
            nxt, nxt_col = ordered[index]
            joined = merge_lines([item.text, nxt.text])
            if not nxt.is_table and nxt_col == col and _on_same_line(item, nxt) and level_of(joined):
                item = Item(joined, item.x0, min(item.y0, nxt.y0), nxt.x1, max(item.y1, nxt.y1))
                level = level_of(joined)
                index += 1
        if level:
            flush()
            text = item.text
            # 标题后面同一行的文字（「问题1」+「什么是身体活动?」），以及写满一行后折下来的短续行
            # （「问题3…有哪些益」+「处?」），都接回标题
            last = item
            for _ in range(2):
                if index >= len(ordered):
                    break
                nxt, nxt_col = ordered[index]
                if nxt.is_table or nxt_col != col or level_of(nxt.text):
                    break
                if _on_same_line(last, nxt) and len(nxt.text) <= 40:
                    text = clean_text(f"{text} {nxt.text}")
                elif len(nxt.text) <= 20 and not _new_paragraph(
                    (last, col), (nxt, nxt_col), bounds, line_height
                ):
                    text = merge_lines([text, nxt.text])
                else:
                    break
                last = nxt
                index += 1
            segments.append(
                Segment("text", page.number, page.method, text=text, heading_level=level)
            )
            prev = None
            continue

        # 同一视觉行上的片段（两端对齐拉开的单词、无框线表格的一行）始终属于同一段
        same_line = prev is not None and prev[1] == col and _on_same_line(prev[0], item)
        if prev is not None and not same_line and _new_paragraph(prev, (item, col), bounds, line_height):
            flush()
        paragraph.append(item)
        prev = (item, col)

    flush()
    return segments


def _drop_references(segments: list[Segment]) -> list[Segment]:
    """参考文献列表对检索只有噪声：从「参考文献 / REFERENCES」起，丢到下一个标题或表格为止。"""
    kept: list[Segment] = []
    skipping = False
    for segment in segments:
        normalized = re.sub(r"\s+", "", segment.text).lower()
        if segment.kind == "text" and normalized in _REFERENCE_TITLES:
            skipping = True
            continue
        if skipping and (segment.kind == "table" or segment.heading_level):
            skipping = False
        if not skipping:
            kept.append(segment)
    return kept


def _is_title_like(text: str) -> bool:
    """一行短标题：不含句末标点、不超过一个短行。"""
    return "\n" not in text and len(text) <= _HEADING_MAX_LEN and not _SENTENCE_END_RE.search(text)


def _adopt_table_captions(segments: list[Segment]) -> list[Segment]:
    """表题补全。

    - 没有表题的表格：紧挨在前面的一行短标题（如「（2）跳远模块阶段性学业质量水平（第一学年）」）作为表题；
    - 「续表」：沿用上一张表的表题，标为「（续表）」。
    """
    out: list[Segment] = []
    last_caption = ""
    for segment in segments:
        if segment.kind == "table":
            if not segment.text and out:
                before = out[-1]
                if (
                    before.kind == "text"
                    and not before.heading_level
                    and before.page == segment.page
                    and _is_title_like(before.text)
                    and len(before.text) >= _CAPTION_MIN_LEN
                ):
                    segment.text = before.text
                    out.pop()
            if _CONTINUED_RE.match(segment.text) and last_caption:
                segment.text = f"{_CONTINUED_RE.sub('', last_caption).strip()}（续表）"
            elif segment.text:
                last_caption = segment.text
        out.append(segment)
    return out


def _drop_sections(segments: list[Segment], rules: DocRules) -> list[Segment]:
    """doc_rules.toml 的 drop_sections：标题命中的章节（直到同级或更高级的下一个标题）整体丢弃，
    表题命中的表格丢弃。"""
    if not rules.drop_sections:
        return segments
    kept: list[Segment] = []
    dropped: dict[int, bool] = {}  # 当前各级标题是否被丢弃
    for segment in segments:
        if segment.heading_level:
            dropped = {lvl: flag for lvl, flag in dropped.items() if lvl < segment.heading_level}
            dropped[segment.heading_level] = rules.drops_section(segment.text)
        if any(dropped.values()):
            continue
        if segment.kind == "table" and segment.text and rules.drops_section(segment.text):
            continue
        kept.append(segment)
    return kept


def _apply_line_rules(segments: list[Segment], rules: DocRules) -> list[Segment]:
    """doc_rules.toml 的 replacements（字形修正等）与 drop_lines，作用于正文、标题、表题和单元格。"""
    kept: list[Segment] = []
    for segment in segments:
        lines = [rules.replace(line) for line in segment.text.split("\n")]
        segment.text = "\n".join(line for line in lines if not (line and rules.drops_line(line))).strip()
        segment.rows = [[rules.replace(cell) for cell in row] for row in segment.rows]
        if segment.text or segment.kind == "table":
            kept.append(segment)
    return kept
