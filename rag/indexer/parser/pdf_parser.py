"""PDF 解析。

- 有文字层的页：按单词取字，再按「视觉行」聚合（上标引用、错位的标点都能回到原位）；
  用 PyMuPDF 的表格识别定位表格，单元格内容同样按视觉行取字；
- 扫描页（几乎没有文字、主要是图片）：本地 OCR 识别；
- 去掉页眉、页脚和页码，小字号的脚注排到本页正文之后，按阅读顺序（支持期刊双栏排版）
  把折行拼回段落，识别章节标题，丢弃参考文献列表。
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
from .models import Item, ParsedDocument, Segment
from .ocr import OcrEngine
from .text_utils import clean_text, is_page_number, merge_lines

logger = get_logger()

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
_CAPTION_MAX_LEN = 80

_HAS_CONTENT_RE = re.compile(r"[0-9A-Za-z一-鿿]")
# 目录条目：短行，以「／页码」或「……」「……页码」「……(页码)」结尾，如「（二）基本理念／2」「八、来回奔跑测试……(202)」
_TOC_ENTRY_RE = re.compile(r"(?:[／/]\s*\d{1,3}|(?:…+|\.{3,}|·{3,})\s*[(（]?\s*\d{0,3}\s*[)）]?)$")
_REFERENCE_TITLES = {"references", "reference", "参考文献"}

_HEADING_MAX_LEN = 40
_HEADING_PATTERNS = [
    (1, re.compile(r"^第[一二三四五六七八九十百零\d]+[篇章部分]")),
    (2, re.compile(r"^[一二三四五六七八九十]+、")),
    (3, re.compile(r"^[（(][一二三四五六七八九十]+[）)]")),
    (3, re.compile(r"^问题\s*\d+")),
    (4, re.compile(r"^练习\s*\d+\s*[：:]")),
]


def heading_level(text: str, *, allow_caps: bool = True) -> int:
    """章节标题的层级（1 最高）；不是标题返回 0。"""
    if len(text) > _HEADING_MAX_LEN:
        return 0
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

    def parse(self, path: Path, base_dir: Path) -> ParsedDocument:
        data = path.read_bytes()
        doc_md5 = hashlib.md5(data).hexdigest()
        pages: list[_Page] = []
        skipped_scans = 0

        with pymupdf.open(stream=data, filetype="pdf") as doc:
            page_count = doc.page_count
            for page in doc:
                number = page.number + 1
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

        _strip_page_furniture(pages)
        segments = [segment for page in pages for segment in _page_segments(page)]
        return ParsedDocument(
            doc_name=path.name,
            doc_path_name=path.resolve().relative_to(base_dir.resolve()).as_posix(),
            doc_type=path.suffix.lstrip(".").lower(),
            doc_md5=doc_md5,
            doc_length=len(data),
            page_count=page_count,
            ocr_pages=sum(1 for page in pages if page.method == "ocr"),
            segments=_drop_references(segments),
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
    tables = _extract_tables(page, words)

    fragments = [
        fragment
        for fragment in _line_fragments(words)
        if not any(_center_inside(fragment, table.rect) for table in tables)
    ]
    rows = [row for row in _visual_rows(fragments) if _HAS_CONTENT_RE.search(row.text)]
    _mark_minor(rows)

    for table in tables:
        caption = _take_caption(rows, table.rect)
        rows.append(Item(caption, *table.rect, rows=table.rows))
    return rows


def _mark_minor(rows: list[Item]) -> None:
    sized = [row.line_height for row in rows if len(row.text) >= 5]
    if len(sized) < 5:
        return
    body = statistics.median(sized)
    for row in rows:
        if row.line_height < _MINOR_RATIO * body:
            row.minor = True


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
        tables.append(_Table(pymupdf.Rect(table.bbox), rows))
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


def _take_caption(items: list[Item], table_rect: pymupdf.Rect) -> str:
    """从表格正上方的短行里取出表题（如「表1-1 男生体重指数（BMI）单项评分表」），并从正文中移除。"""
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
    caption = merge_lines([item.text for item in sorted(near, key=lambda i: (round(i.y0), i.x0))])
    if not re.search(r"表|^table", caption, re.IGNORECASE):
        return ""
    items[:] = [item for item in items if all(item is not taken for taken in near)]
    return caption


# ---------------------------------------------------------------------------
# 页眉页脚与页码
# ---------------------------------------------------------------------------


def _margin_key(item: Item, page: _Page) -> str | None:
    if item.is_table:
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


def _page_segments(page: _Page) -> list[Segment]:
    ordered = _reading_order(page.items, page.width)
    heights = [item.unit_height for item, col in ordered if not item.is_table and col != "M"]
    line_height = statistics.median(heights) if heights else 12.0
    bounds = _column_bounds(ordered)
    allow_caps = page.method == "text"

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

        level = heading_level(item.text, allow_caps=allow_caps) if col != "M" else 0
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
                if nxt.is_table or nxt_col != col or heading_level(nxt.text, allow_caps=allow_caps):
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
