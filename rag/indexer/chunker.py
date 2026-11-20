"""切片：把解析出的段落、标题与表格组装成适合向量检索的文本块。

- 正文：按段落累积到 CHUNK_SIZE（折算字数）为止，相邻两块重叠上一块末尾的若干整句；
  超长段落按句子再切；遇到新标题或表格就另起一块，保证一块只属于一个章节。
- 表格：逐行展开成「表头：值；表头：值」，每块都带上表题（没有表题时用所在章节的标题），
  脱离上下文也读得懂。
- 每块记录所属章节路径（如「第一篇 体能素质篇 > 一、速度 > 练习3：…」）与起止页码。
- 不足 MIN_CHUNK 折算字数的小块：表注（「注：」「*」开头）接到上一块；章节开头的一小段
  并入紧跟的同章或下级章节的正文；孤立的一行短标题（「附件：…」「课程目标」）丢弃；
  其余带有完整句子的（如器材章节的「功能：发展速度、灵敏、协调能力。」）保留。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parser.models import ParsedDocument, Segment
from .parser.text_utils import text_units

_SENTENCE_RE = re.compile(r"(?<=[。！？；!?;])|(?<=\.\s)")
_SEPARATORS = ["\n", "。", "！", "？", "；", "!", "?", ";", ". ", "，", ", ", " ", ""]

MIN_CHUNK = 50
# 小块合并后允许超出 CHUNK_SIZE 的比例
_MERGE_SLACK = 1.2
_NOTE_RE = re.compile(r"^(?:注|备注)\s*\d*\s*[:：]|^\*")
_TABLE_NOTE_MAX = 200
_SENTENCE_END_RE = re.compile(r"[。！？；!?;]")


@dataclass
class Chunk:
    index: int
    kind: str  # text / table
    section: str
    page_start: int
    page_end: int
    parse_method: str  # text / ocr / mixed
    content: str
    headings: tuple[str, ...] = field(default=(), repr=False)  # 章节路径的各级标题，合并小块时用


class Chunker:
    def __init__(self, size: int, overlap: int):
        self.size = size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=text_units,
            separators=_SEPARATORS,
            keep_separator="end",
        )

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        chunks = _tidy_small_chunks(_Builder(self).run(doc.segments), self.size)
        return [replace(chunk, index=index) for index, chunk in enumerate(chunks)]

    def split_long(self, text: str) -> list[str]:
        return [piece.strip() for piece in self._splitter.split_text(text) if piece.strip()]

    def tail(self, page: int, method: str, text: str) -> Segment | None:
        """取文本末尾不超过 overlap 的若干整句，作为下一块开头的重叠内容。"""
        picked: list[str] = []
        total = 0
        for sentence in reversed([s for s in _SENTENCE_RE.split(text) if s.strip()]):
            units = text_units(sentence)
            if total + units > self.overlap:
                break
            picked.insert(0, sentence)
            total += units
        if not picked:
            return None
        return Segment("text", page, method, text="".join(picked).strip())

    def table_contents(self, segment: Segment, fallback_title: str = "") -> list[str]:
        if not segment.rows:
            return []
        header, *body = segment.rows
        title = segment.text or fallback_title or "表格"

        lines: list[str] = []
        for row in body:
            pairs = [
                f"{name}：{value}" if name and name != value else value
                for name, value in zip(header, row)
                if value
            ]
            if pairs:
                lines.append("；".join(pairs))
        if not lines:
            lines = ["；".join(cell for cell in header if cell)]

        contents: list[str] = []
        current: list[str] = []
        used = text_units(title)
        for line in lines:
            units = text_units(line)
            if units > self.size * 2:
                pieces = self.split_long(line)
            else:
                pieces = [line]
            for piece in pieces:
                piece_units = text_units(piece)
                if current and used + piece_units > self.size:
                    contents.append(title + "\n" + "\n".join(current))
                    current, used = [], text_units(title)
                current.append(piece)
                used += piece_units
        if current:
            contents.append(title + "\n" + "\n".join(current))
        return [content for content in contents if content.strip()]


class _Builder:
    def __init__(self, chunker: Chunker):
        self.chunker = chunker
        self.chunks: list[Chunk] = []
        self.headings: dict[int, str] = {}
        self.buffer: list[Segment] = []
        self.carry: Segment | None = None  # 上一块的结尾，接到下一块开头

    def run(self, segments: list[Segment]) -> list[Chunk]:
        size = self.chunker.size
        previous: Segment | None = None
        for segment in segments:
            after_table, previous = previous is not None and previous.kind == "table", segment
            if segment.kind == "table":
                self.flush(keep_tail=False)
                fallback = self.headings[max(self.headings)] if self.headings else ""
                for content in self.chunker.table_contents(segment, fallback):
                    self.emit("table", [segment], content)
            elif (
                after_table
                and not segment.heading_level
                and _NOTE_RE.match(segment.text)
                and text_units(segment.text) <= _TABLE_NOTE_MAX
                and self.chunks
                and self.chunks[-1].kind == "table"
            ):
                # 紧跟在表格后面的表注（「注：…」「*ECC = eccentric…」）归入该表的最后一块
                last = self.chunks[-1]
                last.content += "\n" + segment.text
                last.page_end = max(last.page_end, segment.page)
            elif segment.heading_level:
                self.flush(keep_tail=False)
                self.headings = {
                    level: text for level, text in self.headings.items() if level < segment.heading_level
                }
                self.headings[segment.heading_level] = segment.text
            elif text_units(segment.text) > size:
                self.flush(keep_tail=False)
                pieces = self.chunker.split_long(segment.text)
                for piece in pieces:
                    self.emit("text", [segment], piece)
                if pieces:
                    self.carry = self.chunker.tail(segment.page, segment.method, pieces[-1])
            else:
                if self.buffer and self.units() + text_units(segment.text) > size:
                    self.flush(keep_tail=True)
                self.add(segment)
        self.flush(keep_tail=False)
        return self.chunks

    def units(self) -> int:
        return sum(text_units(segment.text) for segment in self.buffer)

    def add(self, segment: Segment) -> None:
        if not self.buffer and self.carry is not None:
            self.buffer.append(self.carry)
        self.carry = None
        self.buffer.append(segment)

    def flush(self, *, keep_tail: bool) -> None:
        if self.buffer:
            self.emit("text", self.buffer, "\n".join(segment.text for segment in self.buffer))
            last = self.buffer[-1]
            self.carry = self.chunker.tail(last.page, last.method, last.text) if keep_tail else None
            self.buffer = []
        elif not keep_tail:
            self.carry = None

    def emit(self, kind: str, parts: list[Segment], content: str) -> None:
        content = content.strip()
        if not content:
            return
        methods = {part.method for part in parts}
        headings = tuple(self.headings[level] for level in sorted(self.headings))
        self.chunks.append(
            Chunk(
                index=len(self.chunks),
                kind=kind,
                section=" > ".join(headings),
                page_start=min(part.page for part in parts),
                page_end=max(part.page for part in parts),
                parse_method=methods.pop() if len(methods) == 1 else "mixed",
                content=content,
                headings=headings,
            )
        )


# ---------------------------------------------------------------------------
# 小块整理
# ---------------------------------------------------------------------------


def _is_prefix(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    return len(short) <= len(long) and long[: len(short)] == short


def _title_like(content: str) -> bool:
    """一两行短标题：没有句末标点，每行都很短。"""
    lines = content.split("\n")
    return len(lines) <= 2 and all(len(line) <= 40 for line in lines) and not _SENTENCE_END_RE.search(content)


def _merge(first: Chunk, second: Chunk, headings: tuple[str, ...]) -> Chunk:
    """把相邻两块合成一块，章节取两者共同的上级；较深一块多出的标题写进正文，保留层次。"""

    def body(chunk: Chunk) -> str:
        extra = list(chunk.headings[len(headings) :])
        return "\n".join([*extra, chunk.content])

    methods = {first.parse_method, second.parse_method}
    return Chunk(
        index=first.index,
        kind="table" if "table" in (first.kind, second.kind) else "text",
        section=" > ".join(headings),
        page_start=min(first.page_start, second.page_start),
        page_end=max(first.page_end, second.page_end),
        parse_method=methods.pop() if len(methods) == 1 else "mixed",
        content=f"{body(first)}\n{body(second)}",
        headings=headings,
    )


def _tidy_small_chunks(chunks: list[Chunk], size: int) -> list[Chunk]:
    """不足 MIN_CHUNK 的小块：表注接上一块，章节开头的一小段并入下一块，孤立的短标题丢弃。"""
    limit = int(size * _MERGE_SLACK)
    pending = list(chunks)
    result: list[Chunk] = []
    k = 0
    while k < len(pending):
        chunk = pending[k]
        k += 1
        units = text_units(chunk.content)
        if units >= MIN_CHUNK or chunk.kind == "table":
            result.append(chunk)
            continue
        prev = result[-1] if result else None
        nxt = pending[k] if k < len(pending) else None

        # 表注、注释（「注：体重指数…」「*ECC = eccentric…」）：接到同一章节的上一块（含表格）
        if (
            prev is not None
            and _NOTE_RE.match(chunk.content)
            and prev.headings == chunk.headings
            and text_units(prev.content) + units <= limit
        ):
            result[-1] = _merge(prev, chunk, prev.headings)
            continue
        # 章节开头的一小段，后面紧跟同章或下级章节的正文：并入下一块（如「二、主要内容」的引言 +「（一）…」）
        if (
            nxt is not None
            and nxt.kind == "text"
            and chunk.headings
            and _is_prefix(chunk.headings, nxt.headings)
            and text_units(nxt.content) + units <= limit
        ):
            pending[k] = _merge(chunk, nxt, chunk.headings)
            continue
        # 上一块是同章或上级章节的正文：接到上一块
        if (
            prev is not None
            and prev.kind == "text"
            and prev.headings
            and _is_prefix(prev.headings, chunk.headings)
            and text_units(prev.content) + units <= limit
        ):
            result[-1] = _merge(prev, chunk, prev.headings)
            continue
        # 孤立的一行短标题（「附件：国家学生体质健康测试标准」「课程目标」「组合一」）：没有可检索的内容
        if _title_like(chunk.content):
            continue
        result.append(chunk)
    return result
