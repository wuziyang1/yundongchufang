"""切片：把解析出的段落、标题与表格组装成适合向量检索的文本块。

- 正文：按段落累积到 CHUNK_SIZE（折算字数）为止，相邻两块重叠上一块末尾的若干整句；
  超长段落按句子再切；遇到新标题或表格就另起一块，保证一块只属于一个章节。
- 表格：逐行展开成「表头：值；表头：值」，每块都带上表题，脱离上下文也读得懂。
- 每块记录所属章节路径（如「第一篇 体能素质篇 > 一、速度 > 练习3：…」）与起止页码。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parser.models import ParsedDocument, Segment
from .parser.text_utils import text_units

_SENTENCE_RE = re.compile(r"(?<=[。！？；!?;])|(?<=\.\s)")
_SEPARATORS = ["\n", "。", "！", "？", "；", "!", "?", ";", ". ", "，", ", ", " ", ""]


@dataclass
class Chunk:
    index: int
    kind: str  # text / table
    section: str
    page_start: int
    page_end: int
    parse_method: str  # text / ocr / mixed
    content: str


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
        return _Builder(self).run(doc.segments)

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

    def table_contents(self, segment: Segment) -> list[str]:
        if not segment.rows:
            return []
        header, *body = segment.rows
        title = segment.text or "表格"

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
        for segment in segments:
            if segment.kind == "table":
                self.flush(keep_tail=False)
                for content in self.chunker.table_contents(segment):
                    self.emit("table", [segment], content)
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
        self.chunks.append(
            Chunk(
                index=len(self.chunks),
                kind=kind,
                section=" > ".join(self.headings[level] for level in sorted(self.headings)),
                page_start=min(part.page for part in parts),
                page_end=max(part.page for part in parts),
                parse_method=methods.pop() if len(methods) == 1 else "mixed",
                content=content,
            )
        )
