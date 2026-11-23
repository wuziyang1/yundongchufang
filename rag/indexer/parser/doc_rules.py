"""按文档定制的规则，配置在 rag/doc_rules.toml。

通用规则（前置页、论文标题区与作者脚注、OCR 噪点、小切片等）写在解析器和切片器里，对所有文档生效；
个别文档特有的版式问题（某本书的章节编号写法、某篇期刊的字形映射错误、不该入库的附录）
在这里按文件名配置。规则在解析阶段生效，重新入库时脏数据不会再回来。

这里还登记每份文档的「身份」：文档类别、适用学段、是否按动作条目入库。它们随切片写入 Milvus，
在线检索据此按学段过滤、区分评分表与动作条目，后端不需要认识具体的文件名。
"""

from __future__ import annotations

import fnmatch
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..logger import get_logger

logger = get_logger()

# 适用学段的固定取值，与需求文档「按年龄段分别构建知识库」的四档一致
STAGES = ("小学低年级", "小学高年级", "初中", "高中")


@dataclass
class DocRules:
    skip_pages: set[int] = field(default_factory=set)
    headings: list[tuple[re.Pattern, int]] = field(default_factory=list)
    drop_sections: list[re.Pattern] = field(default_factory=list)
    drop_lines: list[re.Pattern] = field(default_factory=list)
    replacements: list[tuple[re.Pattern, str]] = field(default_factory=list)
    # 文档身份：类别（标准 / 指南 / 课程标准 / 训练手册 / 立场声明…）、适用学段、动作编号前缀
    kind: str = ""
    stages: list[str] = field(default_factory=lambda: list(STAGES))
    exercise_prefix: str = ""
    # 表题命中这些正则的表格是「查表数据」（评分表），入库时类型记为 lookup，不参与相似度检索
    lookup_tables: list[re.Pattern] = field(default_factory=list)

    def is_lookup_table(self, title: str) -> bool:
        return any(p.search(title) for p in self.lookup_tables)

    def heading_level(self, text: str) -> int:
        for pattern, level in self.headings:
            if pattern.search(text):
                return level
        return 0

    def drops_section(self, text: str) -> bool:
        compact = re.sub(r"\s+", "", text)
        return any(p.search(text) or p.search(compact) for p in self.drop_sections)

    def drops_line(self, text: str) -> bool:
        return any(p.search(text) for p in self.drop_lines)

    def replace(self, text: str) -> str:
        for pattern, repl in self.replacements:
            text = pattern.sub(repl, text)
        return text


def _pages(values: list) -> set[int]:
    pages: set[int] = set()
    for value in values:
        if isinstance(value, int):
            pages.add(value)
            continue
        start, _, end = str(value).partition("-")
        pages.update(range(int(start), int(end or start) + 1))
    return pages


def _compile(pattern: str, where: str) -> re.Pattern:
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"doc_rules.toml 中 {where} 的正则无效：{pattern!r}（{exc}）") from exc


class DocRuleBook:
    """读取 doc_rules.toml；一个文件可匹配多条 [[doc]]，规则依次合并。"""

    def __init__(self, path: Path):
        self._entries: list[dict] = []
        if path.is_file():
            with path.open("rb") as handle:
                self._entries = tomllib.load(handle).get("doc", [])
            for entry in self._entries:
                if "match" not in entry:
                    raise ValueError(f"{path} 中有一条 [[doc]] 缺少 match")
        else:
            logger.warning("未找到文档清洗规则 %s，只使用通用规则", path)

    def for_document(self, rel_path: str) -> DocRules:
        rules = DocRules()
        name = Path(rel_path).name
        for entry in self._entries:
            pattern = entry["match"]
            if not (fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(name, pattern)):
                continue
            where = f"match={pattern!r}"
            rules.skip_pages |= _pages(entry.get("skip_pages", []))
            rules.headings += [(_compile(p, where), int(level)) for p, level in entry.get("headings", [])]
            rules.drop_sections += [_compile(p, where) for p in entry.get("drop_sections", [])]
            rules.drop_lines += [_compile(p, where) for p in entry.get("drop_lines", [])]
            rules.replacements += [(_compile(p, where), repl) for p, repl in entry.get("replacements", [])]
            if "kind" in entry:
                rules.kind = str(entry["kind"]).strip()
            if "stages" in entry:
                unknown = [stage for stage in entry["stages"] if stage not in STAGES]
                if unknown or not entry["stages"]:
                    raise ValueError(
                        f"doc_rules.toml 中 {where} 的 stages 只能取 {'、'.join(STAGES)}，且不能为空：{entry['stages']!r}"
                    )
                rules.stages = [stage for stage in STAGES if stage in entry["stages"]]
            rules.lookup_tables += [_compile(p, where) for p in entry.get("lookup_tables", [])]
            if "exercise_prefix" in entry:
                prefix = str(entry["exercise_prefix"]).strip()
                if not re.fullmatch(r"[A-Z]{1,4}", prefix):
                    raise ValueError(f"doc_rules.toml 中 {where} 的 exercise_prefix 须为 1 至 4 个大写字母：{prefix!r}")
                rules.exercise_prefix = prefix
        return rules
