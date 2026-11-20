"""知识库离线入库：扫描 rag/knowledge 下的 PDF → 解析（文字层 / OCR / 表格）→ 切片 → 向量化 → 写入 Milvus。

用法（在 rag/ 目录下执行，或用 rag/scripts/run_index.sh 包装）：
    uv run python -m indexer.main                        # backfill（默认）：只处理新增或内容有变化的文件
    uv run python -m indexer.main --mode full-refresh    # 全部文件重新解析、重新入库
    uv run python -m indexer.main --mode rebuild         # 清空集合后全量入库（换嵌入模型/维度时用）
    uv run python -m indexer.main --mode dry-run         # 只解析和切片，不调嵌入、不写库，预览写到 cache/preview/
    uv run python -m indexer.main --mode stats           # 查看库中每份文档的切片数
    uv run python -m indexer.main --mode query --text "青少年每天应该运动多久"   # 入库后自检检索效果
    uv run python -m indexer.main --file 中国儿童青少年身体活动指南-.pdf       # 只处理指定文件

文档在库中的唯一标识是「相对 knowledge 目录的路径」。backfill / full-refresh 会顺带删除
已从 knowledge 目录移走的文档的向量（指定 --file 时不做这一步）。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path

from .chunker import Chunk, Chunker
from .config import Settings, settings
from .logger import get_logger
from .parser.models import ParsedDocument
from .parser.pdf_parser import PDFParser
from .parser.text_utils import text_units

logger = get_logger()

_SUPPORTED_SUFFIXES = {".pdf"}
_IGNORED_NAMES = {".DS_Store", "Thumbs.db", ".gitkeep"}
_INSERT_BATCH = 64


def document_title(path: Path) -> str:
    """文件名去掉扩展名、下载重复后缀「 (1)」和首尾连接符，作为文档标题。"""
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    return stem.strip(" -_—") or path.stem


def embedding_text(title: str, chunk: Chunk) -> str:
    """向量化的文本带上文档标题与章节路径，让切片脱离上下文也能被准确召回。"""
    head = f"《{title}》"
    if chunk.section:
        head += " " + chunk.section
    return f"{head}\n{chunk.content}"


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(text: str, max_bytes: int) -> str:
    """Milvus 的 VARCHAR 长度按 UTF-8 字节计，超长时截断。"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class KnowledgeIndexer:
    def __init__(self, config: Settings):
        self.settings = config
        self.base_dir = config.knowledge_dir.resolve()
        if not self.base_dir.is_dir():
            raise FileNotFoundError(f"知识库目录不存在：{self.base_dir}")
        self.parser = PDFParser(config)
        self.chunker = Chunker(config.chunk_size, config.chunk_overlap)
        self._store = None
        self._embedder = None

    # dry-run 不连 Milvus、不调嵌入接口，所以两者按需创建
    @property
    def store(self):
        if self._store is None:
            from .milvus_api import MilvusAPI

            self._store = MilvusAPI(self.settings)
        return self._store

    @property
    def embedder(self):
        if self._embedder is None:
            from .embedding import EmbeddingTool

            self._embedder = EmbeddingTool(self.settings)
        return self._embedder

    # ------------------------------------------------------------------
    # 文件
    # ------------------------------------------------------------------

    def list_files(self) -> list[Path]:
        files: list[Path] = []
        for path in sorted(self.base_dir.rglob("*")):
            if not path.is_file() or path.name in _IGNORED_NAMES:
                continue
            parts = path.relative_to(self.base_dir).parts
            if any(part.startswith((".", "~$")) for part in parts):
                continue
            if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                logger.warning("暂只支持 PDF，跳过：%s", self.rel_path(path))
                continue
            files.append(path)
        return files

    def select_files(self, names: list[str]) -> list[Path]:
        if not names:
            return self.list_files()
        selected: list[Path] = []
        for name in names:
            candidates = [Path(name).expanduser()]
            if not candidates[0].is_absolute():
                candidates = [self.base_dir / name, Path.cwd() / name]
            path = next((c.resolve() for c in candidates if c.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"找不到文件：{name}")
            if not path.is_relative_to(self.base_dir):
                raise ValueError(f"文件必须放在知识库目录 {self.base_dir} 下：{path}")
            selected.append(path)
        return selected

    def rel_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.base_dir).as_posix()

    def parse_and_chunk(self, path: Path) -> tuple[ParsedDocument, list[Chunk]]:
        parsed = self.parser.parse(path, self.base_dir)
        return parsed, self.chunker.split(parsed)

    # ------------------------------------------------------------------
    # 入库
    # ------------------------------------------------------------------

    def index_file(self, path: Path) -> int:
        started = time.time()
        parsed, chunks = self.parse_and_chunk(path)
        if not chunks:
            logger.warning("  未提取到任何文本，跳过")
            return 0

        title = document_title(path)
        logger.info("  解析完成：%d 页（OCR %d 页）→ %d 个切片，开始向量化...", parsed.page_count, parsed.ocr_pages, len(chunks))
        vectors = self.embedder.embed([embedding_text(title, chunk) for chunk in chunks])
        rows = [
            {
                "doc_name": _fit(parsed.doc_name, 500),
                "doc_title": _fit(title, 500),
                "doc_path_name": parsed.doc_path_name,
                "doc_type": parsed.doc_type,
                "doc_md5": parsed.doc_md5,
                "doc_length": parsed.doc_length,
                "chunk_index": chunk.index,
                "chunk_type": chunk.kind,
                "section": _fit(chunk.section, 1000),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "parse_method": chunk.parse_method,
                "content": _fit(chunk.content, 65535),
                "content_vector": vector,
                "embedding_model": self.settings.embedding_model,
            }
            for chunk, vector in zip(chunks, vectors)
        ]

        # 向量全部算好后再替换旧数据；写入中途失败则删掉半份，避免 backfill 误判为已入库
        self.store.delete_document(parsed.doc_path_name)
        try:
            for start in range(0, len(rows), _INSERT_BATCH):
                self.store.insert_chunks(rows[start : start + _INSERT_BATCH])
        except Exception:
            self.store.delete_document(parsed.doc_path_name)
            raise

        tables = sum(1 for chunk in chunks if chunk.kind == "table")
        logger.info("  已写入 %d 个切片（其中表格 %d 个），耗时 %.1fs", len(rows), tables, time.time() - started)
        return len(rows)

    def sync(self, files: list[Path], *, force: bool, prune: bool) -> int:
        existing = self.store.indexed_documents()
        done = skipped = failed = 0
        for number, path in enumerate(files, 1):
            rel = self.rel_path(path)
            logger.info("[%d/%d] %s", number, len(files), rel)
            if not force and existing.get(rel) == file_md5(path):
                logger.info("  已入库且内容未变，跳过")
                skipped += 1
                continue
            try:
                self.index_file(path)
                done += 1
            except Exception as exc:  # noqa: BLE001 —— 单个文件失败不影响其余文件
                logger.error("  入库失败：%s", exc, exc_info=True)
                failed += 1

        removed = 0
        if prune:
            current = {self.rel_path(path) for path in files}
            for rel in sorted(set(existing) - current):
                self.store.delete_document(rel)
                removed += 1
                logger.info("knowledge 目录中已没有 %s，已删除它的向量", rel)

        self.store.flush()
        logger.info(
            "=== 完成：入库 %d，跳过 %d，失败 %d，清理 %d；库中现有 %d 个切片 ===",
            done, skipped, failed, removed, self.store.count_chunks(),
        )
        return 1 if failed else 0

    # ------------------------------------------------------------------
    # 预览、统计、自检
    # ------------------------------------------------------------------

    def dry_run(self, files: list[Path]) -> int:
        preview_dir = self.settings.cache_dir / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for number, path in enumerate(files, 1):
            logger.info("[%d/%d] %s", number, len(files), self.rel_path(path))
            started = time.time()
            parsed, chunks = self.parse_and_chunk(path)
            title = document_title(path)
            output = preview_dir / f"{title}.md"
            output.write_text(_render_preview(title, parsed, chunks), encoding="utf-8")
            sizes = [text_units(chunk.content) for chunk in chunks] or [0]
            logger.info(
                "  %d 页（OCR %d 页）→ %d 个切片（表格 %d），折算字数 平均 %d / 最大 %d，耗时 %.1fs → %s",
                parsed.page_count, parsed.ocr_pages, len(chunks),
                sum(1 for chunk in chunks if chunk.kind == "table"),
                sum(sizes) // len(sizes), max(sizes), time.time() - started, output,
            )
        return 0

    def stats(self) -> int:
        existing = self.store.indexed_documents()
        if not existing:
            logger.info("集合 %s 中还没有数据", self.store.collection)
            return 0
        for rel in sorted(existing):
            logger.info("%s：%d 个切片", rel, self.store.count_chunks(rel))
        logger.info("共 %d 份文档、%d 个切片", len(existing), self.store.count_chunks())
        return 0

    def query(self, text: str, top_k: int) -> int:
        vector = self.embedder.embed([text])[0]
        hits = self.store.search(vector, top_k)
        print(f"\n查询：{text}\n")
        for rank, hit in enumerate(hits, 1):
            entity = hit["entity"]
            pages = f"第 {entity['page_start']} 页"
            if entity["page_end"] != entity["page_start"]:
                pages = f"第 {entity['page_start']}-{entity['page_end']} 页"
            print(f"#{rank}  相似度 {hit['distance']:.3f}  《{entity['doc_title']}》{pages}  [{entity['chunk_type']}]")
            if entity["section"]:
                print(f"    章节：{entity['section']}")
            content = entity["content"].replace("\n", " ")
            print(f"    {content[:200]}{'…' if len(content) > 200 else ''}\n")
        return 0


def _render_preview(title: str, parsed: ParsedDocument, chunks: list[Chunk]) -> str:
    lines = [
        f"# {title}",
        "",
        f"- 文件：{parsed.doc_path_name}",
        f"- 页数：{parsed.page_count}（OCR {parsed.ocr_pages} 页）",
        f"- 切片：{len(chunks)}（表格 {sum(1 for c in chunks if c.kind == 'table')}）",
        "",
    ]
    for chunk in chunks:
        pages = f"p{chunk.page_start}" + (f"-{chunk.page_end}" if chunk.page_end != chunk.page_start else "")
        lines.append(
            f"## #{chunk.index} · {chunk.kind} · {pages} · {chunk.parse_method} · {text_units(chunk.content)} 字"
        )
        if chunk.section:
            lines.append(f"> {chunk.section}")
        lines += ["", chunk.content, ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m indexer.main",
        description="运动处方知识库离线入库：PDF → 解析/OCR → 切片 → 向量化 → Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "full-refresh", "rebuild", "dry-run", "stats", "query"],
        default="backfill",
        help="运行模式，默认 backfill",
    )
    parser.add_argument("--file", action="append", default=[], help="只处理指定文件，可重复指定")
    parser.add_argument("--text", help="query 模式下的查询语句")
    parser.add_argument("--top-k", type=int, default=5, help="query 模式返回的条数，默认 5")
    args = parser.parse_args(argv)

    if args.mode == "query" and not args.text:
        parser.error("query 模式需要 --text")
    if args.mode == "rebuild" and args.file:
        parser.error("rebuild 会清空整个集合，不能与 --file 同时使用")

    try:
        indexer = KnowledgeIndexer(settings)
        if args.mode == "query":
            return indexer.query(args.text, args.top_k)
        if args.mode == "stats":
            return indexer.stats()

        files = indexer.select_files(args.file)
        if args.mode == "dry-run":
            return indexer.dry_run(files)
        if args.mode == "rebuild":
            indexer.store.recreate_collection()
        return indexer.sync(files, force=args.mode != "backfill", prune=not args.file)
    except KeyboardInterrupt:
        logger.warning("已中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("运行失败：%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
