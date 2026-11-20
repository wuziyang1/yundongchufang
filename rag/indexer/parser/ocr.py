"""扫描页 OCR：本地 RapidOCR（PP-OCR 模型 + onnxruntime 推理），不联网、不需要密钥。

识别结果按「文档 MD5 + 页码 + DPI」缓存在 rag/cache/ocr/，重复入库时不必再识别。
"""

from __future__ import annotations

import json
import logging
import statistics

import numpy as np
import pymupdf

from ..config import Settings
from .models import Item
from .text_utils import clean_text

# 置信度低于此值的识别结果多为插图里的噪点，丢弃
_MIN_SCORE = 0.5


class OcrEngine:
    def __init__(self, settings: Settings):
        self._dpi = settings.ocr_dpi
        self._cache_dir = settings.cache_dir / "ocr"
        self._engine = None

    def recognize(self, page: pymupdf.Page, doc_md5: str) -> list[Item]:
        cache_file = self._cache_dir / doc_md5 / f"page{page.number + 1:04d}_{self._dpi}dpi.json"
        if cache_file.is_file():
            lines = json.loads(cache_file.read_text(encoding="utf-8"))
        else:
            lines = self._run(page)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
        return _lines_to_items(lines, scale=72 / self._dpi)

    def _run(self, page: pymupdf.Page) -> list[dict]:
        pix = page.get_pixmap(dpi=self._dpi, colorspace=pymupdf.csRGB, alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        result = self._get_engine()(image)
        if result.boxes is None or result.txts is None:
            return []

        lines = []
        for box, text, score in zip(result.boxes, result.txts, result.scores):
            text = str(text).strip()
            if not text or float(score) < _MIN_SCORE:
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            lines.append(
                {
                    "text": text,
                    "box": [min(xs), min(ys), max(xs), max(ys)],
                    "score": round(float(score), 3),
                }
            )
        return lines

    def _get_engine(self):
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR()
            logging.getLogger("RapidOCR").setLevel(logging.WARNING)
        return self._engine


def _center(line: dict) -> float:
    return (line["box"][1] + line["box"][3]) / 2


def _height(line: dict) -> float:
    return line["box"][3] - line["box"][1]


def _lines_to_items(lines: list[dict], scale: float) -> list[Item]:
    """把同一视觉行上的识别框（如表格的各个单元格）按从左到右拼成一行，坐标换算为 pt。"""
    rows: list[list[dict]] = []
    for line in sorted(lines, key=_center):
        if rows:
            last = rows[-1]
            last_center = statistics.mean(_center(item) for item in last)
            last_height = min(_height(item) for item in last)
            if abs(_center(line) - last_center) < 0.5 * min(_height(line), last_height):
                last.append(line)
                continue
        rows.append([line])

    items: list[Item] = []
    for row in rows:
        row.sort(key=lambda item: item["box"][0])
        text = clean_text(" ".join(item["text"] for item in row))
        if not text:
            continue
        items.append(
            Item(
                text=text,
                x0=min(item["box"][0] for item in row) * scale,
                y0=min(item["box"][1] for item in row) * scale,
                x1=max(item["box"][2] for item in row) * scale,
                y1=max(item["box"][3] for item in row) * scale,
                line_height=statistics.median(_height(item) for item in row) * scale,
            )
        )
    return items
