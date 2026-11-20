"""文本整理的小工具：行合并、页码识别、长度折算。"""

from __future__ import annotations

import re

# 页码：纯数字，或带装饰的数字，如「·3·」「-15-」「— 15 —」
_PAGE_NUMBER_RE = re.compile(r"^[\s\-—–·•.]*\d{1,4}[\s\-—–·•.]*$")

# 全角数字与字母转半角，「１．５」「ＲPE」这类检索词才能精确命中；中文标点保持全角
_FULLWIDTH_ALNUM = {
    code: code - 0xFEE0
    for code in [*range(0xFF10, 0xFF1A), *range(0xFF21, 0xFF3B), *range(0xFF41, 0xFF5B)]
}
# 全角句点用作小数点：「1．5」「1 ．5」→「1.5」
_DECIMAL_DOT_RE = re.compile(r"(?<=\d)\s*．\s*(?=\d)")
# PDF 抽取常在汉字之间留下空格，如「等 级」「了 解」
_HAN_GAP_RE = re.compile(r"(?<=[一-鿿])[ \t　]+(?=[一-鿿])")
_SPACES_RE = re.compile(r"[ \t　]+")
# 西文连字（ﬁ、ﬂ 等）拆回普通字母，检索词「fitness」才能命中；控制字符多是缺字形的符号，去掉
_LIGATURES = {0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl"}
_CONTROL_CHARS = {code: None for code in [*range(0x00, 0x09), *range(0x0B, 0x20), 0x7F]}


def clean_text(text: str) -> str:
    text = text.translate(_LIGATURES).translate(_CONTROL_CHARS)
    text = text.translate(_FULLWIDTH_ALNUM)
    text = _DECIMAL_DOT_RE.sub(".", text)
    text = _HAN_GAP_RE.sub("", text)
    return _SPACES_RE.sub(" ", text).strip()


def is_han(ch: str) -> bool:
    return 0x4E00 <= ord(ch) <= 0x9FFF


def is_cjk(ch: str) -> bool:
    """汉字，以及中文标点和全角符号。"""
    code = ord(ch)
    return (
        is_han(ch)
        or 0x3000 <= code <= 0x303F  # 中文标点
        or 0xFF00 <= code <= 0xFFEF  # 全角字符
        or ch in "“”‘’—…·"
    )


def is_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER_RE.match(text))


def merge_lines(lines: list[str]) -> str:
    """把 PDF 里因版面宽度折断的多行接回一段。

    中文行直接相连；西文行之间补一个空格；行尾是连字符时不补空格，
    这样 "strength-" + "training" 还原为 "strength-training"。
    """
    merged = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not merged:
            merged = line
        elif is_cjk(merged[-1]) or is_cjk(line[0]) or merged[-1] == "-":
            merged += line
        else:
            merged += " " + line
    return merged


def text_units(text: str) -> int:
    """折算字数，近似 token 数：汉字 1 个算 1，其余字符约 3 个算 1。

    用它衡量切片长度，中文和英文切片的信息量才大致相当。
    """
    han = sum(1 for ch in text if is_han(ch))
    return han + (len(text) - han + 2) // 3
