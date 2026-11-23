"""数据资产加载：学生档案、急症词表，以及生成结果的留档。

启动时一次性读入内存，v1 不引入数据库。

学生档案只有一个来源：data/mock_students.json（前端录入的新档案也追加进来）。
生成留档只有两个目录：画像进 data/profiles，处方进 data/prescriptions，
同一学号重复生成直接覆盖，均以 <student_id>.json 命名。

处方用到的专业知识（评分表、权威资料、动作）不在这里：它们全部来自 Milvus 知识库，
见 app/services/kb。data/exercises.json 与 data/knowledge.json 是早期由 AI 整理的动作库与
知识卡片，文件保留，但已不参与检索与处方生成。
"""

from __future__ import annotations

import json
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import DATA_DIR


def _load(filename: str) -> dict[str, Any]:
    with (DATA_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


STUDENTS_FILE = DATA_DIR / "mock_students.json"


@lru_cache(maxsize=1)
def mock_students() -> list[dict[str, Any]]:
    return _load("mock_students.json")["students"]


@lru_cache(maxsize=1)
def emergency_rules() -> dict[str, Any]:
    return _load("emergency_keywords.json")


def all_students() -> list[dict[str, Any]]:
    return mock_students()


def save_student(record: dict[str, Any]) -> dict[str, Any]:
    """把前端录入的档案追加进 mock_students.json，并返回带学号的完整档案。

    学号始终由服务端生成（CU 前缀 + 时间戳 + 随机串），不采用用户输入，
    且循环避开已存在的学号，避免覆盖已有档案。
    """
    with STUDENTS_FILE.open(encoding="utf-8") as handle:
        data = json.load(handle)

    existing_ids = {item["student_id"] for item in data["students"]}
    while True:
        student_id = f"CU{time.strftime('%Y%m%d%H%M%S')}{secrets.token_hex(2).upper()}"
        if student_id not in existing_ids:
            break
    record["student_id"] = student_id
    data["students"].append(record)
    data["updated_at"] = _now_iso()[:10]

    with STUDENTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    mock_students.cache_clear()
    return record


def find_student(student_id: str) -> dict[str, Any] | None:
    for student in all_students():
        if student["student_id"] == student_id:
            return student
    return None


PROFILE_DIR = DATA_DIR / "profiles"
PRESCRIPTION_DIR = DATA_DIR / "prescriptions"

_UNSAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]")

_CST = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(_CST).isoformat(timespec="seconds")


def _archive_result(directory: Path, payload: dict[str, Any]) -> Path:
    """留档一份生成结果，同一学号仅保留最新一份，重复生成直接覆盖。

    学号来自请求体，先清洗为安全字符集，避免用户输入拼进文件路径。
    """
    directory.mkdir(parents=True, exist_ok=True)
    student_id = _UNSAFE_ID_RE.sub("", str(payload.get("student_id", ""))) or "unknown"
    path = directory / f"{student_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def save_profile_result(profile: dict[str, Any]) -> Path:
    """把档案理解生成的画像 JSON 落盘到 data/profiles，同一学生仅保留最新一份。"""
    return _archive_result(PROFILE_DIR, profile)


def save_prescription_result(prescription: dict[str, Any]) -> Path:
    """把生成的处方 JSON 落盘到 data/prescriptions，同一学生仅保留最新一份。"""
    return _archive_result(PRESCRIPTION_DIR, prescription)


def _safe_student_id(student_id: str) -> str | None:
    """只接受档案库中真实存在的学号，避免用户输入拼进文件路径。"""
    return student_id if find_student(student_id) is not None else None


def _archived_ids(directory: Path) -> set[str]:
    """目录下留档文件对应的学号集合，仅保留档案库中真实存在的学号。"""
    ids: set[str] = set()
    if directory.is_dir():
        for path in directory.glob("*.json"):
            if _safe_student_id(path.stem):
                ids.add(path.stem)
    return ids


def _load_archived(directory: Path, safe_id: str) -> dict[str, Any] | None:
    path = directory / f"{safe_id}.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def saved_result_ids() -> list[str]:
    """有留档结果可载入的学号：画像 ∪ 处方 ∪ 熔断记录。不做缓存，生成结束即可见。"""
    ids = _archived_ids(PROFILE_DIR) | _archived_ids(PRESCRIPTION_DIR)
    ids |= set(_load_emergency_records())
    return sorted(ids)


def load_saved_result(student_id: str) -> dict[str, Any] | None:
    """拼装一份可载入结果，供前端选中学生时自动加载。

    - 有熔断记录：只返回急症提示，不返回旧画像与处方（熔断后旧结果不可信，
      但文件保留在磁盘上不删除，待重新生成成功后覆盖并解除熔断）
    - 无熔断记录：从 profiles/<学号>.json 与 prescriptions/<学号>.json 拼装，
      两份留档由服务层在生成成功时各自落盘（线上生成与脚本预生成共用同一路径）
    """
    safe_id = _safe_student_id(student_id)
    if safe_id is None:
        return None

    emergency = _load_emergency_records().get(safe_id)
    if emergency is not None:
        return {
            "student_id": safe_id,
            "name": emergency.get("name", ""),
            "generated_at": emergency.get("generated_at", ""),
            "profile": None,
            "prescription": None,
            "emergency": emergency,
        }

    profile = _load_archived(PROFILE_DIR, safe_id)
    prescription = _load_archived(PRESCRIPTION_DIR, safe_id)
    if profile is None and prescription is None:
        return None
    timestamps = [
        str(item.get("generated_at") or "")
        for item in (profile, prescription)
        if item is not None
    ]
    return {
        "student_id": safe_id,
        "name": (profile or prescription or {}).get("name", ""),
        "generated_at": max(timestamps) if timestamps else "",
        "profile": profile,
        "prescription": prescription,
        "emergency": None,
    }


EMERGENCY_RECORDS_FILE = DATA_DIR / "emergency_records.json"


def _load_emergency_records() -> dict[str, dict[str, Any]]:
    """学号 → 最近一次急症熔断记录。文件由 record_emergency 维护。"""
    if not EMERGENCY_RECORDS_FILE.is_file():
        return {}
    try:
        with EMERGENCY_RECORDS_FILE.open(encoding="utf-8") as handle:
            records = json.load(handle)
        return records if isinstance(records, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_emergency_records(records: dict[str, dict[str, Any]]) -> None:
    with EMERGENCY_RECORDS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def record_emergency(student_id: str, name: str, emergency: dict[str, Any]) -> None:
    """急症熔断留档：记录最近一次熔断，供前端选中该学生时展示就医提示。

    不动 profiles/ 与 prescriptions/ 里已有的画像与处方文件；同一学号
    重复熔断只保留最新一份。落盘是旁路留档，磁盘问题不中断生成流程。
    """
    safe_id = _safe_student_id(str(student_id or ""))
    if safe_id is None:
        return
    record = dict(emergency)
    record.update({"student_id": safe_id, "name": name, "generated_at": _now_iso()})
    records = _load_emergency_records()
    records[safe_id] = record
    try:
        _write_emergency_records(records)
    except OSError:
        pass


def clear_emergency(student_id: str) -> None:
    """生成成功即解除熔断：删除该学号的熔断记录，恢复正常的留档展示。"""
    safe_id = _safe_student_id(str(student_id or ""))
    if safe_id is None:
        return
    records = _load_emergency_records()
    if records.pop(safe_id, None) is None:
        return
    try:
        _write_emergency_records(records)
    except OSError:
        pass
