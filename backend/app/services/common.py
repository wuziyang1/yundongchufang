"""画像与处方两个模块共用的小助手：时间戳与版本信息。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.config import settings
from app.schemas.common import Versions

CST = timezone(timedelta(hours=8))


def now_iso() -> str:
    return datetime.now(CST).isoformat(timespec="seconds")


def current_versions(knowledge_base: str = "") -> Versions:
    return Versions(
        knowledge_base=knowledge_base,
        prompt=settings.prompt_version,
        model=settings.active_model,
    )
