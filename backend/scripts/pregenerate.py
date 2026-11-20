"""预生成若干学生的画像与处方并留档，供前端秒开演示。

用法：
    uv run python -u scripts/pregenerate.py              # 生成默认的 5 份
    uv run python -u scripts/pregenerate.py BJ2025090314 # 只生成指定学生
    uv run python -u scripts/pregenerate.py --all        # 全部学生

画像与处方由服务层在生成成功时落盘到 app/data/profiles/<学号>.json 与
app/data/prescriptions/<学号>.json（同一学号覆盖），由 /api/v1/pregenerated 读取，
与线上实时生成共用同一套留档。
串行执行：并发会触发上游限流，单次调用会从 1 分钟拖到十几分钟。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.schemas.record import HealthRecord  # noqa: E402
from app.services.data_loader import find_student, mock_students, record_emergency  # noqa: E402
from app.services.prescription_service import stream_prescription  # noqa: E402
from app.services.profile_service import analyze_profile  # noqa: E402
from app.services.safety import check_emergency  # noqa: E402

# 默认预生成的 5 份，覆盖不同学段、性别、体能水平与约束类型
DEFAULT_IDS = [
    "BJ2025090314",  # 李沐阳 初二男 肥胖 + 哮喘 + 右踝陈旧伤
    "BJ2025070221",  # 陈若溪 高一女 偏瘦 柔韧好但心肺差
    "BJ2025050137",  # 王子轩 小学五年级男 正常 身高促进
    "BJ2025120331",  # 黄俊豪 高一男 体能优秀 爆发力与 50 米专项
    "BJ2025110267",  # 刘星月 初二女 膝髌股疼痛恢复期，需低冲击
]


async def generate_one(student_id: str) -> bool:
    student = find_student(student_id)
    if student is None:
        print(f"  跳过 {student_id}：档案不存在")
        return False

    record = HealthRecord.model_validate(student)
    name = record.name

    emergency = check_emergency(student)
    if emergency is not None:
        record_emergency(student_id, name, emergency.model_dump())
        print(f"  {name}：命中急症熔断，已留档熔断记录（不调用模型）")
        return True

    # 画像与处方两个模块各自独立生成，互不传递结果；
    # 各自成功后由服务层覆盖留档到 profiles/ 与 prescriptions/
    t0 = time.time()
    await analyze_profile(record)
    t_profile = time.time() - t0

    t1 = time.time()
    result = None
    retries = 0
    async for kind, data in stream_prescription(record):
        if kind == "retry":
            retries += 1
        elif kind == "result":
            result = data
        elif kind == "error":
            print(f"  {name}：处方生成失败 — {data}")
            return False
    t_rx = time.time() - t1

    if result is None:
        print(f"  {name}：处方未产出结果")
        return False

    days = sum(1 for d in result.weekly_plan if d.is_training_day)
    print(
        f"  {name}：完成（画像 {t_profile:.0f}s + 处方 {t_rx:.0f}s"
        f"，重试 {retries} 次，训练日 {days} 天）"
    )
    return True


async def main() -> None:
    args = sys.argv[1:]
    if args == ["--all"]:
        ids = [s["student_id"] for s in mock_students()]
    elif args:
        ids = args
    else:
        ids = DEFAULT_IDS

    print(
        f"模型 {settings.active_model}，思考{'开' if settings.thinking_enabled else '关'}，"
        f"共 {len(ids)} 份，串行执行\n"
    )
    t0 = time.time()
    ok = 0
    for index, student_id in enumerate(ids, 1):
        print(f"[{index}/{len(ids)}] {student_id}", flush=True)
        if await generate_one(student_id):
            ok += 1
    print(f"\n完成 {ok}/{len(ids)}，总耗时 {time.time() - t0:.0f}s")
    print("留档目录：app/data/profiles 与 app/data/prescriptions")


if __name__ == "__main__":
    asyncio.run(main())
