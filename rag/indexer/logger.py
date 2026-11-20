import logging

# 每次嵌入请求、OCR 初始化都会打 INFO 日志，入库时会刷屏
_NOISY_LOGGERS = ("httpx", "httpx2", "openai", "RapidOCR")


def get_logger(name: str = "rag") -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)
