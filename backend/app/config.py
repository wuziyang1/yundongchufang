"""运行配置。所有可调项集中在此，便于交付期替换为本地推理服务。"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR.parent.parent / "frontend"

load_dotenv(BASE_DIR.parent / ".env")


class Settings:
    """进程级配置，启动时读取一次。"""

    llm_provider: str = os.getenv("LLM_PROVIDER", "zhipu").strip().lower()

    # 智谱 Anthropic 兼容端点
    anthropic_base_url: str = os.getenv(
        "ANTHROPIC_COMPAT_BASE_URL", "https://open.bigmodel.cn/api/anthropic"
    ).strip()
    anthropic_api_key: str = os.getenv("ANTHROPIC_COMPAT_API_KEY", "").strip()
    anthropic_model: str = os.getenv("ANTHROPIC_COMPAT_MODEL", "glm-5.3").strip()

    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "").strip()
    zhipu_base_url: str = os.getenv(
        "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
    ).strip()
    zhipu_model: str = os.getenv("ZHIPU_MODEL", "glm-5.3").strip()

    compat_base_url: str = os.getenv(
        "OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:8000/v1"
    ).strip()
    compat_api_key: str = os.getenv("OPENAI_COMPATIBLE_API_KEY", "EMPTY").strip()
    compat_model: str = os.getenv("OPENAI_COMPATIBLE_MODEL", "glm-4-9b-chat").strip()

    # 嵌入服务，默认复用智谱 OpenAI 兼容端点与密钥；可独立指向其他服务。
    # 查询向量必须与知识库入库时的模型、维度一致（默认 embedding-3、1024 维）。
    embedding_base_url: str = os.getenv(
        "EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
    ).strip()
    embedding_api_key: str = (
        os.getenv("EMBEDDING_API_KEY", "").strip() or os.getenv("ZHIPU_API_KEY", "").strip()
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embedding-3").strip()
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

    # 知识库（Milvus）：rag/ 离线入库的国家标准、指南、手册、文献。处方的专业依据只从这里取：
    # 体测评分表、权威资料摘录、可选动作都来自它。知识库不可用时处方不生成（mock 模式除外）。
    milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1").strip()
    milvus_port: str = os.getenv("MILVUS_PORT", "19530").strip()
    milvus_user: str = os.getenv("MILVUS_USER", "").strip()
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "").strip()
    milvus_database: str = os.getenv("MILVUS_DATABASE", "yundongchufang").strip()
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "knowledge_chunks").strip()
    # 一次处方的全部检索（评分表、摘录、动作）的总超时（秒）
    kb_timeout: float = float(os.getenv("KB_TIMEOUT", "20"))

    # 相关性把关：检索回来的切片是否真的回答了问题。向量相似度区分不开（相关 0.70-0.85、
    # 不相关 0.63-0.69 挨得很近），所以先用向量 + 关键词两路召回候选，再交给重排模型逐条打分，
    # 低于阈值的不注入 prompt，该问题记为「知识库未覆盖」。
    # 默认用智谱 rerank（与嵌入共用密钥），取原始分。阈值 17；针对个人的主题（短板、健康状况）从严，
    # 再加 1。实测 20 份档案：有针对性资料的主题最高分都在 18.2 以上，没有的最高 17.7（短板）、
    # 16.4（健康状况）。换用其他重排服务或嵌入模型后分值尺度不同，须用
    # scripts/check_kb.py --all 重新标定 KB_RERANK_MIN_SCORE。
    kb_rerank_enabled: bool = os.getenv("KB_RERANK_ENABLED", "true").strip().lower() == "true"
    rerank_base_url: str = os.getenv("RERANK_BASE_URL", "").strip() or embedding_base_url
    rerank_api_key: str = os.getenv("RERANK_API_KEY", "").strip() or embedding_api_key
    rerank_model: str = os.getenv("RERANK_MODEL", "rerank").strip()
    kb_rerank_min_score: float = float(os.getenv("KB_RERANK_MIN_SCORE", "17"))
    # 重排关闭或不可用时退回向量相似度把关（COSINE），把关能力弱一些
    kb_dense_min_score: float = float(os.getenv("KB_DENSE_MIN_SCORE", "0.72"))

    # 可选动作的筛选：学生不填器材清单，按家庭与校园普遍可得的器材过滤（徒手动作始终可用）；
    # 器材名以知识库训练手册里的写法为准
    kb_equipment: list[str] = [
        item.strip()
        for item in os.getenv("KB_EQUIPMENT", "瑜伽垫,垫子,标志线,标志碟,锥形筒").split(",")
        if item.strip()
    ]
    # 写进 prompt 的动作条数上限；超出时优先保留针对短板检索到的动作，其余按素质类别均衡取
    kb_exercise_limit: int = int(os.getenv("KB_EXERCISE_LIMIT", "100"))

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "65536"))
    timeout: float = float(os.getenv("LLM_TIMEOUT", "300"))
    # 思考模型开关。开启时约 80% 的输出 token 用于思考，一次完整流程从 84s 涨到 404s。
    # 开启后思考内容会流式显示在前端「AI 生成过程」面板；.env 中默认开启，代码兜底为关闭。
    thinking_enabled: bool = os.getenv("LLM_THINKING", "false").strip().lower() == "true"

    # prompt 版本号，随每份画像与处方一并返回；知识库版本用入库文档的内容指纹表示，随处方留档
    prompt_version: str = "v3.1.0"

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def active_model(self) -> str:
        if self.llm_provider == "zhipu":
            return self.zhipu_model
        if self.llm_provider == "anthropic_compatible":
            return self.anthropic_model
        if self.llm_provider == "openai_compatible":
            return self.compat_model
        return "mock"


settings = Settings()
