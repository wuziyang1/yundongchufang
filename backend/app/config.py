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

    # RAG 检索：仅用于处方生成阶段，知识卡片与动作候选按学生情况检索子集注入 prompt。
    # 关闭开关即可回到全量拼接，便于 A/B 对比效果与耗时。档案理解阶段不使用 RAG。
    rag_enabled: bool = os.getenv("RAG_ENABLED", "true").strip().lower() == "true"
    rag_knowledge_top_k: int = int(os.getenv("RAG_KNOWLEDGE_TOP_K", "12"))
    rag_exercise_limit: int = int(os.getenv("RAG_EXERCISE_LIMIT", "30"))

    # 嵌入服务，默认复用智谱 OpenAI 兼容端点与密钥；可独立指向其他服务
    embedding_base_url: str = os.getenv(
        "EMBEDDING_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
    ).strip()
    embedding_api_key: str = (
        os.getenv("EMBEDDING_API_KEY", "").strip() or os.getenv("ZHIPU_API_KEY", "").strip()
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "embedding-3").strip()
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    embedding_timeout: float = float(os.getenv("EMBEDDING_TIMEOUT", "15"))

    # 权威资料知识库检索（Milvus）：rag/ 离线入库的国家标准、指南、文献切片。仅处方阶段使用，
    # 作为知识卡片的补充，以「权威资料摘录」注入 prompt，每条带出处。查询向量沿用上方
    # EMBEDDING_* 配置，必须与入库时的模型、维度一致（默认 embedding-3、1024 维）。
    # Milvus 未启动、无嵌入密钥、mock 模式或超时都会跳过检索并记 warning，不影响处方生成。
    kb_enabled: bool = os.getenv("KB_RETRIEVAL_ENABLED", "true").strip().lower() == "true"
    milvus_host: str = os.getenv("MILVUS_HOST", "127.0.0.1").strip()
    milvus_port: str = os.getenv("MILVUS_PORT", "19530").strip()
    milvus_user: str = os.getenv("MILVUS_USER", "").strip()
    milvus_password: str = os.getenv("MILVUS_PASSWORD", "").strip()
    milvus_database: str = os.getenv("MILVUS_DATABASE", "yundongchufang").strip()
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "knowledge_chunks").strip()
    kb_top_k: int = int(os.getenv("KB_TOP_K", "5"))
    # COSINE 相似度阈值：低于此值的摘录不注入。embedding-3 下相关切片多在 0.75 以上，
    # 知识库没有覆盖的伤病（如膝、踝）最高也只有 0.63-0.71，0.7 能把它们挡在外面
    kb_score_threshold: float = float(os.getenv("KB_SCORE_THRESHOLD", "0.7"))
    # 同一份文档最多取几条，避免一份长文档占满名额
    kb_max_per_doc: int = int(os.getenv("KB_MAX_PER_DOC", "3"))
    # 查询嵌入 + 向量检索的总超时（秒）
    kb_timeout: float = float(os.getenv("KB_TIMEOUT", "8"))

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "65536"))
    timeout: float = float(os.getenv("LLM_TIMEOUT", "300"))
    # 思考模型开关。开启时约 80% 的输出 token 用于思考，一次完整流程从 84s 涨到 404s。
    # 开启后思考内容会流式显示在前端「AI 生成过程」面板；.env 中默认开启，代码兜底为关闭。
    thinking_enabled: bool = os.getenv("LLM_THINKING", "false").strip().lower() == "true"

    # 资产版本号，随每份画像与处方一并返回，便于定位是哪一版规则/库产出的结果
    exercise_library_version: str = "v1.0.0"
    knowledge_base_version: str = "v1.0.0"
    prompt_version: str = "v2.0.0"

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
