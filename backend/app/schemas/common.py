"""跨模块共享的基础 schema。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Versions(BaseModel):
    """产出这份结果所用的 prompt、模型与知识库版本。

    knowledge_base 是知识库指纹（入库文档内容变了指纹就变）。
    """

    model_config = ConfigDict(extra="allow")

    knowledge_base: str = ""
    prompt: str = ""
    model: str = ""


class AssessmentItem(BaseModel):
    """体质评定条目（S 条目）：一项体测成绩按国标评分表查表的结果。"""

    model_config = ConfigDict(extra="allow")

    id: str = ""
    key: str = ""  # 档案字段名，BMI 为 "BMI"
    name: str = ""
    raw: str = ""
    unit: str = ""
    score: int = 0
    level: str = ""
    basis: str = ""
    source: str = ""


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str = ""
    kind: str = ""
    md5: str = ""


class KnowledgeTopic(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = ""
    covered: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    top_score: float | None = None


class KnowledgeBaseInfo(BaseModel):
    """生成这份结果时知识库的状态：哪一版资料；处方还记录怎么检索的、各主题查到了什么。"""

    model_config = ConfigDict(extra="allow")

    available: bool = False
    fingerprint: str = ""
    collection: str = ""
    embedding_model: str = ""
    retrieval_mode: str = ""  # rerank（重排把关）/ dense（降级为向量相似度把关）
    stage: str = ""
    documents: list[KnowledgeDocument] = Field(default_factory=list)
    topics: list[KnowledgeTopic] = Field(default_factory=list)
    candidate_exercises: int = 0
