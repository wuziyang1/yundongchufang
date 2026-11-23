"""跨模块共享的基础 schema。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Versions(BaseModel):
    """产出这份结果所用的 prompt、模型与知识库版本。

    knowledge_base 是知识库指纹（入库文档内容变了指纹就变），只有处方会填；画像不使用知识库。
    """

    model_config = ConfigDict(extra="allow")

    knowledge_base: str = ""
    prompt: str = ""
    model: str = ""
