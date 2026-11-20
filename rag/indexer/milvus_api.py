"""Milvus 向量库的建库、建表、写入与查询。"""

from __future__ import annotations

from pymilvus import DataType, MilvusClient

from .config import Settings
from .logger import get_logger

logger = get_logger()

_VECTOR_FIELD = "content_vector"


def _quote(value: str) -> str:
    """把字符串写进过滤表达式：转义反斜杠与双引号。"""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


class MilvusAPI:
    def __init__(self, settings: Settings):
        self.collection = settings.milvus_collection
        self._index_name = settings.milvus_index_name
        self._dim = settings.embedding_dimensions

        auth = {}
        if settings.milvus_user:
            auth = {"user": settings.milvus_user, "password": settings.milvus_password}

        try:
            admin = MilvusClient(uri=settings.milvus_uri, **auth)
        except Exception as exc:
            raise ConnectionError(
                f"连不上 Milvus（{settings.milvus_uri}）。请先启动：bash rag/scripts/milvus_up.sh"
            ) from exc

        if settings.milvus_database not in admin.list_databases():
            admin.create_database(settings.milvus_database)
            logger.info("已创建数据库 %s", settings.milvus_database)
        admin.close()

        self.client = MilvusClient(
            uri=settings.milvus_uri, db_name=settings.milvus_database, **auth
        )
        self._ensure_collection()

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            self._check_dimension()
            self.client.load_collection(self.collection)
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        # 文档级信息：同一文档的所有切片相同
        schema.add_field("doc_name", DataType.VARCHAR, max_length=500)
        schema.add_field("doc_title", DataType.VARCHAR, max_length=500)
        schema.add_field("doc_path_name", DataType.VARCHAR, max_length=1000)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=50)
        schema.add_field("doc_md5", DataType.VARCHAR, max_length=32)
        schema.add_field("doc_length", DataType.INT64)
        # 切片级信息
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=20)  # text / table
        schema.add_field("section", DataType.VARCHAR, max_length=1000)  # 章节路径
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("parse_method", DataType.VARCHAR, max_length=20)  # text / ocr / mixed
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field(_VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=100)

        # 知识库是几千到几万条的规模，HNSW 无需按数据量调聚类数，召回也比 IVF 稳
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=_VECTOR_FIELD,
            index_name=self._index_name,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self.client.create_collection(self.collection, schema=schema, index_params=index_params)
        logger.info("已创建集合 %s（%d 维，HNSW / COSINE）", self.collection, self._dim)

    def _check_dimension(self) -> None:
        for field in self.client.describe_collection(self.collection)["fields"]:
            if field["name"] == _VECTOR_FIELD:
                existing = int(field["params"]["dim"])
                if existing != self._dim:
                    raise ValueError(
                        f"集合 {self.collection} 的向量是 {existing} 维，与当前配置的 {self._dim} 维不符。"
                        "更换嵌入模型或维度后请用 rebuild 模式重建。"
                    )

    def recreate_collection(self) -> None:
        """清空重建：删除集合及其全部数据，再按当前配置建表。"""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
            logger.info("已删除集合 %s", self.collection)
        self._ensure_collection()

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    def indexed_documents(self) -> dict[str, str]:
        """库中已有的文档：{doc_path_name: doc_md5}。"""
        rows = self.client.query(
            self.collection,
            filter="chunk_index == 0",
            output_fields=["doc_path_name", "doc_md5"],
            limit=16384,
            consistency_level="Strong",
        )
        return {row["doc_path_name"]: row["doc_md5"] for row in rows}

    def count_chunks(self, doc_path_name: str | None = None) -> int:
        rows = self.client.query(
            self.collection,
            filter=f"doc_path_name == {_quote(doc_path_name)}" if doc_path_name else "",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(rows[0]["count(*)"])

    def delete_document(self, doc_path_name: str) -> None:
        if not doc_path_name.strip():
            raise ValueError("文档路径为空，拒绝执行删除")
        self.client.delete(self.collection, filter=f"doc_path_name == {_quote(doc_path_name)}")

    def insert_chunks(self, rows: list[dict]) -> None:
        self.client.insert(self.collection, data=rows)

    def flush(self) -> None:
        self.client.flush(self.collection)

    def search(self, vector: list[float], limit: int = 5) -> list[dict]:
        """向量相似度检索，仅供入库后的自检使用。"""
        hits = self.client.search(
            self.collection,
            data=[vector],
            anns_field=_VECTOR_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit,
            output_fields=["doc_title", "section", "page_start", "page_end", "chunk_type", "content"],
        )
        return hits[0] if hits else []
