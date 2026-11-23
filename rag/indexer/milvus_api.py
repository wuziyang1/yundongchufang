"""Milvus 向量库的建库、建表、写入与查询。"""

from __future__ import annotations

from pymilvus import DataType, Function, FunctionType, MilvusClient

from .config import Settings
from .logger import get_logger

logger = get_logger()

_VECTOR_FIELD = "content_vector"
_SPARSE_FIELD = "content_sparse"
# 关键词检索（BM25）的分词：jieba 切中文，英文转小写并取词干，丢掉纯符号。
# 「哮喘」「肥胖」「仰卧起坐」「RPE」这类必须字面命中的词，光靠向量相似度召回不稳，要靠它兜住
_ANALYZER = {
    "tokenizer": "jieba",
    "filter": ["lowercase", {"type": "stemmer", "language": "english"}, "cnalphanumonly"],
}
# 在线检索依赖的字段；旧集合缺少它们时必须 rebuild
_REQUIRED_FIELDS = ("doc_kind", "stages", "meta", "search_text", _SPARSE_FIELD)


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
        # 旧结构的集合（缺少学段、动作条目、关键词检索等字段）：只读操作照常，写入前必须 rebuild
        self.outdated: list[str] = []
        self._ensure_collection()

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            self._check_schema()
            self.client.load_collection(self.collection)
            return

        self.outdated = []
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        # 文档级信息：同一文档的所有切片相同
        schema.add_field("doc_name", DataType.VARCHAR, max_length=500)
        schema.add_field("doc_title", DataType.VARCHAR, max_length=500)
        schema.add_field("doc_path_name", DataType.VARCHAR, max_length=1000)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=50)
        schema.add_field("doc_md5", DataType.VARCHAR, max_length=32)
        schema.add_field("doc_length", DataType.INT64)
        # 文档身份（doc_rules.toml）：类别与适用学段，在线检索按学段过滤
        schema.add_field("doc_kind", DataType.VARCHAR, max_length=50)
        schema.add_field(
            "stages", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=8, max_length=20
        )
        # 切片级信息
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=20)  # text / table / lookup / exercise
        schema.add_field("section", DataType.VARCHAR, max_length=1000)  # 章节路径
        schema.add_field("page_start", DataType.INT64)
        schema.add_field("page_end", DataType.INT64)
        schema.add_field("parse_method", DataType.VARCHAR, max_length=20)  # text / ocr / mixed
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        # 动作条目的结构化信息（编号、名称、素质类别、器材、各段文字）；其余切片为空对象
        schema.add_field("meta", DataType.JSON)
        # 向量化与关键词检索共用的文本：「《文档标题》章节路径 + 正文」
        schema.add_field(
            "search_text", DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params=_ANALYZER
        )
        schema.add_field(_VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field(_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=100)
        # Milvus 内置 BM25：写入时由 search_text 自动生成稀疏向量，查询时直接传查询文本
        schema.add_function(
            Function(
                name="search_text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["search_text"],
                output_field_names=[_SPARSE_FIELD],
            )
        )

        # 知识库是几千到几万条的规模，HNSW 无需按数据量调聚类数，召回也比 IVF 稳
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=_VECTOR_FIELD,
            index_name=self._index_name,
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name=_SPARSE_FIELD,
            index_name=f"{_SPARSE_FIELD}_bm25",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
        )

        self.client.create_collection(self.collection, schema=schema, index_params=index_params)
        logger.info("已创建集合 %s（%d 维，HNSW / COSINE + BM25 关键词检索）", self.collection, self._dim)

    def _check_schema(self) -> None:
        fields = {field["name"]: field for field in self.client.describe_collection(self.collection)["fields"]}
        existing = int(fields[_VECTOR_FIELD]["params"]["dim"]) if _VECTOR_FIELD in fields else None
        if existing is not None and existing != self._dim:
            raise ValueError(
                f"集合 {self.collection} 的向量是 {existing} 维，与当前配置的 {self._dim} 维不符。"
                "更换嵌入模型或维度后请用 rebuild 模式重建。"
            )
        self.outdated = [name for name in _REQUIRED_FIELDS if name not in fields]

    def require_current(self) -> None:
        """写入前调用：集合还是旧结构时给出明确提示，而不是在写入时报字段不匹配。"""
        if self.outdated:
            raise ValueError(
                f"集合 {self.collection} 是旧结构，缺少字段 {'、'.join(self.outdated)}。"
                "请用 rebuild 模式重建：bash rag/scripts/run_index.sh --mode rebuild"
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

    def keyword_search(self, text: str, limit: int = 5) -> list[dict]:
        """BM25 关键词检索，仅供入库后的自检使用。旧结构的集合没有这一路。"""
        if self.outdated:
            return []
        hits = self.client.search(
            self.collection,
            data=[text],
            anns_field=_SPARSE_FIELD,
            search_params={"metric_type": "BM25"},
            limit=limit,
            output_fields=["doc_title", "section", "page_start", "page_end", "chunk_type", "content"],
        )
        return hits[0] if hits else []

    def document_summary(self) -> list[dict]:
        """每份文档的类别、适用学段与各类切片数，供 stats 模式展示。"""
        fields = ["doc_path_name", "chunk_type"]
        if not self.outdated:
            fields += ["doc_kind", "stages"]
        summary: dict[str, dict] = {}
        iterator = self.client.query_iterator(self.collection, filter="", output_fields=fields, batch_size=1000)
        try:
            while batch := iterator.next():
                for row in batch:
                    item = summary.setdefault(
                        row["doc_path_name"],
                        {
                            "doc_path_name": row["doc_path_name"],
                            "doc_kind": row.get("doc_kind", ""),
                            "stages": list(row.get("stages") or []),
                            "counts": {},
                        },
                    )
                    item["counts"][row["chunk_type"]] = item["counts"].get(row["chunk_type"], 0) + 1
        finally:
            iterator.close()
        return [summary[name] for name in sorted(summary)]
