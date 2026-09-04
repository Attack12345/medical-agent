"""Qdrant 向量库访问层（企业级向量数据库）。

- 集合：entities（实体向量，payload: name/label/aliases）、
         qa_pairs（问答对，payload: qid/question/answer）、
         symptoms（症状，payload: name/body_part）。
- 距离度量：Cosine；HNSW m=16/ef_construct=100 显式配置（企业级可审计）。
- 失败抛 VectorDbError，调用方捕获后降级（§4.4）。
"""
from __future__ import annotations

from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

from app.config import settings

ENTITIES_COLLECTION = "entities"
QA_PAIRS_COLLECTION = "qa_pairs"
SYMPTOMS_COLLECTION = "symptoms"

COLLECTIONS = [ENTITIES_COLLECTION, QA_PAIRS_COLLECTION, SYMPTOMS_COLLECTION]


class VectorDbError(RuntimeError):
    pass


class VectorDb:
    def __init__(self, host: str | None = None, port: int | None = None):
        self.client = QdrantClient(
            url=f"http://{(host or settings.qdrant_host)}:{(port or settings.qdrant_port)}",
            timeout=60,
            # client 1.19 vs server 1.15：实测接口兼容，跳过版本检查避免噪音
            check_compatibility=False,
        )

    # ---------- 集合管理 ----------

    def ensure_collection(self, name: str, dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(name)
        if recreate and exists:
            self.client.delete_collection(name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
            )

    def count(self, name: str) -> int:
        return self.client.count(name).count

    # ---------- 写入 ----------

    def upsert(self, name: str, points: Iterable[PointStruct]) -> None:
        self.client.upsert(collection_name=name, points=list(points))

    # ---------- 检索 ----------

    def search(self, name: str, vector: list[float], top_k: int = 5) -> list[tuple[int, float, dict[str, Any]]]:
        """返回 [(id, score, payload)]，score 为余弦相似度（越大越相似）。"""
        try:
            hits = self.client.query_points(
                collection_name=name,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as e:
            raise VectorDbError(f"Qdrant 检索失败: {e}") from e
        return [(h.id, h.score, h.payload or {}) for h in hits]

    def search_text(self, name: str, text: str, top_k: int = 5) -> list[tuple[int, float, dict[str, Any]]]:
        """文本直接检索（内部先 embedding）。"""
        from app.retrieval.embedding import embed_query

        return self.search(name, embed_query(text), top_k=top_k)
