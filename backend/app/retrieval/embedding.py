"""Embedding 层：百炼 text-embedding-v3。

- 实现：openai SDK 直调（langchain-openai 的 OpenAIEmbeddings 请求格式与百炼兼容
  端点不兼容，实测 400 InvalidParameter；对话/评估层仍走 langchain，见 app/llm/）。
- 批量上限 10（实测 >10 报 InvalidParameter）。
- 失败抛 EmbeddingError；调用方捕获后降级（§4.4）。
- EMBEDDING_OFFLINE=1（CI/无 key 环境显式开启）：确定性哈希向量兜底，
  只保证向量库构建/检索链路可运行，无语义质量；不隐式嗅探 key，生产严禁开启。
"""
from __future__ import annotations

import hashlib
import math

from openai import OpenAI

from app.config import settings

CHUNK_SIZE = 10  # 与 settings.embedding_batch_size 一致（百炼批量上限）


class EmbeddingError(RuntimeError):
    pass


_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.llm_base_url,
            timeout=60,
            max_retries=2,
        )
    return _client


def _offline_vectors(texts: list[str]) -> list[list[float]]:
    """确定性特征哈希向量（EMBEDDING_OFFLINE=1 时启用，构建与查询共用同一空间）。

    字 2-gram + 空格分词哈希到固定维度（符号哈希减碰撞），L2 归一化。
    确定性保证同输入同向量（幂等重建/可复现评估）。
    """
    dim = settings.embedding_dim

    def one(text: str) -> list[float]:
        text = str(text)
        feats = [text[i:i + 2] for i in range(max(len(text) - 1, 1))] + text.split()
        vec = [0.0] * dim
        for f in feats:
            h = int(hashlib.md5(f.encode("utf-8")).hexdigest()[:8], 16)
            vec[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    return [one(t) for t in texts]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量编码（list[list[float]]，维度=EMBEDDING_DIM）。"""
    if not texts:
        return []
    if settings.embedding_offline:
        return _offline_vectors(texts)
    out: list[list[float]] = []
    try:
        client = get_client()
        for i in range(0, len(texts), CHUNK_SIZE):
            batch = texts[i:i + CHUNK_SIZE]
            resp = client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
                dimensions=settings.embedding_dim,
            )
            out.extend(d.embedding for d in resp.data)
    except Exception as e:
        raise EmbeddingError(f"embedding API 调用失败: {e}") from e
    return out


def embed_query(text: str) -> list[float]:
    if settings.embedding_offline:
        return _offline_vectors([text])[0]
    try:
        resp = get_client().embeddings.create(
            model=settings.embedding_model,
            input=[text],
            dimensions=settings.embedding_dim,
        )
        return resp.data[0].embedding
    except Exception as e:
        raise EmbeddingError(f"embedding API 调用失败: {e}") from e
