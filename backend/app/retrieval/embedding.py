"""Embedding 层：百炼 text-embedding-v3。

- 实现：openai SDK 直调（langchain-openai 的 OpenAIEmbeddings 请求格式与百炼兼容
  端点不兼容，实测 400 InvalidParameter；对话/评估层仍走 langchain，见 app/llm/）。
- 批量上限 10（实测 >10 报 InvalidParameter）。
- 失败抛 EmbeddingError；调用方捕获后降级（§4.4）。
"""
from __future__ import annotations

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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量编码（list[list[float]]，维度=EMBEDDING_DIM）。"""
    if not texts:
        return []
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
    try:
        resp = get_client().embeddings.create(
            model=settings.embedding_model,
            input=[text],
            dimensions=settings.embedding_dim,
        )
        return resp.data[0].embedding
    except Exception as e:
        raise EmbeddingError(f"embedding API 调用失败: {e}") from e
