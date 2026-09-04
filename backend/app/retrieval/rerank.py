"""Rerank 重排（M8.8 实验）：bge-reranker CrossEncoder 对融合候选精排。

实验导向：先以 scripts/experiment_rerank.py 在检索评估集上对照
（A=RRF Top5 现主链路；B=RRF Top20 → rerank 精排 Top5），
有提升才接入主链路（pipeline 融合后调用本模块），无提升则记录证伪。
模型：BAAI/bge-reranker-base（~1.1GB，HF_ENDPOINT=https://hf-mirror.com 镜像下载）。
"""
from __future__ import annotations

MODEL_ID = "BAAI/bge-reranker-base"

_model = None  # 进程内单例（CrossEncoder 加载约 3-5s）


def get_reranker():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_ID, max_length=512)
    return _model


def rerank(query: str, candidates: list[dict], text_key: str = "quote",
           top_n: int = 5) -> list[dict]:
    """对候选按 (query, text) 相关性重排，返回 top_n（附带 rerank_score）。

    candidates 为证据池条目（含 quote/ref/type）；文本截 400 字防超长。
    模型加载失败时原样返回前 top_n（降级不阻塞主链路，§4.4）。
    """
    if not candidates:
        return []
    try:
        model = get_reranker()
        pairs = [(query, str(c.get(text_key, ""))[:400]) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -float(x[1]))
    except Exception:
        return [dict(c) for c in candidates[:top_n]]
    out: list[dict] = []
    for c, s in ranked[:top_n]:
        c = dict(c)
        c["rerank_score"] = round(float(s), 4)
        out.append(c)
    return out
