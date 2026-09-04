"""RRF 融合排序（多路召回融合，复用已验证实现）。"""
from __future__ import annotations

from typing import Sequence


def rrf_fuse(lists: Sequence[list[tuple[int, float]]], k: int = 60,
             top_n: int = 5) -> list[tuple[int, float]]:
    """多路 (doc_id, score) 列表按 RRF 融合：score = Σ 1/(k+rank)。

    - 输入顺序代表路优先级（rank 从 1 起）；
    - 返回 [(doc_id, rrf_score)]，按分数降序取 top_n。
    """
    agg: dict[int, float] = {}
    for ranked in lists:
        for rank, (doc_id, _) in enumerate(ranked, 1):
            agg[doc_id] = agg.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(agg.items(), key=lambda kv: -kv[1])[:top_n]
