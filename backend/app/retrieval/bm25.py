"""自研 BM25 关键词检索（零新依赖，复用已验证实现）。

- 分词：英文按非字母数字切分（小写）；中文连续串按 2-gram 切分（避免分词器依赖）。
- 公式：score = IDF * f*(k1+1) / (f + k1*(1 - b + b*dl/avgdl))；IDF = ln(1 + (N-n+0.5)/(n+0.5))。
- 设计要点：BM25 的 k1/b 参数意义、IDF 平滑、与向量检索的互补。
"""
from __future__ import annotations

import math
import re
from collections import Counter

_ASCII_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文 2-gram + 英文单词。"""
    tokens: list[str] = []
    text = text.lower()
    for eng in _ASCII_RE.findall(text):
        tokens.append(eng)
    cjk_runs = re.split(r"[^\u4e00-\u9fff]+", text)
    for run in cjk_runs:
        run = run.strip()
        if len(run) == 1:
            tokens.append(run)
        elif len(run) >= 2:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("BM25Index 需要至少一个文档")
        self.k1, self.b = k1, b
        self.docs = docs
        self.doc_terms: list[Counter] = []
        self.doc_len: list[int] = []
        self.N = len(docs)
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        df: Counter[str] = Counter()
        for doc in self.docs:
            terms = tokenize(doc)
            self.doc_terms.append(Counter(terms))
            self.doc_len.append(len(terms))
            df.update(set(terms))
        self.avgdl = sum(self.doc_len) / self.N
        for term, n in df.items():
            self.idf[term] = math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """返回 [(doc_index, score)]，按分数降序。"""
        q_terms = tokenize(query)
        scores: list[float] = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_len[i]
            s = 0.0
            for t in set(q_terms):
                f = self.doc_terms[i].get(t, 0)
                if f == 0 or t not in self.idf:
                    continue
                s += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            scores[i] = s
        ranked = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, scores[i]) for i in ranked if scores[i] > 0]
