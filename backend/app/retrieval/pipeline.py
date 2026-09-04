"""检索编排（三路召回 + RRF 融合）。

流程：
  1. 实体链接路：Q 分句 → 名称/别名匹配 + 语义相似 ≥0.85 → 实体集合 E
  2. 图谱路：E 沿关系扩展一跳（症状→PRESENTS→疾病→VISITS→科室等）→ 候选三元组（带 source）
  3. 向量路：Q embedding → Qdrant 三集合各 Top5
  4. BM25 路：Q 分词（中文 2-gram）→ entities/qa_pairs 文本 Top5
  5. 融合：RRF（k=60）合并 图谱候选+向量+BM25，Top5 进上下文（证据候选池）

降级（§4.4）：Qdrant/embedding 不可用 → 向量/语义路空；Neo4j 不可用 → 图谱路空；
全链路不可用 → 返回空池（上层按 §5.3 refusal 处理）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.retrieval.bm25 import BM25Index  # noqa: E402
from app.retrieval.fusion import rrf_fuse  # noqa: E402
from app.retrieval.link import Linker  # noqa: E402
from app.retrieval.vector_db import (  # noqa: E402
    ENTITIES_COLLECTION,
    QA_PAIRS_COLLECTION,
    SYMPTOMS_COLLECTION,
    VectorDb,
    VectorDbError,
)

RRF_K = 60
TOPN = 5

# 图谱一跳扩展：实体 label → (方向, 关系, 对方 label) 列表
GRAPH_EXPANSIONS: dict[str, list[tuple[str, str, str]]] = {
    "Symptom": [
        ("out", "VISITS", "Department"),     # 症状→科室
        ("in", "PRESENTS", "Disease"),       # 症状←疾病
        ("out", "ACCOMPANIES", "Symptom"),
    ],
    "Disease": [
        ("out", "PRESENTS", "Symptom"),
        ("out", "REQUIRES_EXAM", "Exam"),
        ("out", "COMPLICATES", "Disease"),
        ("in", "TREATS", "Drug"),
        ("out", "AVOIDS_FOOD", "Food"),
        ("out", "AFFECTS", "Population"),
        ("out", "ADMITS_TO", "Hospital"),
    ],
    "Drug": [
        ("out", "TREATS", "Disease"),
        ("out", "CONTRAINDICATES", "Disease"),
    ],
    "Department": [],
    "Exam": [("out", "DIAGNOSES", "Disease")],
}

# BM25 文本索引（进程内懒加载，entities 文档名 + qa 文本）
_bm25: dict[str, BM25Index] = {}
_bm25_docs: dict[str, list[str]] = {}
# qa_pairs 的医生回答文本（与 qa_docs 按下标对应）：BM25 用患者提问做匹配，但证据内容
# 必须用医生的回答——患者原话（含口语/粗俗表述）绝不能进入证据池被 LLM 复述（M8.5）
_bm25_qa_answers: list[str] = []


def _get_bm25(collection: str, docs: list[str] | None = None) -> BM25Index:
    global _bm25, _bm25_docs
    if collection not in _bm25:
        if docs is None:
            raise RuntimeError(f"BM25 索引未初始化: {collection}")
        _bm25[collection] = BM25Index(docs)
        _bm25_docs[collection] = docs
    return _bm25[collection]


def init_bm25(entities_docs: list[str], qa_docs: list[str], qa_answers: list[str] | None = None) -> None:
    """预构建 BM25 索引（build_vector_db / M4 agent 启动时调用）。

    qa_docs=患者提问（口语化，匹配用）；qa_answers=医生回答（进证据池用）。
    """
    global _bm25_qa_answers
    _get_bm25("entities", entities_docs)
    _get_bm25("qa_pairs", qa_docs)
    _bm25_qa_answers = list(qa_answers) if qa_answers else []


def _graph_hop(repo, name: str, label: str) -> list[dict]:
    """实体一跳扩展 → [{subject, relation, object, source}]。"""
    triples: list[dict] = []
    for direction, rel, obj_label in GRAPH_EXPANSIONS.get(label, []):
        if direction == "out":
            rows = repo.query(
                f"MATCH (a:{label} {{name: $name}})-[r:{rel}]->(b:{obj_label}) "
                "RETURN b.name AS obj, r.source AS source LIMIT 10",
                {"name": name},
            )
            for row in rows:
                triples.append({"subject": name, "relation": rel, "object": row["obj"], "source": row.get("source") or "graph"})
        else:
            rows = repo.query(
                f"MATCH (a:{label} {{name: $name}})<-[r:{rel}]-(b:{obj_label}) "
                "RETURN b.name AS subj, r.source AS source LIMIT 10",
                {"name": name},
            )
            for row in rows:
                triples.append({"subject": row["subj"], "relation": rel, "object": name, "source": row.get("source") or "graph"})
    return triples


def retrieve(question: str, linker: Linker | None = None,
             db: VectorDb | None = None, repo=None) -> dict:
    """§4.2 全流程检索。返回：
    {
      "entities": [(name, label, confidence)],
      "graph_evidence": [{subject, relation, object, source}],
      "retrieval_evidence": [{doc_type, text, score}],   # 向量+BM25 命中
      "evidence_pool": [{type, ref, quote, score}],      # RRF Top5（证据候选池）
    }
    各路子模块失败自动降级（§4.4），不抛出。
    """
    from app.retrieval.embedding import EmbeddingError, embed_query

    entities: list[tuple[str, str, float]] = []
    graph_evidence: list[dict] = []
    retrieval_evidence: list[dict] = []
    vector_hits: list[tuple[int, float, dict]] = []
    bm25_hits: list[tuple[int, float, dict]] = []

    # 1+2) 实体链接 + 图谱扩展
    q_vector = None
    if linker is not None:
        try:
            q_vector = embed_query(question)
            entities = linker.link_text(question, query_vector=q_vector)
        except (EmbeddingError, VectorDbError):
            try:
                entities = linker.link_text(question)
            except Exception:
                entities = []
        if repo is not None:
            try:
                for name, label, _conf in entities:
                    graph_evidence.extend(_graph_hop(repo, name, label))
            except Exception:
                graph_evidence = []

    # 3) 向量路（三集合 Top5）
    if db is not None and q_vector is not None:
        for collection in [ENTITIES_COLLECTION, QA_PAIRS_COLLECTION, SYMPTOMS_COLLECTION]:
            try:
                vector_hits.extend((doc_id, score, payload) for doc_id, score, payload in
                                   db.search(collection, q_vector, top_k=5))
            except VectorDbError:
                pass

    # 4) BM25 路（qa_pairs 命中后取医生回答，患者提问原文绝不入池——M8.5）
    for collection, docs in [("entities", _bm25_docs.get("entities")),
                             ("qa_pairs", _bm25_docs.get("qa_pairs"))]:
        if not docs:
            continue
        try:
            for idx, score in _get_bm25(collection, docs).search(question, top_k=5):
                text = docs[idx][:200]
                if collection == "qa_pairs" and idx < len(_bm25_qa_answers) and _bm25_qa_answers[idx]:
                    text = _bm25_qa_answers[idx][:200]
                bm25_hits.append((idx, score, {"doc_type": collection, "text": text}))
        except Exception:
            pass

    # 统一为 retrieval_evidence（向量+BM25 去重保留）
    seen_text: set[str] = set()
    for _id, score, payload in vector_hits:
        text = str(payload.get("answer") or payload.get("name") or payload.get("question") or "")[:200]
        doc_type = str(payload.get("label") or payload.get("doc_type") or "vector")
        if text and text not in seen_text:
            seen_text.add(text)
            retrieval_evidence.append({"doc_type": doc_type, "text": text, "score": round(float(score), 4)})
    for _id, score, payload in bm25_hits:
        text = payload.get("text", "")
        if text and text not in seen_text:
            seen_text.add(text)
            retrieval_evidence.append({"doc_type": payload["doc_type"], "text": text, "score": round(float(score), 4)})

    # 5) RRF 融合（图谱三元组 / 向量命中 / BM25 命中 各为一路，拼接索引映射）
    graph_ranked = [(i, 1.0) for i in range(len(graph_evidence))]
    vec_ranked = [(len(graph_evidence) + i, 1.0) for i in range(len(vector_hits))]
    bm_ranked = [(len(graph_evidence) + len(vector_hits) + i, 1.0) for i in range(len(bm25_hits))]
    fused = rrf_fuse([graph_ranked, vec_ranked, bm_ranked], k=RRF_K, top_n=TOPN)

    n_g, n_v = len(graph_evidence), len(vector_hits)
    evidence_pool: list[dict] = []
    for doc_id, _score in fused:
        if doc_id < n_g:
            g = graph_evidence[doc_id]
            # quote 用自然短语（"流涕 → 中医科"），禁止泄露技术关系名（§5.3 展示边界：
            # 回答/证据文本不得出现 VISITS 等技术节点名）；ref 保留审计格式。
            # text 字段 = quote 别名（S102 规则 $chat.evidence_pool[].text 引用）
            q = f"{g['subject']} → {g['object']}"
            evidence_pool.append({"type": "GRAPH_NODE", "ref": f"{g['relation']}:{g['subject']}→{g['object']}",
                                  "quote": q, "text": q, "score": 1.0})
        elif doc_id < n_g + n_v:
            payload = vector_hits[doc_id - n_g][2]
            text = str(payload.get("answer") or payload.get("name") or "")[:200]
            evidence_pool.append({"type": "RETRIEVAL", "ref": f"vector:{payload.get('label', '')}:{payload.get('name', '')}",
                                  "quote": text, "text": text, "score": 1.0})
        else:
            payload = bm25_hits[doc_id - n_g - n_v][2]
            evidence_pool.append({"type": "RETRIEVAL", "ref": f"bm25:{payload['doc_type']}",
                                  "quote": payload["text"], "text": payload["text"], "score": 1.0})

    return {
        "entities": entities,
        # 图谱候选全量返回（每实体每类关系 LIMIT 10，实体通常 1-2 个，上限 ~80 条；
        # M4 注入 LLM 上下文时按需裁剪）
        "graph_evidence": graph_evidence,
        "retrieval_evidence": retrieval_evidence[:20],
        "evidence_pool": evidence_pool,
    }
