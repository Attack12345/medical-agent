"""向量库构建（幂等重建）。

集合（全部 Cosine + HNSW m=16/ef_construct=100）：
  entities  {name, label, aliases}   向量=实体名+描述（Disease 用 summary，其他用 name）
  qa_pairs  {qid, question, answer} 向量=question
  symptoms  {name, body_part}       向量=症状名+部位

用法：python build_vector_db.py [--recreate]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.graph.repo import GraphRepo  # noqa: E402
from app.retrieval.embedding import embed_texts  # noqa: E402
from app.retrieval.pipeline import init_bm25  # noqa: E402
from app.retrieval.vector_db import (  # noqa: E402
    ENTITIES_COLLECTION,
    QA_PAIRS_COLLECTION,
    SYMPTOMS_COLLECTION,
    VectorDb,
)
from qdrant_client.models import PointStruct  # noqa: E402

BATCH = 200  # 每批 upsert 条数（embedding 内部分 10 条/请求）


def log(msg: str) -> None:
    print(f"[build_vector_db] {msg}", flush=True)


def load_graph_entities(repo: GraphRepo) -> list[dict]:
    """图谱实体：Disease（含 summary）/Drug/Department/Exam 名。"""
    rows = repo.query(
        """
        MATCH (n)
        WHERE any(l IN labels(n) WHERE l IN ['Disease', 'Drug', 'Department', 'Exam'])
        RETURN labels(n)[0] AS label, n.name AS name, n.severity AS severity,
               n.summary AS summary, n.aliases AS aliases
        """
    )
    out = []
    for r in rows:
        label = r["label"]
        if label == "Disease":
            text = f"{r['name']}：{(r.get('summary') or '')[:100]}"
        else:
            text = r["name"]
        out.append({
            "name": r["name"], "label": label,
            "aliases": list(r.get("aliases") or []),
            "text": text,
        })
    return out


def load_symptoms(repo: GraphRepo) -> list[dict]:
    rows = repo.query("MATCH (n:Symptom) RETURN n.name AS name, n.body_part AS body_part")
    return [{"name": r["name"], "body_part": r.get("body_part") or "", "text": r["name"]} for r in rows]


def load_qa() -> list[dict]:
    with open(PROJECT_ROOT / "data" / "cleaned" / "qa_pairs.json", "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_batched(db: VectorDb, collection: str, points: list[dict], text_key: str,
                   payload_keys: list[str], id_key: str = "idx") -> None:
    """分批 embedding + upsert；id 用顺序号（重建场景幂等）。"""
    texts = [p[text_key] for p in points]
    total = len(texts)
    for start in range(0, total, BATCH):
        chunk = points[start:start + BATCH]
        vectors = embed_texts([p[text_key] for p in chunk])
        pts = []
        for i, (p, vec) in enumerate(zip(chunk, vectors)):
            payload = {k: p.get(k) for k in payload_keys if k in p}
            pts.append(PointStruct(id=start + i, vector=vec, payload=payload))
        db.upsert(collection, pts)
        log(f"{collection}: {min(start + BATCH, total)}/{total}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="向量库构建")
    parser.add_argument("--recreate", action="store_true", help="重建集合（默认只补缺失）")
    args = parser.parse_args()

    repo = GraphRepo()
    db = VectorDb()
    try:
        for c in [ENTITIES_COLLECTION, QA_PAIRS_COLLECTION, SYMPTOMS_COLLECTION]:
            db.ensure_collection(c, settings.embedding_dim, recreate=args.recreate)

        log("加载图谱实体…")
        entities = load_graph_entities(repo)
        log(f"图谱实体: {len(entities)}")
        upsert_batched(db, ENTITIES_COLLECTION, entities, "text", ["name", "label", "aliases"])

        log("加载症状…")
        symptoms = load_symptoms(repo)
        log(f"症状: {len(symptoms)}")
        upsert_batched(db, SYMPTOMS_COLLECTION, symptoms, "text", ["name", "body_part"])

        log("加载问答对…")
        qa = load_qa()
        log(f"问答对: {len(qa)}")
        upsert_batched(db, QA_PAIRS_COLLECTION, qa, "question", ["qid", "question", "answer"])

        # BM25 文本索引预热（M4 检索直接可用）
        init_bm25([e["text"] for e in entities], [q["question"] for q in qa])
        log("BM25 索引就绪")

        for c in [ENTITIES_COLLECTION, QA_PAIRS_COLLECTION, SYMPTOMS_COLLECTION]:
            log(f"{c}: {db.count(c)} 条")
        log("✅ 向量库构建完成")
        return 0
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
