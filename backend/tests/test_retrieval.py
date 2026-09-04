"""M3 检索层测试：BM25/RRF/实体链接/端到端检索。

- BM25/RRF 纯单测；
- 实体链接/端到端依赖 Neo4j + Qdrant + 百炼 key，缺环境自动 skip。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.retrieval.bm25 import BM25Index, tokenize  # noqa: E402
from app.retrieval.fusion import rrf_fuse  # noqa: E402

HAS_ENV = bool(settings.dashscope_api_key)


# ---------- BM25（§4.2 第4路） ----------

def test_bm25_tokenize_chinese_2gram():
    assert "头疼" in tokenize("头疼怎么办")
    assert tokenize("headache") == ["headache"]


def test_bm25_search_ranks_relevant_first():
    idx = BM25Index(["高血压患者需要低盐饮食", "感冒要多喝水多休息", "发烧可以物理降温"])
    hits = idx.search("高血压饮食", top_k=3)
    assert hits and hits[0][0] == 0, "相关文档应排第一"


# ---------- RRF 融合（§4.2 第5步） ----------

def test_rrf_fuse_merges_lists():
    a = [(0, 1.0), (1, 1.0), (2, 1.0)]
    b = [(1, 1.0), (3, 1.0)]
    fused = rrf_fuse([a, b], k=60, top_n=3)
    assert fused[0][0] == 1, "两路都命中的文档应排第一"
    assert len(fused) == 3


def test_rrf_fuse_topn():
    a = [(i, 1.0) for i in range(10)]
    assert len(rrf_fuse([a], top_n=5)) == 5


# ---------- 实体链接（§4.2 第1路，需 Neo4j） ----------

@pytest.mark.skipif(not HAS_ENV, reason="无百炼 key")
def test_linker_exact_and_alias():
    from app.retrieval.link import Linker

    linker = Linker()
    hits = dict((name, (label, conf)) for name, label, conf in
                [h for h in linker.link_text("我头痛而且发烧了")])
    assert "头痛" in hits, "实体名精确匹配应命中"
    assert "发热" in hits, "别名'发烧'应归一命中'发热'"


# ---------- 端到端检索（§4.2 全流程，需 Neo4j+Qdrant+key） ----------

@pytest.mark.skipif(not HAS_ENV, reason="无百炼 key")
def test_retrieve_end_to_end():
    from app.graph.repo import GraphRepo
    from app.retrieval.link import Linker
    from app.retrieval.pipeline import init_bm25, retrieve
    from app.retrieval.vector_db import VectorDb

    with open(PROJECT_ROOT / "data" / "cleaned" / "diseases.json", "r", encoding="utf-8") as f:
        import json
        diseases = json.load(f)
    with open(PROJECT_ROOT / "data" / "cleaned" / "qa_pairs.json", "r", encoding="utf-8") as f:
        qa = json.load(f)
    init_bm25([d["name"] for d in diseases], [q["question"] for q in qa])

    repo = GraphRepo()
    db = VectorDb()
    linker = Linker()
    try:
        result = retrieve("高血压吃什么药", linker=linker, db=db, repo=repo)
        assert result["evidence_pool"], "证据池不应为空"
        rels = {g["relation"] for g in result["graph_evidence"]}
        assert "TREATS" in rels, "图谱路应命中 TREATS"
        names = {name for name, _l, _c in result["entities"]}
        assert "高血压" in names, "实体链接应命中'高血压'"
    finally:
        repo.close()
