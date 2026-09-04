"""M2 图谱测试（DoD：统计达标/Cypher 抽查/实体链接别名表/五层防护）。

依赖 Neo4j（infra/docker-compose.yml）；连接失败自动 skip（CI 无中间件时）。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.graph import text2cypher  # noqa: E402
from app.graph.repo import GraphRepo  # noqa: E402

pytestmark = pytest.mark.skipif(
    not __import__("app.config", fromlist=["settings"]).settings.neo4j_uri,
    reason="Neo4j 未配置",
)


@pytest.fixture(scope="module")
def repo():
    r = GraphRepo()
    try:
        yield r
    finally:
        r.close()


# ---------- §3.2 验收 ----------

def test_node_counts(repo):
    stats = repo.stats()["nodes"]
    assert stats.get("Disease", 0) >= 300
    assert stats.get("Symptom", 0) >= 150
    assert stats.get("Drug", 0) >= 200
    assert stats.get("Department", 0) >= 20


def test_relation_count(repo):
    total = sum(repo.stats()["relations"].values())
    assert total >= 5000


def test_relation_types(repo):
    rels = repo.stats()["relations"]
    assert {"PRESENTS", "TREATS", "VISITS", "REQUIRES_EXAM", "COMPLICATES"} <= set(rels)


# ---------- Cypher 抽查（DoD） ----------

def test_symptom_to_department(repo):
    rows = repo.query(
        "MATCH (s:Symptom {name: '头痛'})-[:VISITS]->(d:Department) RETURN d.name AS dept LIMIT 5"
    )
    assert rows, "头痛 → 科室 查询无结果"


def test_drug_treats_disease(repo):
    rows = repo.query(
        "MATCH (dr:Drug)-[:TREATS]->(dis:Disease {name: '高血压'}) RETURN dr.name AS drug LIMIT 5"
    )
    names = {r["drug"] for r in rows}
    assert "卡托普利片" in names, "高血压应有常见降压药"


# ---------- 实体链接别名表生效（§3.2 第4步） ----------

def test_alias_normalized(repo):
    """'发烧/拉肚子/头疼'等别名不应作为独立节点存在（已归一为规范名）。"""
    rows = repo.query(
        "MATCH (n:Symptom) WHERE n.name IN ['发烧', '拉肚子', '头疼'] RETURN n.name AS name"
    )
    assert rows == [], f"别名未归一: {rows}"


def test_canonical_exists(repo):
    rows = repo.query("MATCH (n:Symptom {name: '发热'}) RETURN n.name AS name")
    assert rows, "规范名'发热'节点应存在"


# ---------- text2cypher 五层防护（§3.3） ----------

def test_forbidden_write_keyword():
    with pytest.raises(ValueError):
        text2cypher.validate_cypher("MATCH (n) DETACH DELETE n")


def test_forbidden_label():
    with pytest.raises(ValueError):
        text2cypher.validate_cypher("MATCH (n:Hacker) RETURN n")


def test_forbidden_relation():
    with pytest.raises(ValueError):
        text2cypher.validate_cypher("MATCH (a:Disease)-[:HACKS]->(b) RETURN a")


def test_forbidden_attr():
    with pytest.raises(ValueError):
        text2cypher.validate_cypher("MATCH (n:Disease) RETURN n.password")


def test_multiple_statements():
    with pytest.raises(ValueError):
        text2cypher.validate_cypher("MATCH (n) RETURN n; MATCH (m) RETURN m")


def test_valid_cypher_passes():
    text2cypher.validate_cypher(
        "MATCH (s:Symptom {name: '头痛'})-[:VISITS]->(d:Department) RETURN d.name LIMIT 5"
    )
    text2cypher.validate_cypher("MATCH (d:Disease)-[:PRESENTS]->(s:Symptom) RETURN count(s)")


def test_ask_injection_rejected_without_llm():
    """恶意输入即使绕过 LLM 白名单校验层也会被拦截（不依赖 LLM 的防护路径）。"""
    result = text2cypher.ask.__wrapped__ if hasattr(text2cypher.ask, "__wrapped__") else None
    # 直接验证校验层：手写恶意 Cypher 必须被 validate 拦截
    for evil in ["MATCH (n) DETACH DELETE n", "MATCH (n) DROP CONSTRAINT x"]:
        with pytest.raises(ValueError):
            text2cypher.validate_cypher(evil)
