"""Neo4j 访问封装。

职责：连接管理 / 只读查询 / 写操作 / 统计。建图管线编排在 scripts/build_graph.py。
"""
from __future__ import annotations

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from app.config import settings


class GraphRepo:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def execute(self, cypher: str, params: dict | None = None) -> None:
        """写操作（建图/清空）。"""
        with self.driver.session() as session:
            session.run(cypher, params or {})

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """只读查询，返回 dict 列表（键与 Cypher 返回列一致）。"""
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def clear(self) -> None:
        """幂等重建第一步：清空图谱。"""
        self.execute("MATCH (n) DETACH DELETE n")

    def stats(self) -> dict[str, dict]:
        """节点/关系统计（按 label/type 计数）。"""
        nodes = {row["label"]: row["n"]
                 for row in self.query("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS n")}
        rels = {row["type"]: row["n"]
                for row in self.query("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n")}
        return {"nodes": nodes, "relations": rels}

    def is_ready(self) -> bool:
        try:
            return bool(self.query("RETURN 1 AS ok"))
        except Neo4jError:
            return False
