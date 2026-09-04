"""实体链接（第1路）：问句 → 图谱实体（疾病/症状/药物/科室）。

- 确定性优先：问句粗切分 → 实体名/别名精确匹配（confidence 0.95/0.7）。
- 语义兜底：未命中且给了向量走语义相似度 ≥0.85 才建链（§4.2 锁定）。
- 实体词表从 Neo4j 加载（含 medical_aliases.yaml 别名），进程内缓存。
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

SEMANTIC_THRESHOLD = 0.85  # §4.2 锁定

# 同名实体跨标签冲突时的路由优先级（M8.2）：症状主诉（"我头疼"）应路由到 Symptom 走分诊，
# 而非 Disease 走"疾病亚型罗列"。数字越小越优先。
_LABEL_PREFERENCE = {"Symptom": 0, "Drug": 1, "Department": 2, "Disease": 3}


class Linker:
    def __init__(self, entities: list[dict] | None = None,
                 entity_vectors: dict[str, list[float]] | None = None):
        """entities: [{name, label, aliases[]}]；缺省从图谱加载。
        entity_vectors: {实体名: 向量}（Qdrant symptoms 集合构造），语义匹配兜底用
        （口语描述如"心跳很快"精确/别名匹配不上时，语义相似 ≥0.85 建链，§4.2）。"""
        self.entities = entities if entities is not None else self._load_entities()
        self.vectors: dict[str, list[float]] = entity_vectors or {}
        self.name_index: dict[str, str] = {}      # 实体名 → (name, label)
        self.alias_index: dict[str, tuple[str, str]] = {}  # 别名 → (name, label)
        self._build_index()
        self._build_vector_matrix()

    def _build_vector_matrix(self) -> None:
        """语义向量矩阵化（预归一化），查询一次点积完成全量余弦（5000+ 症状零延迟）。"""
        import numpy as np

        self._vec_names: list[str] = []
        rows: list[list[float]] = []
        for name, vec in self.vectors.items():
            if name not in self.name_index:
                continue  # 语义命中的名字必须能映射到图谱标签
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm <= 0:
                continue
            rows.append(arr / norm)
            self._vec_names.append(name)
        if rows:
            self._vec_matrix = np.vstack(rows)
        else:
            self._vec_matrix = None

    def _build_index(self) -> None:
        # 同名多标签（如"头痛"既是 Disease 又是 Symptom）时按路由优先级选标签，
        # 避免 dict 覆盖导致非确定性地落到 Disease（症状主诉误走疾病亚型罗列）。
        best_label: dict[str, str] = {}
        for e in self.entities:
            name, label = e["name"], e["label"]
            cur = best_label.get(name)
            if cur is None or _LABEL_PREFERENCE.get(label, 9) < _LABEL_PREFERENCE.get(cur, 9):
                best_label[name] = label
        for e in self.entities:
            name = e["name"]
            label = best_label[name]
            self.name_index[name] = (name, label)
            for a in e.get("aliases", []):
                self.alias_index.setdefault(a, (name, label))

    @staticmethod
    def _load_entities() -> list[dict]:
        """从 Neo4j 加载全部实体（名称 + 别名表扩充）。"""
        from app.graph.repo import GraphRepo

        repo = GraphRepo()
        try:
            rows = repo.query(
                """
                MATCH (n)
                WHERE any(l IN labels(n) WHERE l IN ['Disease', 'Symptom', 'Drug', 'Department'])
                RETURN labels(n)[0] AS label, n.name AS name, n.aliases AS aliases
                """
            )
        finally:
            repo.close()

        alias_extras: dict[str, list[str]] = {}
        alias_file = Path(__file__).resolve().parents[1] / "graph/medical_aliases.yaml"
        if alias_file.exists():
            raw = yaml.safe_load(alias_file.read_text(encoding="utf-8")) or {}
            for key, mapping in raw.items():
                for canonical, aliases in (mapping or {}).items():
                    alias_extras.setdefault(canonical, []).extend(aliases)

        entities = []
        for r in rows:
            label = r["label"]
            aliases = list(r.get("aliases") or [])
            aliases += alias_extras.get(r["name"], [])
            entities.append({"name": r["name"], "label": label, "aliases": aliases})
        return entities

    def link_text(self, text: str, query_vector: list[float] | None = None) -> list[tuple[str, str, float]]:
        """链接文本中的实体：先名称/别名精确匹配；再词表子串扫描（无标点长句）；
        未命中且给了向量走语义匹配（≥0.85）。

        返回 [(实体名, label, confidence)]，按 confidence 降序去重。
        """
        hits: dict[str, tuple[str, float]] = {}  # 实体名 → (label, confidence)
        # 1) 切分词 token 精确匹配
        for token in _split_terms(text):
            if token in self.name_index:
                name, label = self.name_index[token]
                hits[name] = (label, max(hits.get(name, (label, 0))[1], 0.95))
            elif token in self.alias_index:
                name, label = self.alias_index[token]
                hits[name] = (label, max(hits.get(name, (label, 0))[1], 0.7))
        # 2) 词表子串扫描（"头痛应该挂什么科"无标点也能命中）
        for name, (canonical, label) in self.name_index.items():
            if len(name) >= 2 and name in text and name not in hits:
                hits[canonical] = (label, 0.95)
        for alias, (name, label) in self.alias_index.items():
            if len(alias) >= 2 and alias in text and name not in hits:
                hits[name] = (label, max(hits.get(name, (label, 0))[1], 0.7))
        # 3) 语义兜底（§4.2）：口语描述精确/别名匹配不上时，余弦 ≥0.85 建链。
        #    矩阵化一次点积（预归一化），5000+ 症状零延迟。
        if query_vector is not None and self._vec_matrix is not None:
            import numpy as np

            qv = np.asarray(query_vector, dtype=np.float32)
            qnorm = float(np.linalg.norm(qv))
            if qnorm > 0:
                sims = self._vec_matrix @ (qv / qnorm)  # 全量余弦
                for i in np.nonzero(sims >= SEMANTIC_THRESHOLD)[0]:
                    name = self._vec_names[int(i)]
                    if name in hits:
                        continue
                    label = self.name_index.get(name, ("", ""))[1]
                    if not label:
                        continue
                    hits[name] = (label, max(hits.get(name, (label, 0))[1], float(sims[i])))
        return [(name, label, conf) for name, (label, conf) in
                sorted(hits.items(), key=lambda kv: -kv[1][1])]


def _split_terms(text: str) -> list[str]:
    """粗切分：按标点/空白切 + 长度过滤（≥2 字，实体名最短 2 字）。"""
    return [t for t in re.split(r"[，。！？、,.;:()\s]+", text) if len(t) >= 2]
