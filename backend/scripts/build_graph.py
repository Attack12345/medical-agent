"""建图管线（幂等重建）。

用法：
  python build_graph.py            # 清空重建 + 统计
  python build_graph.py --dry-run  # 只统计当前图谱（不重建）

执行顺序（每步打印统计）：
  1. 清空图谱（MATCH (n) DETACH DELETE n）
  2. diseases.json → Disease 节点（含 severity/aliases/summary）
  3. relations.json → 其余节点 + 全部关系（MERGE 幂等，关系带 source）
  4. 实体链接归一：medical_aliases.yaml + entity_norm.json（别名→规范名）
  5. 打印统计（验收：Disease≥300 / Symptom≥150 / Drug≥200 / Department≥20 / 关系≥5000）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.graph.repo import GraphRepo  # noqa: E402

CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
ALIASES_FILE = PROJECT_ROOT / "backend" / "app" / "graph" / "medical_aliases.yaml"
NORM_FILE = PROJECT_ROOT / "data" / "entity_norm.json"

BATCH = 10_000  # 每批 UNWIND 行数（避免单事务过大）

# §3.1 关系 → (subject_label, object_label) 映射（CONTRAINDICATES 的 object 按数据出现动态处理）
REL_LABELS: dict[str, tuple[str, str]] = {
    "PRESENTS": ("Disease", "Symptom"),
    "TREATS": ("Drug", "Disease"),
    "VISITS": ("Symptom", "Department"),
    "DIAGNOSES": ("Exam", "Disease"),
    "AVOIDS_FOOD": ("Disease", "Food"),
    "AFFECTS": ("Disease", "Population"),
    "CONTRAINDICATES": ("Drug", "Disease"),
    "REQUIRES_EXAM": ("Disease", "Exam"),
    "COMPLICATES": ("Disease", "Disease"),
    "ACCOMPANIES": ("Symptom", "Symptom"),
    "ADMITS_TO": ("Disease", "Hospital"),
}


def log(msg: str) -> None:
    print(f"[build_graph] {msg}", flush=True)


def load_aliases() -> dict[str, dict[str, str]]:
    """别名表 → {type: {别名: 规范名}}，实体链接归一用。"""
    with open(ALIASES_FILE, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, dict[str, str]] = {}
    for etype, mapping in raw.items():
        alias_map: dict[str, str] = {}
        for canonical, aliases in (mapping or {}).items():
            for a in aliases:
                alias_map[a] = canonical
        out[etype] = alias_map
    return out


def load_norm() -> dict[str, str]:
    if NORM_FILE.exists():
        with open(NORM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_diseases(repo: GraphRepo, diseases: list[dict]) -> None:
    """步骤 2：Disease 节点（含 severity/aliases/summary）。"""
    for i in range(0, len(diseases), BATCH):
        rows = diseases[i:i + BATCH]
        repo.execute(
            """
            UNWIND $rows AS d
            MERGE (n:Disease {name: d.name})
            SET n.severity = d.severity,
                n.summary = coalesce(d.summary, ''),
                n.aliases = coalesce(d.aliases, [])
            """,
            {"rows": rows},
        )
    log(f"Disease 节点建入 {len(diseases)} 条")


def _normalize_entity(name: str, alias_map: dict[str, str], norm: dict[str, str]) -> str:
    """实体归一：medical_aliases.yaml（别名）→ entity_norm.json（同义词）→ 原样。"""
    return alias_map.get(name, norm.get(name, name))


# 节点 label → 别名表键（无别名表的 label 不归一）
LABEL_ALIAS_KEY = {
    "Disease": "diseases", "Symptom": "symptoms", "Drug": "drugs",
    "Department": "departments", "Exam": None, "Food": None, "Population": None, "Hospital": None,
}


def build_relations(repo: GraphRepo, relations: list[dict], alias_maps: dict[str, dict[str, str]], norm: dict[str, str]) -> None:
    """步骤 3+4：其余节点 + 关系（MERGE 幂等，source 属性），实体名先归一。"""
    grouped: dict[str, list[dict]] = {}
    for r in relations:
        rel = r["relation"]
        if rel not in REL_LABELS:
            continue
        sub_label, obj_label = REL_LABELS[rel]
        sub_map = alias_maps.get(LABEL_ALIAS_KEY[sub_label]) if LABEL_ALIAS_KEY[sub_label] else {}
        obj_map = alias_maps.get(LABEL_ALIAS_KEY[obj_label]) if LABEL_ALIAS_KEY[obj_label] else {}
        subject = _normalize_entity(r["subject"], sub_map, norm)
        obj = _normalize_entity(r["object"], obj_map, norm)
        grouped.setdefault(rel, []).append({"subject": subject, "object": obj, "source": r["source"]})

    for rel, rows in grouped.items():
        sub_label, obj_label = REL_LABELS[rel]
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i + BATCH]
            repo.execute(
                f"""
                UNWIND $rows AS r
                MERGE (a:{sub_label} {{name: r.subject}})
                MERGE (b:{obj_label} {{name: r.object}})
                MERGE (a)-[rel:{rel} {{source: r.source}}]->(b)
                """,
                {"rows": batch},
            )
        log(f"{rel}: {len(rows)} 条（{sub_label}→{obj_label}）")


def print_stats(repo: GraphRepo) -> None:
    stats = repo.stats()
    nodes, rels = stats["nodes"], stats["relations"]
    log("=== 图谱统计 ===")
    for label in ["Disease", "Symptom", "Drug", "Department", "Exam", "Food", "Population", "Hospital"]:
        log(f"  {label}: {nodes.get(label, 0)}")
    log(f"  关系合计: {sum(rels.values())} | 明细 {rels}")
    # §3.2 验收
    checks = [
        ("Disease", nodes.get("Disease", 0) >= 300),
        ("Symptom", nodes.get("Symptom", 0) >= 150),
        ("Drug", nodes.get("Drug", 0) >= 200),
        ("Department", nodes.get("Department", 0) >= 20),
        ("关系总数", sum(rels.values()) >= 5000),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        log(f"❌ 验收未达标: {failed}")
        sys.exit(1)
    log("✅ §3.2 验收全部达标")


def main() -> int:
    parser = argparse.ArgumentParser(description="建图管线")
    parser.add_argument("--dry-run", action="store_true", help="只统计当前图谱，不重建")
    args = parser.parse_args()

    repo = GraphRepo()
    try:
        if args.dry_run:
            print_stats(repo)
            return 0

        with open(CLEANED_DIR / "diseases.json", "r", encoding="utf-8") as f:
            diseases = json.load(f)
        with open(CLEANED_DIR / "relations.json", "r", encoding="utf-8") as f:
            relations = json.load(f)

        # 1. 清空
        repo.clear()
        log("图谱已清空")

        # 2. Disease 节点
        build_diseases(repo, diseases)

        # 3+4. 关系节点 + 实体链接归一
        alias_maps = load_aliases()
        norm = load_norm()
        build_relations(repo, relations, alias_maps, norm)

        # 5. 统计
        print_stats(repo)
        return 0
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
