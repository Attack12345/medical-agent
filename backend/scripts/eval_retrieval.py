"""检索评估：30 条评估集 → 实体链接准确率 + 关系命中率 → 基线落盘。

用法：python eval_retrieval.py [--json data/retrieval_eval.json]
指标（§4.3）：
  entity_accuracy  = mean(命中实体数 / expected 实体数)   目标 ≥0.8
  relation_accuracy = 非 null 用例中 expected_relation 在图谱候选出现的比例
  evidence_coverage = retrieve 返回 evidence_pool 非空比例
输出：data/retrieval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.graph.repo import GraphRepo  # noqa: E402
from app.retrieval.link import Linker  # noqa: E402
from app.retrieval.pipeline import init_bm25, retrieve  # noqa: E402
from app.retrieval.vector_db import VectorDb  # noqa: E402

EVAL_FILE = PROJECT_ROOT / "data" / "retrieval_eval.json"
BASELINE_FILE = PROJECT_ROOT / "data" / "retrieval_baseline.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="检索评估")
    parser.add_argument("--json", default=str(EVAL_FILE), help="评估集路径")
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # 数据装载（与 build_vector_db 同源）
    with open(PROJECT_ROOT / "data" / "cleaned" / "diseases.json", "r", encoding="utf-8") as f:
        diseases = json.load(f)
    with open(PROJECT_ROOT / "data" / "cleaned" / "qa_pairs.json", "r", encoding="utf-8") as f:
        qa = json.load(f)
    repo = GraphRepo()
    db = VectorDb()
    linker = Linker()
    try:
        # BM25 索引（entities 用疾病名；qa 用问题）
        init_bm25([d["name"] for d in diseases], [q["question"] for q in qa])

        per_case: list[dict] = []
        entity_hits_total = 0.0
        relation_hits_total = 0.0
        relation_cases = 0
        covered = 0

        for i, case in enumerate(cases, 1):
            result = retrieve(case["query"], linker=linker, db=db, repo=repo)
            linked = {name for name, _label, _conf in result["entities"]}
            expected = set(case["expected_entities"])
            if expected:
                hit = len(expected & linked) / len(expected)
                entity_hits_total += hit
            else:
                hit = None  # 无 expected 不参与实体指标
            rel = case.get("expected_relation")
            rel_hit = None
            if rel:
                relation_cases += 1
                rel_hit = rel in {g["relation"] for g in result["graph_evidence"]}
                relation_hits_total += 1.0 if rel_hit else 0.0
            if result["evidence_pool"]:
                covered += 1
            per_case.append({
                "query": case["query"], "expected_entities": case["expected_entities"],
                "expected_relation": rel, "entity_hit": hit, "relation_hit": rel_hit,
                "linked": sorted(linked), "graph_relations": sorted({g["relation"] for g in result["graph_evidence"]}),
            })
            status = "PASS" if (hit is None or hit >= 1.0) else ("PART" if hit and hit > 0 else "FAIL")
            print(f"[{i:02d}] {status} {case['query']} | entity_hit={hit} relation_hit={rel_hit}", flush=True)

        entity_accuracy = entity_hits_total / max(1, sum(1 for c in cases if c["expected_entities"]))
        relation_accuracy = relation_hits_total / max(1, relation_cases)
        evidence_coverage = covered / len(cases)
        metrics = {
            "entity_accuracy": round(entity_accuracy, 4),
            "relation_accuracy": round(relation_accuracy, 4),
            "evidence_coverage": round(evidence_coverage, 4),
            "targets": {"entity_accuracy": 0.8},
        }
        passed = entity_accuracy >= 0.8
        baseline = {
            "metrics": metrics,
            "per_case": per_case,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
        }
        BASELINE_FILE.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== 基线（§4.3）===")
        print(f"实体链接准确率: {metrics['entity_accuracy']}（目标 ≥0.8）")
        print(f"关系命中率: {metrics['relation_accuracy']}")
        print(f"证据覆盖率: {metrics['evidence_coverage']}")
        print(f"{'✅ 达标' if passed else '❌ 未达标'} → {BASELINE_FILE.name}")
        return 0 if passed else 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())
