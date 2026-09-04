"""rerank 有无提升对照实验（M8.8，补充实验）。

对照：
  A（基线）= 现主链路：四路召回 + RRF 融合 Top5
  B（实验）= RRF Top20 候选 → bge-reranker 精排 Top5
口径：检索评估集（data/retrieval_eval.json 30 条），
  实体命中率 = expected_entities 出现在证据池（quote/ref/text）中的比例（每条求均值）。
输出：docs/rerank_experiment.md（结论：有提升→接入主链路；无提升→证伪记录）。

用法：python scripts/experiment_rerank.py [--top-candidates 20]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.agent.nodes_domain import init_retrieval_env, get_retrieval_env  # noqa: E402
from app.retrieval.pipeline import retrieve  # noqa: E402
from app.retrieval.rerank import rerank  # noqa: E402

EVAL_FILE = PROJECT_ROOT / "data" / "retrieval_eval.json"
REPORT_FILE = PROJECT_ROOT / "docs" / "rerank_experiment.md"


def pool_text(pool: list[dict]) -> str:
    return " ".join(f"{p.get('quote','')} {p.get('ref','')} {p.get('text','')}" for p in pool)


def entity_hit(pool: list[dict], expected: list[str]) -> float:
    if not expected:
        return 1.0
    text = pool_text(pool)
    return sum(1 for e in expected if e in text) / len(expected)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--top-candidates", type=int, default=20)
    args = parser.parse_args()

    cases = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    init_retrieval_env()
    env = get_retrieval_env()

    rows = []
    base_sum = rr_sum = 0.0
    base_hit_n = rr_hit_n = 0
    base_ms = rr_ms = 0.0
    changed = 0

    for i, case in enumerate(cases, 1):
        q = case["query"]
        expected = case.get("expected_entities", [])

        t0 = time.time()
        pool_a = retrieve(q, linker=env["linker"], db=env["db"], repo=env["repo"],
                          top_n=5)["evidence_pool"]
        base_ms += (time.time() - t0) * 1000

        t0 = time.time()
        pool_c = retrieve(q, linker=env["linker"], db=env["db"], repo=env["repo"],
                          top_n=args.top_candidates)["evidence_pool"]
        pool_b = rerank(q, pool_c, top_n=5)
        rr_ms += (time.time() - t0) * 1000

        h_a = entity_hit(pool_a, expected)
        h_b = entity_hit(pool_b, expected)
        base_sum += h_a
        rr_sum += h_b
        if h_a >= 1.0:
            base_hit_n += 1
        if h_b >= 1.0:
            rr_hit_n += 1
        if [p.get("ref") for p in pool_a] != [p.get("ref") for p in pool_b]:
            changed += 1
        rows.append((q, h_a, h_b, len(pool_a), len(pool_b)))
        print(f"[{i:02d}] base={h_a:.2f} rerank={h_b:.2f}  {q}", flush=True)

    n = len(cases)
    base_avg, rr_avg = base_sum / n, rr_sum / n
    delta = rr_avg - base_avg

    lines = [
        "# rerank 有无提升对照实验（M8.8）",
        "",
        f"- 模型：BAAI/bge-reranker-base（CPU）",
        f"- 候选池：RRF Top{args.top_candidates} → 精排 Top5；基线：RRF Top5",
        f"- 评估集：retrieval_eval.json（{n} 条）",
        "",
        "| 指标 | 基线 RRF Top5 | RRF Top20 + rerank | Δ |",
        "|---|---|---|---|",
        f"| 实体命中率（均分） | {base_avg:.4f} | {rr_avg:.4f} | {delta:+.4f} |",
        f"| 实体全命中率（用例数） | {base_hit_n}/{n} | {rr_hit_n}/{n} | {rr_hit_n - base_hit_n:+d} |",
        f"| 平均延迟 ms | {base_ms / n:.0f} | {rr_ms / n:.0f} | +{(rr_ms - base_ms) / n:.0f} |",
        f"| Top5 集合发生变化的比例 | - | {changed}/{n} | - |",
        "",
        "## 逐条",
        "",
        "| query | base | rerank |",
        "|---|---|---|",
    ]
    for q, h_a, h_b, _la, _lb in rows:
        lines.append(f"| {q} | {h_a:.2f} | {h_b:.2f} |")
    verdict = ("结论：rerank 有提升，接入主链路（配置开关）。"
               if delta > 0.02 else
               "结论：rerank 无实质提升（Δ≤0.02），不接入主链路，作为实验证伪记录。"
               if abs(delta) <= 0.02 else
               "结论：rerank 为负收益，不接入主链路，记录证伪。")
    lines += ["", verdict, ""]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"报告 → {REPORT_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
