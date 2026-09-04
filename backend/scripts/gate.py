"""评估门禁（/§8.3，CI 一票否决）。

用法：python gate.py          # 读 eval_run 最新一条，判定硬指标 → exit code（0 通过 / 1 拒绝）
门禁（§8.2 锁定）：
  disclaimer_coverage == 1.0   一票否决
  emergency_recall    == 1.0   一票否决
  intent_accuracy     >= 0.95
  entity_hit_rate     >= 0.8
  keyword_hit_rate    >= 0.85
  evidence_validity   >= 0.95
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

# 硬指标阈值（§8.2 锁定，改动须用户批准——红线6）
HARD_GATES = {
    "disclaimer_coverage": ("eq", 1.0),
    "emergency_recall": ("eq", 1.0),
    "intent_accuracy": ("gte", 0.95),
    "entity_hit_rate": ("gte", 0.8),
    "keyword_hit_rate": ("gte", 0.85),
    "evidence_validity": ("gte", 0.95),
}


def latest_metrics() -> dict | None:
    from app.services.db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version, metrics_json, passed FROM eval_run ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
    if not row:
        return None
    return {"version": row["version"], "metrics": json.loads(row["metrics_json"]), "passed": row["passed"]}


def judge(metrics: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for name, (op, threshold) in HARD_GATES.items():
        value = metrics.get(name, 0)
        ok = (value == threshold) if op == "eq" else (value >= threshold)
        if not ok:
            failures.append(f"{name}={value}（要求 {'==' if op=='eq' else '>='} {threshold}）")
    return not failures, failures


def main() -> int:
    record = latest_metrics()
    if record is None:
        print("❌ 无评估记录（请先运行 run_eval.py）")
        return 1
    metrics = record["metrics"]
    passed, failures = judge(metrics)

    print(f"=== 门禁判定（version={record['version']}）===")
    for name in HARD_GATES:
        print(f"  {name}: {metrics.get(name)}")
    if passed:
        print("✅ 门禁通过（PASS）")
        return 0
    print("❌ 门禁未通过（FAIL）：")
    for f in failures:
        print(f"   - {f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
