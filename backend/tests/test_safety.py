"""M5 安全层测试：六条规则命中/短路 + 恶意用例拦截 + 引用提取。

规则层为纯单测（不调 LLM）；端到端急症横幅由 test_agent.py 覆盖。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.agent.nodes_safety import extract_citations  # noqa: E402
from app.engine.dsl import load_rules  # noqa: E402
from app.engine.executor import execute  # noqa: E402
from app.engine.parser import compile_rules  # noqa: E402
from app.engine.trail import build_context  # noqa: E402

RULES = compile_rules(load_rules(Path(__file__).resolve().parents[2] / "backend" / "app" / "engine" / "safety_rules.yaml"))


def run(intent="KNOWLEDGE", entities=None, answer="", pool=None, quotes=None,
        high_risk=False, disclaimer_added=False, risk_level="NONE"):
    ctx = build_context({
        "intent": intent,
        "entities": entities or [],
        "answer": answer,
        "evidence_pool": pool or [],
        "evidence_quotes": quotes or [],
        "high_risk_query": high_risk,
        "disclaimer_added": disclaimer_added,
        "risk_level": risk_level,
    })
    return execute(ctx, RULES)


def actions(r):
    return set(r.actions)


# ---------- S001 免责声明 ----------

def test_s001_medical_requires_disclaimer():
    r = run(intent="DEPARTMENT", answer="根据知识库，建议就诊神经内科。")
    assert "REQUIRE_DISCLAIMER" in actions(r)


def test_s001_skipped_when_added():
    r = run(intent="DEPARTMENT", answer="x", disclaimer_added=True)
    assert "REQUIRE_DISCLAIMER" not in actions(r)


def test_s001_not_for_chat():
    r = run(intent="CHAT", answer="你好呀")
    assert "REQUIRE_DISCLAIMER" not in actions(r)


# ---------- S002 急症置顶 ----------

def test_s002_emergency_high_risk():
    r = run(entities=[{"name": "胸痛", "label": "Symptom", "confidence": 0.95}], answer="x")
    assert "SET_HIGH_RISK" in actions(r)


def test_s002_any_item_match():
    r = run(entities=[{"name": "感冒", "label": "Disease", "confidence": 0.9},
                      {"name": "呼吸困难", "label": "Symptom", "confidence": 0.9}], answer="x")
    assert "SET_HIGH_RISK" in actions(r)


def test_s002_no_emergency():
    r = run(entities=[{"name": "感冒", "label": "Disease", "confidence": 0.9}], answer="x")
    assert "SET_HIGH_RISK" not in actions(r)


# ---------- S003 用药提醒 ----------

def test_s003_drug_notice():
    r = run(intent="DRUG", answer="可以吃阿司匹林。")
    assert "REQUIRE_DRUG_NOTICE" in actions(r)


def test_s003_not_for_knowledge():
    r = run(intent="KNOWLEDGE", answer="感冒多喝水。")
    assert "REQUIRE_DRUG_NOTICE" not in actions(r)


# ---------- S004 高风险检索落空拒答 ----------

def test_s004_refuse_when_no_evidence():
    r = run(high_risk=True, pool=[], answer="")
    assert "REFUSE" in actions(r)


def test_s004_not_refuse_with_evidence():
    r = run(high_risk=True, pool=[{"text": "阿司匹林 高血压"}], answer="x")
    assert "REFUSE" not in actions(r)


# ---------- S101 无证据回答无效（恶意用例） ----------

def test_s101_answer_without_citation_invalid():
    """恶意用例：回答声称来自知识库但无任何引用 → INVALIDATE_ANSWER。"""
    r = run(answer="根据知识库信息，建议立即服用某种特效药。")
    assert "INVALIDATE_ANSWER" in actions(r)


def test_s101_with_citation_ok():
    r = run(answer="根据知识库，头痛建议就诊神经内科。", quotes=["头痛"],
            pool=[{"text": "头痛 神经内科"}])
    assert "INVALIDATE_ANSWER" not in actions(r)


# ---------- S102 证据引用造假（恶意用例） ----------

def test_s102_out_of_pool_citation_invalid():
    """恶意用例：回答引用池外实体（'肺癌'不在证据池）→ INVALIDATE_ANSWER。"""
    r = run(answer="根据知识库，肺癌建议手术切除。", quotes=["肺癌"],
            pool=[{"text": "感冒 发热"}])
    assert "INVALIDATE_ANSWER" in actions(r)


def test_s102_in_pool_citation_ok():
    r = run(answer="根据知识库，感冒建议就诊呼吸科。", quotes=["感冒"],
            pool=[{"text": "感冒 发热"}])
    assert "INVALIDATE_ANSWER" not in actions(r)


# ---------- §6.2 执行语义 ----------

def test_combination_s001_s002_s003():
    """组合式要求：免责声明命中不短路急症/用药提醒（v1.2 同 action 短路语义）。"""
    r = run(intent="DRUG", entities=[{"name": "胸痛", "label": "Symptom", "confidence": 1.0}],
            answer="x")
    assert {"REQUIRE_DISCLAIMER", "SET_HIGH_RISK", "REQUIRE_DRUG_NOTICE"} <= actions(r)


def test_priority_order_veto_before_validate():
    """priority 降序：S004(85) 在 S101(60) 之前执行（轨迹顺序断言）。"""
    r = run(high_risk=True, pool=[], answer="")
    order = [t["rule_id"] for t in r.trails]
    assert order.index("S004") < order.index("S101")


# ---------- §5.3 引用提取 ----------

def test_extract_citations_from_answer():
    pool = [{"quote": "头痛 神经内科", "type": "GRAPH_NODE", "ref": "VISITS:头痛→神经内科", "score": 1.0}]
    graph_ev = [{"subject": "头痛", "relation": "VISITS", "object": "神经内科", "source": "graph"}]
    quotes = extract_citations("根据知识库，头痛建议就诊神经内科。", pool, graph_ev,
                               [{"name": "头痛", "label": "Symptom", "confidence": 0.95}])
    assert "头痛" in quotes and "神经内科" in quotes


def test_extract_citations_empty_when_no_match():
    pool = [{"quote": "高血压 卡托普利片", "type": "GRAPH_NODE", "ref": "TREATS:卡托普利片→高血压", "score": 1.0}]
    graph_ev = [{"subject": "卡托普利片", "relation": "TREATS", "object": "高血压", "source": "graph"}]
    quotes = extract_citations("今天天气不错。", pool, graph_ev, [])
    assert quotes == []
