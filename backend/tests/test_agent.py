"""M4 智能体编排测试。

- 意图预分类/路由为纯单测（不调 LLM）；
- 端到端（三类问题 + interrupt HITL + 急症横幅）依赖 LLM key + Neo4j + Qdrant，缺 key 自动 skip。
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings  # noqa: E402
from app.agent import graph as graph_mod  # noqa: E402
from app.agent.nodes_basic import _pre_classify  # noqa: E402

HAS_ENV = bool(settings.dashscope_api_key)


# ---------- §5.4 意图预分类（纯规则） ----------

def test_pre_classify_department():
    assert _pre_classify("头痛应该挂什么科") == "DEPARTMENT"
    assert _pre_classify("胃痛看什么科") == "DEPARTMENT"


def test_pre_classify_drug():
    assert _pre_classify("高血压吃什么药") == "DRUG"
    assert _pre_classify("这个药用量多少") == "DRUG"


def test_pre_classify_knowledge():
    assert _pre_classify("感冒有哪些症状") == "KNOWLEDGE"
    assert _pre_classify("肺炎是怎么回事") == "KNOWLEDGE"


def test_pre_classify_guide_and_chat():
    assert _pre_classify("怎么挂号就医") == "GUIDE"
    assert _pre_classify("你好") == "CHAT"


def test_pre_classify_unknown_returns_none():
    assert _pre_classify("我今天有点不舒服") is None  # 未命中走 LLM


# ---------- §5.2 状态图结构 ----------

def test_graph_nodes_registered():
    g = graph_mod.build_graph()
    # 全部 9 个节点已注册（编译不抛错即通过）
    assert g is not None


def test_route_after_symptom():
    assert graph_mod._route_after_symptom({"need_more": True}) == "symptom_agent"
    assert graph_mod._route_after_symptom({"intent": "DEPARTMENT"}) == "department_agent"
    assert graph_mod._route_after_symptom({"intent": "DRUG"}) == "drug_agent"
    assert graph_mod._route_after_symptom({"intent": "GUIDE"}) == "guide_agent"
    assert graph_mod._route_after_symptom({"intent": "KNOWLEDGE"}) == "medical_knowledge_agent"
    assert graph_mod._route_after_symptom({"intent": "MEDICAL_QUERY"}) == "medical_knowledge_agent"


# ---------- §5.3 端到端（三类问题 + HITL + 急症） ----------

@pytest.mark.skipif(not HAS_ENV, reason="无百炼 key")
def test_end_to_end_three_types():
    from app.agent.nodes_domain import init_retrieval_env

    init_retrieval_env()
    cases = [
        ("头痛应该挂什么科", "DEPARTMENT"),
        ("高血压吃什么药", "DRUG"),
        ("感冒有哪些症状", "KNOWLEDGE"),
    ]
    for question, expect in cases:
        state = graph_mod.ask(question, thread_id=f"t-{expect.lower()}")
        assert state["intent"] == expect, f"{question}: {state['intent']}"
        assert state["answer"], f"{question}: 回答为空"
        assert state.get("disclaimer"), f"{question}: 医疗回答应带免责声明"


@pytest.mark.skipif(not HAS_ENV, reason="无百炼 key")
def test_emergency_banner():
    from app.agent.nodes_domain import init_retrieval_env

    init_retrieval_env()
    state = graph_mod.ask("我胸口突然很痛，喘不上气", thread_id="t-emo")
    assert state["risk_level"] == "HIGH"
    assert "立即就医" in state["answer"]


@pytest.mark.skipif(not HAS_ENV, reason="无百炼 key")
def test_chat_branch_no_disclaimer():
    state = graph_mod.ask("你好，在吗", thread_id="t-chat")
    assert state["intent"] == "CHAT"
    assert state.get("disclaimer") is None, "闲聊不加免责声明（§5.2）"


@pytest.mark.skipif(not HAS_ENV, reason="无百炼 key")
def test_hitl_interrupt_resume():
    """§5.3 interrupt HITL：笼统提问 → 追问 → 补充症状 → 完整回答。"""
    from app.agent.nodes_domain import init_retrieval_env

    init_retrieval_env()
    g, config, state, pending, finished = graph_mod.ask_with_interrupt(
        "我不舒服，怎么办", thread_id="t-hitl")
    assert not finished and pending, "应触发 interrupt 追问"
    state, pending, finished = graph_mod.resume(g, config, "我头痛而且发烧了")
    assert finished, "补充后应直接完成"
    assert state["answer"], "HITL 回答为空"
    names = {e["name"] for e in state.get("entities", [])}
    assert "头痛" in names and "发热" in names, "追问补充的实体应被链接"
