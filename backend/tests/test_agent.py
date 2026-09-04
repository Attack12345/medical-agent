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


# ---------- 无 LLM 环境兜底（M8.16：CI/离线不把"LLM 失败"升格为"拒答"） ----------

def test_draft_llm_unavailable_falls_back_to_pool_summary(monkeypatch):
    """LLM 不可用 + 证据池非空 → 确定性摘 要回答（拒绝=False，实体名在答案内，箭头不外泄）。"""
    from app.agent import nodes_domain

    def boom(*a, **k):
        raise RuntimeError("no api key")

    monkeypatch.setattr(nodes_domain, "chat_json", boom)
    state = {"question": "什么是糖尿病", "collected_text": "什么是糖尿病",
             "entities": [{"name": "2型糖尿病", "label": "Disease"}],
             "evidence_pool": [
                 {"type": "GRAPH_NODE", "quote": "2型糖尿病 → 多饮", "score": 1.0},
                 {"type": "RETRIEVAL", "quote": "2型糖尿病建议：就诊内分泌科，控制主食量。", "score": 1.0}]}
    out = nodes_domain._draft(state, "p")
    assert out["refusal"] is False
    assert "2型糖尿病" in out["answer"] and "多饮" in out["answer"]
    assert "→" not in out["answer"]
    assert not out["answer"].endswith("。。")


def test_draft_llm_unavailable_high_risk_still_refuses(monkeypatch):
    """LLM 不可用 + 高风险问题 + 证据池空 → 仍拒答（S004 语义不放松）。"""
    from app.agent import nodes_domain

    def boom(*a, **k):
        raise RuntimeError("no api key")

    monkeypatch.setattr(nodes_domain, "chat_json", boom)
    out = nodes_domain._draft({"question": "药物过量怎么办", "entities": [], "evidence_pool": []}, "p")
    assert out["refusal"] is True


def test_fusion_invalidated_empty_pool_keeps_honest_answer(monkeypatch):
    """S101 重生成失败 + 证据池为空 → 保留分支节点如实说明，不升格为拒答。"""
    from app.agent import nodes_safety

    monkeypatch.setattr(nodes_safety, "chat_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
    st = {"answer": "根据知识库信息，暂未检索到相关就医指导，建议前往医院咨询。",
          "evidence_pool": [], "evidence_quotes": [], "invalidated": True,
          "risk_level": "NONE", "disclaimer": nodes_safety.DISCLAIMER_TEXT, "entities": []}
    out = nodes_safety.fusion_agent(st)
    assert "暂未检索到" in out["answer"]
    assert nodes_safety.REFUSAL_TEXT not in out["answer"]


def test_fusion_invalidated_with_pool_still_refuses(monkeypatch):
    """S101 重生成失败 + 有证据池 → 仍拒答（防无引用内容外泄，原语义保留）。"""
    from app.agent import nodes_safety

    monkeypatch.setattr(nodes_safety, "chat_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no key")))
    st = {"answer": "某个无引用回答", "evidence_pool": [{"type": "RETRIEVAL", "quote": "x"}],
          "evidence_quotes": [], "invalidated": True,
          "risk_level": "NONE", "disclaimer": "", "entities": []}
    out = nodes_safety.fusion_agent(st)
    assert out["answer"] == nodes_safety.REFUSAL_TEXT


# ---------- 多轮否定剔除（M8.18："头不疼了，手疼"不得再按头痛给建议） ----------

def test_negated_fragments_extraction():
    from app.agent.nodes_basic import _negated_fragments
    assert "头" in _negated_fragments("头不疼了，手疼")
    assert "热" in _negated_fragments("发热退了，现在咳嗽")
    assert _negated_fragments("应该吃什么药") == ""


def test_carry_forward_drops_negated_entities(monkeypatch):
    """上轮头痛+头晕，本轮"头不疼了，手疼"且链接为空 → 全部剔除，不追问、不继承。"""
    from app.agent import nodes_basic as nb

    monkeypatch.setattr(nb, "get_linker",
                        lambda: type("L", (), {"link_text": lambda self, text, query_vector=None: []})())
    state = {"question": "头不疼了，手疼", "collected_text": "头不疼了，手疼",
             "entities": [{"name": "头痛", "label": "Symptom"}, {"name": "头晕", "label": "Symptom"}],
             "follow_up_left": 2}
    out = nb.symptom_agent(state)
    assert out["entities"] == [] and out["need_more"] is False
    # 对照：无否定线索时正常继承
    state2 = dict(state, question="应该吃什么药", collected_text="应该吃什么药")
    out2 = nb.symptom_agent(state2)
    assert [e["name"] for e in out2["entities"]] == ["头痛", "头晕"]


def test_fusion_neutral_refusal_for_non_s004():
    """非 S004 的 grounded 拒答用中性话术+行动指导，不再套用药安全措辞。"""
    from app.agent import nodes_safety

    st = {"refusal": True, "risk_level": "NONE", "evidence_quotes": [],
          "safety_trail": [{"rule_id": "S102", "hit": False}], "answer": "", "disclaimer": ""}
    out = nodes_safety.fusion_agent(st)
    assert out["answer"] == nodes_safety.NEUTRAL_REFUSAL_TEXT
    st2 = dict(st, safety_trail=[{"rule_id": "S004", "hit": True}])
    assert nodes_safety.fusion_agent(st2)["answer"] == nodes_safety.REFUSAL_TEXT
