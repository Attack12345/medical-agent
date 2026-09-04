"""LangGraph 状态图组装（锁定）。

图结构：
START → intent_agent
intent_agent → CHAT → chat_agent → END
intent_agent → 医疗意图 → symptom_agent（实体链接，含问诊 HITL 追问）
symptom_agent ──need_more──→ symptom_agent（interrupt 追问后重入）
symptom_agent → 条件路由（按 intent）:
  DEPARTMENT → department_agent
  DRUG → drug_agent
  KNOWLEDGE / MEDICAL_QUERY → medical_knowledge_agent（grounded）
  GUIDE → guide_agent
每个医疗分支 → safety_agent（强制汇聚，纯规则）→ fusion_agent → END
"""
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.nodes_basic import intent_agent, symptom_agent
from app.agent.nodes_domain import (
    chat_agent,
    department_agent,
    drug_agent,
    guide_agent,
    medical_knowledge_agent,
)
from app.agent.nodes_safety import fusion_agent, safety_agent
from app.agent.state import ChatState

_GRAPH: Any | None = None


def _route_after_intent(state: dict) -> str:
    if state.get("intent") == "CHAT":
        return "chat_agent"
    return "symptom_agent"


def _route_after_symptom(state: dict) -> str:
    if state.get("need_more"):
        return "symptom_agent"  # interrupt 追问后重入实体链接
    intent = state.get("intent", "KNOWLEDGE")
    return {
        "DEPARTMENT": "department_agent",
        "DRUG": "drug_agent",
        "GUIDE": "guide_agent",
    }.get(intent, "medical_knowledge_agent")  # KNOWLEDGE / MEDICAL_QUERY 兜底


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("intent_agent", intent_agent)
    g.add_node("symptom_agent", symptom_agent)
    g.add_node("department_agent", department_agent)
    g.add_node("drug_agent", drug_agent)
    g.add_node("medical_knowledge_agent", medical_knowledge_agent)
    g.add_node("guide_agent", guide_agent)
    g.add_node("chat_agent", chat_agent)
    g.add_node("safety_agent", safety_agent)
    g.add_node("fusion_agent", fusion_agent)

    g.add_edge(START, "intent_agent")
    g.add_conditional_edges("intent_agent", _route_after_intent,
                            {"chat_agent": "chat_agent", "symptom_agent": "symptom_agent"})
    g.add_conditional_edges("symptom_agent", _route_after_symptom, {
        "symptom_agent": "symptom_agent",
        "department_agent": "department_agent",
        "drug_agent": "drug_agent",
        "medical_knowledge_agent": "medical_knowledge_agent",
        "guide_agent": "guide_agent",
    })
    for node in ["department_agent", "drug_agent", "medical_knowledge_agent", "guide_agent"]:
        g.add_edge(node, "safety_agent")   # 医疗分支强制汇聚
    g.add_edge("chat_agent", END)
    g.add_edge("safety_agent", "fusion_agent")
    g.add_edge("fusion_agent", END)

    return g.compile(checkpointer=MemorySaver())


def get_graph():
    """全局单例（M6 API 场景必需）：MemorySaver 跨请求共享，按 thread_id 恢复会话。"""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def ask(question: str, thread_id: str = "default") -> dict:
    """单轮问答：新问题直接走完整图（M4 用）。返回最终 state。"""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke({"question": question}, config)
    return state


def _has_pending_interrupt(thread_id: str) -> bool:
    """判断该 thread 是否有上一轮遗留的未完成 interrupt（问诊追问）。"""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception:
        return False
    if snapshot is None or not snapshot.next:
        return False
    for task in snapshot.tasks:
        if getattr(task, "interrupts", None):
            return True
    return False


def ask_interrupt_aware(question: str, thread_id: str) -> tuple[dict, str | None]:
    """中断感知问答（M8.3 修复"复读上一轮"）：

    - 若该 thread 有上一轮遗留 interrupt → 本轮输入作为追问回答（resume）；
    - 否则作为新问题跑完整图。
    返回 (state, pending_clarify | None)。pending_clarify 非空表示图停在追问点，
    调用方必须输出追问问题，绝不能输出 state 里残留的上一轮 answer。
    """
    from langgraph.types import Command

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    if _has_pending_interrupt(thread_id):
        state = graph.invoke(Command(resume=question), config)
    else:
        # M8.12 新问显式重置跨轮残留瞬态字段：MemorySaver 同 thread_id 会累积上一轮
        # 的 answer/risk_level/refusal 等，若不清空，上一轮急症 HIGH 会传染本轮普通
        # 问题（横幅误挂）、上一轮拒答会传染（S004 误判）。分支节点未输出的字段由
        # 此处兜底归零；分支节点输出的值正常覆盖。
        clean_input = {
            "question": question,
            "answer": "", "answer_sections": [], "answer_tags": {},
            "risk_level": "NONE", "refusal": False, "invalidated": False,
            "high_risk_query": False, "disclaimer": "", "drug_notice": "",
            "evidence_quotes": [], "evidence_pool": [], "graph_evidence": [],
            "retrieval_evidence": [], "safety_trail": [],
        }
        state = graph.invoke(clean_input, config)
    pending = state.get("__interrupt__") or []
    if pending:
        return state, pending[0].value
    return state, None


def ask_with_interrupt(question: str, thread_id: str = "default"):
    """HITL 对话：执行到第一个 interrupt（问诊追问）或直接结束。

    返回 (graph, config, state, pending_question | None, finished)。
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.invoke({"question": question}, config)
    pending = state.get("__interrupt__") or []
    if pending:
        return graph, config, state, pending[0].value, False
    return graph, config, state, None, True


def resume(graph, config: dict, answer: str):
    """提交 interrupt 的追问回答并继续执行。返回 (state, pending | None, finished)。"""
    from langgraph.types import Command

    state = graph.invoke(Command(resume=answer), config)
    pending = state.get("__interrupt__") or []
    if pending:
        return state, pending[0].value, False
    return state, None, True
