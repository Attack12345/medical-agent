"""对话状态定义（锁定字段）。

注意：LangGraph 会丢弃 TypedDict 未声明的键——所有节点输出的字段
（含 interrupt 桥接字段）必须先在此声明。
"""
from typing import Optional, TypedDict


class ChatState(TypedDict, total=False):
    question: str                        # 用户原始问题
    intent: str                          # MEDICAL_QUERY/DEPARTMENT/DRUG/KNOWLEDGE/GUIDE/CHAT
    entities: list[dict]                 # [{name, label, confidence}]
    graph_evidence: list[dict]           # [{subject, relation, object, source}]
    retrieval_evidence: list[dict]       # [{doc_type, text, score}]
    evidence_pool: list[dict]            # 融合后的证据候选（Top5）
    answer: str                          # 生成中的回答
    answer_sections: list[dict]          # 结构化小节 [{title, points[]}]（M6.1 卡片渲染）
    answer_tags: dict                    # 标签 {symptoms[], departments[]}
    risk_level: str                      # NONE/LOW/MEDIUM/HIGH（安全层输出）
    disclaimer: str                      # 免责声明文本
    drug_notice: str                     # 用药提醒文本（safety 输出，S003）
    refusal: bool                        # 是否拒答（检索落空+高风险）
    invalidated: bool                    # S101/S102 命中：回答无效（fusion 重生成）
    evidence_quotes: list[str]           # 确定性引用提取（§5.3 第4步，供审计）
    safety_trail: list[dict]             # 规则执行轨迹（§6.2，M6 附 evidence_json 落库）
    safety_passed: bool
    high_risk_query: bool                # 问题含高风险词（剂量/禁忌/致死等）
    audit: list[str]
    # interrupt 桥接字段（LangGraph 丢弃未声明键——必须声明）
    pending_input: Optional[str]
    # M4 问诊 HITL（symptom_agent 追问用）
    collected_text: str                  # 累计用户输入（含追问补充）
    follow_up_left: int                  # 追问剩余次数（默认 2）
    need_more: bool                      # 本轮是否需要重入 symptom_agent 再链接
