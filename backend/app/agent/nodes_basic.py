"""基础节点：intent_agent（意图分类）+ symptom_agent（实体链接，含问诊 HITL 追问）。

- intent_agent：§5.4 关键词预筛（data/intent_keywords.json）命中即分类，未命中 LLM（附录 A1）。
- symptom_agent：复用 link.py 实体链接；无命中且追问未用尽 → interrupt 追问具体症状
  （M4 问诊 HITL，§5.3 interrupt 模式），最多 2 次，仍无命中降级走知识分支。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings  # noqa: E402
from app.llm.client import chat_json  # noqa: E402
from app.retrieval.link import Linker  # noqa: E402

INTENT_KEYWORDS_FILE = Path(__file__).resolve().parents[3] / "data" / "intent_keywords.json"
MAX_FOLLOW_UP = 2  # 追问上限（M4 锁定）

MEDICAL_INTENTS = {"DEPARTMENT", "DRUG", "KNOWLEDGE", "GUIDE", "MEDICAL_QUERY"}

_linker: Linker | None = None


def get_linker() -> Linker:
    """取共享 Linker：必须用 init_retrieval_env 构建的实例（带语义向量），
    不能另建裸实例——否则语义兜底失效（M8.6 修复"心跳很快"误触发澄清）。"""
    from app.agent.nodes_domain import get_retrieval_env

    return get_retrieval_env()["linker"]


def _load_keywords() -> dict[str, list[str]]:
    with open(INTENT_KEYWORDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _pre_classify(question: str) -> str | None:
    """§5.4 关键词预筛：命中返回 intent，未命中 None（走 LLM）。"""
    for intent, keywords in _load_keywords().items():
        if any(kw in question for kw in keywords):
            return intent
    return None


def intent_agent(state: dict) -> dict:
    question = state.get("question", "").strip()
    intent = _pre_classify(question)
    if intent is None:
        try:
            data = chat_json(
                "你是意图分类器。将用户问题分类到：DEPARTMENT（问挂什么科）、DRUG（问用药/剂量/禁忌）、"
                "KNOWLEDGE（问疾病/症状知识）、GUIDE（问就医流程/注意事项）、CHAT（闲聊/与医疗无关）。"
                "输出严格 JSON {\"intent\": \"...\"}。",
                f"问题：{question}",
            )
            intent = str(data.get("intent", "KNOWLEDGE")).upper()
        except Exception:
            intent = "KNOWLEDGE"  # LLM 不可用时保守按知识问答处理
    if intent not in MEDICAL_INTENTS | {"CHAT"}:
        intent = "KNOWLEDGE"
    return {"intent": intent, "audit": [f"intent_agent: {intent}"]}


# 否定/缓解线索词（M8.18）：命中时抽取前后 3 字片段，用于继承实体的否定剔除
_NEG_CUE_RE = re.compile(r"不疼|不痛|不痒|不咳|不闷|不麻|不胀|不晕|退了|缓解|好转|消失|减轻|好了")


def _negated_fragments(text: str) -> str:
    """抽取否定词周边片段（否定词前后各 3 字）拼接返回；无否定线索返回空串。

    例："头不疼了，手疼" → 片段含"头"，继承实体"头痛/头晕"按身体部位字剔除。
    """
    frags = [text[max(0, m.start() - 3): m.end() + 3] for m in _NEG_CUE_RE.finditer(text)]
    return " ".join(frags)


def _dedupe_entities(entities: list[dict]) -> list[dict]:
    """同类实体包含去重（M8.18b）：名称互为子串时保留更具体的长名。

    真实图谱存在"头痛头晕"与"头痛"两个节点，口语问句常同时命中，
    不去重会让标签出现"头痛头晕、头痛"式重复，结论句也随之累赘。
    """
    names = [str(e.get("name", "")) for e in entities]
    return [e for e, n in zip(entities, names)
            if not (n and any(n != other and n in other for other in names))]


def symptom_agent(state: dict) -> dict:
    """实体链接 + 问诊 HITL：信息不足时 interrupt 追问（最多 MAX_FOLLOW_UP 次）。

    链接带查询向量启用语义兜底（§4.2）：口语描述（"我心跳很快"）精确/别名匹配
    不上时按余弦 ≥0.85 建链，避免对已描述症状的用户误触发澄清追问。
    """
    from langgraph.types import interrupt

    text = state.get("collected_text") or state.get("question", "")
    left = int(state.get("follow_up_left", MAX_FOLLOW_UP))

    qvec = None
    try:
        from app.retrieval.embedding import embed_query
        qvec = embed_query(text)
    except Exception:
        qvec = None  # embedding 不可用 → 退化纯精确/别名链接（§4.4）
    entities = get_linker().link_text(text, query_vector=qvec)
    linked = _dedupe_entities([{"name": name, "label": label, "confidence": conf}
                               for name, label, conf in entities])

    # 多轮上下文继承（M8.3）：本轮未链接到实体，但上一轮有（如"我头疼"后问"应该吃什么药？"）
    # → 复用上一轮实体，保证追问式跟进有上下文（CHAT 意图不经本节点，不会误继承）。
    # M8.18 否定剔除：用户表达"X不疼了/缓解/好了"时上轮对应症状不再成立，
    # 继承前按否定片段剔除（"头不疼了，手疼"不得再按头痛给建议）。
    if not linked:
        prev = state.get("entities") or []
        if prev:
            neg_frag = _negated_fragments(text)
            carried, dropped = [], []
            for e in prev:
                name = str(e.get("name", ""))
                if neg_frag and any(ch in neg_frag for ch in name if ch not in "不没了的，。、"):
                    dropped.append(name)
                    continue
                carried.append({"name": name, "label": e.get("label", "Symptom"),
                                "confidence": e.get("confidence", 0.9)})
            if carried:
                carried = _dedupe_entities(carried)
                note = f"（否定剔除 {dropped}）" if dropped else ""
                return {"entities": carried, "need_more": False,
                        "audit": [f"symptom_agent: 继承上轮实体 {[e['name'] for e in carried]}{note}"]}
            if dropped:
                # 上轮实体全部被否定，但本轮仍有新主诉（如"头不疼，手疼"余下手疼）且词表
                # 匹配不上 → 定向追问细节（部位/性质/伴随），引导用户给出可匹配的症状词；
                # 这比直接拒答或通用追问都更接近可用答案。追问次数受 follow_up_left 约束。
                if left > 0:
                    segs = re.split(r"[，,。；;！!？?\s]+", text)
                    remaining = "、".join(s for s in segs if s and not _NEG_CUE_RE.search(s))
                    target = remaining or "当前症状"
                    answer = interrupt(
                        f"了解，{'、'.join(dropped)}已缓解。为了给出有依据的建议，"
                        f"请具体描述「{target}」的情况：部位、性质（如胀痛/刺痛/酸痛），"
                        "是否伴随麻木、肿胀或外伤？")
                    return {
                        "collected_text": f"{text} {answer}".strip(),
                        "follow_up_left": left - 1,
                        "need_more": True,
                        "audit": [f"symptom_agent: 否定剔除 {dropped} 后定向追问「{target}」"],
                    }
                return {"entities": [], "need_more": False,
                        "audit": [f"symptom_agent: 上轮实体全部被否定剔除 {dropped}，按新主诉检索"]}

    if not linked and left > 0:
        # 信息不足：interrupt 追问（resume 值并入 collected_text 后重入本节点）
        answer = interrupt("为了给您更准确的建议，请描述一下您的具体症状或不适（如：头痛、发热、胃痛、咳嗽等）。")
        return {
            "collected_text": f"{text} {answer}".strip(),
            "follow_up_left": left - 1,
            "need_more": True,
            "audit": [f"symptom_agent: 追问第 {MAX_FOLLOW_UP - left + 1} 次，补充输入: {answer}"],
        }

    return {
        "entities": linked,
        "need_more": False,
        "audit": [f"symptom_agent: 链接 {len(linked)} 个实体"],
    }
