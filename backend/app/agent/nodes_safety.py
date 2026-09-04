"""安全节点：safety_agent（规则引擎执行）+ fusion_agent（证据校验+答案组装）。

M5 实现：
- safety_agent：构建 chat. 上下文 → 引擎执行 safety_rules.yaml（六条规则）→
  输出 risk_level/disclaimer/drug_notice/refusal/invalidated/safety_trail（§6.3 对齐）。
- fusion_agent：证据引用提取（§5.3 第4步）→ S101/S102 命中则重生成一次，
  仍无引用 → refusal；最终组装（急症横幅 + 回答 + 用药提醒 + 免责声明）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.engine.dsl import load_rules  # noqa: E402
from app.engine.executor import execute  # noqa: E402
from app.engine.parser import compile_rules  # noqa: E402
from app.engine.trail import build_context  # noqa: E402
from app.llm.client import chat_json  # noqa: E402

from app.agent.nodes_domain import EMERGENCY_SYMPTOMS  # noqa: E402

MEDICAL_INTENTS = {"DEPARTMENT", "DRUG", "KNOWLEDGE", "GUIDE"}

DISCLAIMER_TEXT = "以上信息仅供参考，不能替代专业医疗建议，如有不适请及时就医。"
DRUG_NOTICE_TEXT = "请遵医嘱或药师指导使用，切勿自行调整剂量。"
REFUSAL_TEXT = "该问题涉及用药安全，建议咨询医生或药师，本助手暂不回答。"

# 急症词口语变体（medical_aliases.yaml symptoms 别名反向映射，S002 检测用）
_emergency_aliases: dict[str, str] = {}


def _load_emergency_aliases() -> dict[str, str]:
    global _emergency_aliases
    if not _emergency_aliases:
        import yaml

        alias_file = Path(__file__).resolve().parent.parent / "graph" / "medical_aliases.yaml"
        if alias_file.exists():
            raw = yaml.safe_load(alias_file.read_text(encoding="utf-8")) or {}
            for canonical, aliases in (raw.get("symptoms") or {}).items():
                if canonical in EMERGENCY_SYMPTOMS:
                    for a in aliases:
                        _emergency_aliases[a] = canonical
    return _emergency_aliases

# 规则库（进程内缓存）
_rules = None


def _get_rules():
    global _rules
    if _rules is None:
        rules_file = Path(__file__).resolve().parent.parent / "engine" / "safety_rules.yaml"
        _rules = compile_rules(load_rules(rules_file))
    return _rules


def extract_citations(answer: str, evidence_pool: list[dict],
                      graph_evidence: list[dict], entities: list[dict]) -> list[str]:
    """确定性引用提取（§5.3 第4步）：回答中出现的证据来源短语（去重保序）。

    词表 = 图谱三元组 subject/object + 实体名 + 证据池 quote 前 12 字。
    """
    vocab: list[str] = []
    seen: set[str] = set()
    for g in graph_evidence:
        for v in (g.get("subject"), g.get("object")):
            if v and v not in seen:
                seen.add(v)
                vocab.append(str(v))
    for e in entities:
        if e.get("name") and e["name"] not in seen:
            seen.add(e["name"])
            vocab.append(str(e["name"]))
    for p in evidence_pool:
        q = str(p.get("quote", ""))[:12]
        if q and q not in seen:
            seen.add(q)
            vocab.append(q)
    return [v for v in vocab if v and v in answer]


def safety_agent(state: dict) -> dict:
    """安全检查（纯规则引擎，§6）：结论不可被 LLM 推翻（红线2）。"""
    answer = state.get("answer", "")
    pool = state.get("evidence_pool", [])
    graph_ev = state.get("graph_evidence", [])
    # S002 急症检测补充：原文含急症词/口语变体（"胸口很痛"→胸痛、"喘不上气"→呼吸困难）
    # 但实体链接未命中时直接注入急症词实体，保证置顶警告不被绕过（红线2）
    entities = list(state.get("entities", []))
    question = state.get("collected_text") or state.get("question", "")
    emergency_variants = dict(_load_emergency_aliases())
    for canonical in EMERGENCY_SYMPTOMS:
        if canonical in question and not any(e.get("name") == canonical for e in entities):
            entities.append({"name": canonical, "label": "Symptom", "confidence": 1.0})
    for variant, canonical in emergency_variants.items():
        if variant in question and not any(e.get("name") == canonical for e in entities):
            entities.append({"name": canonical, "label": "Symptom", "confidence": 1.0})
    quotes = extract_citations(answer, pool, graph_ev, entities)

    context = build_context({
        "intent": state.get("intent", ""),
        "entities": entities,
        "answer": answer,
        "evidence_pool": pool,
        "evidence_quotes": quotes,
        "high_risk_query": bool(state.get("high_risk_query")),
        "disclaimer_added": False,  # 每轮新状态（融合后置 true 落库）
        "risk_level": state.get("risk_level", "NONE"),
    })
    result = execute(context, _get_rules())
    actions = set(result.actions)

    # 本轮风险等级只由本轮规则命中决定（M8.12：不读 checkpoint 残留，防上一轮急症 HIGH 传染）
    risk_level = "HIGH" if "SET_HIGH_RISK" in actions else "NONE"
    # refusal 读本轮分支节点输出（入口已重置为 False，无残留风险）
    refusal = bool(state.get("refusal")) or "REFUSE" in actions
    return {
        "risk_level": risk_level,
        "disclaimer": DISCLAIMER_TEXT if "REQUIRE_DISCLAIMER" in actions else "",
        "drug_notice": DRUG_NOTICE_TEXT if "REQUIRE_DRUG_NOTICE" in actions else "",
        "refusal": refusal,
        "invalidated": "INVALIDATE_ANSWER" in actions,
        "evidence_quotes": quotes,
        "safety_trail": [t for t in result.trails],
        "safety_passed": result.passed,
        "audit": [f"safety_agent: 规则 {len(result.trails)} 条，命中 {[h.rule_id for h in result.veto_hits + result.validate_hits]}",
                  f"safety_agent: 引用提取 {len(quotes)} 条"],
    }


def _regenerate_with_citation(state: dict) -> tuple[str, list[str]]:
    """S101/S102 命中后重生成：LLM 强制引用证据池具体内容。"""
    pool = state.get("evidence_pool", [])
    ev_text = "\n".join(f"- {p['quote'][:120]}" for p in pool[:8])
    question = state.get("collected_text") or state.get("question", "")
    try:
        data = chat_json(
            "你是医疗知识问答助手。仅根据证据池回答。回答中必须提到证据池里的具体名称"
            "（如症状名、科室名、药物名），但必须用自己的话自然转述："
            "禁止照抄证据池原句，禁止输出'→'箭头或任何技术符号。"
            "例如写'根据知识库信息，XX建议就诊XX科'，而不是复制'XX → XX科'。"
            "若确实无法引用，输出 {\"answer\": \"\"}。输出严格 JSON {\"answer\": \"...\"}。",
            f"证据池：\n{ev_text}\n问题：{question}",
        )
        answer = str(data.get("answer", "")).strip()
    except Exception:
        answer = ""
    quotes = extract_citations(answer, pool, state.get("graph_evidence", []), state.get("entities", []))
    return answer, quotes


def fusion_agent(state: dict) -> dict:
    """答案融合：急症优先 + 拒答/无证据兜底 + 引用校验（重生成一次）+ 风险横幅 + 免责声明。

    急症（risk_level==HIGH）优先于拒答：急症警告不可被无证据拒答绕过（红线2），
    横幅文案保证含"立即就医"（§8 emergency_recall 关键词）。
    """
    quotes = state.get("evidence_quotes", [])
    if state.get("risk_level") == "HIGH":
        # 急症优先：即便无证据/被标记拒答，也输出急症话术（横幅含"立即就医"）
        base = state.get("answer") or "您描述的情况可能属于急症。"
        if state.get("refusal") or not state.get("answer"):
            base = "您描述的情况可能属于急症，请优先前往急诊就医。"
        final = _assemble(state, base)
    elif state.get("refusal"):
        final = REFUSAL_TEXT
    elif not state.get("answer"):
        final = "根据知识库信息，暂未检索到与您问题直接相关的可靠资料，建议前往医院就诊咨询专科医生。"
    else:
        answer = state["answer"]
        # §5.3 第4步：S101/S102 命中 → 重生成一次（强制引用池内内容），仍无引用 → 拒答
        if state.get("invalidated"):
            answer, quotes = _regenerate_with_citation(state)
            if quotes:
                final = _assemble(state, answer)
            elif state.get("evidence_pool"):
                # 有证据却引不出来 → 拒答（防无引用内容外泄，原语义保留）
                final = REFUSAL_TEXT
            else:
                # 证据池为空：原回答是分支节点确定性产出的"如实说明"（模板，非 LLM 生成），
                # 它本就不可能携带引用，重生成失败应原样保留而非升格为拒答
                # （修复：无 LLM 环境下"怎么挂号就医"类 GUIDE 问题被误拒为"用药安全"）。
                final = _assemble(state, state.get("answer", ""))
        else:
            final = _assemble(state, answer)
    return {
        "answer": final,
        "evidence_quotes": quotes,
        "answer_sections": state.get("answer_sections") or [],
        "answer_tags": state.get("answer_tags") or {},
        "audit": [f"fusion_agent: 引用 {len(quotes)} 条" + ("（重生成后通过）" if state.get("invalidated") else "")],
    }


def _assemble(state: dict, answer: str) -> str:
    parts: list[str] = []
    if state.get("risk_level") == "HIGH":
        parts.append("⚠ 您描述的情况可能属于急症，请立即就医，必要时拨打120。")
    parts.append(answer)
    if state.get("drug_notice"):
        parts.append(state["drug_notice"])
    if state.get("disclaimer"):
        parts.append(state["disclaimer"])
    return "\n\n".join(parts)
