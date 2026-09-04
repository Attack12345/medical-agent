"""领域节点：department / drug / medical_knowledge（grounded）/ guide / chat。

grounded-ReAct 核心（§5.3）：medical_knowledge_agent 强制检索（图结构强制，
无"跳过检索"分支），检索结果注入上下文后才允许 LLM 生成；检索落空 +
高风险词 → refusal（§5.3.2 S004 兜底）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.graph.repo import GraphRepo  # noqa: E402
from app.llm.client import chat_json  # noqa: E402
from app.retrieval import drug_db  # noqa: E402
from app.services.history_store import format_history  # noqa: E402
from app.retrieval.pipeline import retrieve  # noqa: E402

# §5.3.2 高风险词（检索落空时触发拒答）
HIGH_RISK_WORDS = ["相互作用", "剂量", "禁忌", "致死", "中毒", "过量", "同服"]

EMERGENCY_SYMPTOMS = {"胸痛", "呼吸困难", "意识模糊", "持续高热", "大量出血", "剧烈头痛"}  # §6 S002 同表

# 敏感词过滤（M8.2）：图谱含大量不宜对患者随意呈现的词条（性/生殖相关）。
# 这些词常因 BM25 关键词重叠被误召回（如"头疼"误配"房事头疼症"），须在证据注入 LLM 前剔除。
# 仅做"呈现层过滤"，不改动图谱数据，不虚构事实。
# 注：耸动罕见病名（癌/转移等）不在此全局过滤（避免误伤肿瘤科与正常肿瘤问答），
#     而是通过"症状分诊不罗列可能疾病"的重构来避免恐吓患者。
_SENSITIVE_PATTERNS = [
    "房事", "性交", "性行为", "遗精", "早泄", "阳痿", "勃起", "手淫", "性欲", "纵欲",
    "避孕", "人流", "性病", "梅毒", "淋病", "尖锐", "艾滋",
    "阴茎", "阴道", "外阴", "睾丸", "射精", "高潮", "调情",
]


def _is_sensitive(text: str) -> bool:
    return any(p in text for p in _SENSITIVE_PATTERNS)


def _filter_sensitive_texts(texts: list[str]) -> list[str]:
    """剔除含敏感词的词条（保持顺序去重）。"""
    out: list[str] = []
    seen: set[str] = set()
    for t in texts:
        t = str(t)
        if not t or _is_sensitive(t) or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _filter_sensitive_evidence(pool: list[dict]) -> list[dict]:
    """过滤证据池中 quote/ref 含敏感词的条目。"""
    return [p for p in pool
            if not _is_sensitive(str(p.get("quote", ""))) and not _is_sensitive(str(p.get("ref", "")))]


# 检索运行环境（模块级单例，run_chat/API 启动时 init_retrieval_env() 注入）
_retrieval_env: dict | None = None


def init_retrieval_env() -> None:
    """初始化检索环境（Linker/VectorDb/GraphRepo/BM25），进程内单例。"""
    global _retrieval_env
    if _retrieval_env is not None:
        return
    import json as _json

    from app.graph.repo import GraphRepo
    from app.retrieval.link import Linker
    from app.retrieval.pipeline import init_bm25
    from app.retrieval.vector_db import VectorDb

    project_root = Path(__file__).resolve().parents[3]  # medical-agent/
    with open(project_root / "data" / "cleaned" / "diseases.json", "r", encoding="utf-8") as f:
        diseases = _json.load(f)
    with open(project_root / "data" / "cleaned" / "qa_pairs.json", "r", encoding="utf-8") as f:
        qa = _json.load(f)
    init_bm25([d["name"] for d in diseases], [q["question"] for q in qa])
    _retrieval_env = {"linker": Linker(), "db": VectorDb(), "repo": GraphRepo()}


def get_retrieval_env() -> dict:
    if _retrieval_env is None:
        init_retrieval_env()
    return _retrieval_env


def _is_high_risk(question: str) -> bool:
    return any(w in question for w in HIGH_RISK_WORDS)


def _draft(state: dict, prompt: str, evidence: list[dict] | None = None) -> dict:
    """通用 grounded 生成：证据池注入 → LLM 生成 → 解析 draft/refusal。

    问题文本用 collected_text（含 HITL 追问补充），保证与证据池对齐。
    输出结构化（M6.1）：answer（自然语言）+ sections（[{title, points}]）+ tags（{symptoms, departments}）。
    """
    question = state.get("collected_text") or state.get("question", "")
    pool = evidence if evidence is not None else state.get("evidence_pool", [])
    if not pool:
        if _is_high_risk(question):
            return {"refusal": True, "answer": "", "high_risk_query": True,
                    "audit": ["grounded: 高风险问题且检索落空 → 拒答（S004）"]}
        return {"refusal": False, "answer": "", "high_risk_query": False,
                "audit": ["grounded: 检索落空 → 无证据不臆测（红线4）"]}
    ev_text = "\n".join(f"- {p['quote'][:120]}" for p in pool[:8])
    hist = format_history(state.get("recent_history"))
    hist_block = f"{hist}\n" if hist else ""
    try:
        data = chat_json(prompt, f"{hist_block}证据池：\n{ev_text}\n问题：{question}")
        answer = str(data.get("answer", "")).strip()
        refusal = bool(data.get("refusal", False))
        if refusal:
            return {"refusal": True, "answer": "", "high_risk_query": _is_high_risk(question),
                    "audit": ["grounded: LLM 判定证据不足拒答"]}
        sections = data.get("sections") or []
        tags = data.get("tags") or {}
        return {"refusal": False, "answer": answer, "high_risk_query": _is_high_risk(question),
                "answer_sections": sections, "answer_tags": tags,
                "audit": [f"grounded: 证据池 {len(pool)} 条 → LLM 生成"]}
    except Exception as e:
        return {"refusal": True, "answer": "", "high_risk_query": _is_high_risk(question),
                "audit": [f"grounded: LLM 失败 {e}"]}


def _query_depts(repo: GraphRepo, names: list[str]) -> list[str]:
    """(症状)-[:VISITS]->(科室)；无则 (疾病)-[:PRESENTS]->(症状)-[:VISITS]->(科室)。"""
    if not names:
        return []
    depts: list[str] = []
    rows = repo.query(
        "MATCH (s:Symptom)-[:VISITS]->(d:Department) WHERE s.name IN $names RETURN DISTINCT d.name AS dept LIMIT 50",
        {"names": names},
    )
    depts = [r["dept"] for r in rows]
    if not depts:
        rows = repo.query(
            "MATCH (dis:Disease {name: $name})-[:PRESENTS]->(s:Symptom)-[:VISITS]->(d:Department) "
            "RETURN DISTINCT d.name AS dept LIMIT 50",
            {"name": names[0]},
        )
        depts = [r["dept"] for r in rows]
    return depts


def department_agent(state: dict) -> dict:
    """科室推荐：图谱查 症状→科室；LLM 组织解释（必须引用科室实体）。"""
    names = [e["name"] for e in state.get("entities", [])]
    repo = GraphRepo()
    try:
        depts = _query_depts(repo, names)
    finally:
        repo.close()
    if not depts:
        return {"answer": "根据知识库信息，暂未检索到与该症状对应的明确科室建议，建议前往医院导诊台咨询。",
                "audit": ["department_agent: 图谱无科室 → 如实说明"]}
    try:
        data = chat_json(
            "你是医院导诊助手。根据给出的推荐科室，用自然语言组织回答（如'根据知识库信息，XX建议就诊XX科'），"
            "必须包含科室名称，不超过 60 字，不要编造图谱外的信息。输出严格 JSON {\"answer\": \"...\"}。",
            f"推荐科室：{'、'.join(depts[:5])}\n问题：{state.get('question')}",
        )
        answer = str(data.get("answer", "")).strip()
    except Exception:
        answer = f"根据知识库信息，{state.get('question')}建议就诊{'、'.join(depts[:3])}。"
    graph_ev = [{"subject": n, "relation": "VISITS", "object": d, "source": "graph"}
                for n in names[:2] for d in depts[:2]]
    return {"answer": answer, "graph_evidence": graph_ev,
            # 证据池：GRAPH_NODE 短语（S101/S102 校验引用基于此）
            "evidence_pool": [{"type": "GRAPH_NODE", "ref": f"VISITS:{n}→{d}",
                               "quote": f"{n} {d}", "score": 1.0}
                              for n in names[:2] for d in depts[:2]],
            "audit": [f"department_agent: 科室 {depts[:5]}"]}


def _symptom_medication_guidance(state: dict, symptom_names: list[str]) -> dict:
    """症状问药（未确诊疾病，如"我头疼→应该吃什么药"）：不推荐具体药物。

    对未明确诊断的症状直接罗列药名既不负责任也违反红线4（图谱无"症状→药"的可靠关系）。
    正确做法：建议先就诊明确病因 + 给出对应科室 + 提醒勿自行服药掩盖病情。
    """
    env = get_retrieval_env()
    repo = env["repo"]
    depts = _rank_depts(_query_depts(repo, symptom_names), symptom_names)
    primary = depts[0] if depts else ""
    dept_text = f"，建议先就诊{primary}明确病因" if primary else ""
    answer = f"根据知识库信息，{'、'.join(symptom_names[:2])}的用药需先明确病因{dept_text}，不建议自行服药。"
    sections = [{"title": "用药安全", "points": [
        f"出现{'、'.join(symptom_names[:2])}时，先明确病因再用药，勿自行服用止痛/对症药物掩盖病情",
        f"建议就诊{primary}，由医生评估后开具用药方案" if primary else "建议就诊相关科室，由医生评估后开具用药方案",
        "如症状突然加重或持续不缓解，请及时就医"]}]
    tags = {"symptoms": symptom_names[:3], "departments": [primary] if primary else []}
    graph_ev = [{"subject": n, "relation": "VISITS", "object": primary, "source": "graph"}
                for n in symptom_names[:1]] if primary else []
    return {"answer": answer, "answer_sections": sections, "answer_tags": tags,
            "graph_evidence": graph_ev,
            "evidence_pool": [{"type": "GRAPH_NODE", "ref": f"VISITS:{symptom_names[0]}→{primary}",
                               "quote": f"{symptom_names[0]} → {primary}", "text": f"{symptom_names[0]} → {primary}",
                               "score": 1.0}] if primary else [],
            "audit": [f"drug_agent: 症状问药安全指导 {symptom_names[:2]} → {primary}"]}


def drug_agent(state: dict) -> dict:
    """用药建议：图谱查 Drug-TREATS->(疾病)；只陈述图谱事实（不推荐剂量）。

    症状问药（实体是 Symptom 而非确诊 Disease）→ 安全指导分支，不硬塞药名。
    """
    entities = state.get("entities", [])
    disease_names = [e["name"] for e in entities if e.get("label") == "Disease"]
    symptom_names = [e["name"] for e in entities if e.get("label") == "Symptom"]
    if not disease_names and symptom_names:
        return _symptom_medication_guidance(state, symptom_names)

    names = disease_names or [e["name"] for e in entities]
    repo = GraphRepo()
    try:
        drugs: list[str] = []
        for n in names:
            rows = repo.query(
                "MATCH (d:Drug)-[:TREATS]->(dis:Disease {name: $name}) RETURN DISTINCT d.name AS drug LIMIT 8",
                {"name": n},
            )
            drugs.extend(r["drug"] for r in rows)
    finally:
        repo.close()
    drugs = list(dict.fromkeys(drugs))
    if not drugs:
        return {"answer": "根据知识库信息，暂未检索到该疾病的对应用药信息，建议咨询医生或药师。",
                "audit": ["drug_agent: 图谱无药物 → 如实说明"]}
    try:
        data = chat_json(
            "你是用药信息助手。根据药物列表输出结构化用药说明：只陈述图谱事实，不给出剂量建议，"
            "回答正文自然语言表述来源（如'根据知识库信息'）。"
            "输出严格 JSON：{\"answer\": \"一句话结论\","
            " \"sections\": [{\"title\": \"常用药物\", \"points\": [\"药物名及用途说明\"]},"
            " {\"title\": \"用药提醒\", \"points\": [\"遵医嘱用药，切勿自行调整剂量\"]}],"
            " \"tags\": {\"symptoms\": [], \"departments\": []}}",
            f"可用药物：{'、'.join(drugs[:8])}\n问题：{state.get('question')}",
        )
        answer = str(data.get("answer", "")).strip()
        sections = data.get("sections") or []
        tags = data.get("tags") or {}
    except Exception:
        answer = f"根据知识库信息，{names[0] if names else '该疾病'}常用药物包括：{'、'.join(drugs[:5])}。"
        sections = [{"title": "常用药物", "points": drugs[:6]},
                    {"title": "用药提醒", "points": ["请遵医嘱或药师指导使用，切勿自行调整剂量"]}]
        tags = {"symptoms": [], "departments": []}
    graph_ev = [{"subject": d, "relation": "TREATS", "object": n, "source": "graph"}
                for d in drugs[:5] for n in names[:1]]
    return {"answer": answer, "answer_sections": sections, "answer_tags": tags,
            "graph_evidence": graph_ev,
            "evidence_pool": [{"type": "GRAPH_NODE", "ref": f"TREATS:{d}→{n}",
                               "quote": f"{d} → {n}", "score": 1.0}
                              for d in drugs[:5] for n in names[:1]],
            "audit": [f"drug_agent: 药物 {drugs[:8]}"]}


def medical_knowledge_agent(state: dict) -> dict:
    """医学知识问答（grounded，§5.3 核心）：强制检索 → 注入证据 → LLM 生成。

    M8.2 分流：
    - 主实体是 Symptom（症状主诉，如"我头疼"）→ 分诊卡：推荐科室 + 就医建议，
      不罗列"可能的疾病"（原始图谱含大量罕见/耸动病名，直接罗列既无用又恐吓患者）。
    - 主实体是 Disease（疾病问答，如"什么是糖尿病"）→ 知识卡：grounded 回答，过滤敏感词。
    检索用 collected_text（含问诊追问补充），保证 HITL 场景证据与完整输入对齐。
    """
    env = get_retrieval_env()
    query = state.get("collected_text") or state.get("question")
    result = retrieve(query, linker=env["linker"], db=env["db"], repo=env["repo"])
    pool = _filter_sensitive_evidence(result["evidence_pool"])  # 呈现层过滤敏感词

    entities = state.get("entities", [])
    symptom_names = [e["name"] for e in entities if e.get("label") == "Symptom"]
    primary_is_symptom = bool(symptom_names) and (
        not any(e.get("label") == "Disease" for e in entities) or entities[0].get("label") == "Symptom"
    )

    if primary_is_symptom:
        out = _symptom_triage(state, symptom_names, env)
    else:
        out = _draft(state, _GROUNDED_PROMPT, pool)

    out["graph_evidence"] = result["graph_evidence"]
    out["retrieval_evidence"] = result["retrieval_evidence"]
    out["evidence_pool"] = pool
    return out


def _symptom_triage(state: dict, symptom_names: list[str], env: dict) -> dict:
    """症状主诉 → 分诊卡：推荐科室 + 就医建议，不罗列可能疾病（避免罕见/耸动病名恐吓）。"""
    repo = env["repo"]
    depts = _rank_depts(_query_depts(repo, symptom_names), symptom_names)
    exams = _filter_sensitive_texts(_query_exams(repo, symptom_names))
    if not depts:
        return {"refusal": False, "answer": "", "high_risk_query": _is_high_risk(state.get("question", "")),
                "audit": ["symptom_triage: 无科室 → 走 grounded 兜底"]}
    primary = depts[0]
    question = state.get("collected_text") or state.get("question", "")
    try:
        data = chat_json(
            "你是医院分诊导诊助手。用户描述了一个症状，请给出实用的就医指导。"
            f"首选科室是{primary}，结论必须以该科室为主。"
            "重要：不要列举或猜测'可能是什么病/哪些疾病会导致该症状'，只做就医指导。"
            "sections 给出 2-4 条实用建议，例如：就诊前记录症状的时间/部位/性质/伴随表现；"
            "近期的诱因（劳累、受凉、情绪、睡眠）；出现哪些加重表现需尽快就医。"
            "回答正文自然语言表述来源（如'根据知识库信息'），不出现技术节点名或箭头符号。"
            "输出严格 JSON：{\"answer\": \"一句话结论（如'根据知识库信息，XX 建议就诊XX科'）\","
            " \"sections\": [{\"title\": \"就医建议\", \"points\": [\"实用建议1\", \"实用建议2\"]}],"
            " \"tags\": {\"symptoms\": [\"症状名\"], \"departments\": [\"科室名\"]}}",
            f"症状：{'、'.join(symptom_names[:3])}\n首选科室：{primary}\n备选科室：{'、'.join(depts[1:4])}"
            f"\n相关检查：{'、'.join(exams[:6]) or '暂无'}\n用户原话：{question}",
        )
        answer = str(data.get("answer", "")).strip()
        sections = data.get("sections") or []
        tags = data.get("tags") or {}
        if not tags.get("departments"):
            tags["departments"] = [primary]
        if not tags.get("symptoms"):
            tags["symptoms"] = symptom_names[:3]
        return {"refusal": False, "answer": answer, "high_risk_query": _is_high_risk(question),
                "answer_sections": sections, "answer_tags": tags,
                "audit": [f"symptom_triage: 科室 {depts[:4]} 检查 {exams[:6]}"]}
    except Exception as e:
        return {"refusal": False,
                "answer": f"根据知识库信息，{'、'.join(symptom_names[:2])}建议就诊{primary}。",
                "high_risk_query": _is_high_risk(question),
                "answer_sections": [{"title": "就医建议", "points": [
                    "就诊前记录症状出现的时间、部位、性质与伴随表现",
                    "回顾近期诱因（劳累、受凉、情绪波动、睡眠）", "如有加重请及时就医"]}],
                "answer_tags": {"symptoms": symptom_names[:3], "departments": [primary]},
                "audit": [f"symptom_triage: LLM 失败兜底 {e}"]}


_GROUNDED_PROMPT = (
    "你是医疗知识问答助手。仅根据下方\"证据池\"回答，不得使用证据池之外的知识。"
    "回答正文用自然语言表述来源（如\"根据知识库信息\"\"资料显示\"），不要输出技术性节点名或引用编号。"
    "输出严格 JSON："
    "{\"answer\": \"完整回答（自然语言，80 字内）\", \"refusal\": false,"
    " \"sections\": [{\"title\": \"小节标题\", \"points\": [\"要点1\", \"要点2\"]}],"
    " \"tags\": {\"symptoms\": [\"症状名\"], \"departments\": [\"科室名\"]}}"
    "sections/tags 无内容时给空数组/空对象；若证据池无法回答，输出 {\"refusal\": true, \"answer\": \"\"}。"
)


def guide_agent(state: dict) -> dict:
    """就医指导：图谱+注意事项；急症症状输出'立即就医'话术（§5.2 S002 兜底）。"""
    names = [e["name"] for e in state.get("entities", [])]
    if set(names) & EMERGENCY_SYMPTOMS:
        return {"answer": "您描述的症状可能属于急症，请立即前往医院急诊科就医，不要自行处理。",
                "risk_level_hint": "HIGH",
                "audit": ["guide_agent: 急症词表命中 → 立即就医话术"]}
    repo = GraphRepo()
    try:
        notices: list[str] = []
        for n in names:
            rows = repo.query(
                "MATCH (d:Disease {name: $name}) RETURN d.summary AS summary LIMIT 1", {"name": n})
            if rows and rows[0].get("summary"):
                notices.append(rows[0]["summary"][:100])
    finally:
        repo.close()
    if not notices:
        return {"answer": "根据知识库信息，暂未检索到相关就医指导，建议前往医院咨询。",
                "audit": ["guide_agent: 无疾病摘要 → 如实说明"]}
    try:
        data = chat_json(
            "你是就医指导助手。根据给出的疾病资料，给出就医建议（就诊科室、注意事项），"
            "用自然语言表述（如'根据知识库信息'），不超过 80 字。输出严格 JSON {\"answer\": \"...\"}。",
            f"疾病资料：{'；'.join(notices[:3])}\n问题：{state.get('question')}",
        )
        answer = str(data.get("answer", "")).strip()
    except Exception:
        answer = f"根据知识库信息，建议前往医院就诊，听从医生安排治疗。"
    return {"answer": answer, "audit": [f"guide_agent: 摘要 {len(notices)} 条"]}


def chat_agent(state: dict) -> dict:
    """通用对话：纯 LLM 闲聊，不调用检索，不加免责声明。"""
    try:
        data = chat_json(
            "你是友好亲切的智能导诊助手。用户正在闲聊（与医疗咨询无关），请简短友好回应，"
            "并提示可以咨询健康问题。不超过 50 字。输出严格 JSON {\"answer\": \"...\"}。",
            f"用户说：{state.get('question')}",
        )
        answer = str(data.get("answer", "")).strip()
    except Exception:
        answer = "您好，我是智能导诊助手，可以问我症状、科室、用药等方面的问题。"
    return {"answer": answer, "audit": ["chat_agent: 闲聊"]}
