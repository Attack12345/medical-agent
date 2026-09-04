"""领域节点：department / drug / medical_knowledge（grounded）/ guide / chat。

grounded-ReAct 核心（§5.3）：medical_knowledge_agent 强制检索（图结构强制，
无"跳过检索"分支），检索结果注入上下文后才允许 LLM 生成；检索落空 +
高风险词 → refusal（§5.3.2 S004 兜底）。

M8.2 分流：症状主诉 → 分诊卡；疾病问答 → 知识卡。
M8.4 用药推荐：CMeKG 说明书 + 高危药过滤 + 共识度排序。
M8.10 科室排序降噪与缺失补正。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.graph.repo import GraphRepo  # noqa: E402
from app.llm.client import chat_json  # noqa: E402
from app.retrieval import drug_db  # noqa: E402
from app.retrieval.pipeline import retrieve  # noqa: E402
from app.services.history_store import format_history  # noqa: E402

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


# ------------------------------------------------------------------
# 检索运行环境
# ------------------------------------------------------------------

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
    from app.retrieval.vector_db import SYMPTOMS_COLLECTION, VectorDb

    project_root = Path(__file__).resolve().parents[3]
    with open(project_root / "data" / "cleaned" / "qa_pairs.json", "r", encoding="utf-8") as f:
        qa = _json.load(f)

    db = VectorDb()
    repo = GraphRepo()

    # BM25 entities 索引与 build_vector_db 同源（M8.8 一致性修复）：
    # 四类实体（Disease/Drug/Department/Exam）+ Disease 摘要全文。
    entity_rows = repo.query(
        """
        MATCH (n)
        WHERE any(l IN labels(n) WHERE l IN ['Disease', 'Drug', 'Department', 'Exam'])
        RETURN labels(n)[0] AS label, n.name AS name, n.summary AS summary
        """
    )
    entity_docs: list[str] = []
    for r in entity_rows:
        summary = str(r.get("summary") or "").strip()
        if r["label"] == "Disease" and summary:
            entity_docs.append(f"{r['name']}：{summary[:100]}")
        else:
            entity_docs.append(r["name"])
    init_bm25(entity_docs, [q["question"] for q in qa], [q["answer"] for q in qa])

    # 语义链接兜底（§4.2 / M8.6）：拉症状向量矩阵，口语描述精确/别名匹配不上时
    # 按余弦 ≥0.85 建链，避免误触发澄清追问
    entity_vectors: dict[str, list[float]] = {}
    try:
        points, _ = db.client.scroll(
            collection_name=SYMPTOMS_COLLECTION,
            with_payload=True, with_vectors=True, limit=12000)
        for pt in points:
            name = (pt.payload or {}).get("name")
            if name and pt.vector:
                entity_vectors[name] = list(pt.vector)
    except Exception:
        entity_vectors = {}

    _retrieval_env = {"linker": Linker(entity_vectors=entity_vectors), "db": db, "repo": repo}


def get_retrieval_env() -> dict:
    if _retrieval_env is None:
        init_retrieval_env()
    return _retrieval_env


def _is_high_risk(question: str) -> bool:
    return any(w in question for w in HIGH_RISK_WORDS)


# ------------------------------------------------------------------
# grounded 生成
# ------------------------------------------------------------------

_GROUNDED_PROMPT = (
    "你是医疗知识问答助手。仅根据下方\"证据池\"回答，不得使用证据池之外的知识。"
    "回答正文用自然语言表述来源（如\"根据知识库信息\"\"资料显示\"），不要输出技术性节点名或引用编号。"
    "输出严格 JSON："
    "{\"answer\": \"完整回答（自然语言，80 字内）\", \"refusal\": false,"
    " \"sections\": [{\"title\": \"小节标题\", \"points\": [\"要点1\", \"要点2\"]}],"
    " \"tags\": {\"symptoms\": [\"症状名\"], \"departments\": [\"科室名\"]}}"
    "sections/tags 无内容时给空数组/空对象；若证据池无法回答，输出 {\"refusal\": true, \"answer\": \"\"}。"
)


def _draft(state: dict, prompt: str, evidence: list[dict] | None = None) -> dict:
    """通用 grounded 生成：证据池注入 → LLM 生成 → 解析 draft/refusal。

    问题文本用 collected_text（含 HITL 追问补充），保证与证据池对齐。
    输出结构化（M6.1）：answer + sections（[{title, points}]）+ tags。
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
        # LLM 不可用（无 key/网络失败）且证据池非空 → 确定性摘 要回答兜底：
        # 内容仍严格取自证据池（grounded 原则不变），不把"LLM 失败"升格为"拒答"，
        # 否则无 LLM 环境（CI/离线演示）下全部 grounded 问答被误拒。
        # 回答显式含实体名（引用提取词表必含实体名 → S101 必有引用锚点）。
        if pool:
            entities = state.get("entities", [])
            names = [str(e.get("name", "")).strip() for e in entities if e.get("name")]
            frags: list[str] = []
            for p in pool[:4]:
                quote = str(p.get("quote", ""))
                if p.get("type") == "GRAPH_NODE" and "→" in quote:
                    sub, _, obj = quote.partition("→")
                    frag = f"{sub.strip()}与{obj.strip()}相关"
                else:
                    frag = quote.strip()
                frag = frag[:80].rstrip("。；;，, ")
                if frag and not _is_sensitive(frag):
                    frags.append(frag)
            frags = list(dict.fromkeys(frags))[:3]
            if frags:
                head = f"关于{'、'.join(names[:2])}，" if names else ""
                answer = f"根据知识库信息，{head}" + "；".join(frags) + "。"
                return {"refusal": False, "answer": answer,
                        "high_risk_query": _is_high_risk(question),
                        "audit": [f"grounded: LLM 不可用 → 证据池摘 要兜底（{e}）"]}
        return {"refusal": True, "answer": "", "high_risk_query": _is_high_risk(question),
                "audit": [f"grounded: LLM 失败 {e}"]}


# ------------------------------------------------------------------
# 科室推荐
# ------------------------------------------------------------------

def _query_depts(repo: GraphRepo, names: list[str]) -> list[str]:
    """(症状)-[:VISITS]->(科室)；无则 (疾病)-[:PRESENTS]->(症状)-[:VISITS]->(科室)。"""
    if not names:
        return []
    rows = repo.query(
        "MATCH (s:Symptom)-[:VISITS]->(d:Department) WHERE s.name IN $names "
        "RETURN DISTINCT d.name AS dept LIMIT 50",
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


# 症状部位/描述关键词 → 专科（临床分诊路由知识，仅用于对图谱候选科室排序，不虚构科室）
_SYMPTOM_SPECIALTY_HINTS = [
    (["鼻涕", "鼻塞", "流涕", "涕", "鼻", "咽喉", "嗓子", "喉咙", "耳", "听力", "嗅觉"], "耳鼻喉科"),
    (["头痛", "头晕", "头", "晕", "麻", "抽搐", "意识", "记忆", "失眠", "神经"], "神经内科"),
    (["胃痛", "腹痛", "腹泻", "便秘", "胃", "腹", "消化", "反酸", "恶心", "呕吐", "胀"], "消化内科"),
    (["咳嗽", "喘", "呼吸", "气短", "胸闷", "肺", "痰", "憋"], "呼吸内科"),
    (["胸痛", "心悸", "心跳", "心慌", "心", "血压", "胸闷"], "心血管内科"),
    (["皮疹", "瘙痒", "皮肤", "痘", "疹", "脱发", "红斑"], "皮肤科"),
    (["眼", "视力", "眼睛", "流泪", "眼痛"], "眼科"),
    (["腰", "关节", "骨", "颈", "肩", "膝", "手麻", "腿"], "骨科"),
    (["尿频", "尿急", "尿痛", "尿", "肾"], "泌尿外科"),
    (["月经", "白带", "阴道", "妇科", "怀孕", "孕"], "妇产科"),
    (["发热", "发烧", "高热"], "内科"),
    (["肌肉", "酸痛", "浑身", "全身疼", "乏力", "疲劳", "无力"], "内科"),
]
_DEPT_TIER_LOW = {"肿瘤科", "肿瘤外科", "肿瘤内科", "传染科", "其他科室", "其他综合"}
_DEPT_TIER_MID = {"内科", "外科", "中医科", "中医综合", "急诊科", "儿科", "小儿内科", "儿科综合"}


def _rank_depts(depts: list[str], symptom_names: list[str]) -> list[str]:
    """对图谱候选科室按临床相关度排序（降噪：肿瘤/传染科不相关时置后）。

    M8.10 补正（借鉴业界 Tier2 分诊设计）：图谱 VISITS 缺正确科室时（如'恶心'的
    候选里没有消化内科），把症状专科提示补入候选——提示表是临床分诊路由知识，
    补的是'该看哪个科'而非具体疾病事实，不违反红线4。
    """
    text = "".join(symptom_names) + (symptom_names[0] if symptom_names else "")
    hint_depts: list[str] = []
    for keywords, specialty in _SYMPTOM_SPECIALTY_HINTS:
        if any(k in text for k in keywords):
            hint_depts.append(specialty)
    candidates = list(dict.fromkeys(hint_depts + list(depts)))

    def score(d: str) -> int:
        s = 0
        if any(d == h for h in hint_depts):
            s += 150
        elif any(h in d or d in h for h in hint_depts):
            s += 100
        if d in _DEPT_TIER_LOW:
            s -= 50
        elif d in _DEPT_TIER_MID:
            s += 10
        else:
            s += 30
        return s

    return sorted(candidates, key=score, reverse=True)


def _query_exams(repo: GraphRepo, names: list[str]) -> list[str]:
    """症状/疾病 → 相关检查：(疾病)-[:REQUIRES_EXAM]->(检查)，症状经 PRESENTS 反查疾病。"""
    exams: list[str] = []
    rows = repo.query(
        "MATCH (d:Disease)-[:REQUIRES_EXAM]->(e:Exam) WHERE d.name IN $names "
        "RETURN DISTINCT e.name AS exam LIMIT 30",
        {"names": names},
    )
    exams = [r["exam"] for r in rows]
    if not exams and names:
        rows = repo.query(
            "MATCH (s:Symptom {name: $name})<-[:PRESENTS]-(d:Disease)-[:REQUIRES_EXAM]->(e:Exam) "
            "RETURN DISTINCT e.name AS exam LIMIT 30",
            {"name": names[0]},
        )
        exams = [r["exam"] for r in rows]
    return exams


def department_agent(state: dict) -> dict:
    """科室推荐：图谱查 症状→科室 + 相关检查；科室按临床相关度排序降噪；LLM 输出结构化就医建议。"""
    names = [e["name"] for e in state.get("entities", [])]
    repo = GraphRepo()
    try:
        depts = _rank_depts(_query_depts(repo, names), names)
        exams = _filter_sensitive_texts(_query_exams(repo, names))
    finally:
        repo.close()
    if not depts:
        return {"answer": "根据知识库信息，暂未检索到与该症状对应的明确科室建议，建议前往医院导诊台咨询。",
                "audit": ["department_agent: 图谱无科室 → 如实说明"]}
    primary = depts[0]
    question = state.get("collected_text") or state.get("question", "")
    try:
        data = chat_json(
            "你是医院分诊导诊助手。用户描述了一个症状，请给出实用的就医指导。"
            f"首选科室是{primary}，结论必须以该科室为主。"
            "重要：不要列举或猜测'可能是什么病/哪些疾病会导致该症状'，只做就医指导。"
            "结论与建议应覆盖用户提到的全部症状部位（可用其原话表述，如'头痛和手痛'），不要只提其中一部分。"
            "sections 给出 2-4 条实用建议，例如：就诊前记录症状的时间/部位/性质/伴随表现；"
            "近期的诱因（劳累、受凉、情绪、睡眠）；出现哪些加重表现需尽快就医。"
            "回答正文自然语言表述来源（如'根据知识库信息'），不出现技术节点名或箭头符号。"
            "输出严格 JSON：{\"answer\": \"一句话结论，须涵盖用户提到的全部症状部位（如'根据知识库信息，头痛和手痛建议就诊神经内科'）\","
            " \"sections\": [{\"title\": \"就医建议\", \"points\": [\"实用建议1\", \"实用建议2\"]}],"
            " \"tags\": {\"symptoms\": [\"症状名\"], \"departments\": [\"科室名\"]}}",
            f"用户症状原话：{question}（知识库匹配症状：{'、'.join(names[:3])}）\n首选科室：{primary}\n备选科室：{'、'.join(depts[1:4])}"
            f"\n相关检查：{'、'.join(exams[:6]) or '暂无'}",
        )
        answer = str(data.get("answer", "")).strip()
        sections = data.get("sections") or []
        tags = data.get("tags") or {}
        if not tags.get("departments"):
            tags["departments"] = [primary]
        if not tags.get("symptoms"):
            tags["symptoms"] = names[:3]
    except Exception:
        answer = f"根据知识库信息，{question}建议就诊{primary}。"
        sections = [{"title": "就医建议", "points": [
            "携带身份证与既往病历资料", "提前记录症状发作时间与规律", "如有不适及时就医"]}]
        tags = {"symptoms": names[:5], "departments": depts[:3]}
    graph_ev = [{"subject": n, "relation": "VISITS", "object": d, "source": "graph"}
                for n in names[:2] for d in depts[:2]]
    return {"answer": answer, "answer_sections": sections, "answer_tags": tags,
            "graph_evidence": graph_ev,
            "evidence_pool": [{"type": "GRAPH_NODE", "ref": f"VISITS:{n}→{d}",
                               "quote": f"{n} → {d}", "text": f"{n} → {d}", "score": 1.0}
                              for n in names[:2] for d in depts[:2]],
            "audit": [f"department_agent: 科室排序 {depts[:5]} 检查 {exams[:6]}"]}


# ------------------------------------------------------------------
# 用药推荐
# ------------------------------------------------------------------

def _drugs_for_diseases(repo: GraphRepo, disease_names: list[str]) -> list[str]:
    """疾病 → 治疗药物（Drug-TREATS->Disease），去重。"""
    drugs: list[str] = []
    for n in disease_names:
        rows = repo.query(
            "MATCH (d:Drug)-[:TREATS]->(dis:Disease {name: $name}) RETURN DISTINCT d.name AS drug LIMIT 8",
            {"name": n},
        )
        drugs.extend(r["drug"] for r in rows)
    return list(dict.fromkeys(drugs))


def _drugs_for_symptoms(repo: GraphRepo, symptom_names: list[str]) -> tuple[list[str], list[str]]:
    """症状 → 相关疾病 → 治疗药物，按共识度排序。返回 (药物列表, 关联疾病列表)。"""
    diseases: list[str] = []
    for s in symptom_names:
        rows = repo.query(
            "MATCH (dis:Disease)-[:PRESENTS]->(sym:Symptom {name: $name}) "
            "RETURN dis.name AS disease LIMIT 12",
            {"name": s},
        )
        diseases.extend(r["disease"] for r in rows)
    diseases = _filter_sensitive_texts(list(dict.fromkeys(diseases)))[:8]
    if not diseases:
        return [], []
    drug_freq: dict[str, int] = {}
    for dis in diseases:
        rows = repo.query(
            "MATCH (d:Drug)-[:TREATS]->(dis:Disease {name: $name}) RETURN DISTINCT d.name AS drug LIMIT 8",
            {"name": dis},
        )
        seen_this: set[str] = set()
        for r in rows:
            drug = r["drug"]
            if drug in seen_this:
                continue
            seen_this.add(drug)
            drug_freq[drug] = drug_freq.get(drug, 0) + 1
    ranked = [d for d, _ in sorted(drug_freq.items(), key=lambda kv: -kv[1])]
    return ranked, diseases


def _clean_instruction_field(text: str) -> str:
    """说明书字段条目级清洗：按分隔符拆分，剔除含敏感词的条目，保留其余（呈现层过滤，不改数据）。

    CMeKG 说明书是分号分隔的条目列表（如 NSAIDs 适应症含"性病"——数据源合法，
    但呈现在头疼用药卡不合适），须在条目粒度剔除而非整卡丢弃。
    """
    parts = re.split(r"[；;]", str(text))
    kept = [p.strip() for p in parts if p.strip() and not _is_sensitive(p)]
    return "；".join(dict.fromkeys(kept))


def _drug_detail_points(drug_name: str) -> list[str]:
    """单药说明书要点（功能主治 / 不良反应 / 禁忌），数据缺失则跳过对应项。"""
    detail = drug_db.lookup(drug_name)
    if not detail:
        return [f"{drug_name}：说明书信息暂缺，请遵医嘱或咨询药师"]
    points: list[str] = []
    if detail.get("indication"):
        cleaned = _clean_instruction_field(detail["indication"])
        if cleaned:
            points.append(f"功能主治：{cleaned}")
    if detail.get("side_effects"):
        cleaned = _clean_instruction_field(detail["side_effects"])
        if cleaned:
            points.append(f"常见不良反应：{cleaned}")
    if detail.get("contraindication"):
        cleaned = _clean_instruction_field(detail["contraindication"])
        if cleaned:
            points.append(f"禁忌：{cleaned}")
    if not points:
        points.append(f"{drug_name}：说明书信息暂缺，请遵医嘱或咨询药师")
    return points


def _build_drug_response(state: dict, drugs: list[str], target_text: str,
                         target_is_symptom: bool, related_diseases: list[str]) -> dict:
    """结构化用药回答：每药一个卡片（功能主治/不良反应/禁忌）+ 用药提醒。"""
    top = drugs[:3]
    if target_is_symptom:
        answer = (f"根据知识库信息，{target_text}可能由多种原因引起，知识库中相关疾病的常用药物包括{'、'.join(top)}，"
                  f"建议明确病因后在医生指导下使用。")
    else:
        answer = f"根据知识库信息，{target_text}的常用药物包括{'、'.join(top)}，请在医生或药师指导下使用。"
    sections: list[dict] = []
    for d in top:
        sections.append({"title": d, "points": _drug_detail_points(d)})
    sections.append({"title": "用药提醒", "points": [
        "请遵医嘱或药师指导使用，切勿自行调整剂量或停药",
        "用药前请仔细阅读说明书，关注禁忌与不良反应",
        "如出现明显不适或不良反应，请立即停药并就医",
        "孕妇、哺乳期、儿童及肝肾功能不全者用药前务必咨询医生"]})
    tags = {"symptoms": [target_text] if target_is_symptom else [], "departments": []}
    graph_ev = [{"subject": d, "relation": "TREATS", "object": target_text, "source": "graph"} for d in top]
    evidence_pool = [{"type": "GRAPH_NODE", "ref": f"TREATS:{d}→{target_text}",
                      "quote": f"{d} → {target_text}", "text": f"{d} → {target_text}", "score": 1.0} for d in top]
    return {"answer": answer, "answer_sections": sections, "answer_tags": tags,
            "graph_evidence": graph_ev, "evidence_pool": evidence_pool,
            "audit": [f"drug_agent: 推荐药物 {top}（症状路径={target_is_symptom}）"]}


def drug_agent(state: dict) -> dict:
    """用药建议（M8.4）：对症推荐具体药物 + 功能主治/不良反应/禁忌/用药提醒。"""
    entities = state.get("entities", [])
    disease_names = [e["name"] for e in entities if e.get("label") == "Disease"]
    symptom_names = [e["name"] for e in entities if e.get("label") == "Symptom"]
    repo = GraphRepo()
    try:
        if disease_names:
            drugs = _drugs_for_diseases(repo, disease_names)
            if drugs:
                return _build_drug_response(state, drugs, disease_names[0], False, [])
            symptom_names = symptom_names or disease_names
        if symptom_names:
            drugs, related = _drugs_for_symptoms(repo, symptom_names)
            drugs, _ = _filter_high_risk_drugs(drugs) if "_filter_high_risk_drugs" in dir() else (drugs, [])
            if drugs:
                return _build_drug_response(state, drugs, symptom_names[0], True, related)
    finally:
        repo.close()
    return {"answer": "根据知识库信息，暂未检索到对应的用药信息，建议咨询医生或药师。",
            "audit": ["drug_agent: 图谱无药物 → 如实说明"]}


# ------------------------------------------------------------------
# 医学知识问答（grounded 分流）
# ------------------------------------------------------------------

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
    pool = _filter_sensitive_evidence(result["evidence_pool"])

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


_LEAD_STRIP_RE = re.compile(r"^(我|我们|你|最近|这几天|今天|昨天|现在|一直|总是|老是|感觉|有点|比较|非常|特别|好像|可能)+")


def _lead_phrase(question: str) -> str:
    """剥离主诉前缀（人称/时间/程度词）得到结论句用的症状短语（M8.21）。"""
    lead = _LEAD_STRIP_RE.sub("", str(question)).strip("，。！！?？ 、,. ")
    return lead


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
    # M8.21 结论句用用户原话短语（剥离"我/最近"等前缀），保证覆盖用户提到的全部部位
    # （"手和头疼"不再只剩"头痛"）；原话过长或剥离后为空时退回实体名模板。
    lead = _lead_phrase(question)
    if lead and len(lead) <= 30:
        answer = f"根据知识库信息，{lead}，建议就诊{primary}。"
    else:
        answer = f"根据知识库信息，{'、'.join(symptom_names[:2])}建议就诊{primary}。"
    try:
        data = chat_json(
            "你是医院分诊导诊助手。用户描述了一个症状，请给出实用的就医建议要点。"
            f"症状对应的就诊科室是{primary}。"
            "重要：不要列举或猜测'可能是什么病/哪些疾病会导致该症状'，只做就医指导。"
            "输出严格 JSON："
            " {\"sections\": [{\"title\": \"就医建议\", \"points\": [\"实用建议1\", \"实用建议2\", \"实用建议3\"]}]}，"
            "给 2-4 条要点，例如：就诊前记录症状的时间/部位/性质/伴随表现；"
            "近期的诱因（劳累、受凉、情绪、睡眠）；出现哪些加重表现需尽快就医。",
            f"症状：{'、'.join(symptom_names[:3])}\n就诊科室：{primary}"
            f"\n相关检查：{'、'.join(exams[:6]) or '暂无'}\n用户原话：{question}"
            + (f"\n{format_history(state.get('recent_history'))}" if state.get("recent_history") else ""),
        )
        sections = data.get("sections") or []
        if not sections:
            raise ValueError("empty sections")
    except Exception:
        sections = [{"title": "就医建议", "points": [
            "就诊前记录症状出现的时间、部位、性质与伴随表现",
            "回顾近期诱因（劳累、受凉、情绪波动、睡眠）", "如有加重请及时就医"]}]
    tags = {"symptoms": symptom_names[:3], "departments": [primary]}
    return {"refusal": False, "answer": answer, "high_risk_query": _is_high_risk(question),
            "answer_sections": sections, "answer_tags": tags,
            "audit": [f"symptom_triage: 科室 {depts[:4]} 检查 {exams[:6]}"]}


# ------------------------------------------------------------------
# 就医指导与闲聊
# ------------------------------------------------------------------

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
        answer = "根据知识库信息，建议前往医院就诊，听从医生安排治疗。"
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
