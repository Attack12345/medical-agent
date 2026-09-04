"""药品说明书索引（M8.4）：从 CMeKG 药品数据构建 药品名 → 说明书详情 的检索索引。

数据源：CMeKG 中文医学知识图谱 drug.json（清华知识工程实验室，镜像 MenglinLu/Web-crawler），
17496 种药物，含 适应症/功能主治、不良反应、禁忌、用法用量、注意事项 结构化字段。

职责：
- build：解析原始 drug.json → data/cleaned/drug_details.json（规范化药名为键）
- lookup：给定图谱药名（可能带剂型/品牌前缀），模糊匹配回说明书详情
仅做数据接入与检索，供 drug_agent 输出"功能主治/副作用/禁忌/用药提醒"（红线4：只陈述数据事实）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # medical-agent/
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "cmekg_drug.json"
CLEANED_FILE = PROJECT_ROOT / "data" / "cleaned" / "drug_details.json"

# 剂型后缀（剥离后做模糊匹配）
_DOSAGE_RE = re.compile(
    r"(缓释片|控释片|分散片|咀嚼片|含片|舌下片|滴丸|滴眼液|滴鼻液|滴耳液|喷雾剂|气雾剂|"
    r"注射液|注射用|葡萄糖注射液|氯化钠注射液|口服液|口服溶液|糖浆|颗粒|胶囊|软胶囊|片|丸|散|"
    r"软膏|乳膏|凝胶|贴剂|贴膏|搽剂|洗剂|栓|溶液|酊|膏|霜|洗液|灌肠剂|植入剂)$"
)

# 说明书字段（值均为短语数组，join 成文本）
_FIELD_MAP = {
    "indication": ["适应症", "适应证", "功能主治"],
    "side_effects": ["不良反应"],
    "contraindication": ["禁忌", "禁忌证"],
    "usage": ["用法用量"],
    "notes": ["注意事项"],
}
_MAX_LEN = 160  # 单字段截断，避免回答过长


def _join_field(rec: dict, keys: list[str]) -> str:
    for k in keys:
        v = rec.get(k)
        if v:
            if isinstance(v, list):
                text = "；".join(str(x).strip() for x in v if str(x).strip())
            else:
                text = str(v).strip()
            if text:
                return text[:_MAX_LEN]
    return ""


def normalize_name(name: str) -> str:
    """规范化药名：去空白、剥离剂型后缀。"""
    n = re.sub(r"\s+", "", str(name))
    return _DOSAGE_RE.sub("", n).strip() or n


def build(raw_path: Path = RAW_FILE, out_path: Path = CLEANED_FILE) -> int:
    """解析原始 CMeKG drug.json → drug_details.json。返回药物条数。"""
    index: dict[str, dict] = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = str(rec.get("中心词", "")).strip()
            if not name:
                continue
            detail = {"name": name}
            for out_key, src_keys in _FIELD_MAP.items():
                detail[out_key] = _join_field(rec, src_keys)
            # 只保留至少有一项有效说明书内容的药物
            if any(detail[k] for k in _FIELD_MAP):
                index[name] = detail
                norm = normalize_name(name)
                if norm != name and norm not in index:
                    index[norm] = detail
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return len(index)


_db: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _db
    if _db is None:
        if not CLEANED_FILE.exists():
            if RAW_FILE.exists():
                build()
            else:
                _db = {}
                return _db
        _db = json.loads(CLEANED_FILE.read_text(encoding="utf-8"))
    return _db


def lookup(name: str) -> dict | None:
    """按药名查说明书详情；先精确，再规范化去剂型匹配。返回 detail dict 或 None。"""
    db = _load()
    if not db or not name:
        return None
    n = re.sub(r"\s+", "", str(name))
    if n in db:
        return db[n]
    norm = normalize_name(n)
    if norm in db:
        return db[norm]
    # 前缀匹配兜底（品牌名+通用名场景，如"达发新(环酯红霉素片)"）
    for key, detail in db.items():
        if len(key) >= 3 and (n.startswith(key) or key.startswith(n)) :
            return detail
    return None
