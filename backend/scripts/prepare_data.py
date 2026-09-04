"""数据集接入+清洗入库（M1 交付）。

用法：
  python prepare_data.py                # 下载真实数据集（D1 疾病库 + D3 cMedQA2）并清洗
  python prepare_data.py --skip-download  # 仅清洗 data/raw/ 已有数据（下载过时用）
  python prepare_data.py --generate     # 模拟数据兜底（§2.2），输出同构 cleaned JSON

输出（§2.4）：
  data/cleaned/diseases.json   疾病主数据（Disease 节点属性 + 关联实体名）
  data/cleaned/relations.json  关系三元组（图谱边，§3.1 关系语义）
  data/cleaned/qa_pairs.json   问答对（向量检索语料）
  data/source_meta.json        数据来源标记 {datasets: [...], cleaned_at}
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # medical-agent/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
SIMULATED_DIR = DATA_DIR / "simulated"

QA_TARGET = 5000          # §2.1 D3 用量
DISEASE_MIN = 300         # §2.1 D1 用量
ENTITY_NORM_FILE = DATA_DIR / "entity_norm.json"

# ---------- 数据源（§2.1） ----------

D1_URL = "https://raw.githubusercontent.com/liuhuanyong/QASystemOnMedicalKG/master/data/medical.json"
D3_QUESTION_URL = "https://github.com/zhangsheng93/cMedQA2/raw/master/question.zip"
D3_ANSWER_URL = "https://github.com/zhangsheng93/cMedQA2/raw/master/answer.zip"
# 药品说明书（CMeKG 中文医学知识图谱镜像，17496 药物，含功能主治/不良反应/禁忌/用法用量）
DRUG_URL = ("https://raw.githubusercontent.com/MenglinLu/Web-crawler/master/cmekg/"
            "%E4%B8%AD%E6%96%87%E5%8C%BB%E5%AD%A6%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%E6%95%B0%E6%8D%AE/drug.json")
# 药品说明书（CMeKG 中文医学知识图谱镜像，17496 药物，含功能主治/不良反应/禁忌/用法用量）
DRUG_URL = ("https://raw.githubusercontent.com/MenglinLu/Web-crawler/master/cmekg/"
            "%E4%B8%AD%E6%96%87%E5%8C%BB%E5%AD%A6%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1%E6%95%B0%E6%8D%AE/drug.json")

# 急症词表（§6 S002 同表，用于真实数据 severity 推断）
EMERGENCY_SYMPTOMS = {"胸痛", "呼吸困难", "意识模糊", "持续高热", "大量出血", "剧烈头痛"}

# 同义归一初始表（§2.4.2，运行后落 data/entity_norm.json 可维护）
DEFAULT_NORM = {
    "高血压病": "高血压",
    "糖尿病2型": "2型糖尿病",
    "二型糖尿病": "2型糖尿病",
    "脑血管疾病": "脑血管病",
    "甲状腺功能亢进症": "甲亢",
    "甲状腺功能亢进": "甲亢",
    "上呼吸道感染": "感冒",
}


# ---------- 工具 ----------

def log(msg: str) -> None:
    print(f"[prepare_data] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_field(items) -> list[str]:
    """§2.4.1 统一字段：字符串列表；去空白、去重、小写归一（英文保留原样）。"""
    if not items:
        return []
    out: list[str] = []
    for it in items:
        if it is None:
            continue
        s = str(it).strip()
        if not s:
            continue
        s = re.sub(r"\s+", "", s)
        if not s:
            continue
        s = s.lower() if re.fullmatch(r"[A-Za-z0-9\s./+-]+", s) else s
        if s not in out:
            out.append(s)
    return out


def normalize_name(name: str, norm: dict[str, str]) -> str:
    """§2.4.2 同义归一：命中归一表即替换。"""
    return norm.get(name, name)


def load_norm() -> dict[str, str]:
    if ENTITY_NORM_FILE.exists():
        return read_json(ENTITY_NORM_FILE)
    write_json(ENTITY_NORM_FILE, DEFAULT_NORM)
    return dict(DEFAULT_NORM)


def download(url: str, dest: Path, timeout: int = 300) -> bool:
    """下载数据文件（读环境变量 https_proxy）；已存在则跳过。"""
    if dest.exists() and dest.stat().st_size > 0:
        log(f"已存在，跳过下载: {dest.name}")
        return True
    log(f"下载 {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        log(f"下载完成: {dest.name} ({dest.stat().st_size} bytes)")
        return True
    except Exception as e:  # 网络失败不致命，提示 --generate 兜底
        log(f"下载失败 {url}: {e}")
        return False


# ---------- D1 解析 ----------

def parse_d1(path: Path) -> list[dict]:
    """解析 medical.json（JSON Lines，MongoDB 导出）。"""
    raw = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return raw


def extract_drugs(rec: dict) -> list[str]:
    """药物：优先 recommand_drug；空则从 drug_detail 提取通用名（括号内）。"""
    drugs = clean_field(rec.get("recommand_drug") or [])
    if drugs:
        return drugs
    out: list[str] = []
    for item in rec.get("drug_detail") or []:
        m = re.search(r"\(([^)]+)\)", str(item))
        name = m.group(1) if m else str(item)
        name = name.strip()
        if name and name not in out:
            out.append(name)
    return out


def parse_d1_to_diseases(raw: list[dict], norm: dict[str, str]) -> list[dict]:
    """原始记录 → 疾病主数据（cleaned schema，§2.3/§3.1 对齐）。"""
    diseases: list[dict] = []
    for rec in raw:
        name = normalize_name(str(rec.get("name", "")).strip(), norm)
        symptoms = [normalize_name(s, norm) for s in clean_field(rec.get("symptom"))]
        if not name or not symptoms:  # 无症状的疾病不建图谱（无可扩展关系）
            continue
        departments = [normalize_name(d, norm) for d in clean_field(rec.get("cure_department"))]
        diseases.append({
            "name": name,
            "aliases": clean_field(rec.get("alias") or []),
            # 真实数据无严重程度字段：症状含急症词表 → "重"，否则 "中"（source_meta 注明口径）
            "severity": "重" if set(symptoms) & EMERGENCY_SYMPTOMS else "中",
            "summary": str(rec.get("desc") or "").strip(),
            "symptoms": symptoms,
            "drugs": [normalize_name(d, norm) for d in extract_drugs(rec)],
            "departments": departments,
            "exams": clean_field(rec.get("check")),
            "foods": [],
            "populations": [],
            "notices": [s.strip() for s in str(rec.get("prevent") or "").split("\n") if s.strip()][:3],
            "complications": [normalize_name(c, norm) for c in clean_field(rec.get("acompany"))],
        })
    return diseases


def expand_relations(diseases: list[dict], source: str) -> list[dict]:
    """疾病主数据 → 关系三元组（§3.1 语义；去重保序）。"""
    rels: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(subject: str, relation: str, obj: str) -> None:
        key = (subject, relation, obj)
        if key not in seen:
            seen.add(key)
            rels.append(key)

    for d in diseases:
        name = d["name"]
        for s in d["symptoms"]:
            add(name, "PRESENTS", s)
            for dep in d["departments"]:  # 症状→科室（疾病带科室，映射到其症状上）
                add(s, "VISITS", dep)
        for drug in d["drugs"]:
            add(drug, "TREATS", name)
        for exam in d["exams"]:
            add(name, "REQUIRES_EXAM", exam)
        for comp in d["complications"]:
            add(name, "COMPLICATES", comp)
    return [{"subject": s, "relation": r, "object": o, "source": source} for s, r, o in rels]


# ---------- D3 解析 ----------

def parse_d3(raw_dir: Path) -> list[dict]:
    """cMedQA2 question/answer zip → [(qid, question, answer)]（每问题取首个回答）。"""
    q_zip = raw_dir / "question.zip"
    a_zip = raw_dir / "answer.zip"
    if not (q_zip.exists() and a_zip.exists()):
        return []

    def read_zip(zf: zipfile.ZipFile, name: str) -> list[str]:
        with zf.open(name) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            return [ln.rstrip("\n") for ln in text]

    with zipfile.ZipFile(q_zip) as zf:
        q_lines = read_zip(zf, "question.csv")[1:]  # 跳过表头
    with zipfile.ZipFile(a_zip) as zf:
        a_lines = read_zip(zf, "answer.csv")[1:]

    answers: dict[str, str] = {}
    for ln in a_lines:
        parts = ln.split(",", 2)
        if len(parts) >= 3 and parts[1] not in answers:
            answers[parts[1]] = parts[2].strip()

    out = []
    for ln in q_lines:
        parts = ln.split(",", 1)
        if len(parts) < 2:
            continue
        qid, question = parts[0].strip(), parts[1].strip()
        ans = answers.get(qid, "")
        if qid and question and ans:
            out.append({"qid": qid, "question": question, "answer": ans})
    return out


def clean_qa(raw_qa: list[dict]) -> list[dict]:
    """§2.4 清洗问答对：去空白/去重/截取 QA_TARGET 条。"""
    seen: set[str] = set()
    out: list[dict] = []
    for item in raw_qa:
        q = re.sub(r"\s+", "", item["question"])
        a = re.sub(r"\s+", "", item["answer"])
        if len(q) < 4 or not a or q in seen:
            continue
        seen.add(q)
        out.append({"qid": item["qid"], "question": q, "answer": a})
        if len(out) >= QA_TARGET:
            break
    return out


# ---------- 模拟数据生成器（§2.2，仅兜底） ----------

# 常见疾病精细预设（词表主干，保证图谱关系非随机噪音）
COMMON_DISEASES = [
    {"name": "高血压", "severity": "中", "symptoms": ["头晕", "头痛", "耳鸣", "心悸", "失眠"],
     "drugs": ["硝苯地平", "氨氯地平", "缬沙坦", "氢氯噻嗪"], "departments": ["心血管内科"],
     "exams": ["血压测量", "心电图", "血常规"], "foods": ["高盐食物", "肥肉"], "populations": ["老年人"],
     "notices": ["低盐低脂饮食", "规律服药不可自行停药"], "complications": ["冠心病", "脑卒中"]},
    {"name": "2型糖尿病", "severity": "中", "symptoms": ["多饮", "多尿", "多食", "体重下降", "乏力"],
     "drugs": ["二甲双胍", "格列本脲", "阿卡波糖"], "departments": ["内分泌科"],
     "exams": ["血糖检测", "糖化血红蛋白", "尿常规"], "foods": ["甜食", "含糖饮料"], "populations": ["中老年人"],
     "notices": ["控制主食量", "餐后适当运动"], "complications": ["糖尿病肾病", "糖尿病足"]},
    {"name": "感冒", "severity": "轻", "symptoms": ["鼻塞", "流鼻涕", "打喷嚏", "咽喉痛", "咳嗽", "发热"],
     "drugs": ["感冒灵颗粒", "对乙酰氨基酚", "板蓝根颗粒"], "departments": ["呼吸内科"],
     "exams": ["血常规"], "foods": ["辛辣食物", "冷饮"], "populations": [],
     "notices": ["多喝水多休息", "注意保暖"], "complications": ["支气管炎", "肺炎"]},
    {"name": "胃炎", "severity": "中", "symptoms": ["胃痛", "胃胀", "反酸", "嗳气", "恶心"],
     "drugs": ["奥美拉唑", "铝碳酸镁", "多潘立酮"], "departments": ["消化内科"],
     "exams": ["胃镜", "幽门螺杆菌检测"], "foods": ["辛辣食物", "油炸食物", "咖啡"], "populations": [],
     "notices": ["规律三餐", "避免过饱", "戒烟限酒"], "complications": ["胃溃疡"]},
    {"name": "偏头痛", "severity": "中", "symptoms": ["头痛", "恶心", "畏光", "畏声", "视觉模糊"],
     "drugs": ["布洛芬", "对乙酰氨基酚", "佐米曲普坦"], "departments": ["神经内科"],
     "exams": ["脑电图", "头颅CT"], "foods": ["巧克力", "红酒"], "populations": ["女性"],
     "notices": ["规律作息", "避免精神紧张"], "complications": []},
    {"name": "冠心病", "severity": "重", "symptoms": ["胸痛", "胸闷", "心悸", "呼吸困难", "乏力"],
     "drugs": ["阿司匹林", "硝酸甘油", "美托洛尔", "阿托伐他汀"], "departments": ["心血管内科"],
     "exams": ["心电图", "冠脉造影", "心脏彩超"], "foods": ["肥肉", "油炸食物"], "populations": ["老年人"],
     "notices": ["避免剧烈运动", "随身携带急救药物"], "complications": ["心肌梗死", "心力衰竭"]},
    {"name": "肺炎", "severity": "重", "symptoms": ["发热", "咳嗽", "咳痰", "胸痛", "呼吸困难"],
     "drugs": ["阿莫西林", "头孢呋辛", "左氧氟沙星"], "departments": ["呼吸内科"],
     "exams": ["胸部CT", "血常规", "痰培养"], "foods": ["辛辣食物"], "populations": ["老年人", "儿童"],
     "notices": ["充分休息", "多饮水", "遵医嘱完成疗程"], "complications": ["胸膜炎"]},
    {"name": "哮喘", "severity": "重", "symptoms": ["喘息", "咳嗽", "胸闷", "呼吸困难", "气短"],
     "drugs": ["沙丁胺醇", "布地奈德", "孟鲁司特"], "departments": ["呼吸内科"],
     "exams": ["肺功能检查", "支气管激发试验"], "foods": ["海鲜", "冷饮"], "populations": ["儿童"],
     "notices": ["远离过敏原", "规范使用吸入剂"], "complications": ["呼吸衰竭"]},
    {"name": "胃溃疡", "severity": "重", "symptoms": ["胃痛", "反酸", "恶心", "黑便", "呕血"],
     "drugs": ["奥美拉唑", "枸橼酸铋钾", "阿莫西林"], "departments": ["消化内科"],
     "exams": ["胃镜", "幽门螺杆菌检测"], "foods": ["辛辣食物", "咖啡", "浓茶"], "populations": [],
     "notices": ["规律饮食", "避免空腹饮酒"], "complications": ["胃出血", "胃穿孔"]},
    {"name": "急性心肌梗死", "severity": "急症", "symptoms": ["胸痛", "呼吸困难", "大汗", "恶心", "濒死感"],
     "drugs": ["阿司匹林", "硝酸甘油", "阿托伐他汀"], "departments": ["心血管内科", "急诊科"],
     "exams": ["心电图", "心肌酶谱", "冠脉造影"], "foods": [], "populations": ["老年人"],
     "notices": ["立即就医", "保持安静平卧"], "complications": ["心力衰竭", "心源性休克"]},
    {"name": "脑出血", "severity": "急症", "symptoms": ["剧烈头痛", "意识模糊", "呕吐", "肢体偏瘫", "言语不清"],
     "drugs": ["甘露醇"], "departments": ["神经外科", "急诊科"],
     "exams": ["头颅CT", "头颅MRI"], "foods": [], "populations": ["老年人"],
     "notices": ["立即就医", "保持呼吸道通畅"], "complications": ["脑疝"]},
    {"name": "脑梗死", "severity": "急症", "symptoms": ["口角歪斜", "言语不清", "肢体无力", "意识模糊", "头晕"],
     "drugs": ["阿司匹林", "氯吡格雷", "阿托伐他汀"], "departments": ["神经内科", "急诊科"],
     "exams": ["头颅CT", "头颅MRI"], "foods": ["高盐食物"], "populations": ["老年人"],
     "notices": ["立即就医", "把握溶栓时间窗"], "complications": ["偏瘫"]},
    {"name": "甲状腺功能亢进症", "severity": "中", "symptoms": ["心悸", "多汗", "体重下降", "手抖", "易怒"],
     "drugs": ["甲巯咪唑", "丙硫氧嘧啶"], "departments": ["内分泌科"],
     "exams": ["甲状腺功能", "甲状腺彩超"], "foods": ["海带", "紫菜"], "populations": ["女性"],
     "notices": ["避免高碘饮食", "规律复查"], "complications": ["甲亢性心脏病"]},
    {"name": "类风湿关节炎", "severity": "中", "symptoms": ["关节肿痛", "晨僵", "关节变形", "乏力"],
     "drugs": ["甲氨蝶呤", "布洛芬", "来氟米特"], "departments": ["风湿免疫科"],
     "exams": ["类风湿因子", "关节X线"], "foods": [], "populations": ["中老年女性"],
     "notices": ["注意关节保暖", "适度功能锻炼"], "complications": ["关节畸形"]},
    {"name": "青光眼", "severity": "重", "symptoms": ["眼胀", "眼痛", "视力下降", "头痛", "恶心"],
     "drugs": ["毛果芸香碱", "噻吗洛尔", "布林佐胺"], "departments": ["眼科"],
     "exams": ["眼压测量", "眼底检查"], "foods": [], "populations": ["老年人"],
     "notices": ["避免长时间低头", "定期测眼压"], "complications": ["失明"]},
    {"name": "过敏性休克", "severity": "急症", "symptoms": ["呼吸困难", "血压下降", "意识模糊", "皮肤瘙痒", "喉头水肿"],
     "drugs": ["肾上腺素", "地塞米松"], "departments": ["急诊科"],
     "exams": [], "foods": [], "populations": [],
     "notices": ["立即就医", "远离过敏原"], "complications": []},
    {"name": "肾结石", "severity": "中", "symptoms": ["腰痛", "血尿", "恶心", "尿频", "尿急"],
     "drugs": ["坦索罗辛", "布洛芬"], "departments": ["泌尿外科"],
     "exams": ["泌尿系彩超", "尿常规"], "foods": ["菠菜", "浓茶"], "populations": [],
     "notices": ["多饮水", "限制高草酸食物"], "complications": ["肾积水"]},
    {"name": "胆囊炎", "severity": "中", "symptoms": ["右上腹痛", "恶心", "呕吐", "发热", "黄疸"],
     "drugs": ["山莨菪碱", "头孢曲松", "熊去氧胆酸"], "departments": ["肝胆外科"],
     "exams": ["腹部彩超", "血常规"], "foods": ["油炸食物", "肥肉", "蛋黄"], "populations": [],
     "notices": ["低脂饮食", "规律进食"], "complications": ["胆囊穿孔"]},
    {"name": "带状疱疹", "severity": "中", "symptoms": ["皮肤水疱", "神经痛", "发热", "乏力"],
     "drugs": ["阿昔洛韦", "加巴喷丁", "维生素B1"], "departments": ["皮肤科"],
     "exams": [], "foods": [], "populations": ["老年人"],
     "notices": ["保持皮疹清洁", "避免抓挠"], "complications": ["带状疱疹后遗神经痛"]},
    {"name": "缺铁性贫血", "severity": "轻", "symptoms": ["乏力", "头晕", "面色苍白", "心悸", "注意力不集中"],
     "drugs": ["硫酸亚铁", "维生素C"], "departments": ["血液科"],
     "exams": ["血常规", "血清铁蛋白"], "foods": [], "populations": ["女性"],
     "notices": ["补充含铁食物", "餐后服用铁剂"], "complications": []},
]

# 组合用词表（§2.2：300 疾病 = 精细预设 + 词表组合）
DISEASE_POOL = [
    "支气管炎", "咽喉炎", "扁桃体炎", "鼻炎", "鼻窦炎", "中耳炎", "结膜炎", "角膜炎", "牙周炎", "口腔溃疡",
    "食管炎", "十二指肠溃疡", "肠炎", "阑尾炎", "痔疮", "肝炎", "脂肪肝", "肝硬化", "胰腺炎", "结肠炎",
    "膀胱炎", "前列腺炎", "尿道炎", "痛风", "骨质疏松", "腰椎间盘突出", "颈椎病", "肩周炎", "腱鞘炎", "膝关节炎",
    "湿疹", "荨麻疹", "痤疮", "银屑病", "白癜风", "真菌感染", "鸡眼", "疖肿", "神经性皮炎", "脂溢性皮炎",
    "抑郁症", "焦虑症", "失眠症", "神经衰弱", "癫痫", "帕金森病", "面神经麻痹", "三叉神经痛", "坐骨神经痛", "多发性硬化",
    "心肌炎", "心律失常", "心力衰竭", "心包炎", "风湿性心脏病", "先天性心脏病", "肺动脉高压", "静脉曲张", "血栓闭塞性脉管炎", "高血压危象",
    "肺气肿", "肺心病", "肺结核", "支气管扩张", "肺栓塞", "气胸", "肺纤维化", "尘肺", "胸膜炎", "睡眠呼吸暂停",
    "胃炎急性发作", "胃下垂", "胃食管反流", "功能性消化不良", "便秘", "腹泻", "肠易激综合征", "溃疡性结肠炎", "克罗恩病", "直肠炎",
    "肾炎", "肾病综合征", "肾功能不全", "尿路感染", "肾盂肾炎", "多囊肾", "前列腺增生", "附睾炎", "睾丸炎", "包皮龟头炎",
    "甲减", "甲状腺结节", "甲状腺炎", "肾上腺皮质功能减退", "高脂血症", "肥胖症", "骨质疏松症", "更年期综合征", "多囊卵巢综合征", "痛经",
]
SYMPTOM_POOL = [
    "发热", "咳嗽", "咳痰", "咽痛", "鼻塞", "流涕", "打喷嚏", "声音嘶哑", "胸痛", "胸闷",
    "气短", "喘息", "心悸", "心慌", "头晕", "头痛", "眩晕", "耳鸣", "视力模糊", "眼干",
    "眼痛", "流泪", "牙痛", "口腔疼痛", "口苦", "口干", "食欲不振", "恶心", "呕吐", "反酸",
    "嗳气", "胃痛", "胃胀", "腹痛", "腹胀", "腹泻", "便秘", "便血", "黑便", "黄疸",
    "尿频", "尿急", "尿痛", "血尿", "腰痛", "腰酸", "关节痛", "关节肿", "肌肉酸痛", "乏力",
    "疲劳", "失眠", "多梦", "烦躁", "焦虑", "情绪低落", "注意力不集中", "记忆力下降", "手抖", "麻木",
    "刺痛", "皮肤瘙痒", "皮疹", "水疱", "脱发", "面色苍白", "多汗", "盗汗", "怕冷", "怕热",
    "多饮", "多尿", "多食", "体重下降", "体重增加", "水肿", "淋巴结肿大", "皮下出血", "紫癜", "烧心",
    "里急后重", "肛门坠胀", "阴茎疼痛", "月经不调", "白带异常", "痛经", "乳房胀痛", "早搏", "晕厥", "抽搐",
    "口角歪斜", "言语不清", "肢体无力", "肢体偏瘫", "吞咽困难", "颈肩酸痛", "腰腿痛", "膝盖疼痛", "足跟痛", "手指僵硬",
    "晨僵", "眼睛干涩", "畏光", "畏声", "视觉模糊", "飞蚊症", "夜尿增多", "遗尿", "阳痿", "早泄",
]
DRUG_POOL = [
    "阿莫西林", "头孢拉定", "头孢克肟", "阿奇霉素", "克拉霉素", "左氧氟沙星", "环丙沙星", "甲硝唑", "奥司他韦", "利巴韦林",
    "布洛芬", "双氯芬酸钠", "洛索洛芬", "塞来昔布", "对乙酰氨基酚", "阿司匹林", "氯吡格雷", "华法林", "氨氯地平", "硝苯地平",
    "美托洛尔", "比索洛尔", "卡托普利", "缬沙坦", "厄贝沙坦", "氢氯噻嗪", "呋塞米", "螺内酯", "阿托伐他汀", "辛伐他汀",
    "二甲双胍", "格列美脲", "阿卡波糖", "胰岛素", "甲巯咪唑", "优甲乐", "奥美拉唑", "兰索拉唑", "雷贝拉唑", "铝碳酸镁",
    "多潘立酮", "莫沙必利", "蒙脱石散", "双歧杆菌", "乳果糖", "开塞露", "洛哌丁胺", "复方甘草片", "氨溴索", "右美沙芬",
    "氯雷他定", "西替利嗪", "孟鲁司特", "沙丁胺醇", "布地奈德", "氨茶碱", "泼尼松", "地塞米松", "甲氨蝶呤", "来氟米特",
]
DEPARTMENT_POOL = [
    "呼吸内科", "消化内科", "心血管内科", "神经内科", "内分泌科", "血液科", "肾内科", "风湿免疫科", "感染科", "肿瘤科",
    "普通外科", "肝胆外科", "泌尿外科", "骨科", "神经外科", "胸外科", "皮肤科", "眼科", "耳鼻喉科", "口腔科",
    "妇科", "儿科", "精神科", "康复科", "急诊科", "全科医学科",
]
EXAM_POOL = [
    "血常规", "尿常规", "便常规", "肝功能", "肾功能", "血糖检测", "血脂检查", "甲状腺功能", "心电图", "心脏彩超",
    "胸部CT", "胸部X线", "腹部彩超", "腹部CT", "头颅CT", "头颅MRI", "胃镜", "肠镜", "肺功能检查", "骨密度检查",
    "眼科检查", "耳内镜检查", "鼻内镜检查", "喉镜检查", "皮肤镜", "关节X线", "腰椎CT", "颈椎CT", "泌尿系彩超", "膀胱镜",
    "前列腺彩超", "妇科彩超", "HPV检测", "幽门螺杆菌检测", "过敏原检测", "血气分析", "心肌酶谱", "脑电图", "肌电图", "病理活检",
]
FOOD_POOL = ["辛辣食物", "油炸食物", "高盐食物", "甜食", "冷饮", "咖啡", "浓茶", "酒精", "海鲜", "肥肉", "腌制食品", "高嘌呤食物"]
POPULATION_POOL = ["老年人", "儿童", "孕妇", "女性", "中老年人", "青少年", "糖尿病患者", "高血压患者"]
NOTICE_POOL = [
    "规律作息，避免熬夜",
    "清淡饮食，少油少盐",
    "戒烟限酒",
    "适量运动，避免剧烈活动",
    "遵医嘱用药，不可自行停药",
    "定期复查，监测病情变化",
    "保持良好心态，避免情绪波动",
    "注意保暖，预防感染",
    "多饮水，促进代谢",
    "避免久坐，适当活动",
]
QA_TEMPLATES = [
    "{symptom}怎么办？",
    "{symptom}是什么原因引起的？",
    "{symptom}应该看什么科？",
    "最近总是{symptom}，需要去医院吗？",
    "{disease}有什么症状？",
    "{disease}需要注意什么？",
    "{disease}平时饮食要注意什么？",
    "得了{disease}挂什么科？",
    "{disease}可以吃什么药？",
    "{symptom}是{disease}的症状吗？",
    "有{symptom}会不会是{disease}？",
    "{disease}出现{symptom}要紧吗？",
]


def _gen_disease_pool() -> list[dict]:
    """词表组合生成剩余疾病（name 来自 DISEASE_POOL 的"原版/急性/慢性"变体，关联实体从词表抽样）。

    COMMON_DISEASES(20) + 组合(280) = 300（§2.2）。
    """
    rng = random.Random(20260821)
    out: list[dict] = []
    variants: list[str] = []
    for n in DISEASE_POOL:
        variants += [n, f"急性{n}", f"慢性{n}"]
    rng.shuffle(variants)
    for name in variants[:280]:
        n_sym = rng.randint(3, 8)  # 症状 3-8（保证模拟关系量 ≥5000）
        symptoms = rng.sample(SYMPTOM_POOL, min(n_sym, len(SYMPTOM_POOL)))
        n_drug = rng.randint(1, 5)
        severity = "重" if set(symptoms) & EMERGENCY_SYMPTOMS else rng.choice(["轻", "轻", "中", "中", "中", "重"])
        complications = [c for c in rng.sample(list(DISEASE_POOL), min(rng.randint(1, 2), len(DISEASE_POOL)))
                         if c != name]  # 并发症从疾病池抽样（避免自指）
        out.append({
            "name": name,
            "severity": severity,
            "symptoms": symptoms,
            "drugs": rng.sample(DRUG_POOL, min(n_drug, len(DRUG_POOL))),
            "departments": rng.sample(DEPARTMENT_POOL, min(rng.randint(1, 2), len(DEPARTMENT_POOL))),
            "exams": rng.sample(EXAM_POOL, min(rng.randint(0, 3), len(EXAM_POOL))),
            "foods": rng.sample(FOOD_POOL, min(rng.randint(0, 2), len(FOOD_POOL))),
            "populations": rng.sample(POPULATION_POOL, min(rng.randint(0, 1), len(POPULATION_POOL))),
            "notices": rng.sample(NOTICE_POOL, min(rng.randint(0, 3), len(NOTICE_POOL))),
            "complications": complications,
        })
    return out


def generate_simulated() -> tuple[list[dict], list[dict]]:
    """§2.2 模拟数据生成：300 疾病（词表非随机噪音）+ 5000 问答模板。"""
    rng = random.Random(20260821)
    diseases = [dict(d) for d in COMMON_DISEASES] + _gen_disease_pool()
    assert len(diseases) == 300, f"simulated disease count={len(diseases)}"

    # 问答对：模板 × 症状 笛卡尔积遍历（确定性，保证 5000 条不重复）
    qa: list[dict] = []
    seen_q: set[str] = set()
    qid = 1
    for d in diseases:
        if len(qa) >= QA_TARGET:
            break
        department = rng.choice(d["departments"]) if d["departments"] else "全科医学科"
        notice = rng.choice(d["notices"]) if d["notices"] else "遵医嘱治疗"
        for template in QA_TEMPLATES:
            for symptom in (d["symptoms"] or ["不适"]):
                if len(qa) >= QA_TARGET:
                    break
                question = template.format(symptom=symptom, disease=d["name"])
                if question in seen_q:
                    continue
                seen_q.add(question)
                answer = f"{d['name']}建议：就诊{department}，{notice}。"
                qa.append({"qid": str(qid), "question": question, "answer": answer})
                qid += 1
    assert len(qa) == QA_TARGET, f"simulated qa count={len(qa)}"

    # §2.2 原始模拟数据落 data/simulated/（与 cleaned 同构，便于核对）
    write_json(SIMULATED_DIR / "diseases.json", diseases)
    write_json(SIMULATED_DIR / "qa_pairs.json", qa)
    log(f"模拟数据已生成: 300 疾病 + {len(qa)} 问答 → data/simulated/")
    return diseases, qa


# ---------- 主流程 ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="数据集接入+清洗")
    parser.add_argument("--generate", action="store_true", help="用模拟数据兜底（§2.2）")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，仅清洗 data/raw/ 已有数据")
    args = parser.parse_args()

    norm = load_norm()
    t0 = datetime.now(timezone.utc)

    if args.generate:
        raw_diseases, raw_qa = generate_simulated()
        source = "simulated"
        source_name = "SIMULATED_V1"
    else:
        source = "download"
        source_name = "QASystemOnMedicalKG+cMedQA2"
        ok1 = ok2 = ok3 = True
        if not args.skip_download:
            ok1 = download(D1_URL, RAW_DIR / "medical.json")
            ok2 = download(D3_QUESTION_URL, RAW_DIR / "question.zip")
            ok3 = download(D3_ANSWER_URL, RAW_DIR / "answer.zip")
        if not (ok1 and ok2 and ok3):
            log("真实数据集下载不完整；运行 `python prepare_data.py --generate` 用模拟数据兜底")
            return 1

        # D1 解析清洗
        raw1 = parse_d1(RAW_DIR / "medical.json")
        raw_diseases = parse_d1_to_diseases(raw1, norm)
        # D3 解析清洗
        raw_qa = parse_d3(RAW_DIR)
        source_name = "QASystemOnMedicalKG+cMedQA2"

        # 药品说明书（CMeKG）：下载 + 构建详情索引；失败不致命（用药卡退化为"信息暂缺"）
        if not args.skip_download:
            download(DRUG_URL, RAW_DIR / "cmekg_drug.json")
        if (RAW_DIR / "cmekg_drug.json").exists():
            try:
                from app.retrieval import drug_db
                n = drug_db.build()
                log(f"药品说明书索引构建完成: {n} 条")
            except Exception as e:
                log(f"药品说明书索引构建失败（不影响主流程）: {e}")

    # ---------- 清洗落盘（§2.4） ----------
    diseases = []
    for d in raw_diseases:
        cleaned = dict(d)
        for key in ("symptoms", "drugs", "departments", "exams", "foods", "populations", "complications"):
            cleaned[key] = clean_field(d.get(key))
        diseases.append(cleaned)

    if not args.generate:  # 下载模式才做 QA 清洗（模拟模式 QA 已是最终形态）
        raw_qa = clean_qa(raw_qa)

    relations = expand_relations(diseases, source)

    d1_count = len(diseases)
    d3_count = len(raw_qa)
    assert d1_count >= DISEASE_MIN, f"D1 疾病数不足: {d1_count} < {DISEASE_MIN}"
    assert d3_count >= QA_TARGET, f"D3 问答数不足: {d3_count} < {QA_TARGET}"

    write_json(CLEANED_DIR / "diseases.json", diseases)
    write_json(CLEANED_DIR / "relations.json", relations)
    write_json(CLEANED_DIR / "qa_pairs.json", raw_qa)

    meta = {
        "datasets": [
            {"name": "D1 中文疾病知识库", "file": "raw/medical.json",
             "source": source, "count": d1_count,
             "note": "severity 口径: 症状含急症词表→重，否则中（原始数据无严重程度字段）"},
            {"name": "D3 cMedQA2 医患问答", "file": "raw/question.zip+answer.zip",
             "source": source, "count": d3_count},
        ],
        "cleaned_at": t0.isoformat(),
        "norm_file": str(ENTITY_NORM_FILE.relative_to(DATA_DIR)),
    }
    write_json(DATA_DIR / "source_meta.json", meta)

    log(f"完成: 疾病 {d1_count}（≥{DISEASE_MIN}）| 关系 {len(relations)} | 问答 {d3_count}（={QA_TARGET}）")
    log(f"输出: {CLEANED_DIR} | 数据源: {source_name} ({source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
