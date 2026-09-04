"""M1 骨架冒烟测试（M1 DoD：pytest 骨架绿）。

覆盖：目录结构、schema.sql、cleaned 数据量、source_meta、config 加载。
MySQL 表存在性为可选测试（设 MEDICAL_AGENT_DB_TEST=1 时执行，CI 无 MySQL 时跳过）。
"""
import json
import os
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # medical-agent/
DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DIR = DATA_DIR / "cleaned"


# ---------- 目录结构（§0.4） ----------

def test_directory_skeleton():
    expected = [
        PROJECT_ROOT / "sql" / "schema.sql",
        PROJECT_ROOT / "backend" / "app" / "config.py",
        PROJECT_ROOT / "backend" / "scripts" / "prepare_data.py",
        PROJECT_ROOT / "backend" / "tests",
        PROJECT_ROOT / ".env.example",
        PROJECT_ROOT / ".gitignore",
    ]
    for p in expected:
        assert p.exists(), f"缺少 {p.relative_to(PROJECT_ROOT)}"
    for sub in ["api", "agent", "engine", "graph", "retrieval", "services", "models"]:
        assert (PROJECT_ROOT / "backend" / "app" / sub / "__init__.py").exists(), f"缺少 app/{sub}/__init__.py"


# ---------- schema.sql（§2.3） ----------

def test_schema_has_five_tables():
    sql = (PROJECT_ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
    for table in ["user", "conversation", "message", "golden_case", "eval_run"]:
        assert re.search(rf"CREATE TABLE IF NOT EXISTS {table}\b", sql), f"缺少表 {table}"


# ---------- 清洗数据（§2.4 / §11 M1 DoD） ----------

@pytest.fixture(scope="module")
def cleaned():
    return {
        "diseases": json.loads((CLEANED_DIR / "diseases.json").read_text(encoding="utf-8")),
        "relations": json.loads((CLEANED_DIR / "relations.json").read_text(encoding="utf-8")),
        "qa": json.loads((CLEANED_DIR / "qa_pairs.json").read_text(encoding="utf-8")),
        "meta": json.loads((DATA_DIR / "source_meta.json").read_text(encoding="utf-8")),
    }


def test_disease_count(cleaned):
    assert len(cleaned["diseases"]) >= 300, "D1 疾病数 < 300（§2.1）"


def test_qa_count(cleaned):
    assert len(cleaned["qa"]) == 5000, "D3 问答数 != 5000（§2.1）"


def test_relation_count(cleaned):
    assert len(cleaned["relations"]) >= 5000, "关系数 < 5000（§3.2 建图验收线）"


def test_disease_schema(cleaned):
    for d in cleaned["diseases"]:
        for key in ["name", "severity", "symptoms", "drugs", "departments", "exams",
                    "foods", "populations", "notices", "complications"]:
            assert key in d, f"疾病 {d.get('name')} 缺字段 {key}"
        assert d["severity"] in ("轻", "中", "重", "急症"), f"非法 severity: {d['name']}"
        assert d["symptoms"], f"疾病 {d['name']} 无症状"


def test_relation_schema(cleaned):
    allowed = {"PRESENTS", "TREATS", "VISITS", "DIAGNOSES", "AVOIDS_FOOD", "AFFECTS",
               "CONTRAINDICATES", "REQUIRES_EXAM", "COMPLICATES", "ACCOMPANIES", "ADMITS_TO"}
    for r in cleaned["relations"]:
        assert r["relation"] in allowed, f"非法关系类型: {r['relation']}"
        assert r["source"], "关系缺 source 属性"


def test_qa_schema(cleaned):
    for item in cleaned["qa"]:
        assert item["qid"] and item["question"] and item["answer"], "问答对缺字段"


def test_source_meta(cleaned):
    assert cleaned["meta"]["datasets"], "source_meta 缺 datasets"
    for ds in cleaned["meta"]["datasets"]:
        assert ds["source"] in ("download", "simulated"), "source 必须 download/simulated"


# ---------- config（§0.2） ----------

def test_config_load():
    sys_path_backup = __import__("sys").path[:]
    try:
        __import__("sys").path.insert(0, str(PROJECT_ROOT / "backend"))
        from app.config import settings
        assert settings.mysql_db == "medical_agent"
        assert settings.llm_model == "qwen-plus"
        assert settings.embedding_dim == 1024
    finally:
        __import__("sys").path[:] = sys_path_backup


# ---------- MySQL 表存在性（可选，需本机/CI 配置） ----------

@pytest.mark.skipif(not os.getenv("MEDICAL_AGENT_DB_TEST"), reason="未设 MEDICAL_AGENT_DB_TEST=1 跳过 DB 测试")
def test_mysql_tables():
    import pymysql

    sys_path_backup = __import__("sys").path[:]
    try:
        __import__("sys").path.insert(0, str(PROJECT_ROOT / "backend"))
        from app.config import settings
    finally:
        __import__("sys").path[:] = sys_path_backup
    conn = pymysql.connect(
        host=settings.mysql_host, port=settings.mysql_port,
        user=settings.mysql_user, password=settings.mysql_password,
        database=settings.mysql_db, charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            tables = {row[0] for row in cur.fetchall()}
        assert {"user", "conversation", "message", "golden_case", "eval_run"} <= tables, f"缺表: {tables}"
    finally:
        conn.close()
