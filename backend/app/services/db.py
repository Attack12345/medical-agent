"""MySQL 数据访问层（：对话/消息/评估落库）。

- pymysql 直连 + 每请求短连接（本机开发规模，连接数可控）；
- 密码哈希：sha256(salt + password)（§2.3 user 表）；
- 全部写操作带幂等/异常处理，供 API 层复用。
"""
from __future__ import annotations

import hashlib
import json
import secrets
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

import pymysql

from app.config import settings


class DbError(RuntimeError):
    pass


@contextmanager
def get_conn() -> Iterator[pymysql.Connection]:
    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_db,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def is_ready() -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


# ---------- 用户（§7.1 #1/#2） ----------

def create_user(username: str, password: str) -> int:
    salt = secrets.token_hex(8)
    pwd_hash = hashlib.sha256((salt + password).encode()).hexdigest()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user (username, password_hash, salt) VALUES (%s, %s, %s)",
                (username, pwd_hash, salt),
            )
            return int(cur.lastrowid)


def get_user(username: str) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, password_hash, salt, role FROM user WHERE username = %s", (username,))
            return cur.fetchone()


def verify_user(username: str, password: str) -> dict | None:
    user = get_user(username)
    if not user:
        return None
    pwd_hash = hashlib.sha256((user["salt"] + password).encode()).hexdigest()
    if pwd_hash != user["password_hash"]:
        return None
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


# ---------- 会话（§7.1 #3/#4） ----------

def create_conversation(user_id: int, title: str = "") -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversation (user_id, title) VALUES (%s, %s)",
                (user_id, title),
            )
            return int(cur.lastrowid)


def list_conversations(user_id: int) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, status, created_at, updated_at "
                "FROM conversation WHERE user_id = %s ORDER BY updated_at DESC LIMIT 50",
                (user_id,),
            )
            return cur.fetchall()


def maybe_set_title(conv_id: int, question: str, default_titles: set[str] | None = None) -> None:
    """首问自动生成会话标题（M8.7）：标题仍为默认值时，用首个问题截断作为标题。

    default_titles 视为"未命名"（如 {'', '新会话'}）；已有有效标题则不覆盖。
    """
    import re as _re

    title = _re.sub(r"\s+", " ", str(question)).strip()[:20]
    if not title:
        return
    defaults = default_titles or {"", "新会话"}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT title FROM conversation WHERE id = %s", (conv_id,))
            row = cur.fetchone()
            if not row or (row["title"] or "").strip() not in defaults:
                return  # 已有有效标题（或会话不存在），不覆盖
            cur.execute("UPDATE conversation SET title = %s WHERE id = %s", (title, conv_id))


def get_conversation(conv_id: int, user_id: int) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, title, status, user_id FROM conversation WHERE id = %s AND user_id = %s",
                (conv_id, user_id),
            )
            return cur.fetchone()


# ---------- 消息（§7.1 #5/#6） ----------

def insert_message(conversation_id: int, role: str, content: str, intent: str | None = None,
                   evidence_json: dict | None = None, risk_level: str = "NONE",
                   disclaimer_added: int = 0) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO message
                   (conversation_id, role, content, intent, evidence_json, risk_level, disclaimer_added)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (conversation_id, role, content, intent,
                 json.dumps(evidence_json or {}, ensure_ascii=False), risk_level, disclaimer_added),
            )
            return int(cur.lastrowid)


def list_messages(conversation_id: int) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, role, content, intent, evidence_json, risk_level, disclaimer_added, created_at "
                "FROM message WHERE conversation_id = %s ORDER BY id",
                (conversation_id,),
            )
            rows = cur.fetchall()
            for r in rows:
                if r.get("evidence_json"):
                    try:
                        r["evidence_json"] = json.loads(r["evidence_json"])
                    except (TypeError, json.JSONDecodeError):
                        r["evidence_json"] = {}
            return rows


# ---------- 管理端（§7.1 #7/#8） ----------

def recent_messages(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT m.id, m.conversation_id, m.role, m.content, m.intent, m.risk_level, "
                "m.disclaimer_added, m.created_at, u.username "
                "FROM message m JOIN conversation c ON m.conversation_id = c.id "
                "JOIN user u ON c.user_id = u.id "
                "ORDER BY m.id DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()


def admin_stats() -> dict[str, Any]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM conversation")
            convs = cur.fetchone()["n"]
            cur.execute("SELECT count(*) AS n FROM message")
            msgs = cur.fetchone()["n"]
            cur.execute("SELECT intent, count(*) AS n FROM message WHERE intent IS NOT NULL GROUP BY intent")
            intent_dist = {r["intent"]: r["n"] for r in cur.fetchall()}
            cur.execute("SELECT risk_level, count(*) AS n FROM message GROUP BY risk_level")
            risk_dist = {r["risk_level"]: r["n"] for r in cur.fetchall()}
            return {
                "conversation_count": convs,
                "message_count": msgs,
                "intent_distribution": intent_dist,
                "risk_distribution": risk_dist,
            }


# ---------- 评估（§7.1 #9） ----------

def latest_eval_report() -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version, metrics_json, passed, created_at FROM eval_run ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            if not row:
                return None
            row["metrics_json"] = json.loads(row["metrics_json"])
            return row
