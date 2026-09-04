"""M6 API 工具层测试：JWT 签发/校验/过期 + 用户落库。

DB 用例依赖本机 MySQL（medical_agent 库），连接失败自动 skip。
"""
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.services import db  # noqa: E402
from app.services.jwt_util import create_token, verify_token  # noqa: E402


# ---------- JWT（§7.1 认证） ----------

def test_jwt_roundtrip():
    token = create_token(1, "USER")
    payload = verify_token(token)
    assert payload["user_id"] == 1 and payload["role"] == "USER"


def test_jwt_tampered_rejected():
    token = create_token(1, "USER")
    assert verify_token(token + "x") is None
    assert verify_token(token[:-2] + "ab") is None


def test_jwt_expired_rejected():
    token = create_token(1, "USER", expire_hours=-1)
    assert verify_token(token) is None


def test_jwt_garbage_rejected():
    assert verify_token("not.a.jwt") is None
    assert verify_token("") is None


# ---------- 用户落库（§2.3 user 表） ----------

@pytest.mark.skipif(not db.is_ready(), reason="MySQL 不可用")
def test_user_create_and_verify():
    import uuid

    username = f"ut_{uuid.uuid4().hex[:8]}"
    db.create_user(username, "secret123")
    user = db.verify_user(username, "secret123")
    assert user and user["username"] == username
    assert db.verify_user(username, "wrongpass") is None
    assert db.verify_user("no_such_user_x", "secret123") is None


@pytest.mark.skipif(not db.is_ready(), reason="MySQL 不可用")
def test_conversation_and_message_flow():
    import uuid

    username = f"ut_{uuid.uuid4().hex[:8]}"
    user_id = db.create_user(username, "secret123")
    conv_id = db.create_conversation(user_id, "测试会话")
    assert db.get_conversation(conv_id, user_id)["title"] == "测试会话"
    db.insert_message(conv_id, "USER", "头痛挂什么科", intent="DEPARTMENT")
    msg_id = db.insert_message(conv_id, "ASSISTANT", "建议就诊神经内科",
                               intent="DEPARTMENT",
                               evidence_json={"evidence_quotes": ["头痛"]},
                               risk_level="NONE", disclaimer_added=1)
    msgs = db.list_messages(conv_id)
    assert len(msgs) == 2
    assert msgs[1]["id"] == msg_id and msgs[1]["evidence_json"]["evidence_quotes"] == ["头痛"]
    assert db.get_conversation(conv_id, user_id + 999) is None  # 越权访问
