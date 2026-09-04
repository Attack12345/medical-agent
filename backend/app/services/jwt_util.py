"""JWT 工具（§7.1 简单 JWT，零新依赖：hmac_sha256 签名，密钥 .env）。

payload: {user_id, role, exp(epoch)}；header 固定 {"alg":"HS256","typ":"JWT"}。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from app.config import settings


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _sign(header_b64: str, payload_b64: str) -> str:
    msg = f"{header_b64}.{payload_b64}".encode()
    return _b64url(hmac.new(settings.jwt_secret.encode(), msg, hashlib.sha256).digest())


def create_token(user_id: int, role: str, expire_hours: int | None = None) -> str:
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = _b64url(json.dumps({
        "user_id": user_id,
        "role": role,
        "exp": int(time.time()) + (expire_hours or settings.jwt_expire_hours) * 3600,
    }, separators=(",", ":")).encode())
    return f"{header}.{payload}.{_sign(header, payload)}"


def verify_token(token: str) -> dict | None:
    """校验签名与有效期；非法/过期返回 None。"""
    try:
        header_b64, payload_b64, sig = token.split(".")
        if _sign(header_b64, payload_b64) != sig:
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None
