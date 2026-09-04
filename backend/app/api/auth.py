"""认证路由（§7.1 #1/#2）：register/login + JWT 依赖注入。"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.models.api_models import LoginRequest, RegisterRequest  # noqa: E402
from app.services import db  # noqa: E402
from app.services.jwt_util import create_token, verify_token  # noqa: E402

router = APIRouter(prefix="/auth", tags=["auth"])


def get_current_user(request: Request) -> dict:
    """FastAPI 依赖：解析 Authorization: Bearer <token>。"""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"code": 40101, "message": "未认证"})
    payload = verify_token(auth[7:])
    if not payload:
        raise HTTPException(status_code=401, detail={"code": 40101, "message": "登录已过期，请重新登录"})
    return payload


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail={"code": 40101, "message": "需要管理员权限"})
    return user


@router.post("/register")
def register(body: RegisterRequest):
    try:
        user_id = db.create_user(body.username, body.password)
    except Exception as e:
        if "Duplicate" in str(e) or "duplicate" in str(e):
            raise HTTPException(status_code=409, detail={"code": 40901, "message": "用户名已存在"})
        raise HTTPException(status_code=500, detail={"code": 50001, "message": "注册失败"})
    return {"code": 0, "user_id": user_id}


@router.post("/login")
def login(body: LoginRequest):
    user = db.verify_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail={"code": 40101, "message": "用户名或密码错误"})
    return {
        "code": 0,
        "token": create_token(user["id"], user["role"]),
        "user_id": user["id"],
        "role": user["role"],
    }
