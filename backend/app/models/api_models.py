"""API 请求/响应模型（pydantic 校验）。"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=64)


class LoginRequest(BaseModel):
    username: str
    password: str


class ConversationCreate(BaseModel):
    title: str = ""


class ChatRequest(BaseModel):
    conversation_id: int
    question: str = Field(min_length=1, max_length=2000)


class ApiError(BaseModel):
    code: int
    message: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    intent: Optional[str] = None
    evidence_json: dict[str, Any] = {}
    risk_level: str = "NONE"
    disclaimer_added: int = 0
    created_at: Optional[str] = None


class ConversationOut(BaseModel):
    id: int
    title: str
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
