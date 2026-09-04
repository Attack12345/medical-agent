"""规则执行轨迹（：safety_trail 审计）。

M5：轨迹由 safety_agent 返回并写入 state（M6 落库时附 message.evidence_json 旁，
不单独建表）。build_context 组装 chat. 单命名空间。
"""
from __future__ import annotations

from typing import Any


def build_context(chat: dict[str, Any]) -> dict[str, Any]:
    """组装执行上下文（chat. 命名空间，§6.1 字段）。"""
    return {"chat": chat or {}}
