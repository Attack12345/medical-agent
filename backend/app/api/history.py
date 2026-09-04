"""管理端/评估/图谱问答路由（§7.1 #7-#10）。"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.api.auth import get_current_user, require_admin  # noqa: E402
from app.services import db  # noqa: E402

router = APIRouter(tags=["admin"])


@router.get("/history")
def history(limit: int = 20, _user: dict = Depends(get_current_user)):
    """最近对话摘要（管理端审计用，§7.1 #7）。"""
    return {"code": 0, "messages": db.recent_messages(limit)}


@router.get("/admin/stats")
def stats(_user: dict = Depends(require_admin)):
    """统计：会话数/消息数/意图分布/风险分布（§7.1 #8）。"""
    return {"code": 0, **db.admin_stats()}


@router.get("/eval/report")
def eval_report(_user: dict = Depends(require_admin)):
    """最近评估报告（§7.1 #9，M7 起有数据）。"""
    report = db.latest_eval_report()
    return {"code": 0, "report": report}


@router.get("/query")
def graph_query(q: str, _user: dict = Depends(require_admin)):
    """Text2Cypher 图谱问答（§7.1 #10，五层防护见 §3.3）。"""
    from app.graph.text2cypher import ask as t2c_ask

    result = t2c_ask(q)
    if not result.get("ok"):
        raise HTTPException(status_code=500, detail={"code": 50001, "message": result.get("error", "查询失败")})
    return {"code": 0, "answer": result["answer"], "cypher": result["cypher"], "records": result["records"][:10]}
