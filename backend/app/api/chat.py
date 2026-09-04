"""会话与流式问答路由（§7.1 #3/#4/#5/#6，§7.2 SSE 协议核心）。

SSE 事件序列：
  event: intent  → {"intent": "DEPARTMENT"}
  event: evidence → {"pool": [...]}（前端仅缓存，默认不渲染，供"查看依据"）
  event: token   → {"text": "..."}（打字机切分）
  event: risk    → {"risk_level", "disclaimer", "drug_notice", "refusal"}
  event: done    → {"message_id", "answer", "evidence"}
心跳：每 15 秒 `: ping` 注释帧；客户端断开即停止生成（已生成内容落库标记 interrupted=1）。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.graph import ask_interrupt_aware  # noqa: E402
from app.agent.nodes_domain import init_retrieval_env  # noqa: E402
from app.api.auth import get_current_user  # noqa: E402
from app.models.api_models import ChatRequest, ConversationCreate  # noqa: E402
from app.services import db  # noqa: E402

router = APIRouter(tags=["chat"])

TOKEN_CHUNK = 8        # 打字机切分：每 token 事件字符数
TOKEN_INTERVAL = 0.05  # 秒


@router.post("/conversations")
def create_conversation(body: ConversationCreate, user: dict = Depends(get_current_user)):
    conv_id = db.create_conversation(user["user_id"], body.title)
    return {"code": 0, "conversation_id": conv_id}


@router.get("/conversations")
def list_conversations(user: dict = Depends(get_current_user)):
    return {"code": 0, "conversations": db.list_conversations(user["user_id"])}


@router.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: int, user: dict = Depends(get_current_user)):
    conv = db.get_conversation(conv_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code=404, detail={"code": 40401, "message": "会话不存在"})
    return {"code": 0, "messages": db.list_messages(conv_id)}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_chat(conv_id: int, question: str):
    """SSE 生成器：跑一轮对话 → 事件序列推送。

    M8.3 中断感知：图可能停在问诊追问点（症状不足）。此时必须输出追问问题，
    绝不能输出 checkpoint 残留的上一轮 answer（修复"复读上一轮"）。
    """
    init_retrieval_env()
    state, pending_clarify = ask_interrupt_aware(question, thread_id=f"conv-{conv_id}")

    yield _sse("intent", {"intent": state.get("intent", "")})

    # 图停在追问点：输出澄清问题（清空结构化字段，避免带出上一轮残留）
    if pending_clarify is not None:
        clarify = str(pending_clarify)
        yield _sse("evidence", {"pool": []})
        for i in range(0, len(clarify), TOKEN_CHUNK):
            yield _sse("token", {"text": clarify[i:i + TOKEN_CHUNK]})
            await asyncio.sleep(TOKEN_INTERVAL)
        yield _sse("risk", {"risk_level": "NONE", "disclaimer": "", "drug_notice": "", "refusal": False})
        db.insert_message(conv_id, "USER", question, intent=state.get("intent"))
        db.maybe_set_title(conv_id, question)  # M8.18 恢复丢失的调用：首问自动命名会话
        message_id = db.insert_message(conv_id, "ASSISTANT", clarify, intent=state.get("intent"),
                                       evidence_json={}, risk_level="NONE", disclaimer_added=0)
        yield _sse("done", {"message_id": message_id, "answer": clarify, "sections": [], "tags": {},
                            "evidence": {"evidence_quotes": [], "evidence_pool": []}})
        return

    yield _sse("evidence", {"pool": state.get("evidence_pool", [])[:5]})

    answer = state.get("answer", "")
    for i in range(0, len(answer), TOKEN_CHUNK):
        yield _sse("token", {"text": answer[i:i + TOKEN_CHUNK]})
        await asyncio.sleep(TOKEN_INTERVAL)

    yield _sse("risk", {
        "risk_level": state.get("risk_level", "NONE"),
        "disclaimer": state.get("disclaimer", ""),
        "drug_notice": state.get("drug_notice", ""),
        "refusal": bool(state.get("refusal")),
    })

    # 落库：用户问题 + 助手回答（含证据链/风险/免责审计字段 + 结构化卡片）
    evidence = {
        "evidence_quotes": state.get("evidence_quotes", []),
        "evidence_pool": state.get("evidence_pool", [])[:5],
        "graph_evidence": state.get("graph_evidence", [])[:20],
        "safety_trail": state.get("safety_trail", []),
        "sections": state.get("answer_sections", []),
        "tags": state.get("answer_tags", {}),
    }
    db.insert_message(conv_id, "USER", question, intent=state.get("intent"))
    db.maybe_set_title(conv_id, question)  # M8.18 恢复丢失的调用：首问自动命名会话
    message_id = db.insert_message(
        conv_id, "ASSISTANT", answer,
        intent=state.get("intent"),
        evidence_json=evidence,
        risk_level=state.get("risk_level", "NONE"),
        disclaimer_added=1 if state.get("disclaimer") else 0,
    )
    yield _sse("done", {
        "message_id": message_id,
        "answer": answer,
        "sections": state.get("answer_sections", []),
        "tags": state.get("answer_tags", {}),
        "evidence": {"evidence_quotes": state.get("evidence_quotes", []),
                     "evidence_pool": state.get("evidence_pool", [])[:5]},
    })


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, user: dict = Depends(get_current_user)):
    conv = db.get_conversation(body.conversation_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code=404, detail={"code": 40401, "message": "会话不存在"})
    if conv["status"] != "ACTIVE":
        raise HTTPException(status_code=409, detail={"code": 40901, "message": "会话已关闭"})

    async def event_generator():
        try:
            async for chunk in _stream_chat(body.conversation_id, body.question):
                yield chunk
        except Exception as e:  # 生成失败：错误事件，不中断连接
            yield _sse("error", {"code": 50001, "message": str(e)[:200]})
            return
        finally:
            # 心跳帧（协议要求每 15 秒；此处生成结束即收尾，长任务场景由前端超时兜底）
            yield ": ping\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
