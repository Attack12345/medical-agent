"""API 全流程验证脚本（M6 DoD）。

用法：
  先启动服务：uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8090
  再运行：python verify_api.py [--base http://127.0.0.1:8090]

覆盖 12 个接口：注册→登录→建会话→列表→SSE 问答→消息→history→stats→eval/report→query→health→ready
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8092"


def req(method: str, path: str, body: dict | None = None, token: str | None = None,
        timeout: float = 120) -> tuple[int, dict | str]:
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def sse_chat(conv_id: int, question: str, token: str) -> dict:
    """消费 SSE 流，聚合事件返回。"""
    body = json.dumps({"conversation_id": conv_id, "question": question}).encode()
    r = urllib.request.Request(BASE + "/api/v1/chat/stream", data=body, method="POST")
    r.add_header("Content-Type", "application/json")
    r.add_header("Authorization", f"Bearer {token}")
    events: dict[str, list] = {}
    answer_parts: list[str] = []
    with urllib.request.urlopen(r, timeout=180) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("event: "):
                evt = line[7:]
            elif line.startswith("data: ") and "evt" in locals():
                data = json.loads(line[6:])
                if evt == "token":
                    answer_parts.append(data["text"])
                events.setdefault(evt, []).append(data)
    return {"events": events, "answer": "".join(answer_parts)}


def main() -> int:
    global BASE
    parser = argparse.ArgumentParser(description="API 全流程验证（§7）")
    parser.add_argument("--base", default=BASE)
    BASE = parser.parse_args().base

    ts = int(time.time())
    username = f"test_{ts}"

    # 1 register
    code, r = req("POST", "/api/v1/auth/register", {"username": username, "password": "test123456"})
    assert code == 200 and r["code"] == 0, f"register: {code} {r}"
    print(f"[01] register ✅ {username}")

    # 2 login
    code, r = req("POST", "/api/v1/auth/login", {"username": username, "password": "test123456"})
    assert code == 200 and r["token"], f"login: {code} {r}"
    token = r["token"]
    print(f"[02] login ✅ role={r['role']}")

    # 3 create conversation
    code, r = req("POST", "/api/v1/conversations", {"title": "验证会话"}, token=token)
    assert code == 200 and r["conversation_id"], f"create conv: {code} {r}"
    conv_id = r["conversation_id"]
    print(f"[03] create conversation ✅ id={conv_id}")

    # 4 list conversations
    code, r = req("GET", "/api/v1/conversations", token=token)
    assert code == 200 and any(c["id"] == conv_id for c in r["conversations"]), f"list: {code} {r}"
    print(f"[04] list conversations ✅ {len(r['conversations'])} 个")

    # 6 SSE 问答（核心）
    result = sse_chat(conv_id, "头痛应该挂什么科", token)
    assert "intent" in result["events"] and "done" in result["events"], f"sse events: {result['events'].keys()}"
    assert result["answer"], "sse answer 为空"
    risk = result["events"]["risk"][0]
    print(f"[06] chat/stream ✅ intent={result['events']['intent'][0]['intent']} "
          f"risk={risk['risk_level']} answer={result['answer'][:40]}…")

    # 5 messages（含证据链）
    code, r = req("GET", f"/api/v1/conversations/{conv_id}/messages", token=token)
    assert code == 200 and len(r["messages"]) == 2, f"messages: {code} {r}"
    assistant = r["messages"][1]
    assert assistant["evidence_json"], "evidence_json 为空"
    print(f"[05] messages ✅ 2 条，证据链字段: {list(assistant['evidence_json'].keys())}")

    # 7 history
    code, r = req("GET", "/api/v1/history", token=token)
    assert code == 200 and r["messages"], f"history: {code} {r}"
    print(f"[07] history ✅ {len(r['messages'])} 条")

    # 8 admin stats（非 admin 403）
    code, r = req("GET", "/api/v1/admin/stats", token=token)
    assert code == 403, f"stats 应 403: {code}"
    print(f"[08] admin/stats ✅ 权限拦截 403")

    # 9 eval/report（非 admin 403）
    code, r = req("GET", "/api/v1/eval/report", token=token)
    assert code == 403, f"eval/report 应 403: {code}"
    print(f"[09] eval/report ✅ 权限拦截 403")

    # 10 query（非 admin 403）
    from urllib.parse import quote
    code, r = req("GET", "/api/v1/query?q=" + quote("高血压吃什么药"), token=token)
    assert code == 403, f"query 应 403: {code}"
    print(f"[10] query ✅ 权限拦截 403")

    # 11 health / 12 ready
    code, r = req("GET", "/api/v1/health")
    assert code == 200 and r["status"] == "ok", f"health: {code} {r}"
    print(f"[11] health ✅")
    code, r = req("GET", "/api/v1/ready")
    assert code == 200 and all(r.get(k) for k in ("mysql", "neo4j", "qdrant")), f"ready: {code} {r}"
    print(f"[12] ready ✅ mysql={r['mysql']} neo4j={r['neo4j']} qdrant={r['qdrant']}")

    print("\n✅ API 全流程验证通过（M6 DoD 接口部分）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
