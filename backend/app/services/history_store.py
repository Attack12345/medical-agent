"""会话历史冷热分层（M8.10，对齐业界成熟的冷热分层契约，按本项目结构重写）。

- Redis 热缓存：最近 12 轮会话上下文（List：LPUSH 插头 + LTRIM 保留 + EXPIRE 24h 滑动续期）
- MySQL 冷存储：全量历史（唯一事实源，写入由 db.insert_message 完成，本模块只负责 Redis 侧）
- 读路径三级命中：Redis sess:{sid} → 未命中 MySQL 取最近 N 轮回填 Redis → 返回
- 写路径双写容错：调用方先写 MySQL（事实源）再 cache_turn 写 Redis——MySQL 失败不落 Redis
  （缓存里绝不出现数据库没有的幻影数据）；Redis 失败仅日志降级，绝不打断主链路
- 连接管理：Redis 进程级连接池单例；每轮 JSON 序列化 {role, content, intent, ts}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from app.config import settings

logger = logging.getLogger("history_store")

HOT_TURNS = 12            # 热缓存保留轮次（设计契约）
HOT_TTL_SECONDS = 24 * 3600  # 热缓存 TTL：24h 滑动续期

_pool = None           # redis 连接池单例（懒加载）
_pool_failed = False   # 初始化失败标记（避免每请求重试阻塞）


def _rdb():
    """Redis 连接池单例；不可用时返回 None（调用方降级走 MySQL）。"""
    global _pool, _pool_failed
    if _pool_failed:
        return None
    if _pool is None:
        try:
            import redis as redis_lib

            _pool = redis_lib.ConnectionPool(
                host=settings.redis_host, port=settings.redis_port,
                db=settings.redis_db, decode_responses=True,
                socket_connect_timeout=2, socket_timeout=2,
            )
        except Exception as e:
            logger.warning(f"[history_store] Redis 连接池初始化失败，降级 MySQL：{e}")
            _pool_failed = True
            return None
    return _pool


def _key(conv_id: int) -> str:
    return f"sess:{conv_id}"


def _client():
    pool = _rdb()
    if pool is None:
        return None
    try:
        import redis as redis_lib

        client = redis_lib.Redis(connection_pool=pool)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"[history_store] Redis 不可用，降级 MySQL：{e}")
        return None


def cache_turn(conv_id: int, role: str, content: str, intent: str | None = None) -> None:
    """写路径 Redis 侧（调用方已先写 MySQL 事实源）：LPUSH + LTRIM + EXPIRE。"""
    client = _client()
    if client is None:
        return
    try:
        turn = json.dumps({
            "role": role,
            "content": (content or "")[:200],
            "intent": intent or "",
            "ts": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False)
        pipe = client.pipeline()
        pipe.lpush(_key(conv_id), turn)
        pipe.ltrim(_key(conv_id), 0, HOT_TURNS - 1)
        pipe.expire(_key(conv_id), HOT_TTL_SECONDS)
        pipe.execute()
    except Exception as e:
        logger.warning(f"[history_store] cache_turn 降级（不影响主链路）：{e}")


def get_recent_turns(conv_id: int, limit: int = HOT_TURNS) -> list[dict[str, Any]]:
    """读路径三级命中：Redis → 未命中 MySQL 取最近 limit 轮回填 → 返回。

    永不抛异常（历史读取失败返回 []，主链路照常单轮执行）。
    """
    client = _client()
    if client is not None:
        try:
            raw = client.lrange(_key(conv_id), 0, limit - 1)
            if raw:
                turns = [json.loads(x) for x in raw]
                turns.reverse()  # LPUSH 最新在头 → 反序为时间正序
                return turns
        except Exception as e:
            logger.warning(f"[history_store] Redis 读失败，回填 MySQL：{e}")
    # 未命中/降级：MySQL 冷存储取最近 limit 轮并回填 Redis
    try:
        from app.services import db

        msgs = db.list_messages(conv_id)[-limit:]
        turns = [{"role": m["role"], "content": (m["content"] or "")[:200],
                  "intent": m.get("intent") or "", "ts": str(m.get("created_at") or "")}
                 for m in msgs]
        if turns and client is not None:
            try:
                pipe = client.pipeline()
                for t in reversed(turns):  # LPUSH 最新在头：按时间正序逆序压入
                    pipe.lpush(_key(conv_id), json.dumps(t, ensure_ascii=False))
                pipe.ltrim(_key(conv_id), 0, HOT_TURNS - 1)
                pipe.expire(_key(conv_id), HOT_TTL_SECONDS)
                pipe.execute()
            except Exception:
                pass
        return turns
    except Exception as e:
        logger.warning(f"[history_store] MySQL 历史读取失败：{e}")
        return []


def format_history(turns: list[dict[str, Any]], max_turns: int = 6, max_len: int = 60) -> str:
    """最近对话拼为 prompt 上下文段（供分诊/知识问答注入多轮记忆）。"""
    recent = [t for t in (turns or []) if t.get("content")][-max_turns:]
    if not recent:
        return ""
    lines = [f"{'用户' if t.get('role') == 'USER' else '助手'}：{str(t['content'])[:max_len]}"
             for t in recent]
    return "【对话历史（供理解指代与上下文，不要复述）】\n" + "\n".join(lines)
