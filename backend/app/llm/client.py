"""LLM 调用层（langchain 生态统一，）。

- 单点封装：所有 LLM 调用走本模块，业务方不直接触碰模型对象。
- JSON 模式：chat_json 强制结构化输出，失败自动重试一次，再失败抛 LLMError。
- 切换：cloud（百炼 API）/ local（vLLM 微调模型，M8）只改 .env，代码零改动。
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings


class LLMError(RuntimeError):
    """LLM 调用/解析失败（网络重试耗尽、JSON 不合法、schema 校验失败）。"""


def get_chat_model(temperature: float = 0.0, json_mode: bool = False) -> ChatOpenAI:
    kwargs: dict[str, Any] = dict(
        model=settings.llm_model,
        api_key=settings.dashscope_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        timeout=60,
        max_retries=2,  # 网络层重试
    )
    if json_mode:
        # OpenAI 兼容 json_object 模式（百炼 qwen 系列支持）
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)


def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON：优先整体解析，其次剥离 ```json 代码块，最后截取首尾花括号。"""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    raise LLMError(f"模型输出无法解析为 JSON: {text[:200]}")


def chat_json(system: str, user: str, temperature: float = 0.0) -> dict:
    """调用 LLM 并返回 JSON dict（json_object 模式 + 提取兜底 + 重试一次）。"""
    model = get_chat_model(temperature=temperature, json_mode=True)
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = model.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            return _extract_json(resp.content if isinstance(resp.content, str) else str(resp.content))
        except LLMError as e:
            last_err = e
        except Exception as e:  # 网络/限流等
            last_err = e
    raise LLMError(f"LLM 调用失败（已重试）: {last_err}")
