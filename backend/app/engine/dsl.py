"""规则 DSL 模型：YAML 加载 + pydantic 校验（复用已验证实现模式）。"""
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

Category = Literal["VETO", "VALIDATE"]
Action = Literal["REQUIRE_DISCLAIMER", "SET_HIGH_RISK", "REQUIRE_DRUG_NOTICE",
                  "REFUSE", "INVALIDATE_ANSWER"]

RULES_FILE = Path(__file__).resolve().parent / "safety_rules.yaml"

# 字段路径前缀白名单（§6.1：单命名空间 chat.）
ALLOWED_PREFIXES = ("chat.",)

# 操作符白名单（§6.1：eq/ne/gt/gte/lt/lte/in/contains/not_contains/is_empty/is_not_empty/is_not_null/any/all）
ALLOWED_OPS = {
    "eq", "ne", "gt", "gte", "lt", "lte", "in",
    "contains", "not_contains", "is_empty", "is_not_empty", "is_not_null",
    "any", "all",
}


class Rule(BaseModel):
    id: str
    name: str
    category: Category
    priority: int
    when: dict[str, Any]
    action: Action
    reason: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


def load_rules(path: Path = RULES_FILE) -> list[Rule]:
    """加载 rules.yaml；非法结构直接抛错（fail-fast，不静默跳过）。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "rules" not in raw or not isinstance(raw["rules"], list):
        raise ValueError(f"规则文件格式非法: {path}")
    rules = []
    for item in raw["rules"]:
        try:
            rules.append(Rule.model_validate(item))
        except Exception as e:  # pydantic ValidationError
            raise ValueError(f"规则定义非法（{item.get('id', '?')}）: {e}") from e
    return rules
