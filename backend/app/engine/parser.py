"""DSL 编译校验：操作符白名单、字段路径合法性、$ 引用语法（非法即启动报错）。"""
import re
from typing import Any

from app.engine.dsl import ALLOWED_OPS, ALLOWED_PREFIXES, Rule

# $chat.evidence_pool[].text（空 [] 纯投影）或 $jd.a.b[?f==v].p（过滤+投影，? 可省略）
REF_RE = re.compile(
    r"^\$(?P<path>[a-z_]+(\.[a-z_]+)*)"
    r"(\[(?P<cond>\??\w+\s*==\s*[\w\"']*)?\])?"   # 括号内条件可整体缺省（空 [] 纯投影）
    r"(\.(?P<proj>\w+))?$"
)


class DSLError(ValueError):
    """DSL 编译错误（规则定义不合法）。"""


def compile_rules(rules: list[Rule]) -> list[Rule]:
    """校验全部规则；任何一条非法即抛 DSLError。"""
    seen_ids: set[str] = set()
    for rule in rules:
        if rule.id in seen_ids:
            raise DSLError(f"规则 id 重复: {rule.id}")
        seen_ids.add(rule.id)
        _validate_when(rule.id, rule.when)
    return rules


def _validate_when(rule_id: str, when: dict, depth: int = 0) -> None:
    if depth > 3:
        raise DSLError(f"规则 {rule_id}: when 嵌套过深")
    if not isinstance(when, dict) or not when:
        raise DSLError(f"规则 {rule_id}: when 必须为非空对象")
    for path, cond in when.items():
        if path.startswith("$"):
            raise DSLError(f"规则 {rule_id}: 路径不得以 $ 开头（{path}），$ 只能出现在操作符的值里")
        if not any(path.startswith(p) for p in ALLOWED_PREFIXES):
            raise DSLError(f"规则 {rule_id}: 字段路径前缀非法（{path}），允许: {ALLOWED_PREFIXES}")
        _validate_cond(rule_id, cond, path)


def _validate_cond(rule_id: str, cond: Any, path: str) -> None:
    if not isinstance(cond, dict) or not cond:
        raise DSLError(f"规则 {rule_id}: 条件必须为非空对象（{path}）")
    for op, value in cond.items():
        if op not in ALLOWED_OPS:
            raise DSLError(f"规则 {rule_id}: 操作符非法（{path}.{op}），允许: {sorted(ALLOWED_OPS)}")
        if op in ("any", "all"):
            if not isinstance(value, dict):
                raise DSLError(f"规则 {rule_id}: {path}.{op} 必须为子条件对象")
            for sub_path, sub_cond in value.items():
                # 两种形态：{field: {op: value}} 或 {field: 字面量}（eq 简写）
                if isinstance(sub_cond, dict):
                    _validate_cond(rule_id, sub_cond, f"{path}[].{sub_path}")
                else:
                    _validate_cond(rule_id, {"eq": sub_cond}, f"{path}[].{sub_path}")
        elif isinstance(value, str) and value.startswith("$"):
            if not REF_RE.match(value):
                raise DSLError(f"规则 {rule_id}: $ 引用语法非法（{path}: {value}），"
                               f"期望 $jd.a.b[?f==v].p 形式")
