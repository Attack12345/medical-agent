"""规则执行器：字段路径求值、操作符语义、优先级/短路、reason 渲染、轨迹。

设计要点：
- 执行顺序 priority 降序，同优先级按 id 字典序；
- VETO 命中短路（后续 VETO 跳过，VALIDATE 仍执行）；
- fail-fast：求值异常即抛错，不静默吞。
"""
import re
from typing import Any

from pydantic import BaseModel

from app.engine.dsl import Rule
from app.engine.parser import REF_RE


class RuleHit(BaseModel):
    rule_id: str
    rule_name: str
    reason: str
    matched_fields: dict[str, Any]
    priority: int
    action: str


class RuleResult(BaseModel):
    passed: bool                    # 无 VETO 命中
    decision: str                   # REJECT / PASS（hard_filter 语境）
    veto_hits: list[RuleHit]
    validate_hits: list[RuleHit]
    actions: list[str]              # 全部命中 action（去重保序）
    trails: list[dict[str, Any]]    # 全量轨迹（含未命中与短路跳过）


def get_path(context: dict, path: str) -> Any:
    """按 a.b.c 取嵌套值；不存在返回 None。"""
    node: Any = context
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
            node = node[int(key)]
        else:
            return None
    return node


def eval_ref(context: dict, ref: str) -> Any:
    """求值 $ 引用：$jd.required_skills[?veto==true].name（过滤+投影）。"""
    m = REF_RE.match(ref)
    if not m:
        return None
    value = get_path(context, m.group("path"))
    cond = m.group("cond")
    if cond and isinstance(value, list):
        field, raw = cond.split("==", 1)
        field = field.strip().lstrip("?").strip()
        raw = raw.strip().strip("\"'")
        expect: Any = True if raw == "true" else False if raw == "false" else raw
        value = [e for e in value if isinstance(e, dict) and e.get(field) == expect]
    proj = m.group("proj")
    if proj and isinstance(value, list):
        value = [e[proj] for e in value if isinstance(e, dict) and proj in e]
    return value


def resolve_value(context: dict, value: Any) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return eval_ref(context, value)
    return value


def _to_num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _name_set(v: Any) -> set[str]:
    """归一化为字符串集合：list[dict] 取 name 字段，list[str] 直接用。"""
    if isinstance(v, list):
        out: set[str] = set()
        for e in v:
            if isinstance(e, dict) and "name" in e:
                out.add(str(e["name"]))
            elif isinstance(e, str):
                out.add(e)
        return out
    return {str(v)} if v is not None else set()


def _contains_any(left_set: set[str], right_set: set[str]) -> bool:
    """包含语义：任一 left 元素被任一 right 元素包含（或反向）。"""
    for l in left_set:
        for r in right_set:
            if l and r and (l in r or r in l):
                return True
    return False


def _compare(left: Any, op: str, right: Any) -> bool:
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op in ("gt", "gte", "lt", "lte"):
        l, r = _to_num(left), _to_num(right)
        if l is None or r is None:
            return False
        return {"gt": l > r, "gte": l >= r, "lt": l < r, "lte": l <= r}[op]
    if op == "in":
        return left in right if isinstance(right, (list, tuple, set)) else False
    if op == "contains":
        return _contains_any(_name_set(left), _name_set(right))
    if op == "not_contains":
        return not _contains_any(_name_set(left), _name_set(right))
    if op == "is_empty":
        return left is None or left == "" or left == [] or left == {}
    if op == "is_not_empty":
        return left is not None and left != "" and left != [] and left != {}
    if op == "is_not_null":
        return left is not None and left != ""
    raise ValueError(f"未知操作符: {op}")


def eval_cond(left: Any, cond: dict, context: dict, matched: dict, missing_sink: dict) -> bool:
    """求值单条件 {op: value}；any/all 递归；matched 记录原始值用于轨迹/reason。"""
    for op, raw in cond.items():
        if op in ("any", "all"):
            if not isinstance(left, list):
                return False
            items = [e for e in left if isinstance(e, dict)]
            if op == "any":
                return any(
                    all(eval_cond(e.get(f), c if isinstance(c, dict) else {"eq": c},
                                  context, matched, missing_sink)
                        for f, c in raw.items())
                    for e in items
                )
            return bool(items) and all(
                eval_cond(e.get(f), c if isinstance(c, dict) else {"eq": c},
                          context, matched, missing_sink)
                for e in items for f, c in raw.items()
            )
        right = resolve_value(context, raw)
        result = _compare(left, op, right)
        if op == "not_contains" and result:
            # 命中缺失：计算差集供 reason 渲染 {缺失列表}
            missing = _name_set(right) - _name_set(left)
            missing_sink["缺失列表"] = sorted(missing)
        return result
    return False


def evaluate_when(context: dict, when: dict, matched: dict, missing_sink: dict) -> bool:
    """when 多条件 AND；matched 记录 path→原始值。"""
    for path, cond in when.items():
        left = get_path(context, path)
        matched[path] = left
        if not eval_cond(left, cond, context, matched, missing_sink):
            return False
    return True


def render_reason(rule: Rule, matched: dict, missing_sink: dict, context: dict) -> str:
    """渲染 reason 模板：{path} 从 context 解析（含 when 外的路径，如 $ 引用目标），
    优先 matched（原始值），其次 get_path 求值；{缺失列表} 由 not_contains 命中注入。"""
    reason = rule.reason
    # 1) matched 中的路径（when 左侧）
    for path, value in matched.items():
        reason = reason.replace("{" + path + "}", str(value))
    # 2) 剩余 {前缀路径} 占位符（如 {chat.intent}）从 context 求值
    for m in re.finditer(r"\{((?:chat)\.[a-z_]+)\}", reason):
        path = m.group(1)
        value = get_path(context, path)
        reason = reason.replace("{" + path + "}", str(value))
    # 3) 缺失列表
    if "{缺失列表}" in reason:
        reason = reason.replace("{缺失列表}", "、".join(missing_sink.get("缺失列表", [])) or "无")
    return reason


def execute(context: dict, rules: list[Rule]) -> RuleResult:
    """按 priority 降序执行；VETO 命中短路后续**同 action** 的 VETO（v1.2 细化：
    S001-S004 为组合式要求，免责声明命中不应短路急症警告/用药提醒/拒答；
    同 action 短路避免重复附加同一要求）；任何异常向上抛（fail-fast）。"""
    ordered = sorted(rules, key=lambda r: (-r.priority, r.id))
    trails: list[dict[str, Any]] = []
    veto_hits: list[RuleHit] = []
    validate_hits: list[RuleHit] = []
    blocked_actions: set[str] = set()

    for rule in ordered:
        if rule.category == "VETO" and rule.action in blocked_actions:
            trails.append({"rule_id": rule.id, "rule_name": rule.name,
                           "hit": False, "matched_fields": {"_skipped": "veto blocked (same action)"},
                           "priority": rule.priority})
            continue
        matched: dict[str, Any] = {}
        missing_sink: dict[str, Any] = {}
        hit = evaluate_when(context, rule.when, matched, missing_sink)
        trails.append({"rule_id": rule.id, "rule_name": rule.name, "hit": hit,
                       "matched_fields": matched, "priority": rule.priority})
        if not hit:
            continue
        item = RuleHit(rule_id=rule.id, rule_name=rule.name,
                       reason=render_reason(rule, matched, missing_sink, context),
                       matched_fields=matched, priority=rule.priority, action=rule.action)
        if rule.category == "VETO":
            veto_hits.append(item)
            blocked_actions.add(rule.action)
        else:
            validate_hits.append(item)

    actions = [h.action for h in veto_hits + validate_hits]
    return RuleResult(
        passed=not veto_hits,
        decision="REJECT" if veto_hits else "PASS",
        veto_hits=veto_hits,
        validate_hits=validate_hits,
        actions=list(dict.fromkeys(actions)),  # 去重保序
        trails=trails,
    )
