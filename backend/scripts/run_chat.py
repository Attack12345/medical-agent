"""对话联调脚本（M4 DoD）。

用法：
  python run_chat.py                # 三类问题（科室/用药/知识）各完整一轮
  python run_chat.py --demo-interrupt  # 演示问诊 HITL（interrupt 追问 + resume）
  python run_chat.py --question "头痛挂什么科"  # 单条问题
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.agent.graph import ask, ask_with_interrupt, resume  # noqa: E402
from app.agent.nodes_domain import init_retrieval_env  # noqa: E402

DEMO_QUESTIONS = [
    ("头痛应该挂什么科", "DEPARTMENT"),
    ("高血压吃什么药", "DRUG"),
    ("感冒有哪些症状", "KNOWLEDGE"),
]


def show(state: dict, title: str) -> None:
    print(f"\n--- {title} ---")
    print(f"意图: {state.get('intent')}")
    print(f"实体: {[(e['name'], e['label']) for e in state.get('entities', [])]}")
    print(f"图谱证据: {len(state.get('graph_evidence', []))} 条 | 检索证据: {len(state.get('retrieval_evidence', []))} 条 | 证据池: {len(state.get('evidence_pool', []))} 条")
    print(f"风险: {state.get('risk_level')} | 拒答: {state.get('refusal')}")
    print(f"回答:\n{state.get('answer', '')}")


def run_rounds() -> None:
    for question, expect in DEMO_QUESTIONS:
        state = ask(question, thread_id=f"demo-{expect.lower()}")
        assert state.get("intent") == expect, f"意图不符: {state.get('intent')} != {expect}"
        assert state.get("answer"), "回答为空"
        show(state, f"{question}（期望意图 {expect}）")
    print("\n✅ 三类问题各完整一轮通过（M4 DoD）")


def run_interrupt_demo() -> None:
    """HITL：笼统提问 → interrupt 追问 → resume 补充 → 完整回答。"""
    print("\n=== 问诊 HITL 演示（interrupt + resume）===")
    graph, config, state, pending, finished = ask_with_interrupt("我不舒服，怎么办", thread_id="hitl-demo")
    if not finished:
        print(f"[interrupt] 追问：{pending}")
        state, pending, finished = resume(graph, config, "我头痛而且发烧了")
    if pending:
        print(f"[interrupt] 再次追问：{pending}")
        state, pending, finished = resume(graph, config, "还有点咳嗽")
    show(state, "resume 后最终结果")
    assert state.get("answer"), "interrupt 流程回答为空"
    print("✅ interrupt 对话跑通（M4 DoD）")


def main() -> int:
    parser = argparse.ArgumentParser(description="对话联调")
    parser.add_argument("--question", type=str, default=None, help="单条问题")
    parser.add_argument("--demo-interrupt", action="store_true", help="演示 interrupt 问诊")
    args = parser.parse_args()

    init_retrieval_env()
    if args.question:
        show(ask(args.question), args.question)
        return 0
    if args.demo_interrupt:
        run_interrupt_demo()
        return 0
    run_rounds()
    run_interrupt_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
