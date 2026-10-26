"""
unit3_memory/extended_main.py
─────────────────────────────
Course Concept: Units 3 & 4 Combined Extension.
- Unit 3: Long-Term Memory (ChromaDB Vector Store) + Missing-Data Feedback Loop
- Unit 4: Role-Based Agents + Shared Blackboard + FIPA-ACL + Human Approval Gate

Usage:
  python main.py --mode extended
  python main.py --mode extended --auto-approve
  python main.py --mode extended --answers data/answers.json
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END
from llm_config import get_llm
from unit4_agents.roles import (
    agent_coordinator_init,
    agent_retriever,
    agent_categorizer,
    agent_auditor,
    agent_reporter,
)

logger = logging.getLogger("extended_pipeline")


def agent_confirm_summary(state: dict[str, Any]) -> dict[str, Any]:
    print("\n" + "=" * 65)
    print(" [AGENT: COORDINATOR] — Final Pipeline Status & Execution Summary")
    print("=" * 65)

    status = state.get("workflow_status", "unknown")
    totals = state.get("totals", {})
    exceptions = state.get("exceptions", [])
    msg_log = state.get("message_log", [])
    eml_path = state.get("eml_path", "None")

    print(f"""
  +-------------------------------------------------------------+
  |        EXTENDED MULTI-AGENT EXPENSE REPORT PIPELINE         |
  +-------------------------------------------------------------+
  | Workflow Status : {status.upper()}
  | Period          : {state.get('date_range', ('?', '?'))[0]} -> {state.get('date_range', ('?', '?'))[1]}
  | Grand Total     : ₹{totals.get('grand_total', 0):,.2f}
  | PDF Published   : {state.get('pdf_path', 'N/A')}
  | RFC 822 EML File: {eml_path}
  | Flagged / Audited: {len(exceptions)}
  | ACL Messages    : {len(msg_log)} messages exchanged
  | LLM Provider    : {state.get('llm_label', 'N/A')}
  +-------------------------------------------------------------+""")

    if exceptions:
        print("\n  [AUDIT TRAIL] Flagged items requiring manual intervention:")
        for exc in exceptions:
            print(f"    - [{exc.get('date','?')}] {exc.get('vendor','?')} -> {exc.get('_flag','flagged')}")

    print("\n  [MULTI-AGENT LOG] FIPA-ACL Communication Transcript:")
    for i, m in enumerate(msg_log, 1):
        perf = m.get("performative", "").upper()
        snd = m.get("sender", "")
        rcv = m.get("receiver", "")
        cnt = str(m.get("content", ""))
        if len(cnt) > 75:
            cnt = cnt[:72] + "..."
        print(f"    {i:02d}. [{perf}] {snd} -> {rcv}: {cnt}")

    print("\n  Extended multi-agent pipeline completed successfully.\n")
    return state


def build_extended_graph():
    """Build the role-based multi-agent graph with shared blackboard state."""
    graph = StateGraph(dict)

    # Register role nodes
    graph.add_node("coordinator", agent_coordinator_init)
    graph.add_node("retriever",   agent_retriever)
    graph.add_node("categorizer", agent_categorizer)
    graph.add_node("auditor",     agent_auditor)
    graph.add_node("reporter",    agent_reporter)
    graph.add_node("confirm",     agent_confirm_summary)

    # Sequence nodes through Blackboard
    graph.set_entry_point("coordinator")
    graph.add_edge("coordinator", "retriever")
    graph.add_edge("retriever",   "categorizer")
    graph.add_edge("categorizer", "auditor")
    graph.add_edge("auditor",     "reporter")
    graph.add_edge("reporter",    "confirm")
    graph.add_edge("confirm",     END)

    return graph.compile()


def run_extended(args: Any) -> dict[str, Any]:
    """Runner for extended mode."""
    print("  Initialising LLM for Extended Multi-Agent System …")
    llm, llm_label = get_llm()
    print(f"   Provider resolved: {llm_label}\n")

    app = build_extended_graph()

    csv_path = str(Path(__file__).parent.parent / "data" / "mock_expenses.csv")

    initial_state = {
        "instruction": getattr(args, "instruction", "Generate my travel expense report for last month and email it to finance."),
        "csv_path": csv_path,
        "mode": "extended",
        "llm": llm,
        "llm_label": llm_label,
        "auto_approve": getattr(args, "auto_approve", False),
        "answers_file": getattr(args, "answers", None),
        "step_log": [],
        "message_log": [],
        "retry_count": 0,
    }

    final_state = app.invoke(initial_state)
    return final_state
