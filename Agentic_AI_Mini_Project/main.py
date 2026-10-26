"""
main.py
───────
Course concept: AGENTIC PIPELINE with LangGraph (State Machine / DAG).

This file wires the 9-node LangGraph pipeline:
  retrieve → filter → categorize → handle_missing → calculate
           → generate_pdf → draft_email → send_email → confirm

The shared state (ExpenseState TypedDict) is the "blackboard" that every
node reads from and writes to — demonstrating how agents communicate
through a shared memory structure.

Usage:
  python main.py                         # base mode, Ollama LLM
  python main.py --mode base             # explicit base mode
  python main.py --mode extended         # Phase 1 extensions
  python main.py --correct "Vendor=Cat"  # save a user correction to memory

Run with PYTHONIOENCODING=utf-8 on Windows:
  $env:PYTHONIOENCODING="utf-8"; python main.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from typing_extensions import TypedDict

# ── Windows UTF-8 fix ──────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from langgraph.graph import StateGraph, END

from llm_config import get_llm
from tools import (
    retrieve_expenses,
    filter_last_month,
    categorize_expenses,
    calculate_totals,
    generate_pdf_report,
    draft_email,
    send_email_with_pdf,
    last_month_range,
    get_report_today,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")

# ════════════════════════════════════════════════════════════════════════════
# State Schema  (the shared "blackboard" passed between every node)
# ════════════════════════════════════════════════════════════════════════════

class ExpenseState(TypedDict, total=False):
    """Shared state passed between every LangGraph node."""

    # Inputs
    instruction: str               # Original user instruction
    csv_path:    str               # Path to mock CSV
    mode:        str               # "base" or "extended"

    # Data
    raw_expenses:         list[dict]   # All rows from CSV
    filtered_expenses:    list[dict]   # Rows in last calendar month
    categorized_expenses: list[dict]   # All rows with category filled in
    exceptions:           list[dict]   # Flagged rows (missing data)

    # Computed
    totals:      dict[str, Any]   # {category_totals, grand_total}
    date_range:  tuple[str, str]  # (first_day_str, last_day_str)

    # Outputs
    pdf_path:     str    # Absolute path to generated PDF
    email_body:   str    # Drafted email text
    email_result: dict   # Send result (status, recipient, …)

    # LLM info
    llm:       Any   # The chat model instance
    llm_label: str   # Human-readable provider label

    # Phase 1 — short-term memory (sliding window of step summaries)
    step_log: list[str]


# ════════════════════════════════════════════════════════════════════════════
# Helper: format date range strings
# ════════════════════════════════════════════════════════════════════════════

def _date_range_strings() -> tuple[str, str]:
    first, last = last_month_range()
    return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")


def _log_step(state: ExpenseState, msg: str) -> list[str]:
    """Append a step summary to the sliding window log."""
    log = list(state.get("step_log", []))
    window = int(os.getenv("SHORT_TERM_WINDOW", "5"))
    log.append(msg)
    return log[-window:]   # keep only last N


# ════════════════════════════════════════════════════════════════════════════
# Node 1 — Retrieve
# ════════════════════════════════════════════════════════════════════════════

def node_retrieve(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print(" NODE 1 | RETRIEVE — Loading expenses from CSV")
    print("=" * 60)

    csv_path = state.get("csv_path", "data/mock_expenses.csv")
    raw = retrieve_expenses(csv_path)

    print(f"   => {len(raw)} expense rows loaded from {csv_path}")
    step_log = _log_step(state, f"RETRIEVE: loaded {len(raw)} rows from {csv_path}")
    return {**state, "raw_expenses": raw, "step_log": step_log}


# ════════════════════════════════════════════════════════════════════════════
# Node 2 — Filter
# ════════════════════════════════════════════════════════════════════════════

def node_filter(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print(" NODE 2 | FILTER — Keeping only last calendar month")
    print("=" * 60)

    date_range = _date_range_strings()
    print(f"   => Date range: {date_range[0]} to {date_range[1]}")
    print(f"   => Reference date (REPORT_TODAY): {get_report_today()}")

    valid, date_exceptions = filter_last_month(state["raw_expenses"])

    print(f"   => {len(valid)} rows in range | {len(date_exceptions)} rows flagged/excluded")
    step_log = _log_step(state, f"FILTER: {len(valid)} valid rows, date={date_range[0]}..{date_range[1]}")
    return {
        **state,
        "filtered_expenses": valid,
        "exceptions":        date_exceptions,
        "date_range":        date_range,
        "step_log":          step_log,
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 3 — Categorize (LLM or keyword fallback)
# ════════════════════════════════════════════════════════════════════════════

def node_categorize(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print(" NODE 3 | CATEGORIZE — Inferring missing categories")
    print("=" * 60)

    llm       = state.get("llm")
    llm_label = state.get("llm_label", "Unknown")
    print(f"   => LLM provider: {llm_label}")

    # Phase 1: if extended mode, use memory store (imported lazily)
    few_shot = None
    if state.get("mode") == "extended":
        try:
            from unit3_memory.memory_store import MemoryStore
            ms = MemoryStore()
            # We pass None as query here; per-row retrieval happens inside extended categorize
        except ImportError:
            pass

    categorized, amount_exceptions = categorize_expenses(
        state["filtered_expenses"], llm, llm_label
    )

    existing_exceptions = state.get("exceptions", [])
    all_exceptions = existing_exceptions + amount_exceptions

    print(f"   => {len(categorized)} rows categorised | {len(amount_exceptions)} missing-amount flags")
    step_log = _log_step(state, f"CATEGORIZE: {len(categorized)} rows, {len(amount_exceptions)} flagged")
    return {
        **state,
        "categorized_expenses": categorized,
        "exceptions":           all_exceptions,
        "step_log":             step_log,
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 4 — Handle Missing Data
# ════════════════════════════════════════════════════════════════════════════

def node_handle_missing(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print(" NODE 4 | HANDLE MISSING — Auditing flagged items")
    print("=" * 60)

    exceptions = state.get("exceptions", [])
    print(f"   => Total flagged items: {len(exceptions)}")
    for i, exc in enumerate(exceptions, 1):
        print(
            f"   [{i}] Vendor: {exc.get('vendor','?')} | "
            f"Date: {exc.get('date','?')} | "
            f"Flag: {exc.get('_flag','?')}"
        )

    if not exceptions:
        print("   OK: No exceptions — all rows processed cleanly.")

    step_log = _log_step(state, f"HANDLE_MISSING: {len(exceptions)} flagged items audited")
    return {**state, "step_log": step_log}   # no state change beyond log


# ════════════════════════════════════════════════════════════════════════════
# Node 5 — Calculate (deterministic, NO LLM)
# ════════════════════════════════════════════════════════════════════════════

def node_calculate(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print(" NODE 5 | CALCULATE — Computing totals (deterministic Python)")
    print("=" * 60)

    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    totals = calculate_totals(state["categorized_expenses"], state.get("exceptions", []))

    print(f"   => Grand total: {currency}{totals['grand_total']:,.2f}")
    if totals.get("flagged_count"):
        print(f"      (excludes {totals['flagged_count']} flagged items)")
    for cat, amt in sorted(totals["category_totals"].items()):
        print(f"      {cat}: {currency}{amt:,.2f}")

    step_log = _log_step(state, f"CALCULATE: grand_total={currency}{totals['grand_total']:,.2f}")
    return {**state, "totals": totals, "step_log": step_log}


# ════════════════════════════════════════════════════════════════════════════
# Node 6 — Generate PDF
# ════════════════════════════════════════════════════════════════════════════

def node_generate_pdf(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print("NODE 6 | GENERATE PDF — Building ReportLab PDF")
    print("=" * 60)

    output_path = str(Path(__file__).parent / "expense_report.pdf")

    pdf_path = generate_pdf_report(
        rows       = state["categorized_expenses"],
        totals     = state["totals"],
        exceptions = state.get("exceptions", []),
        date_range = state["date_range"],
        output_path= output_path,
    )

    print(f"   => PDF saved: {pdf_path}")
    step_log = _log_step(state, f"GENERATE_PDF: saved to {pdf_path}")
    return {**state, "pdf_path": pdf_path, "step_log": step_log}


# ════════════════════════════════════════════════════════════════════════════
# Node 7 — Draft Email (LLM)
# ════════════════════════════════════════════════════════════════════════════

def node_draft_email(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print(" NODE 7 | DRAFT EMAIL — Composing email with LLM")
    print("=" * 60)

    llm       = state.get("llm")
    llm_label = state.get("llm_label", "Unknown")
    print(f"   => LLM provider: {llm_label}")

    body = draft_email(
        totals         = state["totals"],
        date_range     = state["date_range"],
        num_exceptions = len(state.get("exceptions", [])),
        llm            = llm,
        llm_label      = llm_label,
        instruction    = state.get("instruction", "Generate my travel expense report for last month and email it to finance."),
        short_term_ctx = state.get("step_log"),
    )

    # Save drafted email to file for easy inspection
    recipient      = os.getenv("FINANCE_EMAIL", "finance@yourcompany.com")
    email_txt_path = Path(__file__).parent / "drafted_email.txt"
    with open(email_txt_path, "w", encoding="utf-8") as f:
        f.write(f"To: {recipient}\n")
        f.write(f"Attachment: {state.get('pdf_path', 'expense_report.pdf')}\n")
        f.write("-" * 60 + "\n")
        f.write(body + "\n")

    # Print the full drafted email in terminal
    print("\n   +------------------------------------------------------------+")
    print("   |  FULL DRAFTED EMAIL CONTENT (TO FINANCE)                   |")
    print("   +------------------------------------------------------------+")
    print(f"   | To         : {recipient}")
    print(f"   | Attachment : {Path(state.get('pdf_path', 'expense_report.pdf')).name}")
    print("   +------------------------------------------------------------+")
    for line in body.split("\n"):
        print(f"   | {line}")
    print("   +------------------------------------------------------------+")
    print(f"   => Saved full email text to: {email_txt_path.name}")

    step_log = _log_step(state, f"DRAFT_EMAIL: body drafted ({len(body)} chars)")
    return {**state, "email_body": body, "step_log": step_log}


# ════════════════════════════════════════════════════════════════════════════
# Node 8 — Send Email
# ════════════════════════════════════════════════════════════════════════════

def node_send_email(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print(" NODE 8 | SEND EMAIL — Dispatching via SMTP")
    print("=" * 60)

    recipient = os.getenv("FINANCE_EMAIL", "finance@yourcompany.com")
    print(f"   => Recipient: {recipient}")

    result = send_email_with_pdf(
        email_body = state["email_body"],
        pdf_path   = state["pdf_path"],
        recipient  = recipient,
    )

    print(f"   => Send status: {result['status']}")
    step_log = _log_step(state, f"SEND_EMAIL: status={result['status']}")
    return {**state, "email_result": result, "step_log": step_log}


# ════════════════════════════════════════════════════════════════════════════
# Node 9 — Confirm (final summary)
# ════════════════════════════════════════════════════════════════════════════

def node_confirm(state: ExpenseState) -> ExpenseState:
    print("\n" + "=" * 60)
    print(" NODE 9 | CONFIRM — Final Summary")
    print("=" * 60)

    totals     = state.get("totals", {})
    exceptions = state.get("exceptions", [])
    result     = state.get("email_result", {})
    date_range = state.get("date_range", ("?", "?"))

    print(f"""
  +-----------------------------------------------+
  |         EXPENSE REPORT AGENT -- DONE          |
  +-----------------------------------------------+
  | Period          : {date_range[0]} -> {date_range[1]}
  | Grand Total     : ₹{totals.get('grand_total', 0):,.2f}
  | PDF Report      : {state.get('pdf_path', 'N/A')}
  | Email Body File : drafted_email.txt
  | Email Status    : {result.get('status', 'N/A')}
  | Sent To         : {result.get('recipient', 'N/A')}
  | Flagged Items   : {len(exceptions)}
  | LLM Provider    : {state.get('llm_label', 'N/A')}
  +-----------------------------------------------+""")

    if exceptions:
        print("\n  WARNING: Flagged rows requiring manual review:")
        for exc in exceptions:
            print(
                f"    - [{exc.get('date','?')}] {exc.get('vendor','?')} — "
                f"{exc.get('_flag','?')}"
            )

    print("\n  Agent pipeline completed successfully.\n")
    return state


# ════════════════════════════════════════════════════════════════════════════
# Build the LangGraph (base pipeline)
# ════════════════════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(ExpenseState)

    # Register nodes
    graph.add_node("retrieve",       node_retrieve)
    graph.add_node("filter",         node_filter)
    graph.add_node("categorize",     node_categorize)
    graph.add_node("handle_missing", node_handle_missing)
    graph.add_node("calculate",      node_calculate)
    graph.add_node("generate_pdf",   node_generate_pdf)
    graph.add_node("draft_email",    node_draft_email)
    graph.add_node("send_email",     node_send_email)
    graph.add_node("confirm",        node_confirm)

    # Define edges (sequential pipeline)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",       "filter")
    graph.add_edge("filter",         "categorize")
    graph.add_edge("categorize",     "handle_missing")
    graph.add_edge("handle_missing", "calculate")
    graph.add_edge("calculate",      "generate_pdf")
    graph.add_edge("generate_pdf",   "draft_email")
    graph.add_edge("draft_email",    "send_email")
    graph.add_edge("send_email",     "confirm")
    graph.add_edge("confirm",        END)

    return graph.compile()


# ════════════════════════════════════════════════════════════════════════════
# CLI helpers
# ════════════════════════════════════════════════════════════════════════════

def _parse_args():
    parser = argparse.ArgumentParser(description="Expense Report Agent")
    parser.add_argument(
        "--mode", choices=["base", "extended"], default="extended",
        help="Run mode: 'extended' (full agentic pipeline with memory, feedback, approval) or 'base' (original 9-node pipeline)",
    )
    parser.add_argument(
        "--correct", type=str, default=None,
        metavar="VENDOR=CATEGORY",
        help="Save a user correction to long-term memory, e.g. --correct 'Olive Garden=Meals'",
    )
    parser.add_argument(
        "--instruction", type=str,
        default="Generate my travel expense report for last month and email it to finance.",
        help="Natural language instruction for the agent.",
    )
    parser.add_argument(
        "--answers", type=str, default=None,
        help="Path to JSON file with scripted answers for missing-data prompts (Phase 1).",
    )
    parser.add_argument(
        "--auto-approve", action="store_true",
        help="Skip human approval gate (Phase 2).",
    )
    return parser.parse_args()


def _handle_correction(correction_str: str):
    """Save a user category correction to long-term memory."""
    if "=" not in correction_str:
        print(f"ERROR: --correct must be in format 'Vendor=Category', got: '{correction_str}'")
        sys.exit(1)

    vendor_part, cat_part = correction_str.split("=", 1)
    vendor = vendor_part.strip()
    cat    = cat_part.strip()

    from tools import VALID_CATEGORIES
    if cat not in VALID_CATEGORIES:
        print(f"ERROR: '{cat}' is not a valid category. Choose from: {sorted(VALID_CATEGORIES)}")
        sys.exit(1)

    try:
        from unit3_memory.memory_store import MemoryStore
        ms = MemoryStore()
        ms.store(
            merchant=vendor,
            description=f"User-corrected category for {vendor}",
            category=cat,
            source="user_correction",
        )
        print(f"  CORRECTION SAVED: '{vendor}' -> '{cat}' (source: user_correction)")
        print("  This correction will be used in future runs via long-term memory.")
    except ImportError:
        print("  ERROR: Phase 1 memory module not found. Run Phase 1 first.")
        sys.exit(1)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = _parse_args()

    # ── Handle --correct shortcut ─────────────────────────────────────────
    if args.correct:
        _handle_correction(args.correct)
        return

    print("\n" + "=" * 60)
    print("   EXPENSE REPORT AGENT — Starting Up")
    print(f"   Mode: {args.mode.upper()}")
    print("=" * 60)
    print(f"\n Instruction: \"{args.instruction}\"\n")

    # ── If extended mode, delegate to extended pipeline ───────────────────
    if args.mode == "extended":
        try:
            from unit3_memory.extended_main import run_extended
            run_extended(args)
            return
        except ImportError:
            print("  WARNING: Phase 1 (extended mode) not yet installed.")
            print("  Falling back to base mode.\n")

    # ── Initialise LLM ─────────────────────────────────────────────────────
    print("  Initialising LLM …")
    llm, llm_label = get_llm()
    print(f"   Provider resolved: {llm_label}\n")

    # ── Build & run graph ──────────────────────────────────────────────────
    app = build_graph()

    csv_path = str(Path(__file__).parent / "data" / "mock_expenses.csv")

    initial_state: ExpenseState = {
        "instruction": args.instruction,
        "csv_path":    csv_path,
        "llm":         llm,
        "llm_label":   llm_label,
        "mode":        args.mode,
        "step_log":    [],
    }

    final_state = app.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    main()
