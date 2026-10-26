"""
unit4_agents/roles.py
─────────────────────
Course Concept: Unit 4 — Role-Based Agents, Shared Blackboard Architecture,
                and Adjustable Autonomy (Human-in-the-Loop Approval Gate).

Why this concept?
A complex end-to-end task (data retrieval, AI classification, audit compliance,
PDF document publishing, external email dispatch) is prone to single-point
failures if modeled as a monolithic script.
Decomposing into specialized roles with distinct responsibilities:
1. Coordinator: Goal decomposition, workflow orchestration, retry management.
2. Retriever: Data ingestion, schema validation, and date window filtering.
3. Categorizer: Semantic memory lookup, few-shot LLM inference, memory updates.
4. Auditor/Monitor: Anomaly detection, duplicate detection, and missing-data feedback.
5. Reporter: Deterministic calculations, PDF generation, draft generation, and
   email dispatch guarded by an explicit Human Approval Gate.

All agents communicate via:
- Shared Blackboard (LangGraph state dictionary)
- FIPA-ACL structured messaging
"""

from __future__ import annotations

import email.message
import email.policy
import email.utils
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
    VALID_CATEGORIES,
)
from unit3_memory.memory_store import ExpenseMemoryStore
from unit3_memory.feedback_loop import FeedbackLoopManager
from access_control import Role, Permission, check_permission, enforce_permission, validate_recipient
from audit_log import get_audit_log
from .acl_message import ACLMessage, ACLPerformative

logger = logging.getLogger(__name__)


def append_acl_message(
    state: dict[str, Any],
    performative: str,
    sender: str,
    receiver: str,
    content: Any,
) -> list[dict[str, Any]]:
    """Log a FIPA-ACL message and print it live in the terminal."""
    msg = ACLMessage(
        performative=performative,
        sender=sender,
        receiver=receiver,
        content=content,
    )
    msg_log = list(state.get("message_log", []))
    msg_log.append(msg.to_dict())
    print(f"   [ACL] {msg.format_line()}")
    return msg_log


def print_blackboard_snapshot(agent_name: str, state: dict[str, Any]) -> None:
    """Print an ASCII snapshot of the shared Blackboard state."""
    import os
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    raw_count = len(state.get("raw_expenses", []))
    filt_count = len(state.get("filtered_expenses", []))
    cat_count = len(state.get("categorized_expenses", []))
    exc_count = len(state.get("exceptions", []))
    totals = state.get("totals", {})
    grand_total = totals.get("grand_total", 0.0)
    msg_count = len(state.get("message_log", []))
    status = state.get("workflow_status", "in_progress")

    print("\n   +--- [BLACKBOARD SNAPSHOT: Post-" + agent_name.upper() + "] " + "-" * (32 - len(agent_name)) + "+")
    print(f"   | Status        : {status.upper()}")
    print(f"   | Active Agent  : {agent_name}")
    print(f"   | Data On Board : raw={raw_count} | filtered={filt_count} | categorized={cat_count} | flagged={exc_count}")
    print(f"   | Running Total : {currency}{grand_total:,.2f}")
    print(f"   | ACL Messages  : {msg_count} messages exchanged")
    print("   +-------------------------------------------------------------+\n")


def print_trace(node_name: str, tool_name: str, inputs: dict[str, Any], outputs: dict[str, Any], extras: Optional[dict[str, Any]] = None) -> None:
    """Print execution trace for viva verification."""
    print(f"\n   [TRACE] NODE: {node_name} | TOOL: {tool_name}")
    print(f"     -> INPUTS : {json.dumps(inputs, default=str)}")
    print(f"     -> OUTPUTS: {json.dumps(outputs, default=str)}")
    if extras:
        for k, v in extras.items():
            print(f"     -> {k.upper()}: {v}")


# ════════════════════════════════════════════════════════════════════════════
# 1. Coordinator Agent
# ════════════════════════════════════════════════════════════════════════════

def agent_coordinator_init(state: dict[str, Any]) -> dict[str, Any]:
    print("\n" + "=" * 65)
    print(" [AGENT: COORDINATOR] — Decomposing Goal & Initiating Workflow")
    print("=" * 65)

    instruction = state.get("instruction", "Generate my travel expense report for last month and email it to finance.")
    print(f"   Goal: \"{instruction}\"")
    print("   Workflow Plan: Retriever -> Categorizer -> Auditor -> Reporter")

    msg_log = append_acl_message(
        state,
        performative=ACLPerformative.REQUEST,
        sender="Coordinator",
        receiver="Retriever",
        content="Retrieve and filter last month's travel expenses from CSV source.",
    )

    new_state = {
        **state,
        "workflow_status": "in_progress",
        "retry_count": state.get("retry_count", 0),
        "message_log": msg_log,
    }
    print_blackboard_snapshot("Coordinator", new_state)
    return new_state


# ════════════════════════════════════════════════════════════════════════════
# 2. Retriever Agent
# ════════════════════════════════════════════════════════════════════════════

def agent_retriever(state: dict[str, Any]) -> dict[str, Any]:
    print("\n" + "=" * 65)
    print(" [AGENT: RETRIEVER] — Ingesting CSV and Applying Date Filtering")
    print("=" * 65)

    csv_path = state.get("csv_path", "data/mock_expenses.csv")
    raw = retrieve_expenses(csv_path)
    valid, date_exceptions = filter_last_month(raw)

    first, last = last_month_range()
    date_range = (first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))

    print_trace(
        node_name="RetrieverAgent",
        tool_name="retrieve_expenses + filter_last_month",
        inputs={"csv_path": csv_path, "report_today": str(get_report_today())},
        outputs={"raw_rows": len(raw), "in_range_rows": len(valid), "excluded_rows": len(date_exceptions)},
    )

    msg_log = append_acl_message(
        state,
        performative=ACLPerformative.INFORM,
        sender="Retriever",
        receiver="Coordinator",
        content=f"Ingested {len(raw)} rows. {len(valid)} rows within target period {date_range[0]} to {date_range[1]}.",
    )
    msg_log = append_acl_message(
        {**state, "message_log": msg_log},
        performative=ACLPerformative.REQUEST,
        sender="Coordinator",
        receiver="Categorizer",
        content=f"Classify categories for {len(valid)} candidate expense items using semantic memory + LLM.",
    )

    step_log = list(state.get("step_log", []))
    step_log.append(f"Retriever: Ingested {len(valid)} expenses for {date_range[0]}..{date_range[1]}")

    new_state = {
        **state,
        "raw_expenses": raw,
        "filtered_expenses": valid,
        "exceptions": date_exceptions,
        "date_range": date_range,
        "message_log": msg_log,
        "step_log": step_log[-5:],
    }
    print_blackboard_snapshot("Retriever", new_state)
    return new_state


# ════════════════════════════════════════════════════════════════════════════
# 3. Categorizer Agent (Unit 3 Memory Enhanced)
# ════════════════════════════════════════════════════════════════════════════

def agent_categorizer(state: dict[str, Any]) -> dict[str, Any]:
    print("\n" + "=" * 65)
    print(" [AGENT: CATEGORIZER] — Long-Term Memory Lookup & In-Context LLM")
    print("=" * 65)

    memory_store = ExpenseMemoryStore()
    rows = [dict(r) for r in state.get("filtered_expenses", [])]
    llm = state.get("llm")
    llm_label = state.get("llm_label", "Unknown")

    categorized: list[dict[str, Any]] = []
    amount_exceptions: list[dict[str, Any]] = []
    mem_hits = 0
    llm_calls = 0

    print(f"   Memory Store: ChromaDB (BAAI/bge-small-en-v1.5) | Threshold: {memory_store.similarity_threshold}")

    for r in rows:
        vendor = r.get("vendor", "")
        desc = r.get("description", "")
        existing_cat = r.get("category", "").strip()

        # Check for missing amount flag
        amt_str = r.get("amount", "").strip()
        has_amount = False
        if amt_str:
            try:
                has_amount = float(amt_str) > 0.0
            except ValueError:
                has_amount = False

        if not has_amount:
            r["_flag"] = "missing amount"
            amount_exceptions.append(dict(r))

        # Categorization logic
        if existing_cat and existing_cat in VALID_CATEGORIES:
            categorized.append(r)
            continue

        # 1. Query long term memory
        resolved_cat, matches, reason = memory_store.resolve_category(vendor, desc, top_k=3)

        if resolved_cat and resolved_cat in VALID_CATEGORIES:
            mem_hits += 1
            r["category"] = resolved_cat
            top_sim = matches[0]["similarity"] if matches else 1.0
            print(f"   [MEMORY HIT] '{vendor}' -> {resolved_cat} (sim={top_sim:.3f}) | {reason}")
            categorized.append(r)
        else:
            # 2. LLM Call with retrieved few-shot examples
            llm_calls += 1
            few_shots = [f"- {m['vendor']} ({m['description']}) -> {m['category']}" for m in matches[:3]]
            few_shot_str = "\n".join(few_shots) if few_shots else "No past records."
            print(f"   [LLM REQUIRED] '{vendor}' (below threshold) -> Querying {llm_label} with {len(matches)} memory few-shots")

            # Perform categorization (falls back to rules if Ollama unavailable)
            single_classified, _ = categorize_expenses([r], llm, llm_label)
            cat_result = single_classified[0].get("category", "Misc")
            r["category"] = cat_result
            categorized.append(r)

            # Store learned result in long-term memory
            memory_store.store_memory(vendor=vendor, description=desc, category=cat_result, source="model")

    print_trace(
        node_name="CategorizerAgent",
        tool_name="ExpenseMemoryStore.resolve_category + categorize_expenses",
        inputs={"num_rows": len(rows), "similarity_threshold": memory_store.similarity_threshold},
        outputs={"categorized_count": len(categorized), "memory_hits": mem_hits, "llm_calls": llm_calls},
    )

    all_exceptions = list(state.get("exceptions", [])) + amount_exceptions

    msg_log = append_acl_message(
        state,
        performative=ACLPerformative.INFORM,
        sender="Categorizer",
        receiver="Coordinator",
        content=f"Categorization complete. {mem_hits} resolved via vector memory, {llm_calls} required LLM/fallback.",
    )
    msg_log = append_acl_message(
        {**state, "message_log": msg_log},
        performative=ACLPerformative.REQUEST,
        sender="Coordinator",
        receiver="Auditor",
        content=f"Audit {len(all_exceptions)} flagged exception(s) and inspect for data anomalies.",
    )

    step_log = list(state.get("step_log", []))
    step_log.append(f"Categorizer: {len(categorized)} items categorized ({mem_hits} from memory, {llm_calls} via LLM)")

    new_state = {
        **state,
        "categorized_expenses": categorized,
        "exceptions": all_exceptions,
        "message_log": msg_log,
        "step_log": step_log[-5:],
    }
    print_blackboard_snapshot("Categorizer", new_state)
    return new_state


# ════════════════════════════════════════════════════════════════════════════
# 4. Auditor / Monitor Agent (Feedback Loop + Anomaly Detection)
# ════════════════════════════════════════════════════════════════════════════

def agent_auditor(state: dict[str, Any]) -> dict[str, Any]:
    print("\n" + "=" * 65)
    print(" [AGENT: AUDITOR / MONITOR] — Anomaly Detection & Feedback Loop")
    print("=" * 65)

    answers_file = state.get("answers_file")
    fb_manager = FeedbackLoopManager(answers_file=answers_file, max_retries=2)

    rows = state.get("categorized_expenses", [])
    raw_exceptions = state.get("exceptions", [])

    # Feedback Loop cycle: Act -> Observe -> Reflect -> Adjust
    missing_items = fb_manager.act_detect_missing(rows)
    obs = fb_manager.observe(missing_items, iteration=state.get("retry_count", 0))
    should_retry, strategy = fb_manager.reflect(obs)

    print(f"   [FEEDBACK LOOP - ACT] Found {len(missing_items)} item(s) missing critical values.")
    print(f"   [FEEDBACK LOOP - OBSERVE] Vendors: {obs['vendors']}")
    print(f"   [FEEDBACK LOOP - REFLECT] Strategy: {strategy}")

    updated_rows, remaining_exceptions, adjust_logs = fb_manager.adjust_resolve_items(
        rows=rows,
        missing_items=missing_items,
        auto_prompt=(state.get("mode") == "extended"),
    )

    for al in adjust_logs:
        print(f"   [FEEDBACK LOOP - ADJUST] {al}")

    # Check for rows without receipt (Receipt = No)
    no_receipt = []
    for r in updated_rows:
        receipt_val = r.get("Receipt", "").strip().lower()
        if receipt_val == "no":
            no_receipt.append(f"{r.get('vendor')} ({r.get('date')})")

    if no_receipt:
        print(f"   [AUDITOR WARNING] Items without receipt attached: {no_receipt}")

    # Check for duplicate transactions (Vendor + Date + Amount)
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    seen = set()
    duplicates = []
    for r in updated_rows:
        key = (r.get("vendor"), r.get("date"), r.get("amount"))
        if key in seen and r.get("amount"):
            duplicates.append(f"{r.get('vendor')} on {r.get('date')} ({currency}{r.get('amount')})")
        seen.add(key)

    if duplicates:
        print(f"   [AUDITOR WARNING] Potential duplicate transactions found: {duplicates}")

    # Check high value expenses (>= 1,000 in configured currency)
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    high_value = [
        f"{r.get('vendor')}: {currency}{float(r.get('amount')):.2f}"
        for r in updated_rows
        if r.get("amount") and float(r.get("amount", 0)) >= 1000.0
    ]
    if high_value:
        print(f"   [AUDITOR AUDIT] High-value expenses (>= {currency}1,000): {high_value}")

    print_trace(
        node_name="AuditorAgent",
        tool_name="FeedbackLoopManager (Act-Observe-Reflect-Adjust)",
        inputs={"initial_exceptions": len(missing_items), "answers_file": answers_file},
        outputs={"resolved_in_loop": len(adjust_logs), "remaining_exceptions": len(remaining_exceptions)},
    )

    msg_log = append_acl_message(
        state,
        performative=ACLPerformative.INFORM,
        sender="Auditor",
        receiver="Coordinator",
        content=f"Audit completed. {len(adjust_logs)} resolved via feedback loop; {len(remaining_exceptions)} escalated.",
    )
    msg_log = append_acl_message(
        {**state, "message_log": msg_log},
        performative=ACLPerformative.REQUEST,
        sender="Coordinator",
        receiver="Reporter",
        content="Calculate totals, generate ReportLab PDF, and draft email for human sign-off.",
    )

    step_log = list(state.get("step_log", []))
    step_log.append(f"Auditor: Feedback loop resolved {len(adjust_logs)} items, {len(remaining_exceptions)} remain flagged")

    new_state = {
        **state,
        "categorized_expenses": updated_rows,
        "exceptions": remaining_exceptions,
        "auditor_warnings": duplicates,
        "message_log": msg_log,
        "step_log": step_log[-5:],
    }
    print_blackboard_snapshot("Auditor", new_state)
    return new_state


# ════════════════════════════════════════════════════════════════════════════
# 5. Reporter Agent (Calculations, PDF, Approval Gate, EML Dispatch)
# ════════════════════════════════════════════════════════════════════════════

def agent_reporter(state: dict[str, Any]) -> dict[str, Any]:
    print("\n" + "=" * 65)
    print(" [AGENT: REPORTER] — ReportLab PDF, Approval Gate & EML Archive")
    print("=" * 65)

    # 1. Deterministic Calculation
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    totals = calculate_totals(state["categorized_expenses"], state.get("exceptions", []))
    print(f"   => Recalculated Grand Total: {currency}{totals['grand_total']:,.2f}")
    if totals.get("flagged_count"):
        print(f"      (excludes {totals['flagged_count']} flagged items)")
    for cat, amt in sorted(totals["category_totals"].items()):
        print(f"      - {cat}: {currency}{amt:,.2f}")

    # 2. PDF Generation
    pdf_out = str(Path(__file__).parent.parent / "expense_report.pdf")
    pdf_path = generate_pdf_report(
        rows=state["categorized_expenses"],
        totals=totals,
        exceptions=state.get("exceptions", []),
        date_range=state["date_range"],
        output_path=pdf_out,
    )
    print(f"   => PDF Published: {pdf_path}")

    # 3. Draft Email
    llm = state.get("llm")
    llm_label = state.get("llm_label", "Unknown")
    email_body = draft_email(
        totals=totals,
        date_range=state["date_range"],
        num_exceptions=len(state.get("exceptions", [])),
        llm=llm,
        llm_label=llm_label,
        instruction=state.get("instruction", ""),
        short_term_ctx=state.get("step_log"),
    )

    recipient = os.getenv("FINANCE_EMAIL", "finance@yourcompany.com")
    email_txt = Path(__file__).parent.parent / "drafted_email.txt"
    with open(email_txt, "w", encoding="utf-8") as f:
        f.write(f"To: {recipient}\nAttachment: {pdf_path}\n{'-'*60}\n{email_body}\n")

    # 4. Human Approval Gate (Adjustable Autonomy)
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    print("\n   " + "=" * 60)
    print("   [HUMAN APPROVAL GATE - ADJUSTABLE AUTONOMY]")
    print("   " + "=" * 60)
    print(f"   Destination  : {recipient}")
    print(f"   Grand Total  : {currency}{totals['grand_total']:,.2f}")
    print(f"   Period       : {state['date_range'][0]} to {state['date_range'][1]}")
    print(f"   Flagged Rows : {len(state.get('exceptions', []))}")
    if state.get("auditor_warnings"):
        print(f"   Auditor Alert: {state['auditor_warnings']}")
    print("   Attachment   : " + Path(pdf_path).name)

    is_auto = state.get("auto_approve", False)
    approved = False

    if is_auto:
        print("   Action       : AUTO-APPROVED (--auto-approve flag)")
        approved = True
    else:
        try:
            user_choice = input("\n   Type 'approve' to send email to finance, or 'reject' to cancel: ").strip().lower()
            if user_choice == "approve":
                approved = True
            else:
                print("   Action       : REJECTED by human operator.")
                approved = False
        except (EOFError, KeyboardInterrupt):
            print("   Action       : Non-interactive session, skipping dispatch.")
            approved = False

    if not approved:
        msg_log = append_acl_message(
            state,
            performative=ACLPerformative.REFUSE,
            sender="HumanOperator",
            receiver="Coordinator",
            content="User rejected expense dispatch during human approval gate.",
        )
        return {
            **state,
            "totals": totals,
            "pdf_path": pdf_path,
            "email_body": email_body,
            "workflow_status": "rejected_by_human",
            "message_log": msg_log,
        }

    # 5. Email Dispatch & EML Archive — with access control enforcement
    msg_log = append_acl_message(
        state,
        performative=ACLPerformative.ACCEPT_PROPOSAL,
        sender="HumanOperator",
        receiver="Reporter",
        content="Human operator approved expense report for finance submission.",
    )

    # ── ACCESS CONTROL: Only Reporter can send_email, and only after human approval ────
    audit = get_audit_log()
    if not check_permission(Role.REPORTER, Permission.SEND_EMAIL):
        print(f"   [ACL] DENIED: Reporter denied send_email permission")
        audit.log_action(
            agent="Reporter",
            action="send_email",
            status="denied",
            detail=f"Permission check failed",
            recipient=recipient,
        )
        msg_log = append_acl_message(
            {**state, "message_log": msg_log},
            performative=ACLPerformative.REFUSE,
            sender="Reporter",
            receiver="Coordinator",
            content="Reporter denied send_email permission by access control.",
        )
        return {
            **state,
            "totals": totals,
            "pdf_path": pdf_path,
            "email_body": email_body,
            "workflow_status": "access_denied",
            "message_log": msg_log,
        }

    # ── RECIPIENT VALIDATION ────
    if not validate_recipient(recipient):
        print(f"   [ACL] BLOCKED: Recipient {recipient} not in allowlist")
        audit.log_action(
            agent="Reporter",
            action="send_email",
            status="blocked",
            detail=f"Recipient not in allowlist",
            recipient=recipient,
        )
        msg_log = append_acl_message(
            {**state, "message_log": msg_log},
            performative=ACLPerformative.REFUSE,
            sender="Reporter",
            receiver="Coordinator",
            content=f"Email recipient {recipient} not in allowlist. Aborting send.",
        )
        return {
            **state,
            "totals": totals,
            "pdf_path": pdf_path,
            "email_body": email_body,
            "workflow_status": "recipient_blocked",
            "message_log": msg_log,
        }

    send_result = send_email_with_pdf(
        email_body=email_body,
        pdf_path=pdf_path,
        recipient=recipient,
    )

    # Save .eml file in /outbox
    outbox_dir = Path(__file__).parent.parent / "outbox"
    outbox_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    eml_path = outbox_dir / f"expense_report_{timestamp_str}.eml"

    msg = email.message.EmailMessage()
    msg["Subject"] = f"Travel Expense Report - {state['date_range'][0]} to {state['date_range'][1]}"
    msg["From"] = os.getenv("SMTP_FROM", "agent@expense-report.dev")
    msg["To"] = recipient
    msg["Date"] = email.utils.formatdate(localtime=True)
    msg.set_content(email_body)

    if Path(pdf_path).exists():
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        msg.add_attachment(
            pdf_bytes,
            maintype="application",
            subtype="pdf",
            filename=Path(pdf_path).name,
        )

    with open(eml_path, "wb") as f:
        f.write(msg.as_bytes(policy=email.policy.default))

    print(f"   => Full RFC 822 Email archived to: {eml_path}")

    # ── AUDIT LOGGING ────
    audit.log_action(
        agent="Reporter",
        action="send_email",
        status=send_result.get("status", "unknown"),
        detail=f"Grand total: {totals.get('grand_total', 0):.2f}, flagged items: {len(state.get('exceptions', []))}",
        recipient=recipient,
    )
    print(f"   [AUDIT] Logged email send: {recipient} ({send_result.get('status', 'unknown')})")

    msg_log = append_acl_message(
        {**state, "message_log": msg_log},
        performative=ACLPerformative.INFORM,
        sender="Reporter",
        receiver="Coordinator",
        content=f"Report sent ({send_result['status']}) and archived to {eml_path.name}.",
    )

    step_log = list(state.get("step_log", []))
    step_log.append(f"Reporter: Email sent ({send_result['status']}) and archived to {eml_path.name}")

    new_state = {
        **state,
        "totals": totals,
        "pdf_path": pdf_path,
        "email_body": email_body,
        "email_result": send_result,
        "eml_path": str(eml_path),
        "workflow_status": "completed",
        "message_log": msg_log,
        "step_log": step_log[-5:],
    }
    print_blackboard_snapshot("Reporter", new_state)
    return new_state
