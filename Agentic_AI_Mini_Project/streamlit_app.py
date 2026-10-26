"""
streamlit_app.py
────────────────
Streamlit Web UI for Expense Report Agent.
Course Concept: Interactive Multi-Agent Observability & Human-in-the-Loop Gate.

Group 4: Aryan Maradwar · Asta Joseph · Dhiraj Sarawale · Ridhima Maliya

Usage:
  streamlit run streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Expense Report Agent | Group 4",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Expense Report Agent — AIA_7 Group 4")
st.caption("Autonomous Agentic Pipeline · LangGraph · ChromaDB · BAAI/bge-small-en-v1.5 · Local Ollama")

# Sidebar
st.sidebar.header("Agent Configuration")
run_mode = st.sidebar.selectbox("Pipeline Mode", ["extended", "base"], index=0)
auto_approve = st.sidebar.checkbox("Auto-Approve Human Gate", value=True)
answers_choice = st.sidebar.selectbox("Scripted Answers", ["data/answers.json", "None"], index=0)
answers_path = None if answers_choice == "None" else answers_choice

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Long-Term Memory Correction")
corr_vendor = st.sidebar.text_input("Merchant Vendor", value="Olive Garden")
corr_cat = st.sidebar.selectbox("Correct Category", ["Meals", "Hotels", "Flights", "Transport", "Misc"], index=0)
if st.sidebar.button("Save Correction to ChromaDB"):
    from unit3_memory.memory_store import ExpenseMemoryStore
    ms = ExpenseMemoryStore()
    ms.add_user_correction(vendor=corr_vendor, category=corr_cat, description=f"Human correction via UI")
    st.sidebar.success(f"✔ Saved: '{corr_vendor}' -> '{corr_cat}' (source=user_correction)")

# Run Agent Pipeline Button
st.markdown("### Run Agent Pipeline")
instruction = st.text_input(
    "User Instruction",
    value="Generate my travel expense report for last month and email it to finance.",
)

if st.button("🚀 Execute Agent Pipeline", type="primary"):
    with st.spinner("Agent pipeline running..."):
        from llm_config import get_llm
        llm, llm_label = get_llm()
        csv_path = str(Path(__file__).parent / "data" / "mock_expenses.csv")

        if run_mode == "extended":
            from unit3_memory.extended_main import build_extended_graph
            app = build_extended_graph()
            initial_state = {
                "instruction": instruction,
                "csv_path": csv_path,
                "mode": "extended",
                "llm": llm,
                "llm_label": llm_label,
                "auto_approve": auto_approve,
                "answers_file": answers_path,
                "step_log": [],
                "message_log": [],
                "retry_count": 0,
            }
        else:
            from main import build_graph
            app = build_graph()
            initial_state = {
                "instruction": instruction,
                "csv_path": csv_path,
                "mode": "base",
                "llm": llm,
                "llm_label": llm_label,
                "step_log": [],
            }

        final_state = app.invoke(initial_state)
        st.session_state["final_state"] = final_state

# Display Results if Available
if "final_state" in st.session_state:
    state = st.session_state["final_state"]
    totals = state.get("totals", {})
    grand_total = totals.get("grand_total", 0.0)
    cat_totals = totals.get("category_totals", {})
    exceptions = state.get("exceptions", [])
    msg_log = state.get("message_log", [])
    pdf_path = state.get("pdf_path")
    email_body = state.get("email_body", "")

    # Currency setup
    import os
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grand Total", f"{currency} {grand_total:,.2f}")
    c2.metric("Period", f"{state.get('date_range', ('?','?'))[0]} to {state.get('date_range', ('?','?'))[1]}")
    c3.metric("Exceptions / Flagged", len(exceptions))
    c4.metric("ACL Messages", len(msg_log))

    # Tabs for detailed views
    tab_summary, tab_blackboard, tab_messages, tab_pdf, tab_email, tab_audit = st.tabs([
        "📊 Financial Breakdown",
        "📋 Shared Blackboard",
        "💬 Multi-Agent ACL Log",
        "📄 Generated PDF Report",
        "✉️ Drafted Email",
        "🔐 Audit Log",
    ])

    with tab_summary:
        st.subheader("Category Totals (Deterministic)")
        cols = st.columns(len(cat_totals) if cat_totals else 1)
        for col, (cat, amt) in zip(cols, sorted(cat_totals.items())):
            col.metric(cat, f"{currency} {amt:,.2f}")

        if exceptions:
            st.warning(f"⚠️ {len(exceptions)} item(s) flagged for manual audit:")
            st.table(exceptions)

    with tab_blackboard:
        st.subheader("Shared Blackboard State Snapshot")
        st.json({
            "workflow_status": state.get("workflow_status", "completed"),
            "raw_expenses_count": len(state.get("raw_expenses", [])),
            "filtered_expenses_count": len(state.get("filtered_expenses", [])),
            "categorized_expenses_count": len(state.get("categorized_expenses", [])),
            "grand_total": grand_total,
            "step_log": state.get("step_log", []),
        })

    with tab_messages:
        st.subheader("FIPA-ACL Multi-Agent Communication Transcript")
        if msg_log:
            for i, m in enumerate(msg_log, 1):
                st.markdown(
                    f"**{i:02d}. `[{m.get('performative','').upper()}]`** "
                    f"**{m.get('sender','')}** ➡️ **{m.get('receiver','')}**: "
                    f"{m.get('content','')}"
                )
        else:
            st.info("No ACL messages recorded (Base pipeline runs sequentially without multi-agent messaging).")

    with tab_pdf:
        st.subheader("ReportLab PDF Artifact")
        if pdf_path and Path(pdf_path).exists():
            st.success(f"✔ PDF generated: `{pdf_path}`")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download expense_report.pdf",
                    data=f.read(),
                    file_name="expense_report.pdf",
                    mime="application/pdf",
                )
        else:
            st.info("PDF has not been generated yet.")

    with tab_email:
        st.subheader("Email Content (Prepared for Finance)")
        st.text_area("Email Body", value=email_body, height=250)
        eml_path = state.get("eml_path")
        if eml_path and Path(eml_path).exists():
            st.success(f"✔ RFC 822 EML File archived: `{eml_path}`")

    with tab_audit:
        st.subheader("SQLite Audit Log (Security & Compliance)")
        st.write("Last 20 actions: approvals, rejections, denied operations, email sends.")
        try:
            from audit_log import get_audit_log
            audit = get_audit_log()
            records = audit.get_recent(limit=20)
            if records:
                st.dataframe(records, use_container_width=True)
            else:
                st.info("No audit log records yet. Run the pipeline to generate records.")
        except Exception as e:
            st.error(f"Could not load audit log: {e}")
