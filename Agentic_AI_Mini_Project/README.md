# Expense Report Agent
**AIA_7 · Group 4** — Aryan Maradwar · Asta Joseph · **Dhiraj Sarawale** · Ridhima Maliya

> _"Generate my travel expense report for last month and email it to finance."_

A fully offline, open-source AI agent built with **LangGraph + local Ollama** that
retrieves expenses, categorises them using an LLM, generates a PDF report, and emails
it — all without any paid API keys.

---

## Quick Start

### Prerequisites
```bash
# 1. Install Ollama (https://ollama.com)
ollama pull qwen2.5:7b        # ~4 GB download

# 2. Clone and setup
git clone <repo>
cd "Agentic AI Mini Project"

pip install -r requirements.txt

# 3. Copy and configure environment
copy .env.example .env        # Windows
# Leave SMTP_USER/PASS empty for demo mode (simulated send)
```

### Run Base Pipeline
```bash
$env:PYTHONIOENCODING="utf-8"   # PowerShell — supports emoji in output
python main.py                  # or: python main.py --mode base
```

### Run Extended Pipeline (Phase 1 — long-term memory + feedback loop)
```bash
python main.py --mode extended
```

### Save a User Category Correction
```bash
python main.py --correct "Olive Garden=Meals"
# Next run will use this from long-term memory, skipping the LLM
```

### Regenerate CSV for any date
```bash
python data/seed.py --date 2026-10-01   # regenerates Sept 2026 data
```

---

## Project Structure

```
.
├── main.py                  # LangGraph 9-node pipeline (base + extended entry)
├── llm_config.py            # Local Ollama config (open-source only, no API keys)
├── tools.py                 # All tool functions (retrieve, filter, categorize, …)
├── requirements.txt
├── .env.example
├── data/
│   ├── mock_expenses.csv    # 20-row demo expense data (September 2026)
│   └── seed.py              # Regenerate CSV with any reference date
├── unit3_memory/            # Phase 1: Long-term memory + feedback loop
│   ├── memory_store.py      # ChromaDB + sentence-transformers embeddings
│   └── extended_main.py     # Extended LangGraph pipeline
├── unit4_agents/            # Phase 2: Role-based agents + blackboard
├── tests/                   # Phase 3: pytest tests
├── run_demo.py              # Phase 3: 3 scripted demo scenarios
└── ARCHITECTURE.md          # Mermaid diagram + design rationale
```

---

## Modes

| Command | What it does |
|---|---|
| `python main.py` | Base pipeline (Ollama LLM for categorise + email) |
| `python main.py --mode base` | Same as above |
| `python main.py --mode extended` | Phase 1: memory + feedback loop |
| `python main.py --correct "Vendor=Cat"` | Save user correction to long-term memory |
| `python main.py --answers answers.json` | Scripted missing-data answers (Phase 1) |
| `python main.py --mode extended --auto-approve` | Skip human approval gate (Phase 2) |

---

## Fallback Mode (Offline Safety)

If Ollama is not running, the agent automatically falls back to:
- **Categorisation**: keyword rules (no LLM needed)
- **Email**: professional template (no LLM needed)
- A clear `⚠ FALLBACK MODE` message is printed

The pipeline **never crashes** due to a missing LLM.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Local Ollama model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `REPORT_TODAY` | (real date) | Override "today" for reproducible demos |
| `FINANCE_EMAIL` | `finance@yourcompany.com` | Email recipient |
| `SMTP_HOST/PORT/USER/PASS` | Mailtrap sandbox | SMTP credentials |
| `CHROMA_PERSIST_DIR` | `chroma_db` | ChromaDB storage folder |
| `MEMORY_TOP_K` | `3` | How many memory matches to retrieve |
| `MEMORY_SIMILARITY_THRESHOLD` | `0.85` | Above this score → skip LLM |
| `SHORT_TERM_WINDOW` | `5` | Steps to keep in sliding window |

---

## Course Concepts Demonstrated

| Phase | Concept | Where |
|---|---|---|
| Base | **Tool use** (ReAct pattern) | `tools.py` |
| Base | **Agentic pipeline** (LangGraph DAG) | `main.py` |
| Base | **Local/open-source LLM** with fallback | `llm_config.py` |
| Phase 1 | **Long-term memory** (ChromaDB + embeddings) | `unit3_memory/memory_store.py` |
| Phase 1 | **Feedback loop** (LangGraph conditional edges) | `unit3_memory/extended_main.py` |
| Phase 2 | **Role-based agents** (Coordinator, Retriever, Auditor, Reporter) | `unit4_agents/` |
| Phase 2 | **Blackboard architecture** (shared LangGraph state) | `unit4_agents/` |
| Phase 2 | **FIPA-ACL messaging** between agents | `unit4_agents/` |
| Phase 2 | **Human approval gate** (adjustable autonomy) | `unit4_agents/` |

---

## Expected Output (Base Run)

```
============================================================
   EXPENSE REPORT AGENT — Starting Up
   Mode: BASE
============================================================

 Instruction: "Generate my travel expense report for last month..."

  Initialising LLM …
   Provider resolved: Local Ollama (qwen2.5:7b)

============================================================
 NODE 1 | RETRIEVE — Loading expenses from CSV
   => 20 expense rows loaded from data/mock_expenses.csv
   [TRACE] TOOL: csv.read 20 rows from data/mock_expenses.csv

 NODE 2 | FILTER — Keeping only last calendar month
   => Date range: 2026-09-01 to 2026-09-30

 NODE 3 | CATEGORIZE — Inferring missing categories
   => LLM provider: Local Ollama (qwen2.5:7b)
   [TRACE] TOOL: ollama.generate 18 categories inferred

 NODE 5 | CALCULATE — Computing totals
   => Grand total: Rs. 3,653.25
      Flights: Rs. 2,075.00
      Hotels:  Rs. 753.30
      Meals:   Rs. 436.55
      Transport: Rs. 388.40
   [TRACE] TOOL: calculate_total grand_total=Rs. 3,653.25 flagged=3

 NODE 6 | GENERATE PDF
   => PDF saved: expense_report.pdf
   [TRACE] TOOL: pdf.generate expense_report.pdf (20 rows, 3 flagged)

 NODE 8 | SEND EMAIL
   => Send status: simulated (demo mode)
   [TRACE] TOOL: smtp.send finance@yourcompany.com (simulated, saved to report_20261020_143025.eml)

  +-----------------------------------------------+
  |         EXPENSE REPORT AGENT -- DONE          |
  +-----------------------------------------------+
  | Grand Total : Rs. 3,653.25 (excludes 3 flagged items)
  | Flagged     : 3 items (missing amount, no receipt)
  | PDF Report  : expense_report.pdf
  | EML Archive : outbox/report_*.eml
  +-----------------------------------------------+
```

---

## Streamlit Observability Dashboard (Phase 3)

Launch the interactive web interface:
```bash
streamlit run streamlit_app.py
```
Provides:
- Live blackboard state inspection
- Real-time FIPA-ACL multi-agent communication log
- Downloadable PDF report and `.eml` message viewer
- Interactive human approval gate (`Approve` / `Reject`)
- Long-term memory category correction panel
- **SQLite Audit Log Panel** — view all approvals, rejections, denials, and email sends

---

## Security & Compliance Features

### Access Control (Role-Based)
- **Retriever**: read_csv only
- **Categorizer**: read + categorize
- **Auditor**: read + flag (cannot send)
- **Reporter**: all permissions (send_email only after human approval)

Every denied action is logged to the ACL trace and `audit.db`.

**Example:** Attempting to call `send_email` from Retriever role:
```python
from access_control import Role, Permission, enforce_permission
enforce_permission(Role.RETRIEVER, Permission.SEND_EMAIL)
# Raises: PermissionError: retriever denied send_email
```

### Audit Logging (SQLite)
All critical actions are logged to `audit.db` with:
- Timestamp
- Agent name
- Action
- Status (approved, rejected, denied, sent, failed)
- Details & recipient

View via Streamlit Dashboard → 🔐 Audit Log tab.

### Tool-Call Tracing
Every tool invocation prints a standardized trace:
```
[TRACE] TOOL: csv.read 20 rows from data/mock_expenses.csv
[TRACE] TOOL: chroma.query vendor='Olive Garden' -> Meals (memory hit, sim=0.92, skip LLM)
[TRACE] TOOL: ollama.generate 18 categories inferred
[TRACE] TOOL: pdf.generate expense_report.pdf (20 rows, 3 flagged)
[TRACE] TOOL: smtp.send finance@yourcompany.com sent
```

### Email Resilience
- Primary: SMTP send via Mailtrap sandbox or configured server
- Fallback 1: If SMTP fails, retry up to 2x
- Fallback 2: Save RFC 822 `.eml` file locally to `outbox/` for offline delivery
- Always archives a copy to `outbox/report_YYYYMMDD_HHMMSS.eml`

### Receipt Validation
Rows with `Receipt = No` are automatically flagged:
```
[TRACE] ⚠ Row flagged — no receipt: vendor=Starbucks
```
These appear in the PDF's Exceptions Appendix.

---

## Currency Configuration

All outputs (PDF, email, console) use configurable currency symbol:
```bash
CURRENCY_SYMBOL=Rs.
```

Set in `.env`. Applies to all numeric values:
- PDF tables
- Email body
- Console output
- Streamlit dashboard

---

## Sentence-Transformers Offline Caching

After the first run, embedding models are cached locally for fully offline operation:
```bash
export HF_HOME=./huggingface_cache
export SENTENCE_TRANSFORMERS_HOME=./huggingface_cache/sentence_transformers
```

The project creates these directories automatically on first run.

---

## Failure & Recovery Scenarios

Test graceful degradation with built-in demo scenarios:

```bash
# Scenario 4: Ollama unavailable → keyword-rule fallback
python run_demo.py --scenario 4
# Output: FAILED: Ollama unavailable → RECOVERED: Using keyword rules

# Scenario 5: SMTP failure → local .eml fallback
python run_demo.py --scenario 5
# Output: FAILED: SMTP auth error → RECOVERED: Saved to outbox/
```

---

## Automated Test Suite (Pytest)

Run the automated test suite covering Units 3 & 4:
```bash
python -m pytest -v tests/test_agent.py
```

Includes:
- Unit tests for memory store, feedback loop, and agent roles
- Integration tests for full pipeline
- Access control enforcement tests (Retriever denied send_email)
- SMTP retry and fallback logic

---

## License
MIT — open-source, educational use.
