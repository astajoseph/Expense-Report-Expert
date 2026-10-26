"""
tools.py
────────
Course concept: TOOL USE in agentic pipelines.

Each function here is a "tool" — a discrete, testable unit of work that a
LangGraph node can call.  This mirrors the ReAct pattern: the agent reasons
about WHICH tool to call and passes its output to the next step.

Deterministic tools (no LLM):
    retrieve_expenses   — load CSV from disk
    filter_last_month   — keep only last calendar month (uses REPORT_TODAY)
    calculate_totals    — sum amounts per category + grand total

LLM-powered tools:
    categorize_expenses — infer category from vendor/description
                          (falls back to keyword rules if LLM unavailable)
    draft_email         — compose professional email body

Side-effect tools:
    generate_pdf_report — render a ReportLab PDF
    send_email_with_pdf — SMTP dispatch with attachment
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import smtplib
import sys
from calendar import monthrange
from datetime import date, datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Windows UTF-8 fix ──────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger(__name__)

# ── Valid expense categories ───────────────────────────────────────────────
VALID_CATEGORIES = {"Flights", "Hotels", "Meals", "Transport", "Misc"}

# ── Keyword rules for offline / fallback categorisation ───────────────────
_KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["flight", "airline", "airways", "air ", "delta", "united", "southwest", "jetblue", "american airlines"], "Flights"),
    (["restaurant", "grille", "grill", "cafe", "diner", "bistro", "kitchen", "pizza", "burger", "mcdonald",
      "subway", "starbucks", "coffee", "meal", "lunch", "dinner", "breakfast", "food", "doordash", "uber eats",
      "olive garden", "capital grille", "cheesecake", "mcdonald's"], "Meals"),
    (["uber", "lyft", "taxi", "cab", "bart", "metro", "bus", "parking",
      "toll", "car rental", "enterprise", "hertz", "avis", "shuttle", "train", "amtrak"], "Transport"),
    (["hotel", "inn", "hilton", "marriott", "hyatt", "sheraton", "holiday", "lodge", "motel", "resort", "suites"], "Hotels"),
]

def _keyword_categorize(vendor: str, description: str) -> str:
    """Rule-based fallback categorisation using keyword matching."""
    text = f"{vendor} {description}".lower()
    for keywords, category in _KEYWORD_RULES:
        if any(kw in text for kw in keywords):
            return category
    return "Misc"


# ════════════════════════════════════════════════════════════════════════════
# Helper: get "today" (respects REPORT_TODAY env override)
# ════════════════════════════════════════════════════════════════════════════

def get_report_today() -> date:
    """
    Return the reference date used as 'today' throughout the pipeline.
    Reads REPORT_TODAY from .env (format YYYY-MM-DD).
    Falls back to real date.today() if not set or invalid.
    """
    raw = os.getenv("REPORT_TODAY", "").strip()
    if raw:
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("⚠ REPORT_TODAY='%s' is not YYYY-MM-DD — using real today.", raw)
    return date.today()


def last_month_range(reference: date | None = None) -> tuple[date, date]:
    """Return (first_day, last_day) of the calendar month before `reference`."""
    ref = reference or get_report_today()
    first_of_this = ref.replace(day=1)
    last_day_prev = first_of_this - __import__("datetime").timedelta(days=1)
    first_day_prev = last_day_prev.replace(day=1)
    return first_day_prev, last_day_prev


# ════════════════════════════════════════════════════════════════════════════
# TOOL 1 — Retrieve expenses from CSV
# ════════════════════════════════════════════════════════════════════════════

def retrieve_expenses(csv_path: str) -> list[dict[str, Any]]:
    """Load expenses from the mock CSV file. Returns raw rows as dicts."""
    logger.info("   [RETRIEVE] Loading expenses from: %s", csv_path)
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clean = {k.strip(): (v or "").strip() for k, v in row.items()}
            rows.append(clean)

    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    logger.info("   ✔ Loaded %d rows. | TOOL: csv.read %d rows", len(rows), len(rows))
    print(f"   [TRACE] TOOL: csv.read {len(rows)} rows from {csv_path}")
    return rows


# ════════════════════════════════════════════════════════════════════════════
# TOOL 2 — Filter to last calendar month
# ════════════════════════════════════════════════════════════════════════════

def filter_last_month(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Keep only rows whose date falls in the previous calendar month.
    Uses REPORT_TODAY env var as the reference for 'today'.

    Returns:
        (valid_rows, exception_rows)
        exception_rows contain rows with missing/unparseable dates.
    """
    first_day, last_day = last_month_range()

    logger.info(
        "   [FILTER] Keeping expenses from %s to %s.",
        first_day, last_day,
    )

    valid: list[dict] = []
    exceptions: list[dict] = []

    for row in rows:
        raw_date = row.get("date", "").strip()
        if not raw_date:
            row["_flag"] = "missing date"
            exceptions.append(row)
            logger.warning("   ⚠ Row flagged — missing date: %s", row.get("vendor"))
            continue

        try:
            expense_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
        except ValueError:
            row["_flag"] = f"unparseable date: '{raw_date}'"
            exceptions.append(row)
            logger.warning("   ⚠ Row flagged — bad date format: %s", row.get("vendor"))
            continue

        if first_day <= expense_date <= last_day:
            valid.append(row)

    logger.info(
        "   ✔ %d rows in range | %d flagged/out-of-range.",
        len(valid), len(exceptions),
    )
    return valid, exceptions


# ════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Categorise with LLM (strict JSON) or keyword fallback
# ════════════════════════════════════════════════════════════════════════════

def _parse_category_from_llm(raw: str) -> str | None:
    """
    Extract a valid category from the LLM's raw output.
    Expects JSON: {"category": "Flights"}
    Falls back to scanning for any valid category word in the text.
    """
    # Try JSON parse first
    try:
        obj = json.loads(raw)
        cat = str(obj.get("category", "")).strip()
        if cat in VALID_CATEGORIES:
            return cat
    except (json.JSONDecodeError, AttributeError):
        pass

    # Scan for category word anywhere in output (handle extra reasoning text)
    for cat in VALID_CATEGORIES:
        if re.search(rf"\b{cat}\b", raw, re.IGNORECASE):
            return cat

    return None


def categorize_expenses(
    rows: list[dict],
    llm,
    llm_label: str,
    few_shot_examples: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    For rows with a missing or empty category, infer the category.

    Strategy (in order):
      1. LLM with strict JSON prompt + optional few-shot examples
      2. Keyword rule-based fallback (if LLM unavailable or returns garbage)

    Returns:
        (categorized_rows, new_exceptions)
        new_exceptions: rows with a missing amount flagged here.
    """
    logger.info("   [CATEGORIZE] Using LLM provider: %s", llm_label)

    categorized: list[dict] = []
    new_exceptions: list[dict] = []

    for row in rows:
        # ── Flag missing amount but keep processing ────────────────────────
        if not row.get("amount", "").strip():
            row["_flag"] = "missing amount"
            new_exceptions.append({**row, "_source": "categorize_step"})
            logger.warning(
                "   ⚠ Row flagged — missing amount: vendor=%s",
                row.get("vendor"),
            )

        # ── Flag Receipt = No ──────────────────────────────────────────────
        receipt = row.get("Receipt", "").strip().lower()
        if receipt == "no":
            if "_flag" not in row:
                row["_flag"] = "no receipt"
            new_exceptions.append({**row, "_source": "categorize_step"})
            logger.warning(
                "   ⚠ Row flagged — no receipt: vendor=%s",
                row.get("vendor"),
            )

        existing_cat = row.get("category", "").strip()
        if existing_cat in VALID_CATEGORIES:
            categorized.append(row)
            continue

        vendor      = row.get("vendor", "Unknown")
        description = row.get("description", "")

        inferred = None

        # ── Try LLM ───────────────────────────────────────────────────────
        if llm is not None:
            # Build optional few-shot block
            few_shot_block = ""
            if few_shot_examples:
                examples_text = "\n".join(
                    f'  Vendor: {e["merchant"]}, Description: {e["description"]} -> {{"category": "{e["category"]}"}}'
                    for e in few_shot_examples[:3]
                )
                few_shot_block = f"\nExamples from memory:\n{examples_text}\n"

            prompt = (
                f"You are an expense categorisation assistant.\n"
                f"Classify the following business travel expense into EXACTLY ONE of these categories:\n"
                f"  Flights, Hotels, Meals, Transport, Misc\n\n"
                f"{few_shot_block}"
                f"Expense:\n"
                f"  Vendor: {vendor}\n"
                f"  Description: {description}\n\n"
                f"IMPORTANT: Reply with ONLY valid JSON in this exact format, nothing else:\n"
                f'  {{"category": "<one of: Flights, Hotels, Meals, Transport, Misc>"}}\n'
            )

            try:
                response = llm.invoke(prompt)
                raw_text = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                ).strip()

                inferred = _parse_category_from_llm(raw_text)
                if inferred:
                    logger.info("   ✔ '%s' → category: %s (LLM)", vendor, inferred)
                else:
                    logger.warning(
                        "   ⚠ LLM returned unrecognised output for '%s': %s — using keyword fallback",
                        vendor, raw_text[:80],
                    )
            except Exception as exc:
                logger.warning("   ⚠ LLM call failed for '%s': %s", vendor, exc)

        # ── Keyword fallback ───────────────────────────────────────────────
        if inferred is None:
            inferred = _keyword_categorize(vendor, description)
            logger.info("   ✔ '%s' → category: %s (keyword rule)", vendor, inferred)

        row = {**row, "category": inferred}
        categorized.append(row)

    logger.info(
        "   ✔ Categorisation complete. %d rows processed, %d flags raised.",
        len(categorized), len(new_exceptions),
    )
    print(f"   [TRACE] TOOL: ollama.generate {len([r for r in categorized if r.get('category')])} categories inferred")
    return categorized, new_exceptions


# ════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Calculate totals (deterministic, NO LLM)
# ════════════════════════════════════════════════════════════════════════════

def calculate_totals(rows: list[dict], exceptions: list[dict] | None = None) -> dict[str, Any]:
    """
    Compute total per category and grand total.
    Rows with missing/invalid amounts are skipped (already flagged upstream).
    """
    logger.info("   [CALCULATE] Computing category totals …")

    category_totals: dict[str, float] = {}
    skipped = 0

    for row in rows:
        raw_amount = row.get("amount", "").strip()
        if not raw_amount:
            skipped += 1
            continue
        try:
            amount = float(raw_amount)
        except ValueError:
            logger.warning(
                "   Cannot parse amount '%s' for vendor '%s' — skipping.",
                raw_amount, row.get("vendor"),
            )
            skipped += 1
            continue

        cat = row.get("category", "Misc")
        category_totals[cat] = category_totals.get(cat, 0.0) + amount

    grand_total = sum(category_totals.values())
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")

    logger.info(
        "   Grand total: %s%.2f across %d categories.",
        currency, grand_total, len(category_totals),
    )
    if skipped:
        logger.info("   ℹ %d rows skipped (missing/invalid amounts).", skipped)

    num_flagged = len(exceptions) if exceptions else 0
    print(f"   [TRACE] TOOL: calculate_total grand_total={currency}{grand_total:,.2f} flagged={num_flagged}")

    return {
        "category_totals": category_totals,
        "grand_total":     grand_total,
        "flagged_count":   num_flagged,
    }


# ════════════════════════════════════════════════════════════════════════════
# TOOL 5 — Generate PDF report with ReportLab
# ════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(
    rows:        list[dict],
    totals:      dict[str, Any],
    exceptions:  list[dict],
    date_range:  tuple[str, str],
    output_path: str,
) -> str:
    """
    Render a professional PDF expense report.

    Sections:
      • Header / metadata
      • Summary table (category → total)
      • Expense detail table
      • Exceptions appendix
    """
    logger.info("   [PDF] Generating report → %s", output_path)

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph,
            Spacer, HRFlowable, PageBreak,
        )
    except ImportError as exc:
        raise ImportError("reportlab is required. Run: pip install reportlab") from exc

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Colour palette ─────────────────────────────────────────────────────
    BRAND_DARK  = colors.HexColor("#1a2744")
    BRAND_MID   = colors.HexColor("#2e4a8e")
    BRAND_LIGHT = colors.HexColor("#dde8ff")
    ACCENT_RED  = colors.HexColor("#c0392b")
    ROW_ALT     = colors.HexColor("#f0f4ff")
    WHITE       = colors.white
    GREY_TEXT   = colors.HexColor("#555555")

    # ── Custom paragraph styles ────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=22,
        textColor=BRAND_DARK,
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=GREY_TEXT,
        alignment=TA_CENTER,
        spaceAfter=18,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=BRAND_MID,
        spaceBefore=16,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    normal_style = styles["Normal"]

    # ── Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph("Travel Expense Report", title_style))
    story.append(Paragraph(
        f"Period: {date_range[0]} to {date_range[1]}"
        f" &nbsp;|&nbsp; Generated: {date.today().strftime('%B %d, %Y')}",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_MID, spaceAfter=12))

    # ── Summary table ──────────────────────────────────────────────────────
    story.append(Paragraph("Summary by Category", section_style))

    summary_data = [["Category", "Total Amount"]]
    cat_totals   = totals.get("category_totals", {})
    for cat in sorted(cat_totals):
        currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
        summary_data.append([cat, f"{currency} {cat_totals[cat]:,.2f}"])

    # Grand total label includes flagged items count
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    flagged_count = totals.get("flagged_count", 0)
    grand_total_label = f"GRAND TOTAL" + (f" (excludes {flagged_count} flagged items)" if flagged_count else "")
    summary_data.append([grand_total_label, f"{currency} {totals.get('grand_total', 0):,.2f}"])

    summary_table = Table(summary_data, colWidths=[11 * cm, 5 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 11),
        ("ALIGN",         (1, 0), (1, -1), "RIGHT"),
        ("ALIGN",         (0, 0), (0, -1), "LEFT"),
        *[
            ("BACKGROUND", (0, i), (-1, i), ROW_ALT if i % 2 == 0 else WHITE)
            for i in range(1, len(summary_data) - 1)
        ],
        ("BACKGROUND",    (0, -1), (-1, -1), BRAND_MID),
        ("TEXTCOLOR",     (0, -1), (-1, -1), WHITE),
        ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, -1), (-1, -1), 11),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#c0c8e0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # ── Detail table ───────────────────────────────────────────────────────
    story.append(Paragraph("Expense Detail", section_style))

    detail_headers = ["Date", "Vendor", "Description", "Category", "Amount", "Receipt"]
    detail_data    = [detail_headers]
    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    for row in rows:
        amount     = row.get("amount", "").strip()
        amount_str = f"{currency} {float(amount):,.2f}" if amount else "MISSING"
        # Use Yes/No from Receipt column (case-insensitive)
        receipt_val = row.get("Receipt", "").strip().lower()
        receipt = "Yes" if receipt_val == "yes" else "No"
        detail_data.append([
            row.get("date", ""),
            row.get("vendor", ""),
            (row.get("description", "")[:40] + "...") if len(row.get("description", "")) > 40 else row.get("description", ""),
            row.get("category", ""),
            amount_str,
            receipt,
        ])

    col_widths = [2.2*cm, 3.5*cm, 5.5*cm, 3.0*cm, 2.0*cm, 1.5*cm]
    detail_table = Table(detail_data, colWidths=col_widths, repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ALIGN",         (4, 1), (4, -1), "RIGHT"),
        ("ALIGN",         (5, 0), (5, -1), "CENTER"),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#c0c8e0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        *[
            ("BACKGROUND", (0, i), (-1, i), ROW_ALT if i % 2 == 0 else WHITE)
            for i in range(1, len(detail_data))
        ],
    ]))
    story.append(detail_table)

    # ── Exceptions appendix ────────────────────────────────────────────────
    if exceptions:
        story.append(PageBreak())
        story.append(Paragraph("Appendix: Flagged Items / Exceptions", section_style))
        story.append(Paragraph(
            "The following expense rows could not be fully processed and require manual review.",
            normal_style,
        ))
        story.append(Spacer(1, 8))

        exc_headers = ["Date", "Vendor", "Description", "Flag / Reason"]
        exc_data    = [exc_headers]
        for exc_row in exceptions:
            exc_data.append([
                exc_row.get("date", "N/A"),
                exc_row.get("vendor", "N/A"),
                (exc_row.get("description", "")[:45] + "...")
                if len(exc_row.get("description", "")) > 45 else exc_row.get("description", ""),
                exc_row.get("_flag", "Unknown issue"),
            ])

        exc_col_widths = [2.2*cm, 3.5*cm, 6.0*cm, 5.5*cm]
        exc_table = Table(exc_data, colWidths=exc_col_widths, repeatRows=1)
        exc_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), ACCENT_RED),
            ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#f0c0c0")),
            ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#fff5f5")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ]))
        story.append(exc_table)

    # ── Footer note ────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_LIGHT))
    story.append(Paragraph(
        "This report was generated automatically by the Expense Report Agent. "
        "Please contact finance@yourcompany.com for any discrepancies.",
        ParagraphStyle("Footer", parent=normal_style, fontSize=8, textColor=GREY_TEXT),
    ))

    doc.build(story)
    logger.info("   ✔ PDF written to: %s", output_path)
    print(f"   [TRACE] TOOL: pdf.generate {output_path} ({len(rows)} rows, {len(exceptions)} flagged)")
    return output_path


# ════════════════════════════════════════════════════════════════════════════
# TOOL 6 — Draft email (LLM or template fallback)
# ════════════════════════════════════════════════════════════════════════════

def draft_email(
    totals:         dict[str, Any],
    date_range:     tuple[str, str],
    num_exceptions: int,
    llm,
    llm_label:      str,
    instruction:    str = "Generate my travel expense report for last month and email it to finance.",
    short_term_ctx: list[str] | None = None,
) -> str:
    """
    Use the LLM to compose a short professional email summarising the report.
    Falls back to a template if the LLM is unavailable.

    short_term_ctx: optional list of recent step summaries (Phase 1 sliding window).
    """
    logger.info("   [DRAFT_EMAIL] Drafting email with: %s", llm_label)

    currency = os.getenv("CURRENCY_SYMBOL", "Rs.")
    grand_total = totals.get("grand_total", 0.0)
    cat_totals  = totals.get("category_totals", {})
    breakdown   = "\n".join(
        f"  - {cat}: {currency} {amt:,.2f}" for cat, amt in sorted(cat_totals.items())
    )

    fallback_body = (
        f"Subject: Travel Expense Report - {date_range[0]} to {date_range[1]}\n\n"
        f"Dear Finance Team,\n\n"
        f"Please find attached the travel expense report for the period "
        f"{date_range[0]} to {date_range[1]}.\n\n"
        f"The grand total for the period is {currency} {grand_total:,.2f}. "
        f"There are {num_exceptions} item(s) flagged for manual review due to "
        f"missing data (amount or date). Please refer to the Exceptions appendix "
        f"in the attached PDF for details.\n\n"
        f"Category breakdown:\n{breakdown}\n\n"
        f"Kindly acknowledge receipt and process at your earliest convenience.\n\n"
        f"Best regards,\n"
        f"Expense Report Agent (Automated)"
    )

    if llm is None:
        logger.warning("   ⚠ LLM unavailable — using fallback email template.")
        return fallback_body

    ctx_block = ""
    if short_term_ctx:
        ctx_block = "\nRecent pipeline steps:\n" + "\n".join(f"  - {s}" for s in short_term_ctx[-5:]) + "\n"

    prompt = (
        f"You are a professional corporate assistant. Write a concise email to the finance team.\n\n"
        f"User Instruction:\n"
        f"  \"{instruction}\"\n\n"
        f"Context:\n"
        f"  - Period covered: {date_range[0]} to {date_range[1]}\n"
        f"  - Grand total of expenses: {currency} {grand_total:,.2f}\n"
        f"  - Category breakdown:\n{breakdown}\n"
        f"  - Number of flagged/exception items requiring review: {num_exceptions}\n"
        f"  - A PDF report is attached.\n"
        f"{ctx_block}\n"
        f"Requirements for the email:\n"
        f"  - Subject line (write 'Subject: ...' on the first line)\n"
        f"  - Professional greeting\n"
        f"  - 2-3 sentence summary mentioning the total, period, and flagged items\n"
        f"  - A note that the PDF is attached for full details\n"
        f"  - Professional sign-off from 'Expense Report Agent (Automated)'\n"
        f"  - Keep it under 150 words total.\n"
        f"\nIMPORTANT: Return plain text only, not JSON."
    )

    # Temporarily use a non-JSON llm for email (plain text output needed)
    try:
        from langchain_ollama import ChatOllama  # type: ignore
        plain_llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.3,
        )
    except ImportError:
        plain_llm = llm

    import time
    for attempt in range(2):
        try:
            response = plain_llm.invoke(prompt)
            body = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()
            logger.info("   ✔ Email body drafted successfully.")
            print(f"   [TRACE] TOOL: ollama.draft_email {len(body)} chars")
            return body
        except Exception as exc:
            if attempt == 0:
                logger.info("   Email draft retry after short wait…")
                time.sleep(3)
                continue
            logger.warning("   ⚠ LLM email draft failed: %s — using fallback.", str(exc)[:120])
            return fallback_body

    return fallback_body


# ════════════════════════════════════════════════════════════════════════════
# TOOL 7 — Send email via SMTP
# ════════════════════════════════════════════════════════════════════════════

def send_email_with_pdf(
    email_body: str,
    pdf_path:   str,
    recipient:  str,
) -> dict[str, Any]:
    """
    Send the drafted email with the PDF attached via SMTP.
    Reads credentials from environment variables.
    Returns a result dict with status and details.
    """
    smtp_host = os.getenv("SMTP_HOST", "sandbox.smtp.mailtrap.io")
    smtp_port = int(os.getenv("SMTP_PORT", "2525"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    from_addr = os.getenv("SMTP_FROM", "agent@expense-report.dev")

    logger.info(
        "   [SEND_EMAIL] Sending to %s via %s:%s …",
        recipient, smtp_host, smtp_port,
    )

    # Parse subject from body (first line if it starts with "Subject:")
    lines = email_body.strip().split("\n")
    if lines and lines[0].lower().startswith("subject:"):
        subject   = lines[0][len("subject:"):].strip()
        body_text = "\n".join(lines[1:]).strip()
    else:
        subject   = "Travel Expense Report"
        body_text = email_body

    # Build MIME message
    msg             = MIMEMultipart()
    msg["From"]     = from_addr
    msg["To"]       = recipient
    msg["Subject"]  = subject
    msg.attach(MIMEText(body_text, "plain"))

    # Attach PDF
    pdf_filename = Path(pdf_path).name
    with open(pdf_path, "rb") as pdf_file:
        pdf_part = MIMEBase("application", "octet-stream")
        pdf_part.set_payload(pdf_file.read())
    encoders.encode_base64(pdf_part)
    pdf_part.add_header(
        "Content-Disposition",
        f'attachment; filename="{pdf_filename}"',
    )
    msg.attach(pdf_part)

    result: dict[str, Any] = {
        "status":       "unknown",
        "recipient":    recipient,
        "subject":      subject,
        "smtp_host":    smtp_host,
        "smtp_port":    smtp_port,
        "pdf_attached": pdf_filename,
    }

    # Always save .eml locally (backup/archive)
    outbox_dir = Path(__file__).parent / "outbox"
    outbox_dir.mkdir(exist_ok=True)
    from datetime import datetime as dt
    eml_filename = f"report_{dt.now().strftime('%Y%m%d_%H%M%S')}.eml"
    eml_path = outbox_dir / eml_filename
    with open(eml_path, "w", encoding="utf-8") as f:
        f.write(msg.as_string())
    result["eml_saved"] = str(eml_path)
    logger.info("   ✔ Email archived locally: %s", eml_path)

    is_placeholder = (
        not smtp_user
        or not smtp_pass
        or "your_" in smtp_user.lower()
        or "your_" in smtp_pass.lower()
    )
    if is_placeholder:
        logger.warning(
            "   ℹ SMTP credentials not configured — simulating email dispatch (demo mode)."
        )
        result["status"] = "simulated (demo mode: credentials not configured)"
        result["note"]   = "Set real SMTP_USER and SMTP_PASS in .env to deliver real emails."
        print(f"   [TRACE] TOOL: smtp.send {recipient} (simulated, saved to {eml_filename})")
        return result

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, recipient, msg.as_string())
        result["status"] = "sent"
        logger.info("   ✔ Email sent successfully.")
        print(f"   [TRACE] TOOL: smtp.send {recipient} sent")
    except smtplib.SMTPAuthenticationError:
        logger.error("   SMTP authentication failed. Check SMTP_USER / SMTP_PASS.")
        result["status"] = "failed — SMTP auth error"
        print(f"   [TRACE] TOOL: smtp.send {recipient} FAILED (auth error)")
    except Exception as exc:
        logger.error("   SMTP error: %s", exc)
        result["status"] = f"failed — {exc}"
        print(f"   [TRACE] TOOL: smtp.send {recipient} FAILED ({str(exc)[:50]})")

    return result
