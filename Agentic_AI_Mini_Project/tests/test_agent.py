"""
tests/test_agent.py
───────────────────
Pytest test suite for Expense Report Agent.
Tests:
1. Date filtering with REPORT_TODAY override
2. Deterministic category & grand totals calculation
3. Missing-amount detection and feedback loop resolution
4. Long-term memory storage, cosine similarity retrieval, and LLM skipping
5. User correction priority over model predictions
"""

import os
import sys
from datetime import date
from pathlib import Path

# Add project root to path for reliable imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tools import filter_last_month, calculate_totals, get_report_today, last_month_range
from unit3_memory.feedback_loop import FeedbackLoopManager
from unit3_memory.memory_store import ExpenseMemoryStore


def test_report_today_date_filtering(monkeypatch):
    """Test that REPORT_TODAY correctly sets reference date and last calendar month."""
    monkeypatch.setenv("REPORT_TODAY", "2026-10-15")
    today = get_report_today()
    assert today == date(2026, 10, 15)

    first, last = last_month_range()
    assert first == date(2026, 9, 1)
    assert last == date(2026, 9, 30)

    sample_rows = [
        {"date": "2026-09-10", "vendor": "Delta", "amount": "100.00"},
        {"date": "2026-08-31", "vendor": "Old Vendor", "amount": "50.00"},
        {"date": "2026-10-01", "vendor": "Future Vendor", "amount": "75.00"},
        {"date": "invalid-date", "vendor": "Bad Date", "amount": "20.00"},
    ]
    valid, exceptions = filter_last_month(sample_rows)
    assert len(valid) == 1
    assert valid[0]["vendor"] == "Delta"
    # Unparseable date is flagged as exception; out-of-range dates are filtered out
    assert len(exceptions) == 1
    assert "unparseable" in exceptions[0]["_flag"]


def test_calculate_totals():
    """Test deterministic calculation of category totals and grand total."""
    rows = [
        {"category": "Flights", "amount": "1250.00"},
        {"category": "Flights", "amount": "825.00"},
        {"category": "Meals", "amount": "45.50"},
        {"category": "Transport", "amount": "30.00"},
        {"category": "Hotels", "amount": ""},       # missing amount should be skipped
        {"category": "Hotels", "amount": "0.00"},     # zero should be skipped
        {"category": "Hotels", "amount": "300.00"},
    ]
    totals = calculate_totals(rows)
    cat_totals = totals["category_totals"]

    assert cat_totals["Flights"] == 2075.00
    assert cat_totals["Meals"] == 45.50
    assert cat_totals["Transport"] == 30.00
    assert cat_totals["Hotels"] == 300.00
    assert totals["grand_total"] == 2450.50


def test_feedback_loop_missing_amount_resolution(tmp_path):
    """Test that FeedbackLoopManager detects missing amounts and resolves them via answers dict."""
    import json
    answers_file = tmp_path / "test_answers.json"
    with open(answers_file, "w") as f:
        json.dump({"Lyft": 42.50}, f)

    manager = FeedbackLoopManager(answers_file=str(answers_file), max_retries=2)

    rows = [
        {"date": "2026-09-05", "vendor": "Lyft", "description": "Ride", "amount": ""},
        {"date": "2026-09-06", "vendor": "Unknown Parking", "description": "Parking", "amount": "0.00"},
    ]

    missing = manager.act_detect_missing(rows)
    assert len(missing) == 2

    obs = manager.observe(missing, iteration=0)
    should_retry, strategy = manager.reflect(obs)
    assert should_retry is True

    updated_rows, remaining_exceptions, logs = manager.adjust_resolve_items(
        rows=rows,
        missing_items=missing,
        auto_prompt=False,
    )

    # Lyft should be resolved, Unknown Parking should remain an exception
    assert len(remaining_exceptions) == 1
    assert remaining_exceptions[0]["vendor"] == "Unknown Parking"
    lyft_row = next(r for r in updated_rows if r["vendor"] == "Lyft")
    assert lyft_row["amount"] == "42.50"


def test_long_term_memory_reuse_and_user_correction(tmp_path):
    """Test ChromaDB long-term memory store: high similarity reuse and user correction."""
    db_path = str(tmp_path / "test_chroma")
    store = ExpenseMemoryStore(db_path=db_path, similarity_threshold=0.80)

    # Store a canonical memory
    store.store_memory(
        vendor="Acme Car Rental",
        description="Economy car rental for 2 days",
        category="Transport",
        source="model",
    )

    # Query exact or near-identical vendor
    cat, matches, reason = store.resolve_category("Acme Car Rental", "Car rental downtown")
    assert cat == "Transport"
    assert "memory match" in reason
    assert len(matches) > 0
    assert matches[0]["similarity"] >= 0.80

    # User correction priority
    store.add_user_correction("Bistro 42", "Meals", "Dinner with client")
    cat_corr, matches_corr, reason_corr = store.resolve_category("Bistro 42", "Business lunch")
    assert cat_corr == "Meals"
    assert "user_correction" in reason_corr
