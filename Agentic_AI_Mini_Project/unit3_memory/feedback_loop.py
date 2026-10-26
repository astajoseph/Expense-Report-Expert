"""
unit3_memory/feedback_loop.py
─────────────────────────────
Course Concept: Unit 3 — Feedback Loops in Agentic Systems (Act - Observe - Reflect - Adjust).

Why this concept?
In real-world data pipelines, missing or corrupted data cannot be guessed
arbitrarily by an LLM (especially financial amounts). A feedback loop allows
the agent to:
1. ACT: Detect missing fields (amount, date, vendor).
2. OBSERVE: Gather the current state of flagged records.
3. REFLECT: Determine what missing data is needed and if a retry budget remains.
4. ADJUST: Request clarification from the human (via CLI or scripted answers.json)
   and apply the fix, or escalate by permanently flagging the row in the audit log.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class FeedbackLoopManager:
    """
    Manages the Act-Observe-Reflect-Adjust cycle for missing expense data.
    Supports interactive CLI input or automated scripted answers (--answers answers.json).
    """

    def __init__(self, answers_file: Optional[str] = None, max_retries: int = 2):
        self.answers_file = answers_file
        self.max_retries = max_retries
        self.scripted_answers: dict[str, Any] = {}
        if answers_file and Path(answers_file).exists():
            try:
                with open(answers_file, "r", encoding="utf-8") as f:
                    self.scripted_answers = json.load(f)
                logger.info(f"Loaded {len(self.scripted_answers)} scripted answers from {answers_file}")
            except Exception as e:
                logger.warning(f"Could not load answers file {answers_file}: {e}")

    def act_detect_missing(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        ACT: Scan rows to find items with missing amounts, invalid dates, or empty vendors.
        """
        missing_items = []
        for row in rows:
            vendor = (row.get("vendor") or "").strip()
            amount_str = str(row.get("amount") or "").strip()
            date_str = (row.get("date") or "").strip()

            is_missing_amount = False
            if not amount_str:
                is_missing_amount = True
            else:
                try:
                    val = float(amount_str)
                    if val <= 0.0:
                        is_missing_amount = True
                except ValueError:
                    is_missing_amount = True

            if is_missing_amount or not date_str or not vendor:
                flag = "missing amount" if is_missing_amount else "missing metadata"
                item = dict(row)
                item["_flag"] = flag
                missing_items.append(item)
        return missing_items

    def observe(self, missing_items: list[dict[str, Any]], iteration: int) -> dict[str, Any]:
        """
        OBSERVE: Summarize the current state of missing items for this feedback loop cycle.
        """
        summary = {
            "iteration": iteration,
            "total_missing": len(missing_items),
            "vendors": [it.get("vendor", "Unknown") for it in missing_items],
            "flags": [it.get("_flag", "flagged") for it in missing_items],
        }
        return summary

    def reflect(self, observation: dict[str, Any]) -> tuple[bool, str]:
        """
        REFLECT: Decide whether to continue the loop (ask human) or escalate to PDF appendix.
        Returns: (should_retry, strategy_explanation)
        """
        total = observation["total_missing"]
        iteration = observation["iteration"]

        if total == 0:
            return False, "All rows resolved cleanly. Ready to calculate totals."

        if iteration >= self.max_retries:
            return False, f"Max retries ({self.max_retries}) reached. Escalating {total} unresolved row(s) to manual audit appendix."

        return True, f"Found {total} row(s) requiring missing information. Attempting recovery cycle {iteration + 1}/{self.max_retries}."

    def adjust_resolve_items(
        self,
        rows: list[dict[str, Any]],
        missing_items: list[dict[str, Any]],
        auto_prompt: bool = True,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        """
        ADJUST: Interactively or script-assisted resolution of missing data.
        Returns: (updated_rows, remaining_exceptions, adjustment_logs)
        """
        resolved_vendors = set()
        adjustment_logs = []

        for item in missing_items:
            vendor = item.get("vendor", "")
            date_val = item.get("date", "")
            desc = item.get("description", "")
            flag = item.get("_flag", "missing amount")

            provided_amount: Optional[float] = None

            # 1. Check scripted answers first
            if vendor in self.scripted_answers:
                try:
                    provided_amount = float(self.scripted_answers[vendor])
                    adjustment_logs.append(
                        f"Resolved '{vendor}' via scripted answers: ₹{provided_amount:.2f}"
                    )
                except (ValueError, TypeError):
                    pass

            # 2. Check interactive user input if in interactive mode and not resolved
            if provided_amount is None and auto_prompt:
                try:
                    print(f"\n   [FEEDBACK LOOP - ADJUST] Missing data detected:")
                    print(f"   Row: {date_val} | Vendor: {vendor} | Desc: {desc} | Issue: {flag}")
                    user_input = input(f"   Enter missing amount for '{vendor}' in ₹ (or press Enter to skip): ").strip()
                    if user_input:
                        try:
                            clean_val = user_input.replace("₹", "").replace("Rs.", "").replace("Rs", "").replace("$", "").replace(",", "").strip()
                            provided_amount = float(clean_val)
                            adjustment_logs.append(
                                f"Resolved '{vendor}' via user input: ₹{provided_amount:.2f}"
                            )
                        except ValueError:
                            print(f"   Invalid amount entered for '{vendor}'. Skipping.")
                except (EOFError, KeyboardInterrupt):
                    pass

            # 3. Apply resolution if amount obtained
            if provided_amount is not None and provided_amount > 0:
                for r in rows:
                    if r.get("vendor") == vendor and r.get("date") == date_val:
                        r["amount"] = f"{provided_amount:.2f}"
                        if "_flag" in r:
                            del r["_flag"]
                resolved_vendors.add(vendor)

        # Remaining exceptions are those not resolved
        remaining_exceptions = [
            it for it in missing_items if it.get("vendor") not in resolved_vendors
        ]

        return rows, remaining_exceptions, adjustment_logs
