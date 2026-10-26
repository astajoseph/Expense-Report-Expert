"""
data/seed.py
────────────
Course concept: Reproducible test data generation with date offsets.

This script regenerates mock_expenses.csv so that all expense dates fall
inside the "last calendar month" relative to any REPORT_TODAY date.

Run:
  python data/seed.py                        # uses REPORT_TODAY from .env
  python data/seed.py --date 2026-10-01      # explicit reference date

The dates are always generated for the month BEFORE the reference date,
so the filter step will always find them regardless of when you run the demo.
"""

import argparse
import csv
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# ── Allow running from repo root or data/ ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_reference_date(override: str | None = None) -> date:
    raw = override or os.getenv("REPORT_TODAY", "").strip()
    if raw:
        try:
            from datetime import datetime
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            pass
    return date.today()


def last_month_dates(ref: date) -> tuple[date, date]:
    """Return (first_day, last_day) of the month before ref."""
    first_of_this = ref.replace(day=1)
    last_day_prev = first_of_this - timedelta(days=1)
    first_day_prev = last_day_prev.replace(day=1)
    return first_day_prev, last_day_prev


def spread_date(first: date, last: date, index: int, total: int) -> str:
    """Spread `total` items roughly evenly across the date range."""
    span = (last - first).days
    offset = int(index * span / max(total - 1, 1))
    d = first + timedelta(days=offset)
    # Avoid weekends for realism
    while d.weekday() >= 5:
        d += timedelta(days=1)
        if d > last:
            d = last
    return d.strftime("%Y-%m-%d")


# ── Template rows (amounts/categories from original spec) ─────────────────
TEMPLATE_ROWS = [
    ("Delta Airlines",       "Flight SFO to NYC round trip",        "Flights",   "1250.00", "true"),
    ("Hilton Garden Inn",    "Hotel stay NYC 3 nights",              "",          "380.00",  "true"),
    ("The Capital Grille",   "Client dinner NYC",                    "",          "185.50",  "true"),
    ("Uber",                 "Ride from JFK to hotel",               "Transport", "42.30",   "true"),
    ("Starbucks",            "Morning coffee and breakfast",         "",          "18.75",   "false"),
    ("Lyft",                 "Airport pickup to hotel",              "",          "",        "false"),   # missing amount
    ("BART",                 "Bay Area transit commute",             "Transport", "12.50",   "true"),
    ("United Airlines",      "Flight NYC to Chicago",                "Flights",   "825.00",  "true"),
    ("Marriott",             "Hotel Chicago 2 nights",               "",          "255.00",  "true"),
    ("Olive Garden",         "Team lunch Chicago",                   "",          "95.40",   "true"),
    ("Lyft",                 "Ride to O'Hare airport",               "Transport", "38.25",   "true"),
    ("Parking Authority",    "Airport parking 3 days",               "",          "",        "false"),  # missing amount
    ("Cheesecake Factory",   "Dinner with client",                   "",          "112.80",  "true"),
    ("Enterprise Rent-A-Car","Car rental 2 days",                    "Transport", "52.10",   "true"),
    ("Subway",               "Quick lunch at airport",               "",          "14.25",   "false"),
    ("Holiday Inn Express",  "Hotel Boston 1 night",                 "",          "118.30",  "true"),
    ("McDonald's",           "Breakfast at airport",                 "",          "9.85",    "false"),
    ("Amtrak",               "Train NYC to Boston",                  "",          "89.00",   "true"),
    ("Southwest Airlines",   "Flight BOS to SFO",                    "Flights",   "0.00",    "true"),   # voucher flight, $0
    ("Uber",                 "Ride SFO to office",                   "Transport", "31.00",   "true"),
]


def generate_csv(ref: date, output_path: Path) -> None:
    first, last = last_month_dates(ref)
    total = len(TEMPLATE_ROWS)

    print(f"  Reference date   : {ref}")
    print(f"  Expense month    : {first} to {last}")
    print(f"  Output           : {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "vendor", "description", "category", "amount", "receipt_attached"])
        for i, (vendor, desc, cat, amount, receipt) in enumerate(TEMPLATE_ROWS):
            d = spread_date(first, last, i, total)
            writer.writerow([d, vendor, desc, cat, amount, receipt])

    print(f"  ✔ Generated {total} rows for month {first.strftime('%B %Y')}.")


def main():
    parser = argparse.ArgumentParser(description="Regenerate mock_expenses.csv")
    parser.add_argument("--date", type=str, default=None,
                        help="Reference 'today' date (YYYY-MM-DD). Defaults to REPORT_TODAY env or real today.")
    args = parser.parse_args()

    ref = get_reference_date(args.date)
    output = Path(__file__).parent / "mock_expenses.csv"
    generate_csv(ref, output)


if __name__ == "__main__":
    main()
