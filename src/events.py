"""Economic and earnings event calendar for trading awareness.

Provides upcoming event context so Claude knows when FOMC, CPI,
and TSLA earnings are approaching. Static macro calendar (updated annually)
plus live earnings dates from yfinance.
"""

from datetime import datetime, timedelta
import yfinance as yf
from .utils import setup_logging

logger = setup_logging("events")

# 2026 FOMC statement dates (second day — when decision released at 2pm ET)
# SEP = Summary of Economic Projections ("dot plot") — highest impact
FOMC_2026 = [
    ("2026-01-28", "FOMC"),
    ("2026-03-18", "FOMC+SEP"),
    ("2026-04-29", "FOMC"),
    ("2026-06-17", "FOMC+SEP"),
    ("2026-07-29", "FOMC"),
    ("2026-09-16", "FOMC+SEP"),
    ("2026-10-28", "FOMC"),
    ("2026-12-09", "FOMC+SEP"),
]

# 2026 CPI release dates (released at 8:30am ET)
CPI_2026 = [
    ("2026-01-13", "CPI"),
    ("2026-02-11", "CPI"),
    ("2026-03-11", "CPI"),
    ("2026-04-10", "CPI"),
    ("2026-05-12", "CPI"),
    ("2026-06-10", "CPI"),
    ("2026-07-14", "CPI"),
    ("2026-08-12", "CPI"),
    ("2026-09-11", "CPI"),
    ("2026-10-14", "CPI"),
    ("2026-11-10", "CPI"),
    ("2026-12-10", "CPI"),
]

# Jobs report (Non-Farm Payrolls) — first Friday of each month, 8:30am ET
# These move markets significantly too
NFP_2026 = [
    ("2026-01-09", "NFP"),
    ("2026-02-06", "NFP"),
    ("2026-03-06", "NFP"),
    ("2026-04-03", "NFP"),
    ("2026-05-08", "NFP"),
    ("2026-06-05", "NFP"),
    ("2026-07-02", "NFP"),
    ("2026-08-07", "NFP"),
    ("2026-09-04", "NFP"),
    ("2026-10-02", "NFP"),
    ("2026-11-06", "NFP"),
    ("2026-12-04", "NFP"),
]

ALL_MACRO_EVENTS = FOMC_2026 + CPI_2026 + NFP_2026


def get_upcoming_macro_events(hours: int = 48) -> list[dict]:
    """Return macro events within the next N hours.

    Returns list of dicts with date, event_type, and hours_until.
    """
    now = datetime.utcnow()
    cutoff = now + timedelta(hours=hours)
    upcoming = []

    for date_str, event_type in ALL_MACRO_EVENTS:
        event_date = datetime.strptime(date_str, "%Y-%m-%d")
        if now <= event_date <= cutoff:
            hours_until = (event_date - now).total_seconds() / 3600
            upcoming.append({
                "date": date_str,
                "event": event_type,
                "hours_until": round(hours_until, 1),
            })

    # Sort by soonest first
    upcoming.sort(key=lambda x: x["hours_until"])
    return upcoming


def get_tsla_earnings_date() -> dict | None:
    """Fetch next TSLA earnings date from yfinance.

    Returns dict with date, eps_estimate, revenue_estimate, days_until.
    Returns None if unavailable.
    """
    try:
        stock = yf.Ticker("TSLA")
        cal = stock.calendar

        if not cal or "Earnings Date" not in cal:
            return None

        earnings_dates = cal["Earnings Date"]
        if not earnings_dates:
            return None

        # First entry is the next earnings date
        next_date = earnings_dates[0]

        # Calculate days until
        if hasattr(next_date, "date"):
            next_date_obj = next_date
        else:
            next_date_obj = datetime.strptime(str(next_date), "%Y-%m-%d")

        now = datetime.utcnow().date()
        if hasattr(next_date_obj, "date"):
            days_until = (next_date_obj.date() - now).days if hasattr(next_date_obj, "date") else (next_date_obj - now).days
        else:
            days_until = (next_date_obj - datetime.utcnow()).days

        result = {
            "date": str(next_date),
            "days_until": days_until,
            "eps_estimate": cal.get("Earnings Average"),
            "revenue_estimate": cal.get("Revenue Average"),
        }

        logger.info(f"TSLA earnings: {result['date']} ({days_until} days away)")
        return result

    except Exception as e:
        logger.warning(f"TSLA earnings calendar failed: {e}")
        return None


def get_upcoming_events(hours: int = 72) -> dict:
    """Get all upcoming events — macro + TSLA earnings.

    Returns a dict suitable for inclusion in the trading brief.
    """
    result = {
        "macro_events": get_upcoming_macro_events(hours),
        "tsla_earnings": None,
    }

    # Always check TSLA earnings (even if > 72 hours away, trader should know)
    try:
        earnings = get_tsla_earnings_date()
        if earnings:
            result["tsla_earnings"] = earnings
    except Exception as e:
        logger.warning(f"Earnings check failed: {e}")

    return result


def format_events_for_brief(events: dict) -> str:
    """Format upcoming events as text for the Claude trading brief."""
    parts = []

    # TSLA earnings
    earnings = events.get("tsla_earnings")
    if earnings:
        days = earnings.get("days_until", 999)
        eps = earnings.get("eps_estimate")
        eps_str = f", EPS estimate ${eps:.2f}" if eps else ""
        if days <= 7:
            parts.append(f">>> TSLA EARNINGS IN {days} DAYS ({earnings['date']}){eps_str} — HIGH VOLATILITY EVENT <<<")
        elif days <= 30:
            parts.append(f"TSLA earnings: {earnings['date']} ({days} days away){eps_str}")

    # Macro events
    macro = events.get("macro_events", [])
    if macro:
        for event in macro:
            hours = event["hours_until"]
            if hours <= 24:
                urgency = ">>> TOMORROW"
                end = " — EXPECT VOLATILITY <<<"
            else:
                urgency = "Upcoming"
                end = ""
            parts.append(f"{urgency}: {event['event']} on {event['date']} ({hours:.0f}h away){end}")

    if not parts:
        return "No major events in the next 72 hours."

    return "\n".join(parts)
