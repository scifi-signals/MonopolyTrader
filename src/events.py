"""Economic and earnings event calendar for trading awareness.

Provides upcoming event context so Claude knows when FOMC, CPI,
and TSLA earnings are approaching. Static macro calendar (updated annually)
plus live earnings dates from yfinance.

v6.1: Added event impact measurement — tracks TSLA price changes on event days
to build historical impact data for the agent's brief.
"""

from datetime import datetime, timedelta
import yfinance as yf
from .utils import setup_logging, load_json, save_json, DATA_DIR

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

# v6.1 Blindspot #10: Event impact tracking
EVENT_IMPACTS_PATH = DATA_DIR / "event_impacts.json"


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
    """Format upcoming events as text for the Claude trading brief.

    v6.1: Includes historical impact data for upcoming event types.
    """
    parts = []

    # Load historical event impacts
    impacts = load_json(EVENT_IMPACTS_PATH, default={})

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

            # Add historical impact data if available
            impact_str = _format_event_impact_history(event["event"], impacts)
            parts.append(
                f"{urgency}: {event['event']} on {event['date']} ({hours:.0f}h away){end}"
                f"{impact_str}"
            )

    if not parts:
        return "No major events in the next 72 hours."

    return "\n".join(parts)


# === Event Impact Tracking (v6.1 Blindspot #10) ===


def get_recent_past_events(days: int = 7) -> list[dict]:
    """Return macro events that happened in the last N days.

    v6.1 Blindspot #10: Identifies recent events for impact measurement.
    """
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days)
    recent = []

    for date_str, event_type in ALL_MACRO_EVENTS:
        event_date = datetime.strptime(date_str, "%Y-%m-%d")
        if cutoff <= event_date <= now:
            days_ago = (now - event_date).days
            recent.append({
                "date": date_str,
                "event": event_type,
                "days_ago": days_ago,
            })

    recent.sort(key=lambda x: x["days_ago"])
    return recent


def record_event_impact(event_type: str, event_date: str, tsla_change_pct: float):
    """Record the TSLA price change on an event day.

    v6.1 Blindspot #10: Builds historical event impact data.

    Args:
        event_type: e.g. "CPI", "FOMC", "NFP"
        event_date: ISO date string
        tsla_change_pct: TSLA price change on that day (percent)
    """
    impacts = load_json(EVENT_IMPACTS_PATH, default={})

    if event_type not in impacts:
        impacts[event_type] = []

    # Don't duplicate
    existing_dates = {e["date"] for e in impacts[event_type]}
    if event_date in existing_dates:
        return

    impacts[event_type].append({
        "date": event_date,
        "tsla_change_pct": round(tsla_change_pct, 2),
    })

    # Keep only last 20 events per type
    if len(impacts[event_type]) > 20:
        impacts[event_type] = impacts[event_type][-20:]

    save_json(EVENT_IMPACTS_PATH, impacts)
    logger.info(
        f"Event impact recorded: {event_type} on {event_date}, "
        f"TSLA {tsla_change_pct:+.2f}%"
    )


def measure_recent_event_impacts(ticker: str = "TSLA"):
    """Check for recent events and record their TSLA price impact.

    v6.1 Blindspot #10: Called during daily tasks to measure impact
    of events that occurred in the past few days.
    """
    recent_events = get_recent_past_events(days=3)
    if not recent_events:
        return

    impacts = load_json(EVENT_IMPACTS_PATH, default={})

    for event in recent_events:
        event_type = event["event"]
        event_date = event["date"]

        # Skip if already recorded
        existing_dates = {e["date"] for e in impacts.get(event_type, [])}
        if event_date in existing_dates:
            continue

        # Fetch TSLA price change on event day
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=event_date, period="5d", interval="1d")
            if len(hist) >= 1:
                # Find the event day row
                for i, idx in enumerate(hist.index):
                    if str(idx.date()) == event_date:
                        if i > 0:
                            prev_close = float(hist["Close"].iloc[i - 1])
                            event_close = float(hist["Close"].iloc[i])
                            change_pct = ((event_close - prev_close) / prev_close) * 100
                        else:
                            event_open = float(hist["Open"].iloc[i])
                            event_close = float(hist["Close"].iloc[i])
                            change_pct = ((event_close - event_open) / event_open) * 100

                        record_event_impact(event_type, event_date, change_pct)
                        break
        except Exception as e:
            logger.debug(f"Failed to measure impact for {event_type} on {event_date}: {e}")


def get_event_impact_summary(event_type: str) -> dict | None:
    """Get historical impact summary for an event type.

    Returns dict with avg_change_pct, count, and recent entries.
    Returns None if no data.
    """
    impacts = load_json(EVENT_IMPACTS_PATH, default={})
    entries = impacts.get(event_type, [])

    if not entries:
        return None

    changes = [e["tsla_change_pct"] for e in entries]
    return {
        "event_type": event_type,
        "count": len(changes),
        "avg_change_pct": round(sum(changes) / len(changes), 2),
        "max_up": round(max(changes), 2),
        "max_down": round(min(changes), 2),
        "recent": entries[-3:],
    }


def _format_event_impact_history(event_type: str, impacts: dict) -> str:
    """Format historical impact data for a specific event type."""
    entries = impacts.get(event_type, [])
    if not entries:
        return ""

    changes = [e["tsla_change_pct"] for e in entries]
    avg = sum(changes) / len(changes)
    n = len(changes)

    return f"\n    Historical TSLA move on {event_type} days: avg {avg:+.2f}% (N={n})"
