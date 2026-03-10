"""Trade journal — one entry per trade, one lesson per close.

The journal is the only persistent learning mechanism in v4.
Every trade gets an entry when opened. When closed, Haiku writes
a 50-word lesson. The last 10 entries appear in every trading brief.
"""

import os
from anthropic import Anthropic
from .utils import load_json, save_json, iso_now, setup_logging, load_config, DATA_DIR

logger = setup_logging("journal")

JOURNAL_PATH = DATA_DIR / "trade_journal.json"

HAIKU_MODEL = "claude-haiku-4-5-20251001"


def load_journal() -> list[dict]:
    """Load all journal entries, ordered oldest-first."""
    return load_json(JOURNAL_PATH, default=[])


def save_journal(entries: list[dict]) -> None:
    """Save journal entries to disk."""
    save_json(JOURNAL_PATH, entries)


def add_entry(
    trade_id: str,
    action: str,
    ticker: str,
    shares: float,
    price: float,
    reasoning: str,
    confidence: float,
    portfolio_value: float,
    market_snapshot: str,
    tags: dict | None = None,
) -> dict:
    """Record a new trade in the journal. Called immediately after a trade executes."""
    entries = load_journal()

    entry = {
        "trade_id": trade_id,
        "timestamp": iso_now(),
        "action": action,
        "ticker": ticker,
        "shares": round(shares, 4),
        "price": round(price, 2),
        "total_value": round(shares * price, 2),
        "reasoning": reasoning[:500],
        "confidence": round(confidence, 2),
        "portfolio_value": round(portfolio_value, 2),
        "market_snapshot": market_snapshot[:200],
        "tags": tags,
        "lesson": None,
        "close_trade_id": None,
        "close_price": None,
        "realized_pnl": None,
        "closed_at": None,
    }

    entries.append(entry)
    save_journal(entries)
    logger.info(f"Journal: recorded {action} {shares:.4f} {ticker} @ ${price:.2f}")
    return entry


def close_entry(
    open_trade_id: str,
    close_trade_id: str,
    close_price: float,
    realized_pnl: float,
) -> dict | None:
    """Mark a journal entry as closed and generate a lesson via Haiku."""
    entries = load_journal()

    target = None
    for entry in entries:
        if entry["trade_id"] == open_trade_id and entry["lesson"] is None:
            target = entry
            break

    if target is None:
        logger.warning(f"Journal: no open entry found for {open_trade_id}")
        return None

    target["close_trade_id"] = close_trade_id
    target["close_price"] = round(close_price, 2)
    target["realized_pnl"] = round(realized_pnl, 2)
    target["closed_at"] = iso_now()

    lesson = _generate_lesson(target)
    target["lesson"] = lesson

    save_journal(entries)
    logger.info(
        f"Journal: closed {open_trade_id} → P&L ${realized_pnl:.2f} | "
        f"Lesson: {lesson[:80]}"
    )
    return target


def _generate_lesson(entry: dict) -> str:
    """Call Haiku to generate a 50-word lesson from a closed trade."""
    config = load_config()
    model = config.get("haiku_model", HAIKU_MODEL)

    pnl = entry.get("realized_pnl", 0)
    outcome = "profit" if pnl >= 0 else "loss"
    pnl_pct = (pnl / entry["total_value"] * 100) if entry["total_value"] > 0 else 0

    prompt = f"""Write a 50-word trading lesson from this trade:

Action: {entry['action']} {entry['shares']} shares of {entry['ticker']}
Entry: ${entry['price']:.2f} | Exit: ${entry.get('close_price', 0):.2f}
Result: ${pnl:.2f} ({pnl_pct:+.1f}%) — {outcome}
Reasoning at entry: {entry['reasoning'][:300]}
Market context at entry: {entry['market_snapshot']}

Write exactly one lesson in 50 words or fewer. Be specific and actionable.
Do not start with "Lesson:" — just state the insight directly."""

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            for path in ["anthropic_api_key.txt", "../anthropic_api_key.txt"]:
                try:
                    with open(path) as f:
                        api_key = f.read().strip()
                        break
                except FileNotFoundError:
                    continue

        if not api_key:
            return f"{'Win' if pnl >= 0 else 'Loss'}: ${pnl:.2f} on {entry['ticker']}. No API key for lesson generation."

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        lesson = response.content[0].text.strip()

        # Truncate to ~50 words if Haiku was verbose
        words = lesson.split()
        if len(words) > 60:
            lesson = " ".join(words[:50]) + "..."

        return lesson

    except Exception as e:
        logger.warning(f"Lesson generation failed: {e}")
        return f"{'Win' if pnl >= 0 else 'Loss'}: ${pnl:.2f}. {entry['reasoning'][:100]}"


def get_recent_entries(n: int = 10) -> list[dict]:
    """Get the last N journal entries, newest-first."""
    entries = load_journal()
    return list(reversed(entries[-n:]))


def get_entries_since(date_str: str) -> list[dict]:
    """Get all journal entries since a given date (inclusive), oldest-first.

    Args:
        date_str: ISO date string, e.g. "2026-03-01"
    """
    entries = load_journal()
    return [e for e in entries if e.get("timestamp", "") >= date_str]


def format_journal_for_brief(entries: list[dict]) -> str:
    """Format journal entries as text for the Claude trading brief."""
    if not entries:
        return "No previous trades recorded."

    lines = []
    for e in entries:
        pnl_str = ""
        lesson_str = ""
        if e.get("realized_pnl") is not None:
            pnl_str = f" → ${e['realized_pnl']:+.2f}"
            if e.get("lesson"):
                lesson_str = f"\n    LESSON: {e['lesson']}"

        lines.append(
            f"  [{e['timestamp'][:16]}] {e['action']} {e['shares']:.2f} {e['ticker']} "
            f"@ ${e['price']:.2f} (conf={e['confidence']:.0%}){pnl_str}"
            f"{lesson_str}"
        )

    return "\n".join(lines)


def get_journal_stats() -> dict:
    """Compute summary statistics from the journal."""
    entries = load_journal()

    closed = [e for e in entries if e.get("realized_pnl") is not None]
    wins = [e for e in closed if e["realized_pnl"] >= 0]
    losses = [e for e in closed if e["realized_pnl"] < 0]

    return {
        "total_trades": len(entries),
        "closed_trades": len(closed),
        "open_trades": len(entries) - len(closed),
        "total_pnl": round(sum(e["realized_pnl"] for e in closed), 2) if closed else 0,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "avg_win": round(sum(e["realized_pnl"] for e in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(e["realized_pnl"] for e in losses) / len(losses), 2) if losses else 0,
        "biggest_win": max((e["realized_pnl"] for e in wins), default=0),
        "biggest_loss": min((e["realized_pnl"] for e in losses), default=0),
        "lessons": [e["lesson"] for e in closed if e.get("lesson")],
    }
