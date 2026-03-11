"""Trade journal — one entry per trade, one lesson per close.

The journal is the only persistent learning mechanism in v4.
Every trade gets an entry when opened. When closed, Haiku writes
a structured lesson with outcome classification. The last 10 entries
appear in every trading brief.

v6.1: Enhanced with outcome_type diagnosis, intra-trade price tracking,
hold_duration, hypothesis testing, and thesis_consistent fields.
"""

import json
import os
from datetime import datetime, timezone
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
    strategy: str = "",
    hypothesis: str = "",
    expected_learning: str = "",
    thesis_consistent: bool | None = None,
) -> dict:
    """Record a new trade in the journal. Called immediately after a trade executes.

    v6: Added strategy, hypothesis, expected_learning fields.
    v6.1: Added thesis_consistent, intra-trade price tracking fields.
    """
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
        "strategy": strategy[:100] if strategy else "",
        "hypothesis": hypothesis[:300] if hypothesis else "",
        "expected_learning": expected_learning[:300] if expected_learning else "",
        "thesis_consistent": thesis_consistent,
        "lesson": None,
        "close_trade_id": None,
        "close_price": None,
        "realized_pnl": None,
        "closed_at": None,
        # Intra-trade price tracking (updated every cycle while position is open)
        "peak_price": round(price, 2),
        "trough_price": round(price, 2),
        "peak_unrealized_pnl": 0.0,
        "trough_unrealized_pnl": 0.0,
        # Outcome diagnosis (populated at close time)
        "outcome_type": None,
        "thesis_correct": None,
        "hypothesis_tested": None,
        "hypothesis_result": None,
        "hold_minutes": None,
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
    """Mark a journal entry as closed and generate a structured lesson via Haiku.

    v6.1: Computes hold_minutes, hold_duration tag, and structured outcome
    diagnosis (outcome_type, thesis_correct, hypothesis_tested, hypothesis_result).
    """
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

    # Compute hold_minutes (Blindspot #8 — trade duration)
    try:
        open_time = datetime.fromisoformat(target["timestamp"])
        close_time = datetime.fromisoformat(target["closed_at"])
        if open_time.tzinfo is None:
            open_time = open_time.replace(tzinfo=timezone.utc)
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        hold_minutes = (close_time - open_time).total_seconds() / 60.0
        target["hold_minutes"] = round(hold_minutes, 1)
    except Exception:
        target["hold_minutes"] = None

    # Add hold_duration tag (Blindspot #8)
    if target.get("tags") is None:
        target["tags"] = {}
    hm = target.get("hold_minutes")
    if hm is not None:
        if hm < 20:
            target["tags"]["hold_duration"] = "quick"
        elif hm < 60:
            target["tags"]["hold_duration"] = "medium"
        elif hm < 240:
            target["tags"]["hold_duration"] = "extended"
        else:
            target["tags"]["hold_duration"] = "overnight"

    # Generate structured lesson with outcome diagnosis
    lesson_data = _generate_lesson(target)
    if isinstance(lesson_data, dict):
        target["lesson"] = lesson_data.get("lesson", str(lesson_data))
        target["outcome_type"] = lesson_data.get("outcome_type")
        target["thesis_correct"] = lesson_data.get("thesis_correct")
        target["hypothesis_tested"] = lesson_data.get("hypothesis_tested")
        target["hypothesis_result"] = lesson_data.get("hypothesis_result")
    else:
        target["lesson"] = lesson_data

    save_journal(entries)
    logger.info(
        f"Journal: closed {open_trade_id} → P&L ${realized_pnl:.2f} | "
        f"Outcome: {target.get('outcome_type', 'unknown')} | "
        f"Lesson: {str(target.get('lesson', ''))[:80]}"
    )
    return target


def _generate_lesson(entry: dict) -> dict | str:
    """Call Haiku to generate a structured lesson from a closed trade.

    v6.1: Returns a dict with lesson text, outcome classification, and hypothesis
    assessment. Falls back to plain text string if JSON parsing fails.

    Combines:
    - Blindspot #1: outcome_type diagnosis (thesis_wrong, timing_wrong, etc.)
    - Blindspot #8: plan adherence (hypothesis_tested, hypothesis_result)
    """
    config = load_config()
    model = config.get("haiku_model", HAIKU_MODEL)

    pnl = entry.get("realized_pnl", 0)
    outcome = "profit" if pnl >= 0 else "loss"
    pnl_pct = (pnl / entry["total_value"] * 100) if entry["total_value"] > 0 else 0

    # Build intra-trade context if available
    peak_price = entry.get("peak_price", entry["price"])
    trough_price = entry.get("trough_price", entry["price"])
    peak_pnl = entry.get("peak_unrealized_pnl", 0)
    trough_pnl = entry.get("trough_unrealized_pnl", 0)
    intra_trade_ctx = ""
    if peak_price != entry["price"] or trough_price != entry["price"]:
        intra_trade_ctx = (
            f"\nIntra-trade: peaked at ${peak_price:.2f} (unrealized {'+' if peak_pnl >= 0 else ''}${peak_pnl:.2f}), "
            f"troughed at ${trough_price:.2f} (unrealized {'+' if trough_pnl >= 0 else ''}${trough_pnl:.2f})"
        )

    # Include hypothesis for plan adherence assessment
    hypothesis = entry.get("hypothesis", "")
    hypothesis_ctx = ""
    if hypothesis and hypothesis != "N/A - observing":
        hypothesis_ctx = f"\nHypothesis at entry: {hypothesis[:200]}"

    hold_ctx = ""
    if entry.get("hold_minutes") is not None:
        hold_ctx = f"\nHold duration: {entry['hold_minutes']:.0f} minutes"

    prompt = f"""Analyze this closed trade and return a JSON object.

Action: {entry['action']} {entry['shares']} shares of {entry['ticker']}
Entry: ${entry['price']:.2f} | Exit: ${entry.get('close_price', 0):.2f}
Result: ${pnl:.2f} ({pnl_pct:+.1f}%) — {outcome}
Reasoning at entry: {entry['reasoning'][:300]}
Market context at entry: {entry['market_snapshot']}{intra_trade_ctx}{hypothesis_ctx}{hold_ctx}

Return ONLY a JSON object with these exact fields:
{{
  "lesson": "<50-word actionable trading lesson>",
  "outcome_type": "<one of: thesis_wrong, timing_wrong, execution_wrong, external_shock, spread_cost>",
  "thesis_correct": <true or false>,
  "hypothesis_tested": <true or false — did the trade actually test the stated hypothesis?>,
  "hypothesis_result": "<one of: confirmed, refuted, inconclusive>"
}}

OUTCOME TYPE DEFINITIONS:
- thesis_wrong: The predicted direction was wrong (bought expecting up, price went down consistently)
- timing_wrong: Direction was right eventually but entry/exit timing was bad (price dipped then recovered after exit, or peaked during trade but exited at wrong time)
- execution_wrong: Thesis and timing were right but execution was poor (exited too early, wrong position size)
- external_shock: An unforeseeable event caused the loss (breaking news, flash crash)
- spread_cost: The move was too small to overcome slippage

For the lesson: be specific and actionable, 50 words max. Do not start with "Lesson:".
For hypothesis_tested: true if the trade conditions actually tested the stated hypothesis, false if conditions changed or hypothesis was vague.
For hypothesis_result: "confirmed" if outcome supports hypothesis, "refuted" if it contradicts, "inconclusive" if data is unclear.

Return ONLY valid JSON, no other text."""

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
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Try to parse as JSON
        try:
            # Strip code fences if present
            clean = raw
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            # Find JSON object
            json_start = clean.find("{")
            json_end = clean.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                clean = clean[json_start:json_end]

            result = json.loads(clean)

            # Validate and sanitize fields
            valid_outcome_types = {
                "thesis_wrong", "timing_wrong", "execution_wrong",
                "external_shock", "spread_cost",
            }
            if result.get("outcome_type") not in valid_outcome_types:
                result["outcome_type"] = "thesis_wrong" if pnl < 0 else None

            valid_hypothesis_results = {"confirmed", "refuted", "inconclusive"}
            if result.get("hypothesis_result") not in valid_hypothesis_results:
                result["hypothesis_result"] = "inconclusive"

            # Ensure boolean fields
            result["thesis_correct"] = bool(result.get("thesis_correct", pnl >= 0))
            result["hypothesis_tested"] = bool(result.get("hypothesis_tested", False))

            # Truncate lesson
            lesson = result.get("lesson", "")
            words = lesson.split()
            if len(words) > 60:
                result["lesson"] = " ".join(words[:50]) + "..."

            return result

        except (json.JSONDecodeError, ValueError):
            # Fall back to plain text lesson
            logger.debug("Haiku returned non-JSON lesson, falling back to plain text")
            words = raw.split()
            if len(words) > 60:
                raw = " ".join(words[:50]) + "..."
            return raw

    except Exception as e:
        logger.warning(f"Lesson generation failed: {e}")
        return f"{'Win' if pnl >= 0 else 'Loss'}: ${pnl:.2f}. {entry['reasoning'][:100]}"


def update_intra_trade_prices(current_price: float, ticker: str) -> bool:
    """Update peak/trough price tracking for open (unclosed) journal entries.

    Called every cycle while a position is held. Tracks the high and low
    water marks during the life of the trade.

    v6.1 Blindspot #3: Intra-trade price tracking.

    Returns True if any entry was updated.
    """
    entries = load_journal()
    updated = False

    for entry in entries:
        # Only update open (unclosed) BUY entries for this ticker
        if (entry.get("action") != "BUY"
            or entry.get("ticker") != ticker
            or entry.get("lesson") is not None):
            continue

        entry_price = entry.get("price", 0)
        shares = entry.get("shares", 0)
        if entry_price <= 0 or shares <= 0:
            continue

        # Initialize fields if missing (backward compat with pre-v6.1 entries)
        if "peak_price" not in entry:
            entry["peak_price"] = entry_price
        if "trough_price" not in entry:
            entry["trough_price"] = entry_price
        if "peak_unrealized_pnl" not in entry:
            entry["peak_unrealized_pnl"] = 0.0
        if "trough_unrealized_pnl" not in entry:
            entry["trough_unrealized_pnl"] = 0.0

        current_unrealized = round((current_price - entry_price) * shares, 2)

        if current_price > entry.get("peak_price", entry_price):
            entry["peak_price"] = round(current_price, 2)
            entry["peak_unrealized_pnl"] = current_unrealized
            updated = True

        if current_price < entry.get("trough_price", entry_price):
            entry["trough_price"] = round(current_price, 2)
            entry["trough_unrealized_pnl"] = current_unrealized
            updated = True

        # Also update if PnL exceeds known extremes even without new price extreme
        if current_unrealized > entry.get("peak_unrealized_pnl", 0):
            entry["peak_unrealized_pnl"] = current_unrealized
            updated = True
        if current_unrealized < entry.get("trough_unrealized_pnl", 0):
            entry["trough_unrealized_pnl"] = current_unrealized
            updated = True

    if updated:
        save_journal(entries)
        logger.debug(f"Journal: updated intra-trade prices at ${current_price:.2f}")

    return updated


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
    """Format journal entries as text for the Claude trading brief.

    v6.1: Shows outcome_type, intra-trade extremes, hypothesis results.
    """
    if not entries:
        return "No previous trades recorded."

    lines = []
    for e in entries:
        pnl_str = ""
        lesson_str = ""
        strategy_str = ""
        hypothesis_str = ""
        outcome_str = ""
        intra_trade_str = ""

        if e.get("realized_pnl") is not None:
            pnl_str = f" -> ${e['realized_pnl']:+.2f}"
            if e.get("lesson"):
                lesson_str = f"\n    LESSON: {e['lesson']}"
            if e.get("outcome_type"):
                outcome_str = f" [{e['outcome_type']}]"
            if e.get("hypothesis_result"):
                outcome_str += f" hypothesis:{e['hypothesis_result']}"

            # Show intra-trade extremes if the trade had significant movement
            peak = e.get("peak_price")
            trough = e.get("trough_price")
            if peak and trough and peak != e["price"]:
                intra_trade_str = (
                    f"\n    INTRA-TRADE: peaked ${peak:.2f}, troughed ${trough:.2f}"
                )

        if e.get("strategy"):
            strategy_str = f" [{e['strategy']}]"

        if e.get("hypothesis") and e["hypothesis"] != "N/A - observing":
            hypothesis_str = f"\n    HYPOTHESIS: {e['hypothesis'][:150]}"

        consistency_str = ""
        if e.get("thesis_consistent") is not None:
            consistency_str = " [MID-aligned]" if e["thesis_consistent"] else " [CONTRARIAN]"

        lines.append(
            f"  [{e['timestamp'][:16]}] {e['action']} {e['shares']:.2f} {e['ticker']} "
            f"@ ${e['price']:.2f} (conf={e['confidence']:.0%}){strategy_str}{consistency_str}{pnl_str}{outcome_str}"
            f"{hypothesis_str}{intra_trade_str}{lesson_str}"
        )

    return "\n".join(lines)


def get_journal_stats() -> dict:
    """Compute summary statistics from the journal.

    v6.1: Added outcome_type breakdown and thesis consistency stats.
    """
    entries = load_journal()

    closed = [e for e in entries if e.get("realized_pnl") is not None]
    wins = [e for e in closed if e["realized_pnl"] >= 0]
    losses = [e for e in closed if e["realized_pnl"] < 0]

    # Outcome type breakdown
    outcome_types = {}
    for e in closed:
        ot = e.get("outcome_type", "unclassified")
        outcome_types[ot] = outcome_types.get(ot, 0) + 1

    # Thesis consistency stats
    consistent_trades = [e for e in closed if e.get("thesis_consistent") is True]
    inconsistent_trades = [e for e in closed if e.get("thesis_consistent") is False]
    consistent_wins = sum(1 for e in consistent_trades if e["realized_pnl"] >= 0)
    inconsistent_wins = sum(1 for e in inconsistent_trades if e["realized_pnl"] >= 0)

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
        "outcome_types": outcome_types,
        "thesis_consistent_count": len(consistent_trades),
        "thesis_consistent_win_rate": round(consistent_wins / len(consistent_trades) * 100, 1) if consistent_trades else None,
        "thesis_inconsistent_count": len(inconsistent_trades),
        "thesis_inconsistent_win_rate": round(inconsistent_wins / len(inconsistent_trades) * 100, 1) if inconsistent_trades else None,
    }
