"""Shadow Journal — tracks HOLD decisions and what would have happened.

Every HOLD decision gets logged with full market context. Each cycle, we
update recent HOLDs with the current price to compute shadow P&L: what
would the outcome have been if we'd traded instead of holding?

This data helps calibrate the agent's hold/trade threshold. If HOLDs
consistently avoid losses, the agent is correctly conservative. If HOLDs
consistently miss gains, the agent is too cautious.
"""

from datetime import datetime, timezone, timedelta
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("shadow_journal")

HOLD_JOURNAL_PATH = DATA_DIR / "hold_journal.json"

# Standard shadow trade size for P&L calculation
SHADOW_TRADE_SHARES = 0.5

# How far back to track shadow prices (hours)
SHADOW_LOOKBACK_HOURS = 4

# Threshold for "meaningful" price move
MEANINGFUL_MOVE_PCT = 0.2


def log_hold_decision(
    decision: dict,
    market_data: dict,
    tags: dict,
) -> dict:
    """Log a HOLD decision to the shadow journal.

    Args:
        decision: The agent's decision dict (action=HOLD, with reasoning etc.)
        market_data: Current market data snapshot
        tags: Current market condition tags

    Returns:
        The logged entry dict.
    """
    current = market_data.get("current", {})
    price = current.get("price", 0)
    regime = market_data.get("regime", {})

    entry = {
        "timestamp": iso_now(),
        "price": price,
        "confidence": decision.get("confidence", 0),
        "strategy": decision.get("strategy", ""),
        "hypothesis": decision.get("hypothesis", ""),
        "expected_learning": decision.get("expected_learning", ""),
        "reasoning": decision.get("reasoning", "")[:500],
        "tags": tags,
        "regime": {
            "trend": regime.get("trend", "unknown"),
            "directional": regime.get("directional", "unknown"),
            "volatility": regime.get("volatility", "unknown"),
            "vix": regime.get("vix", 0),
        },
        "change_pct": current.get("change_pct", 0),
        "volume": current.get("volume", 0),
        # Shadow tracking fields — updated by update_shadow_prices()
        "shadow_updates": [],
        "shadow_peak_price": price,
        "shadow_trough_price": price,
        "shadow_final_price": None,
        "shadow_pnl_buy": None,  # P&L if we'd bought
        "shadow_pnl_sell": None,  # P&L if we'd sold
        "shadow_resolved": False,
    }

    # Load existing journal and append
    journal = load_json(HOLD_JOURNAL_PATH, default=[])
    journal.append(entry)
    save_json(HOLD_JOURNAL_PATH, journal)

    logger.info(
        f"Shadow journal: logged HOLD at ${price:.2f} "
        f"(strategy={entry['strategy']}, conf={entry['confidence']:.2f})"
    )
    return entry


def update_shadow_prices(current_price: float) -> int:
    """Update shadow P&L for all recent unresolved HOLD entries.

    Called every cycle regardless of action. Updates the shadow
    tracking for HOLDs from the last SHADOW_LOOKBACK_HOURS.

    Args:
        current_price: Current TSLA price

    Returns:
        Number of entries updated.
    """
    journal = load_json(HOLD_JOURNAL_PATH, default=[])
    if not journal:
        return 0

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=SHADOW_LOOKBACK_HOURS)
    updated = 0

    for entry in journal:
        if entry.get("shadow_resolved"):
            continue

        # Parse timestamp
        try:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
        except (ValueError, KeyError):
            continue

        # Skip entries older than lookback window — resolve them
        if entry_time < cutoff:
            _resolve_entry(entry, current_price)
            updated += 1
            continue

        # Update shadow tracking
        entry_price = entry.get("price", 0)
        if entry_price <= 0:
            continue

        # Track peak and trough
        if current_price > entry.get("shadow_peak_price", entry_price):
            entry["shadow_peak_price"] = current_price
        if current_price < entry.get("shadow_trough_price", entry_price):
            entry["shadow_trough_price"] = current_price

        # Add shadow update point
        entry["shadow_updates"].append({
            "timestamp": iso_now(),
            "price": current_price,
        })

        # Keep only last 10 update points to prevent unbounded growth
        if len(entry["shadow_updates"]) > 10:
            entry["shadow_updates"] = entry["shadow_updates"][-10:]

        # Compute current shadow P&L
        entry["shadow_pnl_buy"] = round(
            (current_price - entry_price) * SHADOW_TRADE_SHARES, 2
        )
        entry["shadow_pnl_sell"] = round(
            (entry_price - current_price) * SHADOW_TRADE_SHARES, 2
        )

        updated += 1

    if updated > 0:
        save_json(HOLD_JOURNAL_PATH, journal)
        logger.debug(f"Shadow journal: updated {updated} entries at ${current_price:.2f}")

    return updated


def _resolve_entry(entry: dict, final_price: float):
    """Mark a shadow entry as resolved with final P&L."""
    entry_price = entry.get("price", 0)
    if entry_price > 0:
        entry["shadow_final_price"] = final_price
        entry["shadow_pnl_buy"] = round(
            (final_price - entry_price) * SHADOW_TRADE_SHARES, 2
        )
        entry["shadow_pnl_sell"] = round(
            (entry_price - final_price) * SHADOW_TRADE_SHARES, 2
        )
    entry["shadow_resolved"] = True


def get_shadow_summary(hours: int = 24) -> dict:
    """Compute summary statistics from recent HOLD decisions.

    Args:
        hours: How far back to look (default 24h).

    Returns:
        Dict with summary stats.
    """
    journal = load_json(HOLD_JOURNAL_PATH, default=[])
    if not journal:
        return {
            "total_holds": 0,
            "period_hours": hours,
        }

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    recent = []
    for entry in journal:
        try:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            if entry_time >= cutoff:
                recent.append(entry)
        except (ValueError, KeyError):
            continue

    if not recent:
        return {
            "total_holds": 0,
            "period_hours": hours,
        }

    # Classify outcomes for resolved entries
    profitable_buys = 0  # Price went up > threshold (should have bought)
    unprofitable_buys = 0  # Price went down > threshold (correctly avoided buying)
    neutral = 0
    biggest_missed_gain = 0.0
    biggest_avoided_loss = 0.0

    for entry in recent:
        entry_price = entry.get("price", 0)
        if entry_price <= 0:
            neutral += 1
            continue

        # Use shadow_pnl_buy if available (what would have happened if we bought)
        shadow_pnl = entry.get("shadow_pnl_buy")
        if shadow_pnl is None:
            neutral += 1
            continue

        # Compute move percentage from entry price
        final_price = entry.get("shadow_final_price") or entry.get("shadow_peak_price", entry_price)
        if final_price and entry_price > 0:
            move_pct = abs(final_price - entry_price) / entry_price * 100
        else:
            move_pct = 0

        if shadow_pnl > 0 and move_pct > MEANINGFUL_MOVE_PCT:
            profitable_buys += 1
            if shadow_pnl > biggest_missed_gain:
                biggest_missed_gain = shadow_pnl
        elif shadow_pnl < 0 and move_pct > MEANINGFUL_MOVE_PCT:
            unprofitable_buys += 1
            if abs(shadow_pnl) > biggest_avoided_loss:
                biggest_avoided_loss = abs(shadow_pnl)
        else:
            neutral += 1

    total = len(recent)
    hold_quality = 0.0
    if total > 0 and (unprofitable_buys + profitable_buys) > 0:
        # Quality = % of HOLDs that correctly avoided a loss
        hold_quality = unprofitable_buys / (unprofitable_buys + profitable_buys)

    return {
        "total_holds": total,
        "period_hours": hours,
        "profitable_buys": profitable_buys,  # missed gains (price went up)
        "unprofitable_buys": unprofitable_buys,  # avoided losses (price went down)
        "neutral": neutral,
        "biggest_missed_gain": round(biggest_missed_gain, 2),
        "biggest_avoided_loss": round(biggest_avoided_loss, 2),
        "hold_quality": round(hold_quality, 3),
    }


def format_shadow_for_brief(hours: int = 24) -> str:
    """Format shadow journal summary for the agent's brief.

    Returns:
        Human-readable summary string, or empty string if no data.
    """
    summary = get_shadow_summary(hours)
    total = summary.get("total_holds", 0)
    if total == 0:
        return ""

    profitable = summary.get("profitable_buys", 0)
    avoided = summary.get("unprofitable_buys", 0)
    neutral = summary.get("neutral", 0)
    missed = summary.get("biggest_missed_gain", 0)
    saved = summary.get("biggest_avoided_loss", 0)
    quality = summary.get("hold_quality", 0)

    parts = [
        f"Last {hours}h: {total} HOLDs tracked.",
    ]

    if profitable + avoided > 0:
        parts.append(
            f"  {avoided} correctly avoided losses, "
            f"{profitable} missed profitable setups, "
            f"{neutral} neutral."
        )
        if missed > 0:
            parts.append(f"  Biggest missed gain: ${missed:.2f}")
        if saved > 0:
            parts.append(f"  Biggest avoided loss: ${saved:.2f}")
        parts.append(f"  Hold quality (loss avoidance rate): {quality:.0%}")
    else:
        parts.append(f"  All {total} HOLDs had neutral outcomes (price moved <{MEANINGFUL_MOVE_PCT}%).")

    return "\n".join(parts)
