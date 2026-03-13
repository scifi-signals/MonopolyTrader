"""v8 Analyst — nightly narrative + optional flash updates.

Produces a plain-text narrative about what happened today, why, and what
the signals show. Replaces the structured MID with 15 fields.

v8: One Sonnet call per night. Output is plain text, not structured JSON.
"""

import json
from datetime import datetime, timezone, timedelta

from .utils import (
    load_config, load_json, save_json, iso_now, setup_logging,
    call_ai_with_fallback, DATA_DIR,
)
from .journal import get_entries_since
from .market_data import get_current_price
from .news_feed import fetch_news_feed, format_news_for_prompt

logger = setup_logging("analyst")

NARRATIVE_PATH = DATA_DIR / "daily_narrative.json"
MID_PATH = DATA_DIR / "market_intelligence.json"


NIGHTLY_PROMPT = """You are the MonopolyTrader v8 Analyst. Write a market narrative for the trading agent.

The agent uses a quantitative signal engine that computes expected edge from historical price patterns. Your narrative provides the CONTEXT that helps the agent judge whether those patterns apply tomorrow.

Write 3-5 paragraphs covering:
1. What happened today — price action, key moves, volume
2. Why — catalysts, news drivers, macro/geopolitical context
3. Key levels — support/resistance from today's action
4. Active catalysts — what's still moving the stock going forward
5. What the signals show — note any strong or surprising signal weights

Write in plain English. No JSON. Be specific with numbers and price levels. Focus on information that could cause tomorrow's price action to differ from historical patterns.

Keep it under 500 words."""


def load_mid() -> dict:
    """Load the Market Intelligence Document (legacy compatibility).

    v8 keeps this for tags.py _compute_regime_age() compatibility.
    The MID is no longer updated nightly — use daily_narrative.json instead.
    """
    return load_json(MID_PATH, default={})


def run_nightly_update():
    """After market close: generate a narrative about today.

    v8: Plain text narrative instead of structured MID with 15 fields.
    Saved to data/daily_narrative.json, injected into system prompt next day.
    """
    config = load_config()
    analyst_model = config.get(
        "analyst_model",
        config.get("anthropic_model", "claude-sonnet-4-20250514"),
    )

    # Get recent closed trades (last 7 days for context)
    seven_days_ago = (
        datetime.now(timezone.utc) - timedelta(days=7)
    ).strftime("%Y-%m-%d")
    recent_entries = get_entries_since(seven_days_ago)
    closed_trades = [e for e in recent_entries if e.get("realized_pnl") is not None]

    trades_text = "No closed trades in last 7 days."
    if closed_trades:
        lines = []
        for e in closed_trades[-10:]:
            pnl = e.get("realized_pnl", 0)
            lines.append(
                f"  {e.get('action')} {e.get('shares', 0):.2f} "
                f"@ ${e.get('price', 0):.2f} -> ${e.get('close_price', 0):.2f}, "
                f"P&L: ${pnl:+.2f}"
            )
        trades_text = (
            f"{len(closed_trades)} closed trades:\n" + "\n".join(lines)
        )

    # Signal registry summary — top weights by magnitude
    signal_text = "No signal data available."
    try:
        from .signal_engine import load_registry
        registry = load_registry()
        signals = registry.get("signals", {})
        if signals:
            sorted_sigs = sorted(
                signals.items(),
                key=lambda x: abs(x[1].get("weight", 0)),
                reverse=True,
            )
            lines = [
                f"Signal registry: {len(signals)} signals from "
                f"{registry.get('resolved_outcomes', 0)} outcomes"
            ]
            for key, stats in sorted_sigs[:10]:
                lines.append(
                    f"  {key}: edge={stats['expected_edge']:+.4f} "
                    f"(n={stats['n']}, {stats['confidence']})"
                )
            signal_text = "\n".join(lines)
    except Exception:
        pass

    # Daily P&L
    pnl_text = ""
    try:
        from .risk_manager import load_daily_pnl
        daily = load_daily_pnl()
        pnl_text = (
            f"Today's P&L: ${daily.get('realized_pnl', 0):+.2f} "
            f"({daily.get('trades_today', 0)} trades)"
        )
    except Exception:
        pass

    # Current price
    try:
        current = get_current_price(config["ticker"])
        price_text = f"TSLA: ${current['price']} ({current['change_pct']:+.2f}%)"
    except Exception:
        price_text = "TSLA price unavailable"

    # News
    try:
        news_feed = fetch_news_feed(config["ticker"])
        news_text = format_news_for_prompt(news_feed, max_items=8)
    except Exception:
        news_text = "News unavailable."

    user_prompt = f"""=== TODAY'S MARKET DATA ===
{price_text}
{pnl_text}

=== TRADES ===
{trades_text}

=== SIGNAL REGISTRY (top signals by weight) ===
{signal_text}

=== NEWS ===
{news_text}

Write the market narrative for tonight."""

    try:
        raw, model_used = call_ai_with_fallback(
            system=NIGHTLY_PROMPT,
            user=user_prompt,
            max_tokens=800,
            config=config,
            model=analyst_model,
        )

        narrative = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "narrative": raw.strip(),
            "generated_at": iso_now(),
            "model": model_used,
        }

        save_json(NARRATIVE_PATH, narrative)
        logger.info(
            f"Nightly narrative generated "
            f"({len(raw)} chars, model={model_used})"
        )
        return narrative

    except Exception as e:
        logger.error(f"Nightly analyst failed: {e}")
        fallback = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "narrative": f"Nightly analysis unavailable. {price_text}",
            "generated_at": iso_now(),
            "model": "fallback",
        }
        save_json(NARRATIVE_PATH, fallback)
        return fallback


# --- Legacy compatibility stubs ---
# Kept so tags.py _compute_regime_age() and any other references don't crash.
# These do nothing useful in v8.


def format_mid_for_system_prompt(mid: dict) -> str:
    """Legacy stub. v8 uses daily_narrative.json instead."""
    return ""


def format_briefing_for_prompt(briefing: dict) -> str:
    """Legacy stub. v8 doesn't use pre-market briefings."""
    return ""


def run_pre_market():
    """Legacy stub. v8 uses nightly narrative + live cycle data."""
    logger.info("Pre-market briefing skipped (v8 — not used)")
    return {}
