"""AI decision engine — contextual judgment on quantitative signals.

v8: Code computes the signal, AI adds judgment. Brief is ~800 chars.
AI does NOT compute probabilities, set position sizes, or enforce risk rules.
"""

import json
from .utils import (
    load_config, format_currency, iso_now, setup_logging,
    call_ai_with_fallback, load_json, DATA_DIR,
)
from .news_feed import NewsFeed, format_news_for_prompt
from .events import format_events_for_brief
from .signal_engine import get_signal_summary

logger = setup_logging("agent")


SYSTEM_PROMPT = """You are MonopolyTrader v8. You add contextual judgment to a quantitative trading signal.

The signal engine computes expected edge from historical price data. Your job:
- If today's context matches historical patterns: AGREE with the signal
- If news, geopolitics, or catalysts make history unreliable: OVERRIDE with explanation
- When holding a long position: evaluate whether to SELL or HOLD
- When holding a short position: evaluate whether to COVER or HOLD
- On bullish signals: confirm BUY or override to HOLD
- On bearish signals in trending markets: confirm SHORT or override to HOLD

You do NOT compute probabilities, set position sizes, or enforce risk rules. Code handles all of that.

Respond ONLY with valid JSON:
{"action":"BUY|SELL|SHORT|COVER|HOLD","reasoning":"<2-3 sentences>","override":false,"override_reason":null,"stop_pct":1.5,"exit_criteria":"Close if...","confidence":0.5}"""


def build_market_brief(
    market_data: dict,
    world: dict,
    portfolio: dict,
    news_feed: NewsFeed | None,
    config: dict,
    events: dict | None,
    signal: dict,
    sizing: dict,
    current_tags: dict,
    options_data: dict | None = None,
) -> str:
    """Build the compact ~800 char brief for Claude.

    Five sections: Narrative, Signal, News, Position, Reference.
    """
    current = market_data.get("current", {})
    daily = market_data.get("daily_indicators", {})
    regime = market_data.get("regime", {})
    ticker = config["ticker"]
    price = current.get("price", 0)

    parts = []

    # === MARKET NARRATIVE ===
    parts.append("=== MARKET NARRATIVE ===")
    parts.append(_build_narrative(market_data, world, events, news_feed))

    # === SIGNAL ===
    parts.append("")
    parts.append("=== SIGNAL ===")
    parts.append(get_signal_summary(current_tags, signal, sizing))

    # === NEWS ===
    parts.append("")
    parts.append("=== NEWS ===")
    if news_feed and news_feed.items:
        parts.append(format_news_for_prompt(news_feed, max_items=5))
    else:
        parts.append("No news available.")

    # === POSITION ===
    parts.append("")
    parts.append("=== POSITION ===")
    h = portfolio.get("holdings", {}).get(ticker, {})
    shares_held = h.get("shares", 0)

    from .risk_manager import load_daily_pnl, load_active_position
    daily_pnl = load_daily_pnl()

    if shares_held > 0.0001:
        avg_cost = h.get("avg_cost_basis", 0)
        pnl_pct = (
            ((price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
        )
        parts.append(
            f"LONG {shares_held:.4f} shares @ ${avg_cost:.2f} "
            f"(now ${price:.2f}, {pnl_pct:+.1f}%)"
        )
        pos = load_active_position()
        if pos:
            peak = pos.get("peak_price", 0)
            stop_pct = pos.get("trailing_stop_pct", 0.015)
            stop_price = peak * (1 - stop_pct)
            parts.append(
                f"Stop: ${stop_price:.2f} (peak ${peak:.2f}, {stop_pct:.1%})"
            )
            if pos.get("exit_criteria"):
                parts.append(f"Exit: {pos['exit_criteria']}")
    elif shares_held < -0.0001:
        avg_cost = h.get("avg_cost_basis", 0)
        pnl_pct = (
            ((avg_cost - price) / avg_cost * 100) if avg_cost > 0 else 0
        )
        parts.append(
            f"SHORT {abs(shares_held):.4f} shares @ ${avg_cost:.2f} "
            f"(now ${price:.2f}, {pnl_pct:+.1f}%)"
        )
        pos = load_active_position()
        if pos:
            trough = pos.get("peak_price", 0)
            stop_pct = pos.get("trailing_stop_pct", 0.015)
            stop_price = trough * (1 + stop_pct)
            parts.append(
                f"Stop: ${stop_price:.2f} (trough ${trough:.2f}, {stop_pct:.1%})"
            )
            if pos.get("exit_criteria"):
                parts.append(f"Exit: {pos['exit_criteria']}")
    else:
        parts.append("Flat (no position)")

    daily_loss_limit = config.get("v8_risk", {}).get("daily_loss_limit", 50)
    parts.append(
        f"Cash: {format_currency(portfolio.get('cash', 0))} "
        f"| Value: {format_currency(portfolio.get('total_value', 0))} "
        f"| Today P&L: ${daily_pnl.get('realized_pnl', 0):+.2f} "
        f"| Loss budget: ${daily_loss_limit:.0f}"
    )

    # === REFERENCE ===
    parts.append("")
    parts.append("=== REFERENCE ===")

    ref_parts = [f"{ticker} ${price:.2f}"]
    if daily.get("rsi_14") is not None:
        ref_parts.append(f"RSI:{daily['rsi_14']:.0f}")
    if daily.get("macd") is not None:
        ref_parts.append(f"MACD:{daily['macd']:.1f}")
    if daily.get("sma_50") is not None:
        ref_parts.append(f"SMA50:{daily['sma_50']:.0f}")
    if daily.get("adx") is not None:
        ref_parts.append(f"ADX:{daily['adx']:.0f}")
    parts.append(" ".join(ref_parts))

    # World reference line
    world_parts = []
    vix = regime.get("vix", 0)
    if vix:
        world_parts.append(f"VIX:{vix:.0f}")
    for sym in ["SPY", "QQQ"]:
        macro_data = world.get("macro", {}).get(sym, {})
        if macro_data:
            world_parts.append(f"{sym}:{macro_data.get('change_pct', 0):+.1f}%")
    oil = world.get("macro", {}).get("CL=F", {})
    if oil:
        world_parts.append(f"Oil:{oil.get('change_pct', 0):+.1f}%")

    directional = regime.get("directional", "?")
    vol_label = regime.get("volatility", "?")
    world_parts.append(f"| Regime: {directional}, {vol_label}")
    parts.append(" ".join(world_parts))

    # Events reference (compact)
    if events:
        event_text = _format_events_compact(events)
        if event_text:
            parts.append(event_text)

    return "\n".join(parts)


def _build_narrative(
    market_data: dict, world: dict,
    events: dict | None, news_feed,
) -> str:
    """Build 3-5 sentence market narrative from data (no AI call)."""
    current = market_data.get("current", {})
    regime = market_data.get("regime", {})
    price = current.get("price", 0)
    change_pct = current.get("change_pct", 0)
    vix = regime.get("vix", 20)

    sentences = []

    # Price action
    if abs(change_pct) > 2:
        sentences.append(
            f"TSLA is moving sharply today at ${price:.2f} ({change_pct:+.1f}%)."
        )
    elif abs(change_pct) > 0.5:
        direction = "higher" if change_pct > 0 else "lower"
        sentences.append(
            f"TSLA is trading {direction} at ${price:.2f} ({change_pct:+.1f}%)."
        )
    else:
        sentences.append(
            f"TSLA is flat at ${price:.2f} ({change_pct:+.1f}%)."
        )

    # Regime context
    directional = regime.get("directional", "unknown")
    trend = regime.get("trend", "unknown")
    if directional == "trending":
        sentences.append(f"Market is in a trending regime ({trend}).")
    elif directional == "range_bound":
        sentences.append("Market is range-bound.")

    # VIX context
    if vix > 30:
        sentences.append(
            f"Volatility is elevated (VIX {vix:.0f}) — wider price swings expected."
        )
    elif vix > 25:
        sentences.append(f"VIX is above average at {vix:.0f}.")

    # SPY context
    spy = world.get("macro", {}).get("SPY", {})
    if spy:
        spy_change = spy.get("change_pct", 0)
        if abs(spy_change) > 1:
            direction = "rallying" if spy_change > 0 else "selling off"
            sentences.append(f"Broader market {direction} (SPY {spy_change:+.1f}%).")

    # Top news headline
    if news_feed and hasattr(news_feed, "high_impact") and news_feed.high_impact:
        top = news_feed.high_impact[0]
        sentences.append(f"Top headline: {top.title[:80]}.")

    return " ".join(sentences[:5])


def _format_events_compact(events: dict) -> str:
    """One-line events summary for the reference section."""
    items = []
    for ev in events.get("macro_events", [])[:2]:
        items.append(f"{ev['event']} in {ev['hours_until']:.0f}h")
    earnings = events.get("tsla_earnings")
    if earnings and earnings.get("days_until", 999) <= 14:
        items.append(f"TSLA earnings {earnings['days_until']}d")
    if items:
        return "Events: " + ", ".join(items)
    return ""


def make_decision(
    market_data: dict,
    world: dict,
    portfolio: dict,
    news_feed: NewsFeed | None,
    config: dict,
    events: dict | None = None,
    signal: dict = None,
    sizing: dict = None,
    current_tags: dict = None,
    options_data: dict | None = None,
) -> dict:
    """Call Claude for contextual judgment on the signal.

    v8: Brief is ~800 chars. AI adds judgment, doesn't compute.
    """
    brief = build_market_brief(
        market_data, world, portfolio, news_feed,
        config, events, signal, sizing, current_tags, options_data,
    )

    # Add daily narrative + judgment scorecard to system prompt
    system = SYSTEM_PROMPT
    try:
        narrative = load_json(DATA_DIR / "daily_narrative.json", default={})
        if narrative and narrative.get("narrative"):
            system = (
                SYSTEM_PROMPT
                + "\n\n=== DAILY NARRATIVE ===\n"
                + narrative["narrative"][:500]
            )
    except Exception:
        pass

    try:
        from .judgment_scorecard import get_scorecard_for_brief
        scorecard_text = get_scorecard_for_brief()
        if scorecard_text:
            system += "\n\n" + scorecard_text
    except Exception:
        pass

    logger.info(f"Calling Claude (brief ~{len(brief)} chars)")

    raw = ""
    try:
        raw, model_used = call_ai_with_fallback(
            system=system,
            user=brief,
            max_tokens=500,
            config=config,
        )

        # Strip code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        decision = json.loads(raw)

        # Validate
        action = decision.get("action", "HOLD")
        if action not in ("BUY", "SELL", "SHORT", "COVER", "HOLD"):
            decision["action"] = "HOLD"

        decision["_model_version"] = model_used
        decision["_signal_score"] = signal.get("score", 0) if signal else 0
        decision["_brief_length"] = len(brief)

        logger.info(
            f"Decision: {decision.get('action')} "
            f"(override={decision.get('override', False)}, "
            f"confidence={decision.get('confidence', 0):.2f})"
        )

        return decision

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw: {raw[:300]}")
        return _fallback_decision(f"JSON parse error: {e}")
    except Exception as e:
        logger.error(f"Agent decision failed: {e}")
        return _fallback_decision(f"Error: {e}")


def _fallback_decision(reason: str) -> dict:
    """Safe HOLD decision on any error."""
    return {
        "action": "HOLD",
        "reasoning": reason,
        "override": False,
        "override_reason": None,
        "stop_pct": 1.5,
        "exit_criteria": "",
        "confidence": 0,
    }
