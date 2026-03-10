"""AI decision engine — Claude IS the trading brain.

v4: No coded strategies. No signal aggregation. No thesis system.
Raw market data + world context + trade journal → Claude reasons → Claude decides.
"""

import json
from datetime import datetime
from .utils import (
    load_config, format_currency, iso_now, setup_logging,
    call_ai_with_fallback,
)
from .journal import get_recent_entries, format_journal_for_brief
from .news_feed import NewsFeed, format_news_for_prompt
from .web_search import format_search_results
from .events import format_events_for_brief
from .thesis_builder import format_playbook_for_brief
from .analyst import load_mid, format_mid_for_system_prompt, format_briefing_for_prompt

logger = setup_logging("agent")


SYSTEM_PROMPT = """You are MonopolyTrader v5, an AI trader managing $1,000 of Monopoly dollars on TSLA.

You receive a market brief every 15 minutes during market hours. You see raw indicators, world context, news, your trade journal with lessons, your playbook stats, and the Market Intelligence Document with your persistent thesis.

## POSITION LIMITS (enforced by code)
1. Max 50% of portfolio value in any position
2. Keep $100 cash minimum
3. Fractional shares OK
4. No stop losses — you evaluate every position with context each cycle

## YOUR JOB: DEVELOP AND TEST TRADING STRATEGIES

You are not just executing trades — you are a strategist. Your goal is to figure out what works in each type of market environment and build a repertoire of strategies through deliberate experimentation.

Each cycle, think about:
- What type of market am I in right now? (trending, range-bound, volatile, pre-event, etc.)
- What strategy makes sense for THIS environment?
- What specific hypothesis am I testing with this trade?
- What would prove my hypothesis right or wrong?

Your PLAYBOOK section shows your actual performance broken down by market conditions. Your TRADE JOURNAL shows your recent trades with lessons. Your TRADING STATS show your overall track record. Study these — they are YOUR data from YOUR trades. They tell you what's working and what isn't.

When something isn't working, adapt. Try a different approach. When something IS working, lean into it. This is how you learn.

## TRADING COSTS
Every trade has a real cost in slippage. The SPREAD COST section shows your round-trip breakeven. Factor this into every decision — a strategy that targets moves smaller than the spread cost is a losing strategy by design.

## STRATEGY FIELD
In your response, include a "strategy" field naming the strategy you're employing (e.g., "mean_reversion_oversold", "breakout_above_resistance", "pre_catalyst_positioning", "cash_preservation", etc.). This helps you track which strategies work over time.

Respond ONLY with valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": <float, 0 if HOLD>,
  "confidence": <float 0-1>,
  "strategy": "<name of the strategy you're using this cycle>",
  "reasoning": "<your analysis: what market environment you see, what strategy you're applying, what you expect to happen, what would invalidate your thesis>",
  "risk_note": "<what could go wrong>"
}"""


def build_market_brief(
    market_data: dict,
    world: dict,
    portfolio: dict,
    news_feed: NewsFeed | None,
    web_results: list[dict],
    journal_entries: list[dict],
    config: dict,
    events: dict | None = None,
) -> str:
    """Build the complete market brief for Claude.

    This is the single prompt that contains everything Claude needs
    to make a trading decision.
    """
    current = market_data.get("current", {})
    daily = market_data.get("daily_indicators", {})
    intraday = market_data.get("intraday_indicators") or {}
    recent = market_data.get("recent_days", [])
    regime = market_data.get("regime", {})
    ticker = config["ticker"]

    parts = []

    # === TSLA Price & Indicators ===
    parts.append(f"=== {ticker} ===")
    parts.append(f"Price: ${current.get('price', 'N/A')} ({current.get('change_pct', 0):+.2f}%)")
    parts.append(f"Volume: {current.get('volume', 0):,}")
    parts.append("")

    parts.append("Daily Indicators:")
    indicator_keys = [
        "rsi_14", "sma_20", "sma_50",
        "macd", "macd_signal", "macd_histogram", "macd_crossover",
        "bollinger_upper", "bollinger_lower", "bollinger_mid",
        "atr", "adx", "adx_pos", "adx_neg",
        "ema_12", "ema_26", "volume_sma_20", "obv",
    ]
    for key in indicator_keys:
        val = daily.get(key)
        if val is not None:
            parts.append(f"  {key}: {val}")

    if intraday and intraday.get("current_price"):
        parts.append("")
        parts.append("Intraday (5min) Indicators:")
        for key in ["rsi_14", "sma_20", "macd_crossover", "bollinger_upper", "bollinger_lower"]:
            val = intraday.get(key)
            if val is not None:
                parts.append(f"  {key}: {val}")

    if recent:
        parts.append("")
        parts.append("Last 5 Days:")
        for day in recent:
            parts.append(
                f"  {day['date']}: O={day['open']} H={day['high']} "
                f"L={day['low']} C={day['close']} V={day['volume']:,}"
            )

    # === World Context ===
    parts.append("")
    parts.append("=== WORLD ===")

    macro = world.get("macro", {})
    if macro:
        parts.append("Macro:")
        for sym, data in macro.items():
            name = data.get("name", sym)
            parts.append(f"  {name} ({sym}): {data['price']} ({data['change_pct']:+.2f}%)")

    peers = world.get("ev_peers", {})
    if peers:
        parts.append("EV Peers:")
        for sym, data in peers.items():
            parts.append(f"  {sym}: ${data['price']} ({data['change_pct']:+.2f}%)")

    if regime:
        parts.append("")
        parts.append(f"Regime: trend={regime.get('trend','?')} directional={regime.get('directional','?')} "
                     f"volatility={regime.get('volatility','?')} VIX={regime.get('vix',0):.1f} "
                     f"ADX={regime.get('adx',0):.1f}")

    # === News ===
    parts.append("")
    parts.append("=== NEWS ===")
    if news_feed and news_feed.items:
        parts.append(format_news_for_prompt(news_feed, max_items=10))
    else:
        parts.append("No news available.")

    # === Upcoming Events ===
    if events:
        parts.append("")
        parts.append("=== UPCOMING EVENTS ===")
        parts.append(format_events_for_brief(events))

    # === Web Search ===
    if web_results:
        parts.append("")
        parts.append("=== WEB SEARCH (last 24h) ===")
        parts.append(format_search_results(web_results))

    # === Today's Briefing ===
    try:
        from .utils import load_json, DATA_DIR
        briefing = load_json(DATA_DIR / "daily_briefing.json", default={})
        if briefing:
            parts.append("")
            parts.append("=== TODAY'S BRIEFING ===")
            briefing_text = format_briefing_for_prompt(briefing)
            # Check staleness
            import os
            briefing_path = DATA_DIR / "daily_briefing.json"
            if briefing_path.exists():
                mtime = os.path.getmtime(briefing_path)
                age_hours = (datetime.now().timestamp() - mtime) / 3600
                if age_hours > 6:
                    parts.append(f"[STALE — generated {age_hours:.0f}h ago]")
            parts.append(briefing_text)
    except Exception:
        pass

    # === Options Flow ===
    try:
        from .market_data import get_options_snapshot
        options = get_options_snapshot(ticker)
        if options:
            parts.append("")
            parts.append("=== OPTIONS FLOW ===")
            parts.append(f"  Put/Call Ratio: {options.get('put_call_ratio', 'N/A')}")
            parts.append(f"  Max Pain: ${options.get('max_pain', 'N/A')}")
            parts.append(f"  Unusual Volume: {'YES' if options.get('unusual_volume') else 'No'}")
            parts.append(f"  Nearest Expiry: {options.get('nearest_expiry', 'N/A')}")
    except Exception:
        pass

    # === Analyst Consensus ===
    try:
        from .market_data import get_analyst_consensus
        analysts = get_analyst_consensus(ticker)
        if analysts:
            parts.append("")
            parts.append("=== ANALYST CONSENSUS ===")
            buys = analysts.get('strong_buy', 0) + analysts.get('buy', 0)
            holds = analysts.get('hold', 0)
            sells = analysts.get('sell', 0) + analysts.get('strong_sell', 0)
            parts.append(f"  Buy: {buys}, Hold: {holds}, Sell: {sells}")
            if analysts.get('target_mean'):
                vs_current = ""
                if current.get("price") and analysts["target_mean"] > 0:
                    diff = ((analysts["target_mean"] - current["price"]) / current["price"]) * 100
                    vs_current = f" ({diff:+.1f}% vs current)"
                parts.append(f"  Mean Target: ${analysts['target_mean']}{vs_current}")
                parts.append(f"  Range: ${analysts.get('target_low', '?')} - ${analysts.get('target_high', '?')}")
    except Exception:
        pass

    # === Institutional ===
    try:
        from .market_data import get_institutional_data
        inst = get_institutional_data(ticker)
        if inst:
            parts.append("")
            parts.append("=== INSTITUTIONAL ===")
            if inst.get("institutional_pct"):
                parts.append(f"  Institutional Ownership: {inst['institutional_pct']}%")
            if inst.get("short_interest_pct"):
                parts.append(f"  Short Interest: {inst['short_interest_pct']}%")
            if inst.get("top_holders"):
                parts.append(f"  Top Holders: {', '.join(inst['top_holders'][:3])}")
    except Exception:
        pass

    # === Portfolio ===
    parts.append("")
    parts.append("=== PORTFOLIO ===")
    parts.append(f"Cash: {format_currency(portfolio.get('cash', 0))}")
    parts.append(f"Total Value: {format_currency(portfolio.get('total_value', 0))}")
    parts.append(f"P&L: {format_currency(portfolio.get('total_pnl', 0))} ({portfolio.get('total_pnl_pct', 0):+.2f}%)")
    parts.append(f"Trades: {portfolio.get('total_trades', 0)} (W:{portfolio.get('winning_trades', 0)} L:{portfolio.get('losing_trades', 0)})")

    holdings = portfolio.get("holdings", {})
    h = holdings.get(ticker, {})
    if h.get("shares", 0) > 0:
        parts.append(
            f"Position: {h['shares']:.4f} {ticker} @ avg ${h['avg_cost_basis']:.2f} "
            f"(current ${h['current_price']:.2f}, unrealized {format_currency(h['unrealized_pnl'])})"
        )
        pnl_pct = ((h['current_price'] - h['avg_cost_basis']) / h['avg_cost_basis'] * 100) if h['avg_cost_basis'] > 0 else 0
        if pnl_pct < 0:
            parts.append(f"  >>> UNDERWATER: {pnl_pct:.1f}% loss. Evaluate: exit, hold, or buy more? <<<")
    else:
        parts.append("Position: FLAT (all cash)")

    # === Spread Cost (hurdle rate) ===
    slippage_pct = config["risk_params"].get("slippage_per_side_pct", 0.0005)
    round_trip_pct = slippage_pct * 2 * 100  # both sides, as percentage
    price = current.get("price", 400)
    parts.append("")
    parts.append("=== SPREAD COST ===")
    parts.append(f"Round-trip slippage: {round_trip_pct:.2f}% (${price * slippage_pct * 2:.2f} per share)")
    parts.append(f"Breakeven move: >{round_trip_pct:.2f}% (>${price * slippage_pct * 2:.2f}/share)")
    parts.append(f"Minimum target move (2x breakeven): >{round_trip_pct * 2:.2f}% (>${price * slippage_pct * 4:.2f}/share)")

    # === Trading Stats (self-awareness) ===
    try:
        from .portfolio import load_transactions
        all_txns = load_transactions()
        all_sells = [t for t in all_txns if t["action"] == "SELL"]
        if all_sells:
            parts.append("")
            parts.append("=== YOUR TRADING STATS ===")
            total_sells = len(all_sells)
            winners = sum(1 for t in all_sells if (t.get("realized_pnl", 0) or 0) > 0)
            losers = sum(1 for t in all_sells if (t.get("realized_pnl", 0) or 0) < 0)
            win_rate = (winners / total_sells * 100) if total_sells > 0 else 0
            total_realized = sum(t.get("realized_pnl", 0) or 0 for t in all_sells)
            avg_win = 0
            avg_loss = 0
            winning_pnls = [t.get("realized_pnl", 0) for t in all_sells if (t.get("realized_pnl", 0) or 0) > 0]
            losing_pnls = [t.get("realized_pnl", 0) for t in all_sells if (t.get("realized_pnl", 0) or 0) < 0]
            if winning_pnls:
                avg_win = sum(winning_pnls) / len(winning_pnls)
            if losing_pnls:
                avg_loss = sum(losing_pnls) / len(losing_pnls)
            parts.append(f"Closed trades: {total_sells} (W:{winners} L:{losers})")
            parts.append(f"Win rate: {win_rate:.0f}%")
            parts.append(f"Total realized P&L: ${total_realized:.2f}")
            parts.append(f"Avg win: ${avg_win:.2f} | Avg loss: ${avg_loss:.2f}")

            # Recent streak
            consecutive_losses = 0
            for t in reversed(all_sells):
                if (t.get("realized_pnl", 0) or 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            if consecutive_losses > 0:
                streak_pnl = sum(t.get("realized_pnl", 0) or 0 for t in all_sells[-consecutive_losses:])
                parts.append(f"Current streak: {consecutive_losses} consecutive losses (${streak_pnl:.2f})")

            # Today's activity
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_txns = [t for t in all_txns if t["timestamp"].startswith(today)]
            today_sells = [t for t in today_txns if t["action"] == "SELL"]
            today_buys = [t for t in today_txns if t["action"] == "BUY"]
            if today_txns:
                today_pnl = sum(t.get("realized_pnl", 0) or 0 for t in today_sells)
                parts.append(f"Today: {len(today_buys)} buys, {len(today_sells)} sells, P&L: ${today_pnl:.2f}")
    except Exception:
        pass

    # Position limits for Claude
    max_position = portfolio.get("total_value", 1000) * config["risk_params"].get("max_position_pct", 0.50)
    current_holding_value = h.get("shares", 0) * h.get("current_price", 0)
    available_for_position = max_position - current_holding_value
    max_buy_value = min(available_for_position, portfolio.get("cash", 0) - config["risk_params"].get("min_cash_reserve", 100))
    if max_buy_value > 0 and current.get("price", 0) > 0:
        max_shares = max_buy_value / current["price"]
        parts.append(f"Max BUY: {max_shares:.4f} shares (${max_buy_value:.2f})")
    elif h.get("shares", 0) > 0:
        parts.append(f"Can SELL up to {h['shares']:.4f} shares")

    # === Playbook (learning stats) ===
    parts.append("")
    parts.append("=== YOUR PLAYBOOK ===")
    try:
        from .utils import load_json, DATA_DIR
        ledger = load_json(DATA_DIR / "thesis_ledger.json", default={})
        parts.append(format_playbook_for_brief(ledger))
    except Exception:
        parts.append("Playbook not yet available.")

    # === Trade Journal ===
    parts.append("")
    parts.append("=== YOUR TRADE JOURNAL (last 5) ===")
    parts.append(format_journal_for_brief(journal_entries))

    return "\n".join(parts)


def make_decision(
    market_data: dict,
    world: dict,
    portfolio: dict,
    news_feed: NewsFeed | None,
    web_results: list[dict],
    journal_entries: list[dict],
    config: dict,
    events: dict | None = None,
) -> dict:
    """Call Claude to make a trading decision.

    v5: System prompt includes Market Intelligence Document.
    User prompt includes daily briefing + enhanced market data.
    """
    brief = build_market_brief(
        market_data, world, portfolio, news_feed,
        web_results, journal_entries, config, events,
    )

    # Build system prompt with Market Intelligence Document
    system = SYSTEM_PROMPT
    try:
        mid = load_mid()
        mid_text = format_mid_for_system_prompt(mid)
        if mid_text:
            system = SYSTEM_PROMPT + "\n\n" + mid_text
    except Exception as e:
        logger.warning(f"Failed to load MID for system prompt: {e}")

    logger.info(f"Calling Claude for decision (brief ~{len(brief)} chars, system ~{len(system)} chars)")

    try:
        raw, model_used = call_ai_with_fallback(
            system=system,
            user=brief,
            max_tokens=800,
            config=config,
        )

        # Strip code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        decision = json.loads(raw)

        # Validate required fields
        action = decision.get("action", "HOLD")
        if action not in ("BUY", "SELL", "HOLD"):
            logger.warning(f"Invalid action '{action}', defaulting to HOLD")
            decision["action"] = "HOLD"
            decision["shares"] = 0

        if action == "HOLD":
            decision["shares"] = 0

        decision["_model_version"] = model_used

        logger.info(
            f"Decision: {decision.get('action')} "
            f"{decision.get('shares', 0):.4f} shares, "
            f"confidence={decision.get('confidence', 0):.2f}"
        )

        return decision

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response: {e}\nRaw: {raw[:500]}")
        return {
            "action": "HOLD", "shares": 0, "confidence": 0,
            "reasoning": f"JSON parse error: {e}",
            "risk_note": "Defaulting to HOLD due to parse error",
        }
    except Exception as e:
        logger.error(f"Agent decision failed: {e}")
        return {
            "action": "HOLD", "shares": 0, "confidence": 0,
            "reasoning": f"Error: {e}",
            "risk_note": f"Agent error: {e}",
        }
