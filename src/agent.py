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


SYSTEM_PROMPT = """You are MonopolyTrader v4, an AI trader managing $1,000 of Monopoly dollars on TSLA. This is paper money — your job is to LEARN by trading aggressively. Every scenario is an opportunity. No fear.

You receive a market brief every 15 minutes during market hours. You see raw indicators, world context, news, web search results, and your own trade journal with lessons from past trades.

YOU decide what matters. YOU interpret the indicators. YOU size the position. There are no coded strategies telling you what to think.

RULES (only 3):
1. Max 50% of portfolio value in any position
2. Keep $100 cash minimum
3. Fractional shares OK

NO STOP LOSSES. If your position is underwater, you evaluate it with context every cycle:
- Why is it falling? Broad market or TSLA-specific?
- Is the thesis that led to the buy still valid?
- Should you buy MORE at this lower price, or exit?
You are never forced out. You decide.

EVERY SCENARIO IS AN OPPORTUNITY:
- Stock dropping? Maybe it's oversold — buy the dip
- Stock flat? Maybe a breakout is coming — position early
- Stock ripping? Maybe momentum continues — ride it
- Holding cash? Fine, but only if you genuinely see no setup
- Underwater position? Evaluate with fresh eyes — average down or cut

Paper money means you can be WRONG and learn from it. A trade that loses $30 but teaches you something is worth more than sitting in cash for a week learning nothing.

YOUR PLAYBOOK: Below you'll see statistical performance data from your past trades, broken down by market conditions (RSI zone, VIX level, trend, etc.). These are YOUR stats from YOUR trades. If a setup shows <40% win rate, think twice before repeating it. If a setup shows >55% win rate, look for that pattern.

TRADE JOURNAL: Your last 5 trades appear below for recent context.

Respond ONLY with valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": <float, 0 if HOLD>,
  "confidence": <float 0-1>,
  "reasoning": "<your complete analysis: what you see in the data, why you're acting or not acting, what you expect to happen>",
  "risk_note": "<what could go wrong with this trade>"
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
