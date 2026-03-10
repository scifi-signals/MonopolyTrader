"""AI decision engine — Claude IS the trading brain.

v4: No coded strategies. No signal aggregation. No thesis system.
Raw market data + world context + trade journal → Claude reasons → Claude decides.
"""

import json
from .utils import (
    load_config, format_currency, iso_now, setup_logging,
    call_ai_with_fallback,
)
from .journal import get_recent_entries, format_journal_for_brief
from .news_feed import NewsFeed, format_news_for_prompt
from .web_search import format_search_results
from .events import format_events_for_brief

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

TRADE JOURNAL: Your last 10 trades and their lessons appear below. These are YOUR lessons from YOUR experience. Use them. If you lost money buying into high RSI, don't do it again. If you made money buying VIX spikes, look for that setup.

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

    # === Trade Journal ===
    parts.append("")
    parts.append("=== YOUR TRADE JOURNAL (last 10) ===")
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

    One brief, one decision. No multi-step, no analyst/trader split.
    """
    brief = build_market_brief(
        market_data, world, portfolio, news_feed,
        web_results, journal_entries, config, events,
    )

    logger.info(f"Calling Claude for decision (brief ~{len(brief)} chars)")

    try:
        raw, model_used = call_ai_with_fallback(
            system=SYSTEM_PROMPT,
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
