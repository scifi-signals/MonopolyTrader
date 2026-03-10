"""Mechanical trade tagging — objective market conditions at trade time.

Every trade gets tagged with 8 independent market conditions.
Tags are computed from existing data — no LLM, no subjectivity.
Used by thesis_builder.py to aggregate performance stats.
"""


def compute_tags(
    market_data: dict,
    world: dict,
    portfolio: dict,
    config: dict,
    events: dict,
    action: str,
) -> dict:
    """Compute objective tags from market data at trade time.

    Returns a dict of 8 tags, each with a string value.
    All thresholds are explicit and mechanical.
    """
    current = market_data.get("current", {})
    daily = market_data.get("daily_indicators", {})
    regime = market_data.get("regime", {})
    ticker = config["ticker"]
    holdings = portfolio.get("holdings", {}).get(ticker, {})

    price = current.get("price", 0)
    rsi = daily.get("rsi_14", 50)
    vix = regime.get("vix", 20)
    sma50 = daily.get("sma_50", 0)
    macd_cross = daily.get("macd_crossover", "none")
    directional = regime.get("directional", "unknown")

    # SPY change from world data
    spy_change = 0.0
    spy_data = world.get("macro", {}).get("SPY", {})
    if spy_data:
        spy_change = spy_data.get("change_pct", 0.0)

    # Position state — distinguishes adding to winners vs losers
    held_shares = holdings.get("shares", 0)
    avg_cost = holdings.get("avg_cost_basis", 0)

    if held_shares == 0 or held_shares < 0.0001:
        if action == "BUY":
            pos_state = "opening_new"
        else:
            pos_state = "flat"
    elif action == "BUY":
        # Adding to existing position — is it winning or losing?
        if price > avg_cost:
            pos_state = "adding_to_winner"
        else:
            pos_state = "adding_to_loser"
    elif action == "SELL":
        if price > avg_cost:
            pos_state = "taking_profit"
        else:
            pos_state = "cutting_loss"
    else:
        pos_state = "holding"

    # Event proximity
    has_event_24h = False
    has_event_72h = False
    if events:
        for ev in events.get("macro_events", []):
            if ev.get("hours_until", 999) <= 24:
                has_event_24h = True
            elif ev.get("hours_until", 999) <= 72:
                has_event_72h = True
        earnings = events.get("tsla_earnings")
        if earnings:
            days = earnings.get("days_until", 999)
            if days <= 3:
                has_event_24h = True
            elif days <= 7:
                has_event_72h = True

    return {
        "rsi_zone": (
            "oversold" if rsi < 30
            else "overbought" if rsi > 70
            else "neutral"
        ),
        "trend": "above_sma50" if (sma50 > 0 and price > sma50) else "below_sma50",
        "volatility": (
            "low_vix" if vix < 18
            else "high_vix" if vix > 25
            else "normal_vix"
        ),
        "regime": directional if directional in ("trending", "range_bound") else "unknown",
        "macd": (
            "bullish_cross" if macd_cross in ("bullish_crossover", "bullish_cross")
            else "bearish_cross" if macd_cross in ("bearish_crossover", "bearish_cross")
            else "neutral"
        ),
        "market_context": (
            "spy_up" if spy_change > 0.3
            else "spy_down" if spy_change < -0.3
            else "spy_flat"
        ),
        "position_state": pos_state,
        "event_proximity": (
            "pre_event_24h" if has_event_24h
            else "pre_event_72h" if has_event_72h
            else "no_event"
        ),
    }
