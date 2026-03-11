"""Mechanical trade tagging — objective market conditions at trade time.

Every trade gets tagged with independent market conditions.
Tags are computed from existing data — no LLM, no subjectivity.
Used by thesis_builder.py to aggregate performance stats.
"""

from datetime import datetime
from zoneinfo import ZoneInfo


def compute_tags(
    market_data: dict,
    world: dict,
    portfolio: dict,
    config: dict,
    events: dict,
    action: str,
    news_feed=None,
    options_data: dict | None = None,
) -> dict:
    """Compute objective tags from market data at trade time.

    Returns a dict of tags, each with a string value.
    All thresholds are explicit and mechanical.

    Args:
        news_feed: Optional NewsFeed object for news_catalyst tag.
        options_data: Optional options snapshot for options_sentiment tag.

    v6.1: Added options_sentiment and intraday_regime tags.
    """
    current = market_data.get("current", {})
    daily = market_data.get("daily_indicators", {})
    intraday = market_data.get("intraday_indicators") or {}
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

    # Event proximity — actually use the events parameter
    has_event_24h = False
    has_event_72h = False
    if events:
        for ev in events.get("macro_events", []):
            hours_until = ev.get("hours_until", 999)
            if hours_until <= 24:
                has_event_24h = True
            elif hours_until <= 72:
                has_event_72h = True
        earnings = events.get("tsla_earnings")
        if earnings:
            days = earnings.get("days_until", 999)
            if days <= 3:
                has_event_24h = True
            elif days <= 7:
                has_event_72h = True

    # Time of day — Eastern time
    time_tag = _compute_time_of_day()

    # Intraday/daily RSI divergence
    divergence_tag = _compute_intraday_daily_divergence(
        intraday_rsi=intraday.get("rsi_14"),
        daily_rsi=daily.get("rsi_14"),
    )

    # News catalyst
    catalyst_tag = _compute_news_catalyst(news_feed)

    # Options sentiment (Blindspot #4)
    options_tag = _compute_options_sentiment(options_data)

    # Intraday regime (Blindspot #5)
    intraday_regime_tag = _compute_intraday_regime(regime)

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
        "time_of_day": time_tag,
        "intraday_daily_divergence": divergence_tag,
        "news_catalyst": catalyst_tag,
        "options_sentiment": options_tag,
        "intraday_regime": intraday_regime_tag,
    }


def _compute_time_of_day() -> str:
    """Classify current Eastern time into trading session buckets."""
    try:
        et_now = datetime.now(ZoneInfo("America/New_York"))
        hour = et_now.hour
        minute = et_now.minute
        total_minutes = hour * 60 + minute

        # 9:30 - 10:30 ET
        if total_minutes < 630:  # before 10:30
            return "morning_open"
        # 10:30 - 14:00 ET
        elif total_minutes < 840:  # before 14:00
            return "midday"
        # 14:00 - 15:30 ET
        elif total_minutes < 930:  # before 15:30
            return "afternoon"
        # 15:30 - 16:00 ET
        else:
            return "close"
    except Exception:
        return "unknown"


def _compute_intraday_daily_divergence(
    intraday_rsi: float | None,
    daily_rsi: float | None,
) -> str:
    """Compare intraday RSI to daily RSI and classify divergence."""
    if intraday_rsi is None or daily_rsi is None:
        return "no_data"

    diff = abs(intraday_rsi - daily_rsi)
    if diff <= 15:
        return "aligned"
    elif diff <= 30:
        return "mild_divergence"
    else:
        return "strong_divergence"


def _compute_options_sentiment(options_data: dict | None) -> str:
    """Classify options sentiment from put/call ratio.

    v6.1 Blindspot #4: Options data tagging.
    """
    if not options_data:
        return "unavailable"

    pc_ratio = options_data.get("put_call_ratio")
    if pc_ratio is None:
        return "unavailable"

    if pc_ratio > 1.2:
        return "put_heavy"
    elif pc_ratio < 0.8:
        return "call_heavy"
    else:
        return "balanced"


def _compute_intraday_regime(regime: dict) -> str:
    """Classify intraday directional strength from regime data.

    v6.1 Blindspot #5: Uses intraday_directional field added by
    classify_regime() in market_data.py.
    """
    intraday_dir = regime.get("intraday_directional")
    if intraday_dir:
        return intraday_dir

    return "unavailable"


def _compute_news_catalyst(news_feed) -> str:
    """Extract the highest-relevance catalyst type from the news feed."""
    if news_feed is None:
        return "none"

    items = getattr(news_feed, "items", [])
    if not items:
        return "none"

    # Find the highest-relevance item with a known catalyst type
    best_catalyst = "none"
    best_relevance = 0.0
    for item in items:
        relevance = getattr(item, "relevance", 0.0)
        catalyst = getattr(item, "catalyst_type", "unknown")
        if relevance > best_relevance and catalyst != "unknown":
            best_relevance = relevance
            best_catalyst = catalyst

    # Only report if relevance is meaningful
    if best_relevance < 0.3:
        return "none"

    return best_catalyst
