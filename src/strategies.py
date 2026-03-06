"""Strategy engine — signal generators with dynamic weighting."""

from dataclasses import dataclass, field
from .utils import load_config, setup_logging
from .knowledge_base import get_strategy_scores

logger = setup_logging("strategies")


@dataclass
class StrategySignal:
    action: str          # "BUY", "SELL", "HOLD"
    confidence: float    # 0.0 to 1.0 (raw signal strength)
    weight: float        # from strategy_scores.json (earned trust)
    strategy: str        # strategy name
    reasoning: str       # human-readable explanation
    signals: dict = field(default_factory=dict)  # raw indicator values used


def _hold(strategy: str, reasoning: str, weight: float, signals: dict = None) -> StrategySignal:
    return StrategySignal("HOLD", 0.0, weight, strategy, reasoning, signals or {})


# --- Individual Strategies ---

def momentum_signal(indicators: dict, weight: float) -> StrategySignal:
    """Buy when price > SMA20 > SMA50, RSI 50-70; Sell when price < SMA20, RSI > 75."""
    name = "momentum"
    price = indicators.get("current_price")
    sma20 = indicators.get("sma_20")
    sma50 = indicators.get("sma_50")
    rsi = indicators.get("rsi_14")

    if any(v is None for v in [price, sma20, sma50, rsi]):
        return _hold(name, "Insufficient indicator data", weight)

    sigs = {"price": price, "sma_20": sma20, "sma_50": sma50, "rsi_14": rsi}

    # Strong sell: price below SMA20 and RSI overbought
    if price < sma20 and rsi > 75:
        conf = min((rsi - 75) / 25 + 0.5, 1.0)
        return StrategySignal("SELL", round(conf, 2), weight, name,
            f"Bearish: price ${price:.2f} below SMA20 ${sma20:.2f}, RSI overbought at {rsi:.1f}", sigs)

    # Sell signal: price dropped below both MAs
    if price < sma20 and price < sma50:
        conf = 0.4 + min((sma50 - price) / sma50 * 10, 0.4)
        return StrategySignal("SELL", round(conf, 2), weight, name,
            f"Bearish: price ${price:.2f} below both SMA20 ${sma20:.2f} and SMA50 ${sma50:.2f}", sigs)

    # Buy signal: uptrend with RSI in sweet spot
    if price > sma20 > sma50 and 50 <= rsi <= 70:
        # Stronger signal when gap between MAs is widening
        ma_spread = (sma20 - sma50) / sma50 * 100
        conf = 0.5 + min(ma_spread / 5, 0.3) + (rsi - 50) / 100
        conf = min(conf, 1.0)
        return StrategySignal("BUY", round(conf, 2), weight, name,
            f"Bullish momentum: price ${price:.2f} > SMA20 ${sma20:.2f} > SMA50 ${sma50:.2f}, RSI {rsi:.1f}", sigs)

    # Weak buy: price above SMA20 but RSI low (building momentum)
    if price > sma20 and 40 <= rsi < 50:
        return StrategySignal("BUY", 0.3, weight, name,
            f"Early momentum: price above SMA20, RSI building at {rsi:.1f}", sigs)

    return _hold(name, f"No clear momentum signal. Price ${price:.2f}, SMA20 ${sma20:.2f}, RSI {rsi:.1f}", weight, sigs)


def mean_reversion_signal(indicators: dict, weight: float) -> StrategySignal:
    """Buy near lower Bollinger + low RSI; Sell near upper Bollinger + high RSI."""
    name = "mean_reversion"
    price = indicators.get("current_price")
    bb_upper = indicators.get("bollinger_upper")
    bb_lower = indicators.get("bollinger_lower")
    bb_mid = indicators.get("bollinger_mid")
    rsi = indicators.get("rsi_14")

    if any(v is None for v in [price, bb_upper, bb_lower, bb_mid, rsi]):
        return _hold(name, "Insufficient indicator data", weight)

    sigs = {"price": price, "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid, "rsi_14": rsi}

    # Buy: price at or below lower Bollinger, RSI oversold
    if price <= bb_lower and rsi < 35:
        conf = 0.6 + min((bb_lower - price) / bb_lower * 10, 0.3) + (35 - rsi) / 100
        conf = min(conf, 1.0)
        return StrategySignal("BUY", round(conf, 2), weight, name,
            f"Oversold: price ${price:.2f} at lower BB ${bb_lower:.2f}, RSI {rsi:.1f}", sigs)

    # Moderate buy: price near lower Bollinger
    if price <= bb_lower * 1.01 and rsi < 40:
        return StrategySignal("BUY", 0.45, weight, name,
            f"Near lower BB: price ${price:.2f}, BB lower ${bb_lower:.2f}, RSI {rsi:.1f}", sigs)

    # Sell: price at or above upper Bollinger, RSI overbought
    if price >= bb_upper and rsi > 70:
        conf = 0.6 + min((price - bb_upper) / bb_upper * 10, 0.3) + (rsi - 70) / 100
        conf = min(conf, 1.0)
        return StrategySignal("SELL", round(conf, 2), weight, name,
            f"Overbought: price ${price:.2f} at upper BB ${bb_upper:.2f}, RSI {rsi:.1f}", sigs)

    # Moderate sell: price near upper Bollinger
    if price >= bb_upper * 0.99 and rsi > 65:
        return StrategySignal("SELL", 0.4, weight, name,
            f"Near upper BB: price ${price:.2f}, BB upper ${bb_upper:.2f}, RSI {rsi:.1f}", sigs)

    return _hold(name, f"Price ${price:.2f} within Bollinger bands (${bb_lower:.2f}-${bb_upper:.2f}), RSI {rsi:.1f}", weight, sigs)


def technical_signals_signal(indicators: dict, weight: float) -> StrategySignal:
    """MACD crossovers, volume spikes, support/resistance levels."""
    name = "technical_signals"
    price = indicators.get("current_price")
    macd_cross = indicators.get("macd_crossover", "none")
    macd_hist = indicators.get("macd_histogram")
    volume = indicators.get("obv")
    vol_sma = indicators.get("volume_sma_20")
    atr = indicators.get("atr")

    if price is None:
        return _hold(name, "No price data", weight)

    sigs = {
        "price": price, "macd_crossover": macd_cross,
        "macd_histogram": macd_hist, "atr": atr,
    }

    reasons = []
    buy_score = 0
    sell_score = 0

    # MACD crossover
    if macd_cross == "bullish_crossover":
        buy_score += 0.4
        reasons.append("Bullish MACD crossover")
    elif macd_cross == "bearish_crossover":
        sell_score += 0.4
        reasons.append("Bearish MACD crossover")

    # MACD histogram strength
    if macd_hist is not None:
        if macd_hist > 0:
            buy_score += min(abs(macd_hist) / 5, 0.2)
            reasons.append(f"Positive MACD histogram ({macd_hist:.2f})")
        elif macd_hist < 0:
            sell_score += min(abs(macd_hist) / 5, 0.2)
            reasons.append(f"Negative MACD histogram ({macd_hist:.2f})")

    # ATR-based volatility context
    if atr is not None and price > 0:
        atr_pct = atr / price * 100
        if atr_pct > 5:
            reasons.append(f"High volatility (ATR {atr_pct:.1f}%)")
            # High volatility reduces confidence in either direction
            buy_score *= 0.8
            sell_score *= 0.8

    net = buy_score - sell_score
    reasoning = "; ".join(reasons) if reasons else "No strong technical signals"

    if net > 0.2:
        return StrategySignal("BUY", round(min(net, 1.0), 2), weight, name, reasoning, sigs)
    elif net < -0.2:
        return StrategySignal("SELL", round(min(abs(net), 1.0), 2), weight, name, reasoning, sigs)

    return _hold(name, reasoning, weight, sigs)


def range_trader_signal(indicators: dict, regime: dict, weight: float) -> StrategySignal:
    """Range-bound mean reversion optimized for TSLA's sideways volatility.

    Active when regime.directional == "range_bound". Uses wider Bollinger
    thresholds than mean_reversion and incorporates ADX to confirm range.
    This is the primary strategy for exploiting TSLA's bounce-around behavior.
    """
    name = "range_trader"
    directional = regime.get("directional", "range_bound")

    # Only active in range-bound regime
    if directional == "trending":
        return _hold(name, "Regime is trending — range trader inactive", weight)

    price = indicators.get("current_price")
    bb_upper = indicators.get("bollinger_upper")
    bb_lower = indicators.get("bollinger_lower")
    bb_mid = indicators.get("bollinger_mid")
    rsi = indicators.get("rsi_14")
    adx = indicators.get("adx")

    if any(v is None for v in [price, bb_upper, bb_lower, bb_mid, rsi]):
        return _hold(name, "Insufficient indicator data", weight)

    sigs = {
        "price": price, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "bb_mid": bb_mid, "rsi_14": rsi, "adx": adx,
        "regime_directional": directional,
    }

    # ADX confirmation bonus: lower ADX = stronger range signal
    adx_bonus = 0.0
    if adx is not None and adx < 20:
        adx_bonus = 0.1  # Strong range-bound confirmation

    # BUY zone: lower third of Bollinger band + RSI < 40
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        position_in_band = (price - bb_lower) / bb_range  # 0 = lower, 1 = upper

        if position_in_band <= 0.15 and rsi < 38:
            # Strong buy: at the bottom of the range
            conf = 0.65 + adx_bonus + min((38 - rsi) / 50, 0.15)
            conf = round(min(conf, 0.90), 2)
            return StrategySignal("BUY", conf, weight, name,
                f"Range BUY: price ${price:.2f} at bottom of BB range ({position_in_band:.0%}), RSI {rsi:.1f}, ADX {adx or 'N/A'}", sigs)

        if position_in_band <= 0.30 and rsi < 43:
            # Moderate buy: lower third
            conf = 0.45 + adx_bonus
            return StrategySignal("BUY", round(conf, 2), weight, name,
                f"Range BUY: price ${price:.2f} in lower third ({position_in_band:.0%}), RSI {rsi:.1f}", sigs)

        # SELL zone: upper portion of Bollinger band
        # Note: daily RSI rarely exceeds 55 in a range-bound TSLA market,
        # so thresholds are calibrated for daily bars (not intraday)
        if position_in_band >= 0.80 and rsi > 55:
            conf = 0.65 + adx_bonus + min((rsi - 55) / 50, 0.15)
            conf = round(min(conf, 0.90), 2)
            return StrategySignal("SELL", conf, weight, name,
                f"Range SELL: price ${price:.2f} at top of BB range ({position_in_band:.0%}), RSI {rsi:.1f}, ADX {adx or 'N/A'}", sigs)

        if position_in_band >= 0.50 and rsi > 46:
            # Moderate sell: upper half with RSI above midline
            conf = 0.40 + adx_bonus
            return StrategySignal("SELL", round(conf, 2), weight, name,
                f"Range SELL: price ${price:.2f} in upper half ({position_in_band:.0%}), RSI {rsi:.1f}", sigs)

    bb_pos_str = f"{position_in_band:.0%}" if bb_range > 0 else "N/A"
    return _hold(name, f"Price ${price:.2f} in mid-range ({bb_pos_str} of BB), RSI {rsi:.1f} — waiting for extremes", weight, sigs)


def thesis_alignment_signal(
    indicators: dict,
    thesis_direction: str,
    thesis_conviction: float,
    weight: float,
) -> StrategySignal:
    """Generate signal informed by thesis direction + technical confirmation.

    v3: Thesis informs SIZE, not direction blocking. Counter-thesis trades
    are allowed at reduced confidence when technicals are strong. This prevents
    the thesis from permanently blocking profitable setups.
    """
    name = "thesis_alignment"
    price = indicators.get("current_price")
    rsi = indicators.get("rsi_14")
    sma20 = indicators.get("sma_20")
    macd_cross = indicators.get("macd_crossover", "none")

    if price is None or not thesis_direction:
        return _hold(name, "Insufficient data or no thesis", weight)

    sigs = {
        "price": price, "rsi_14": rsi, "sma_20": sma20,
        "thesis_direction": thesis_direction, "thesis_conviction": thesis_conviction,
    }

    # Neutral thesis = no directional signal
    if thesis_direction == "neutral":
        return _hold(name, f"Thesis neutral (conviction={thesis_conviction:.2f}), waiting for clarity", weight, sigs)

    # Low conviction = weak signal regardless of technicals
    if thesis_conviction < 0.3:
        return _hold(name, f"Thesis {thesis_direction} but conviction too low ({thesis_conviction:.2f})", weight, sigs)

    # Count buy and sell technical confirmations
    buy_confirms = 0
    sell_confirms = 0
    reasons = [f"Thesis {thesis_direction} (conviction={thesis_conviction:.2f})"]

    if rsi is not None:
        if rsi < 35:
            buy_confirms += 1
            reasons.append(f"RSI oversold at {rsi:.1f}")
        elif rsi > 65:
            sell_confirms += 1
            reasons.append(f"RSI overbought at {rsi:.1f}")

    if sma20 is not None:
        if price >= sma20 * 0.99:
            buy_confirms += 1
            reasons.append(f"Price ${price:.2f} at/above SMA20 ${sma20:.2f}")
        elif price < sma20:
            sell_confirms += 1
            reasons.append(f"Price ${price:.2f} below SMA20 ${sma20:.2f}")

    if macd_cross == "bullish_crossover":
        buy_confirms += 1
        reasons.append("Bullish MACD crossover")
    elif macd_cross == "bearish_crossover":
        sell_confirms += 1
        reasons.append("Bearish MACD crossover")

    # THESIS-ALIGNED trade: full confidence
    if thesis_direction == "bullish" and buy_confirms >= 1:
        conf = thesis_conviction * (0.5 + buy_confirms * 0.15)
        conf = round(min(conf, 0.95), 2)
        return StrategySignal("BUY", conf, weight, name,
            f"Thesis-aligned BUY: {'; '.join(reasons)}", sigs)

    if thesis_direction == "bearish" and sell_confirms >= 1:
        conf = thesis_conviction * (0.5 + sell_confirms * 0.15)
        conf = round(min(conf, 0.95), 2)
        return StrategySignal("SELL", conf, weight, name,
            f"Thesis-aligned SELL: {'; '.join(reasons)}", sigs)

    # COUNTER-THESIS trade: allowed but at reduced confidence (thesis sizes it down)
    # Only when technicals are strong (2+ confirmations against thesis)
    if thesis_direction == "bearish" and buy_confirms >= 2:
        # Technicals strongly bullish despite bearish thesis — allow at reduced conf
        counter_conf = (1.0 - thesis_conviction) * (0.3 + buy_confirms * 0.15)
        counter_conf = round(min(counter_conf, 0.50), 2)  # Cap at 0.50 for counter-thesis
        if counter_conf >= 0.15:
            return StrategySignal("BUY", counter_conf, weight, name,
                f"Counter-thesis BUY (thesis bearish but technicals strong): {'; '.join(reasons)}", sigs)

    if thesis_direction == "bullish" and sell_confirms >= 2:
        counter_conf = (1.0 - thesis_conviction) * (0.3 + sell_confirms * 0.15)
        counter_conf = round(min(counter_conf, 0.50), 2)
        if counter_conf >= 0.15:
            return StrategySignal("SELL", counter_conf, weight, name,
                f"Counter-thesis SELL (thesis bullish but technicals strong): {'; '.join(reasons)}", sigs)

    # Thesis-aligned but no technical confirmation yet
    if thesis_direction == "bullish" and thesis_conviction >= 0.6:
        return StrategySignal("BUY", round(thesis_conviction * 0.35, 2), weight, name,
            f"Thesis bullish high conviction but no tech confirmation: {'; '.join(reasons)}", sigs)

    if thesis_direction == "bearish" and thesis_conviction >= 0.6:
        return StrategySignal("SELL", round(thesis_conviction * 0.35, 2), weight, name,
            f"Thesis bearish high conviction but no tech confirmation: {'; '.join(reasons)}", sigs)

    return _hold(name, f"Thesis {thesis_direction} but waiting for technical confirmation: {'; '.join(reasons)}", weight, sigs)


# --- Legacy strategy functions kept for replay/ensemble backward compatibility ---

def dca_signal(portfolio: dict, indicators: dict, weight: float) -> StrategySignal:
    """Small periodic buys — DEPRECATED in v2, kept for replay compatibility."""
    return _hold("dca", "DCA disabled in v2 (thesis-driven trading)", weight)


def sentiment_signal(news_sentiment: float | None, weight: float) -> StrategySignal:
    """Score based on news sentiment — DEPRECATED in v2, replaced by thesis_alignment."""
    return _hold("sentiment", "Sentiment disabled in v2 (replaced by thesis_alignment)", weight)


# --- Aggregation ---

def _apply_regime_adjustment(signal: StrategySignal, regime: dict) -> StrategySignal:
    """Adjust strategy confidence based on current market regime.

    Uses both legacy trend (bull/bear/sideways) and new directional
    (trending/range_bound) classification for regime-appropriate boosts.
    """
    if not regime or signal.action == "HOLD":
        return signal

    trend = regime.get("trend", "sideways")
    vol = regime.get("volatility", "normal")
    directional = regime.get("directional", "range_bound")
    multiplier = 1.0

    if signal.strategy == "momentum":
        if directional == "trending":
            multiplier = 1.3  # Momentum thrives in trending regimes
        elif directional == "range_bound":
            multiplier = 0.6  # Momentum fails in range-bound
        if vol == "high":
            multiplier *= 0.85
    elif signal.strategy == "mean_reversion":
        if directional == "range_bound":
            multiplier = 1.3  # Mean reversion thrives in range-bound
        elif directional == "trending":
            multiplier = 0.6  # Mean reversion fails in trends
        if vol == "high":
            multiplier *= 1.1  # Wide bands = bigger reversion opportunities
    elif signal.strategy == "range_trader":
        # Range trader already gates itself on directional regime,
        # so just give a vol boost
        if vol == "high":
            multiplier = 1.15
        elif vol == "low":
            multiplier = 0.9  # Narrow ranges = less opportunity
    elif signal.strategy == "technical_signals":
        if vol == "high":
            multiplier = 0.8
    elif signal.strategy == "thesis_alignment":
        if signal.action == "BUY" and trend == "bull":
            multiplier = 1.15
        elif signal.action == "SELL" and trend == "bear":
            multiplier = 1.15
        elif signal.action == "BUY" and trend == "bear":
            multiplier = 0.75
        elif signal.action == "SELL" and trend == "bull":
            multiplier = 0.80
    elif signal.strategy == "sentiment":
        if signal.action == "BUY" and trend == "bull":
            multiplier = 1.15
        elif signal.action == "SELL" and trend == "bear":
            multiplier = 1.15
        elif signal.action == "BUY" and trend == "bear":
            multiplier = 0.8

    if multiplier != 1.0:
        new_conf = round(min(signal.confidence * multiplier, 1.0), 2)
        signal.confidence = new_conf

    return signal


def evaluate_all_strategies(
    market_data: dict,
    portfolio: dict,
    scores: dict = None,
    news_sentiment: float = None,
    regime: dict = None,
    thesis_direction: str = None,
    thesis_conviction: float = 0.0,
) -> list[StrategySignal]:
    """Run all enabled strategies and return weighted signals.

    v2: accepts thesis_direction and thesis_conviction for the thesis_alignment strategy.
    """
    config = load_config()
    enabled = config["strategies_enabled"]

    if scores is None:
        scores = get_strategy_scores()

    weights = {name: s["weight"] for name, s in scores.get("strategies", {}).items()}

    signals = []

    indicators = market_data.get("daily_indicators", {})
    # Prefer intraday indicators if available
    intraday = market_data.get("intraday_indicators")
    if intraday and intraday.get("current_price"):
        indicators = {**indicators, **intraday}

    # Get regime from market_data if not passed explicitly
    if regime is None:
        regime = market_data.get("regime", {})

    if "momentum" in enabled:
        signals.append(momentum_signal(indicators, weights.get("momentum", 0.2)))

    if "mean_reversion" in enabled:
        signals.append(mean_reversion_signal(indicators, weights.get("mean_reversion", 0.2)))

    if "technical_signals" in enabled:
        signals.append(technical_signals_signal(indicators, weights.get("technical_signals", 0.2)))

    if "thesis_alignment" in enabled:
        signals.append(thesis_alignment_signal(
            indicators,
            thesis_direction or "neutral",
            thesis_conviction,
            weights.get("thesis_alignment", 0.25),
        ))

    # Range trader — always enabled, self-gates on regime
    signals.append(range_trader_signal(
        indicators, regime, weights.get("range_trader", 0.20),
    ))

    # Legacy strategies — still supported for replay/ensemble backward compat
    if "dca" in enabled:
        signals.append(dca_signal(portfolio, indicators, weights.get("dca", 0.2)))

    if "sentiment" in enabled:
        signals.append(sentiment_signal(news_sentiment, weights.get("sentiment", 0.2)))

    # Apply regime adjustments to all signals
    signals = [_apply_regime_adjustment(s, regime) for s in signals]

    for s in signals:
        logger.info(f"  [{s.strategy}] {s.action} conf={s.confidence:.2f} wt={s.weight:.2f} — {s.reasoning}")

    return signals


def calculate_signal_balance(signals: list[StrategySignal]) -> dict:
    """Calculate a weighted signal balance on a -1 (SELL) to +1 (BUY) scale.

    Returns a dict with the overall balance, per-strategy breakdown, and
    a human-readable summary suitable for agent prompts and HOLD analysis.
    """
    if not signals:
        return {
            "balance": 0.0,
            "breakdown": {},
            "buy_pressure": 0.0,
            "sell_pressure": 0.0,
            "summary": "No signals available.",
        }

    total_weight = sum(s.weight for s in signals)
    if total_weight == 0:
        total_weight = 1.0

    buy_pressure = 0.0
    sell_pressure = 0.0
    breakdown = {}

    for s in signals:
        # Map each signal to -1..+1 scale weighted by confidence and strategy weight
        if s.action == "BUY":
            value = s.confidence * s.weight / total_weight
            buy_pressure += value
        elif s.action == "SELL":
            value = -s.confidence * s.weight / total_weight
            sell_pressure += abs(value)
        else:
            value = 0.0

        breakdown[s.strategy] = {
            "action": s.action,
            "confidence": round(s.confidence, 3),
            "weight": round(s.weight, 3),
            "contribution": round(value, 4),
        }

    balance = round(buy_pressure - sell_pressure, 4)

    # Build summary
    parts = []
    for name, b in sorted(breakdown.items(), key=lambda x: abs(x[1]["contribution"]), reverse=True):
        if b["contribution"] != 0:
            direction = "BUY" if b["contribution"] > 0 else "SELL"
            parts.append(f"{name}: {direction} {abs(b['contribution']):.3f}")
        else:
            parts.append(f"{name}: HOLD (neutral)")

    summary = f"Signal Balance: {balance:+.4f} (buy={buy_pressure:.3f}, sell={sell_pressure:.3f}). {'; '.join(parts)}"

    return {
        "balance": balance,
        "breakdown": breakdown,
        "buy_pressure": round(buy_pressure, 4),
        "sell_pressure": round(sell_pressure, 4),
        "summary": summary,
    }


def aggregate_signals(signals: list[StrategySignal]) -> StrategySignal:
    """Weighted combination of all signals into a single recommendation."""
    if not signals:
        return _hold("aggregate", "No signals to aggregate", 1.0)

    buy_score = 0.0
    sell_score = 0.0
    total_weight = 0.0
    buy_reasons = []
    sell_reasons = []
    hold_reasons = []
    all_signals = {}

    for s in signals:
        weighted = s.confidence * s.weight
        total_weight += s.weight

        if s.action == "BUY":
            buy_score += weighted
            buy_reasons.append(f"{s.strategy}({s.confidence:.2f}×{s.weight:.2f})")
        elif s.action == "SELL":
            sell_score += weighted
            sell_reasons.append(f"{s.strategy}({s.confidence:.2f}×{s.weight:.2f})")
        else:
            hold_reasons.append(s.strategy)

        all_signals[s.strategy] = {
            "action": s.action, "confidence": s.confidence,
            "weight": s.weight, "reasoning": s.reasoning,
        }

    # Normalize by weight of ACTIVE signals only. Strategies that return HOLD
    # are abstaining, not voting against. Their weight shouldn't dilute real signals.
    # Floor at 40% of total weight so one weak strategy can't dominate.
    active_weight = sum(
        s.weight for s in signals if s.action != "HOLD"
    )
    divisor = max(active_weight, total_weight * 0.4) if total_weight > 0 else 1.0

    if divisor > 0:
        buy_score /= divisor
        sell_score /= divisor

    net = buy_score - sell_score

    # Need a meaningful threshold to act
    threshold = 0.10
    active_count = len(buy_reasons) + len(sell_reasons)

    if net > threshold:
        reasoning = (
            f"Weighted BUY signal ({net:.3f}, {active_count} active of {len(signals)}). "
            f"Buyers: {', '.join(buy_reasons)}"
        )
        if sell_reasons:
            reasoning += f". Dissenters: {', '.join(sell_reasons)}"
        if hold_reasons:
            reasoning += f". Abstaining: {', '.join(hold_reasons)}"
        return StrategySignal("BUY", round(min(net, 1.0), 2), 1.0, "aggregate", reasoning, all_signals)

    elif net < -threshold:
        reasoning = (
            f"Weighted SELL signal ({net:.3f}, {active_count} active of {len(signals)}). "
            f"Sellers: {', '.join(sell_reasons)}"
        )
        if buy_reasons:
            reasoning += f". Dissenters: {', '.join(buy_reasons)}"
        if hold_reasons:
            reasoning += f". Abstaining: {', '.join(hold_reasons)}"
        return StrategySignal("SELL", round(min(abs(net), 1.0), 2), 1.0, "aggregate", reasoning, all_signals)

    else:
        reasoning = f"Mixed/weak signals (net: {net:.3f}, {active_count} active). "
        if buy_reasons:
            reasoning += f"Buy: {', '.join(buy_reasons)}. "
        if sell_reasons:
            reasoning += f"Sell: {', '.join(sell_reasons)}. "
        if hold_reasons:
            reasoning += f"Abstaining: {', '.join(hold_reasons)}."
        return StrategySignal("HOLD", round(abs(net), 2), 1.0, "aggregate", reasoning, all_signals)
