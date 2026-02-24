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


def dca_signal(portfolio: dict, indicators: dict, weight: float) -> StrategySignal:
    """Small periodic buys, with tactical adjustment based on conditions."""
    name = "dca"
    price = indicators.get("current_price")
    rsi = indicators.get("rsi_14")

    if price is None:
        return _hold(name, "No price data", weight)

    sigs = {"price": price, "rsi_14": rsi}

    # DCA always leans toward small buys if we have cash
    cash = portfolio.get("cash", 0)
    total_value = portfolio.get("total_value", 1)

    if cash / total_value < 0.15:
        return _hold(name, f"Low cash reserve ({cash/total_value*100:.0f}%), skipping DCA", weight, sigs)

    # Base DCA is a low-confidence buy
    conf = 0.25

    # Boost if price is depressed (RSI low = buy more)
    if rsi is not None and rsi < 35:
        conf = 0.5
        return StrategySignal("BUY", conf, weight, name,
            f"DCA buy boosted — RSI oversold at {rsi:.1f}", sigs)

    # Reduce if overbought
    if rsi is not None and rsi > 70:
        conf = 0.1
        return StrategySignal("BUY", conf, weight, name,
            f"DCA buy reduced — RSI overbought at {rsi:.1f}", sigs)

    return StrategySignal("BUY", conf, weight, name,
        f"Regular DCA buy. RSI {f'{rsi:.1f}' if rsi else 'N/A'}", sigs)


def sentiment_signal(news_sentiment: float | None, weight: float) -> StrategySignal:
    """Score based on news sentiment (-1 to +1). Populated by agent.py via Claude."""
    name = "sentiment"
    sigs = {"sentiment_score": news_sentiment}

    if news_sentiment is None:
        return _hold(name, "No sentiment data available", weight, sigs)

    if news_sentiment > 0.5:
        conf = min(news_sentiment, 1.0)
        return StrategySignal("BUY", round(conf * 0.7, 2), weight, name,
            f"Positive sentiment ({news_sentiment:.2f})", sigs)
    elif news_sentiment < -0.5:
        conf = min(abs(news_sentiment), 1.0)
        return StrategySignal("SELL", round(conf * 0.7, 2), weight, name,
            f"Negative sentiment ({news_sentiment:.2f})", sigs)
    elif news_sentiment > 0.2:
        return StrategySignal("BUY", 0.2, weight, name,
            f"Mildly positive sentiment ({news_sentiment:.2f})", sigs)
    elif news_sentiment < -0.2:
        return StrategySignal("SELL", 0.2, weight, name,
            f"Mildly negative sentiment ({news_sentiment:.2f})", sigs)

    return _hold(name, f"Neutral sentiment ({news_sentiment:.2f})", weight, sigs)


# --- Aggregation ---

def _apply_regime_adjustment(signal: StrategySignal, regime: dict) -> StrategySignal:
    """Adjust strategy confidence based on current market regime.

    - Momentum gets boosted in bull/low-vol, penalized in bear/high-vol
    - Mean reversion gets boosted in range-bound/high-vol
    - Technical signals get penalized in high volatility
    - DCA is regime-neutral
    - Sentiment gets boosted when it aligns with regime direction
    """
    if not regime or signal.action == "HOLD":
        return signal

    trend = regime.get("trend", "sideways")
    vol = regime.get("volatility", "normal")
    multiplier = 1.0

    if signal.strategy == "momentum":
        if trend == "bull" and vol == "low":
            multiplier = 1.2
        elif trend == "bear":
            multiplier = 0.7
        elif vol == "high":
            multiplier = 0.8
    elif signal.strategy == "mean_reversion":
        if trend == "sideways" or vol == "high":
            multiplier = 1.2
        elif trend in ("bull", "bear") and vol == "low":
            multiplier = 0.8
    elif signal.strategy == "technical_signals":
        if vol == "high":
            multiplier = 0.8
    elif signal.strategy == "sentiment":
        # Boost when sentiment direction aligns with regime
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
) -> list[StrategySignal]:
    """Run all enabled strategies and return weighted signals."""
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

    # Normalize by total weight
    if total_weight > 0:
        buy_score /= total_weight
        sell_score /= total_weight

    net = buy_score - sell_score

    # Need a meaningful threshold to act
    threshold = 0.10

    if net > threshold:
        reasoning = f"Weighted BUY signal ({net:.3f}). Buyers: {', '.join(buy_reasons)}"
        if sell_reasons:
            reasoning += f". Dissenters: {', '.join(sell_reasons)}"
        return StrategySignal("BUY", round(min(net * 2, 1.0), 2), 1.0, "aggregate", reasoning, all_signals)

    elif net < -threshold:
        reasoning = f"Weighted SELL signal ({net:.3f}). Sellers: {', '.join(sell_reasons)}"
        if buy_reasons:
            reasoning += f". Dissenters: {', '.join(buy_reasons)}"
        return StrategySignal("SELL", round(min(abs(net) * 2, 1.0), 2), 1.0, "aggregate", reasoning, all_signals)

    else:
        reasoning = f"Mixed signals (net: {net:.3f}). "
        if buy_reasons:
            reasoning += f"Buy: {', '.join(buy_reasons)}. "
        if sell_reasons:
            reasoning += f"Sell: {', '.join(sell_reasons)}. "
        if hold_reasons:
            reasoning += f"Hold: {', '.join(hold_reasons)}."
        return StrategySignal("HOLD", round(abs(net), 2), 1.0, "aggregate", reasoning, all_signals)
