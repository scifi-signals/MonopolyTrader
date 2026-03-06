"""Backtest v3 swing trading changes against Feb 18 - Mar 5 TSLA data.

Tests:
1. ADX regime classification — was it trending or range_bound each day?
2. Range trader signals — where would it have fired BUY/SELL?
3. Thesis alignment counter-thesis signals — would they have fired?
4. Compare hypothetical P&L vs actual agent performance
"""
import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import ta

# We need the actual functions
from src.market_data import calculate_indicators
from src.strategies import (
    range_trader_signal, mean_reversion_signal, momentum_signal,
    thesis_alignment_signal, _apply_regime_adjustment,
)


def load_data():
    """Download TSLA + SPY daily data for the backtest period."""
    print("Downloading TSLA data (2025-11-01 to 2026-03-06)...")
    tsla = yf.Ticker("TSLA").history(start="2025-11-01", end="2026-03-06", interval="1d")
    print(f"  Got {len(tsla)} bars")

    print("Downloading SPY data...")
    spy = yf.Ticker("SPY").history(start="2025-11-01", end="2026-03-06", interval="1d")
    print(f"  Got {len(spy)} bars")

    print("Downloading VIX data...")
    vix = yf.Ticker("^VIX").history(start="2025-11-01", end="2026-03-06", interval="1d")
    print(f"  Got {len(vix)} bars")

    return tsla, spy, vix


def classify_regime_for_bar(tsla_df_to_date, spy_df_to_date, vix_df_to_date):
    """Classify regime using data up to (and including) the current bar."""
    regime = {
        "trend": "sideways", "directional": "range_bound",
        "volatility": "normal", "vix": 0.0, "adx": 0.0,
        "strategy_mode": "mean_reversion",
    }

    # VIX
    if len(vix_df_to_date) > 0:
        vix_val = float(vix_df_to_date["Close"].iloc[-1])
        regime["vix"] = round(vix_val, 2)
        if vix_val < 20:
            regime["volatility"] = "low"
        elif vix_val < 30:
            regime["volatility"] = "normal"
        else:
            regime["volatility"] = "high"

    # ADX from TSLA
    if len(tsla_df_to_date) >= 30:
        adx_ind = ta.trend.ADXIndicator(
            tsla_df_to_date["High"], tsla_df_to_date["Low"],
            tsla_df_to_date["Close"], window=14
        )
        adx_series = adx_ind.adx().dropna()
        if len(adx_series) >= 3:
            adx_val = float(adx_series.iloc[-1])
            adx_avg_3 = float(adx_series.tail(3).mean())
            regime["adx"] = round(adx_val, 2)
            regime["adx_avg_3d"] = round(adx_avg_3, 2)

            if adx_avg_3 >= 25:
                regime["directional"] = "trending"
            elif adx_avg_3 <= 20:
                regime["directional"] = "range_bound"
            # else keep range_bound as default

            di_pos = adx_ind.adx_pos().dropna()
            di_neg = adx_ind.adx_neg().dropna()
            if len(di_pos) > 0 and len(di_neg) > 0:
                regime["di_positive"] = round(float(di_pos.iloc[-1]), 2)
                regime["di_negative"] = round(float(di_neg.iloc[-1]), 2)

    # SPY slope for legacy trend
    if len(spy_df_to_date) >= 50:
        close_50 = spy_df_to_date["Close"].tail(50)
        x = np.arange(len(close_50))
        slope = np.polyfit(x, close_50.values, 1)[0]
        avg_price = close_50.mean()
        slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0
        if slope_pct > 0.05:
            regime["trend"] = "bull"
        elif slope_pct < -0.05:
            regime["trend"] = "bear"
        else:
            regime["trend"] = "sideways"

    if regime["directional"] == "trending":
        regime["strategy_mode"] = "momentum"

    return regime


def simulate_range_trader(tsla_df, spy_df, vix_df):
    """Walk through Feb 18 - Mar 5, compute signals on each day."""
    # Only backtest from Feb 18 onward (need lookback data before that)
    start_date = pd.Timestamp("2026-02-18", tz=tsla_df.index.tz)
    end_date = pd.Timestamp("2026-03-06", tz=tsla_df.index.tz)

    test_bars = tsla_df[(tsla_df.index >= start_date) & (tsla_df.index < end_date)]

    results = []
    position = 0  # shares held
    cash = 1000.0
    entry_price = 0.0
    trades = []

    print(f"\n{'='*90}")
    print(f"{'Date':>12} {'Price':>8} {'ADX':>6} {'ADX3d':>6} {'Regime':>12} {'BB%':>6} {'RSI':>6} {'Signal':>12} {'Conf':>6}")
    print(f"{'='*90}")

    for i, (date, row) in enumerate(test_bars.iterrows()):
        # Get all data up to this date for indicator calculation
        tsla_to_date = tsla_df[tsla_df.index <= date]
        spy_to_date = spy_df[spy_df.index <= date] if len(spy_df) > 0 else pd.DataFrame()
        vix_to_date = vix_df[vix_df.index <= date] if len(vix_df) > 0 else pd.DataFrame()

        # Calculate indicators
        indicators = calculate_indicators(tsla_to_date)
        price = indicators.get("current_price", float(row["Close"]))

        # Classify regime
        regime = classify_regime_for_bar(tsla_to_date, spy_to_date, vix_to_date)

        # Get signals from range_trader
        rt_signal = range_trader_signal(indicators, regime, 0.20)
        rt_signal = _apply_regime_adjustment(rt_signal, regime)

        # Also get mean reversion for comparison
        mr_signal = mean_reversion_signal(indicators, 0.20)
        mr_signal = _apply_regime_adjustment(mr_signal, regime)

        # Thesis alignment with counter-thesis (bearish thesis, conviction 0.72)
        ta_signal = thesis_alignment_signal(indicators, "bearish", 0.72, 0.25)
        ta_signal = _apply_regime_adjustment(ta_signal, regime)

        # Calculate BB position
        bb_upper = indicators.get("bollinger_upper", 0)
        bb_lower = indicators.get("bollinger_lower", 0)
        bb_range = bb_upper - bb_lower if bb_upper and bb_lower else 1
        bb_pct = (price - (bb_lower or 0)) / bb_range if bb_range > 0 else 0.5

        rsi = indicators.get("rsi_14", 50)
        adx = regime.get("adx", 0)
        adx_3d = regime.get("adx_avg_3d", 0)

        # Determine combined best signal
        best_signal = rt_signal
        if mr_signal.action != "HOLD" and mr_signal.confidence > best_signal.confidence:
            best_signal = mr_signal

        # Simple simulation: execute range_trader signals
        trade_action = ""
        if rt_signal.action == "BUY" and position == 0:
            shares = min(cash * 0.20 / price, cash * 0.65 / price)  # 20% max trade
            if shares * price > 10:  # min $10 trade
                position = shares
                entry_price = price * 1.0005  # slippage
                cash -= shares * entry_price
                trade_action = f"BUY {shares:.2f}"
                trades.append({"date": str(date.date()), "action": "BUY", "price": entry_price, "shares": shares})

        elif rt_signal.action == "SELL" and position > 0:
            sell_price = price * 0.9995  # slippage
            pnl = (sell_price - entry_price) * position
            cash += position * sell_price
            trade_action = f"SELL {position:.2f} P&L=${pnl:.2f}"
            trades.append({"date": str(date.date()), "action": "SELL", "price": sell_price, "shares": position, "pnl": pnl})
            position = 0

        date_str = str(date.date())
        signal_str = f"{rt_signal.action}"
        if rt_signal.action != "HOLD":
            signal_str += f"({rt_signal.strategy[:5]})"

        print(
            f"{date_str:>12} ${price:>7.2f} {adx:>5.1f} {adx_3d:>5.1f} "
            f"{regime['directional']:>12} {bb_pct:>5.0%} {rsi or 0:>5.1f} "
            f"{signal_str:>12} {rt_signal.confidence:>5.2f}"
            + (f"  ← {trade_action}" if trade_action else "")
        )

        results.append({
            "date": date_str,
            "price": price,
            "adx": adx,
            "adx_avg_3d": adx_3d,
            "directional": regime["directional"],
            "volatility": regime["volatility"],
            "trend": regime["trend"],
            "bb_pct": round(bb_pct, 3),
            "rsi": rsi,
            "rt_action": rt_signal.action,
            "rt_conf": rt_signal.confidence,
            "rt_reasoning": rt_signal.reasoning[:80],
            "mr_action": mr_signal.action,
            "mr_conf": mr_signal.confidence,
            "ta_action": ta_signal.action,
            "ta_conf": ta_signal.confidence,
            "ta_reasoning": ta_signal.reasoning[:80],
        })

    # Final P&L
    final_value = cash
    if position > 0:
        final_price = float(test_bars["Close"].iloc[-1])
        final_value += position * final_price

    print(f"\n{'='*90}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*90}")
    print(f"Period: Feb 18 - Mar 5, 2026")
    print(f"Starting value: $1,000.00")
    print(f"Final value: ${final_value:.2f} ({(final_value/1000-1)*100:+.2f}%)")
    print(f"Total trades: {len(trades)}")

    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) < 0]
    print(f"Wins: {len(wins)}, Losses: {len(losses)}")
    if trades:
        total_pnl = sum(t.get("pnl", 0) for t in trades)
        print(f"Total realized P&L: ${total_pnl:.2f}")

    print(f"\nTrades:")
    for t in trades:
        pnl_str = f" P&L=${t['pnl']:.2f}" if "pnl" in t else ""
        print(f"  {t['date']} {t['action']} {t['shares']:.2f} @ ${t['price']:.2f}{pnl_str}")

    # Regime summary
    print(f"\nRegime Classification Summary:")
    regime_counts = {}
    for r in results:
        key = f"{r['directional']}/{r['volatility']}"
        regime_counts[key] = regime_counts.get(key, 0) + 1
    for key, count in sorted(regime_counts.items()):
        print(f"  {key}: {count} days")

    # Signal summary
    print(f"\nRange Trader Signal Summary:")
    rt_buys = [r for r in results if r["rt_action"] == "BUY"]
    rt_sells = [r for r in results if r["rt_action"] == "SELL"]
    rt_holds = [r for r in results if r["rt_action"] == "HOLD"]
    print(f"  BUY signals: {len(rt_buys)}")
    for r in rt_buys:
        print(f"    {r['date']} @ ${r['price']:.2f} (BB {r['bb_pct']:.0%}, RSI {r['rsi']:.1f}, conf {r['rt_conf']:.2f})")
    print(f"  SELL signals: {len(rt_sells)}")
    for r in rt_sells:
        print(f"    {r['date']} @ ${r['price']:.2f} (BB {r['bb_pct']:.0%}, RSI {r['rsi']:.1f}, conf {r['rt_conf']:.2f})")
    print(f"  HOLD: {len(rt_holds)} days")

    # Counter-thesis signals
    print(f"\nThesis Alignment (bearish 0.72) Counter-Thesis Signals:")
    ta_buys = [r for r in results if r["ta_action"] == "BUY"]
    if ta_buys:
        for r in ta_buys:
            print(f"  {r['date']} BUY @ ${r['price']:.2f} conf={r['ta_conf']:.2f}: {r['ta_reasoning']}")
    else:
        print(f"  No counter-thesis BUY signals fired")

    ta_sells = [r for r in results if r["ta_action"] == "SELL"]
    print(f"  Thesis-aligned SELL signals: {len(ta_sells)}")
    for r in ta_sells:
        print(f"    {r['date']} @ ${r['price']:.2f} conf={r['ta_conf']:.2f}")

    return results, trades


if __name__ == "__main__":
    tsla, spy, vix = load_data()
    results, trades = simulate_range_trader(tsla, spy, vix)
