"""Backfill historical cycle outcomes to train the signal engine.

Pulls TSLA/SPY/VIX data from yfinance, computes technical tags at each
hourly interval, records the actual 1h forward price change, and merges
into cycle_outcomes.json.

This gives the signal engine data from different market regimes (bull,
bear, sideways) instead of only learning from whatever regime happened
to be active when v8 launched.

Usage:
    python -m src.backfill                  # default 120 trading days
    python -m src.backfill --days 200       # longer lookback
    python -m src.backfill --dry-run        # preview without saving
"""

import argparse
import json
import numpy as np
import pandas as pd
import ta
import yfinance as yf
from datetime import datetime, timezone
from .utils import load_json, save_json, setup_logging, DATA_DIR

logger = setup_logging("backfill")

OUTCOMES_PATH = DATA_DIR / "cycle_outcomes.json"


def _compute_tags_from_history(
    tsla_daily: pd.DataFrame,
    spy_daily: pd.DataFrame,
    vix_daily: pd.DataFrame,
    tsla_hourly: pd.DataFrame,
    idx: int,
    ts: pd.Timestamp,
) -> dict:
    """Compute tags from historical data at a specific hourly bar.

    Replicates tags.py logic using pre-computed indicators.
    Tags we can't compute from history get neutral defaults.
    """
    price = float(tsla_hourly["Close"].iloc[idx])

    # Find the matching daily bar (same date or most recent prior)
    bar_date = ts.date()
    daily_mask = tsla_daily.index.date <= bar_date
    if not daily_mask.any():
        return None
    daily_idx = tsla_daily.index[daily_mask][-1]
    d = tsla_daily.loc[daily_idx]

    # RSI zone
    rsi = d.get("rsi_14", 50)
    if rsi < 30:
        rsi_zone = "oversold"
    elif rsi > 70:
        rsi_zone = "overbought"
    else:
        rsi_zone = "neutral"

    # Trend (price vs SMA50)
    sma50 = d.get("sma_50", 0)
    trend = "above_sma50" if (sma50 > 0 and price > sma50) else "below_sma50"

    # SMA20 position
    sma20 = d.get("sma_20", 0)
    if sma20 > 0 and price > sma20:
        sma20_pos = "above_sma20"
    elif sma20 > 0:
        sma20_pos = "below_sma20"
    else:
        sma20_pos = "no_data"

    # Trend direction from ADX + DI
    adx = d.get("adx", 20)
    di_pos = d.get("di_pos", 0)
    di_neg = d.get("di_neg", 0)

    if adx >= 25:
        regime = "trending"
        if di_pos > di_neg:
            trend_dir = "bull"
        else:
            trend_dir = "bear"
    else:
        regime = "range_bound"
        trend_dir = "sideways"

    # VIX / volatility
    vix_val = 20.0
    vix_mask = vix_daily.index.date <= bar_date
    if vix_mask.any():
        vix_idx = vix_daily.index[vix_mask][-1]
        vix_val = float(vix_daily.loc[vix_idx, "Close"])

    if vix_val < 18:
        vol_tag = "low_vix"
    elif vix_val > 25:
        vol_tag = "high_vix"
    else:
        vol_tag = "normal_vix"

    # MACD crossover
    macd_cross = d.get("macd_cross", "neutral")

    # SPY context (daily change)
    spy_change = 0.0
    spy_mask = spy_daily.index.date <= bar_date
    if spy_mask.any():
        spy_idx = spy_daily.index[spy_mask][-1]
        spy_change = float(spy_daily.loc[spy_idx, "pct_change"])

    if spy_change > 0.3:
        mkt_ctx = "spy_up"
    elif spy_change < -0.3:
        mkt_ctx = "spy_down"
    else:
        mkt_ctx = "spy_flat"

    # Time of day (ET)
    try:
        et_hour = ts.hour
        et_minute = ts.minute
        total_min = et_hour * 60 + et_minute
        if total_min < 630:  # before 10:30
            time_tag = "morning_open"
        elif total_min < 840:  # before 14:00
            time_tag = "midday"
        elif total_min < 900:  # before 15:00
            time_tag = "afternoon"
        else:
            time_tag = "power_hour"
    except Exception:
        time_tag = "midday"

    # Intraday regime from hourly ADX (if available)
    intraday_adx = tsla_hourly["adx"].iloc[idx] if "adx" in tsla_hourly.columns else None
    if intraday_adx is not None and not pd.isna(intraday_adx):
        if intraday_adx >= 25:
            intraday_regime = "trending"
        elif intraday_adx <= 18:
            intraday_regime = "range_bound"
        else:
            intraday_regime = "mixed"
    else:
        intraday_regime = "unavailable"

    # Intraday/daily RSI divergence
    intraday_rsi = tsla_hourly["rsi_14"].iloc[idx] if "rsi_14" in tsla_hourly.columns else None
    if intraday_rsi is not None and not pd.isna(intraday_rsi) and not pd.isna(rsi):
        diff = abs(float(intraday_rsi) - float(rsi))
        if diff <= 15:
            div_tag = "aligned"
        elif diff <= 30:
            div_tag = "mild_divergence"
        else:
            div_tag = "strong_divergence"
    else:
        div_tag = "no_data"

    return {
        "rsi_zone": rsi_zone,
        "trend": trend,
        "trend_direction": trend_dir,
        "sma20_position": sma20_pos,
        "volatility": vol_tag,
        "regime": regime,
        "macd": macd_cross,
        "market_context": mkt_ctx,
        "position_state": "flat",
        "event_proximity": "no_event",
        "time_of_day": time_tag,
        "intraday_daily_divergence": div_tag,
        "news_catalyst": "none",
        "options_sentiment": "unavailable",
        "intraday_regime": intraday_regime,
        "regime_age": "unknown",
    }


def _prepare_daily_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily technical indicators on TSLA daily data."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # RSI
    rsi = ta.momentum.RSIIndicator(close, window=14)
    df["rsi_14"] = rsi.rsi()

    # SMAs
    df["sma_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()

    # ADX + DI
    adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
    df["adx"] = adx_ind.adx()
    df["di_pos"] = adx_ind.adx_pos()
    df["di_neg"] = adx_ind.adx_neg()

    # MACD crossover
    macd_ind = ta.trend.MACD(close)
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()
    df["macd_cross"] = "neutral"
    for i in range(1, len(df)):
        prev_diff = macd_line.iloc[i - 1] - signal_line.iloc[i - 1]
        curr_diff = macd_line.iloc[i] - signal_line.iloc[i]
        if prev_diff < 0 and curr_diff > 0:
            df.iloc[i, df.columns.get_loc("macd_cross")] = "bullish_cross"
        elif prev_diff > 0 and curr_diff < 0:
            df.iloc[i, df.columns.get_loc("macd_cross")] = "bearish_cross"

    return df


def _prepare_hourly_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute intraday indicators on TSLA hourly data."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Intraday RSI
    rsi = ta.momentum.RSIIndicator(close, window=14)
    df["rsi_14"] = rsi.rsi()

    # Intraday ADX (for intraday_regime tag)
    try:
        adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
        df["adx"] = adx_ind.adx()
    except Exception:
        df["adx"] = pd.NA

    return df


def backfill(lookback_days: int = 120, dry_run: bool = False) -> int:
    """Pull historical data and generate cycle outcomes.

    Args:
        lookback_days: Trading days of history to pull (max ~730 for hourly).
        dry_run: If True, compute but don't save.

    Returns:
        Number of outcomes generated.
    """
    logger.info(f"Backfilling {lookback_days} days of historical data...")

    # Pull data — hourly for cycle resolution, daily for indicators
    period = f"{lookback_days}d"

    logger.info("Downloading TSLA daily data...")
    tsla_daily_raw = yf.download("TSLA", period=f"{lookback_days + 60}d", interval="1d", progress=False)
    if tsla_daily_raw.empty:
        logger.error("No TSLA daily data")
        return 0

    logger.info("Downloading TSLA hourly data...")
    tsla_hourly_raw = yf.download("TSLA", period=period, interval="1h", progress=False)
    if tsla_hourly_raw.empty:
        logger.error("No TSLA hourly data")
        return 0

    logger.info("Downloading SPY daily data...")
    spy_daily_raw = yf.download("SPY", period=f"{lookback_days + 10}d", interval="1d", progress=False)

    logger.info("Downloading VIX daily data...")
    vix_daily_raw = yf.download("^VIX", period=f"{lookback_days + 10}d", interval="1d", progress=False)

    # Handle MultiIndex columns from yfinance
    for df in [tsla_daily_raw, tsla_hourly_raw, spy_daily_raw, vix_daily_raw]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Compute indicators
    logger.info("Computing daily indicators...")
    tsla_daily = _prepare_daily_indicators(tsla_daily_raw.copy())

    logger.info("Computing hourly indicators...")
    tsla_hourly = _prepare_hourly_indicators(tsla_hourly_raw.copy())

    # SPY daily pct change
    spy_daily = spy_daily_raw.copy()
    spy_daily["pct_change"] = spy_daily["Close"].pct_change() * 100

    # Convert hourly index to ET for time-of-day tagging
    try:
        if tsla_hourly.index.tz is None:
            tsla_hourly.index = tsla_hourly.index.tz_localize("America/New_York")
        else:
            tsla_hourly.index = tsla_hourly.index.tz_convert("America/New_York")
    except Exception:
        pass  # timestamps may already be correct

    logger.info(
        f"Data loaded: {len(tsla_daily)} daily bars, "
        f"{len(tsla_hourly)} hourly bars, "
        f"range {tsla_hourly.index[0].date()} to {tsla_hourly.index[-1].date()}"
    )

    # Generate outcomes: for each hourly bar, compute tags + 1h forward change
    outcomes = []
    existing_ids = set()

    # Load existing outcomes to avoid duplicates
    existing = load_json(OUTCOMES_PATH, default=[])
    for o in existing:
        existing_ids.add(o.get("timestamp", "")[:13])  # dedup by hour

    skipped_existing = 0
    skipped_no_tags = 0

    for i in range(len(tsla_hourly) - 1):  # -1 because we need i+1 for forward return
        ts = tsla_hourly.index[i]
        ts_key = ts.isoformat()[:13]

        # Skip if we already have this hour
        if ts_key in existing_ids:
            skipped_existing += 1
            continue

        # Skip pre/post market hours
        if hasattr(ts, 'hour'):
            if ts.hour < 9 or (ts.hour == 9 and ts.minute < 30) or ts.hour >= 16:
                continue

        price = float(tsla_hourly["Close"].iloc[i])
        future_price = float(tsla_hourly["Close"].iloc[i + 1])

        if price <= 0 or future_price <= 0:
            continue

        change_1h = (future_price - price) / price

        tags = _compute_tags_from_history(
            tsla_daily, spy_daily, vix_daily_raw, tsla_hourly, i, ts
        )
        if tags is None:
            skipped_no_tags += 1
            continue

        outcome = {
            "id": f"bf_{ts.strftime('%Y%m%d_%H%M')}",
            "timestamp": ts.isoformat(),
            "price": round(price, 2),
            "tags": tags,
            "action_taken": "BACKFILL",
            "prices": {"1h": round(future_price, 2)},
            "changes": {"1h": round(change_1h, 6)},
            "resolved": True,
            "backfilled": True,
        }
        outcomes.append(outcome)

    logger.info(
        f"Generated {len(outcomes)} backfill outcomes "
        f"(skipped {skipped_existing} existing, {skipped_no_tags} no-tag)"
    )

    if not outcomes:
        logger.info("No new outcomes to add")
        return 0

    # Summary stats
    bull_tags = sum(1 for o in outcomes if o["tags"].get("trend_direction") == "bull")
    bear_tags = sum(1 for o in outcomes if o["tags"].get("trend_direction") == "bear")
    sideways_tags = sum(1 for o in outcomes if o["tags"].get("trend_direction") == "sideways")
    up_moves = sum(1 for o in outcomes if o["changes"]["1h"] > 0.001)
    dn_moves = sum(1 for o in outcomes if o["changes"]["1h"] < -0.001)

    logger.info(
        f"Regime breakdown: {bull_tags} bull, {bear_tags} bear, {sideways_tags} sideways"
    )
    logger.info(
        f"Direction breakdown: {up_moves} up, {dn_moves} down, "
        f"{len(outcomes) - up_moves - dn_moves} flat"
    )

    if dry_run:
        logger.info("DRY RUN — not saving")
        # Show a sample
        sample = outcomes[len(outcomes) // 2]
        print(json.dumps(sample, indent=2, default=str))
        return len(outcomes)

    # Merge with existing outcomes
    merged = existing + outcomes
    # Sort by timestamp
    merged.sort(key=lambda o: o.get("timestamp", ""))
    save_json(OUTCOMES_PATH, merged)

    logger.info(f"Saved {len(merged)} total outcomes ({len(existing)} existing + {len(outcomes)} new)")

    # Rebuild signal registry with new data
    from .signal_engine import rebuild_signal_registry
    reg = rebuild_signal_registry()
    logger.info(
        f"Signal registry rebuilt: {len(reg.get('signals', {}))} single-tag, "
        f"{len(reg.get('combos', {}))} combo signals from {reg.get('resolved_outcomes', 0)} outcomes"
    )

    return len(outcomes)


def main():
    parser = argparse.ArgumentParser(description="Backfill historical cycle outcomes")
    parser.add_argument("--days", type=int, default=120, help="Trading days to look back")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    count = backfill(lookback_days=args.days, dry_run=args.dry_run)
    print(f"Backfill complete: {count} outcomes generated")


if __name__ == "__main__":
    main()
