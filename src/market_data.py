"""Market data service — price fetching, technical indicators, macro regime."""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from .utils import load_config, setup_logging

logger = setup_logging("market_data")


def get_current_price(ticker: str) -> dict:
    """Fetch current/latest price data for a ticker."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2d", interval="1m")
    if hist.empty:
        # Fallback to daily if intraday unavailable (market closed)
        hist = stock.history(period="5d")
        if hist.empty:
            raise ValueError(f"No price data available for {ticker}")

    latest = hist.iloc[-1]
    prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest["Close"]
    change = latest["Close"] - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0

    return {
        "price": round(float(latest["Close"]), 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": int(latest["Volume"]),
        "change": round(change, 2),
        "change_pct": round(change_pct, 2),
        "timestamp": str(hist.index[-1]),
    }


def get_price_history(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data."""
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No historical data for {ticker} ({period}/{interval})")
    return df


def get_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame:
    """Fetch intraday candles (last 1-2 days)."""
    stock = yf.Ticker(ticker)
    df = stock.history(period="2d", interval=interval)
    if df.empty:
        raise ValueError(f"No intraday data for {ticker}")
    return df


def calculate_indicators(df: pd.DataFrame) -> dict:
    """Calculate technical indicators from an OHLCV DataFrame.

    Expects columns: Open, High, Low, Close, Volume.
    Returns a dict of the latest indicator values.
    """
    if len(df) < 50:
        logger.warning(f"Only {len(df)} rows — some indicators may be NaN")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].astype(float)

    indicators = {}

    # RSI
    rsi = ta.momentum.RSIIndicator(close, window=14)
    indicators["rsi_14"] = _last(rsi.rsi())

    # Moving averages
    indicators["sma_20"] = _last(ta.trend.SMAIndicator(close, window=20).sma_indicator())
    indicators["sma_50"] = _last(ta.trend.SMAIndicator(close, window=50).sma_indicator())
    indicators["ema_12"] = _last(ta.trend.EMAIndicator(close, window=12).ema_indicator())
    indicators["ema_26"] = _last(ta.trend.EMAIndicator(close, window=26).ema_indicator())

    # MACD
    macd_ind = ta.trend.MACD(close)
    macd_val = _last(macd_ind.macd())
    macd_sig = _last(macd_ind.macd_signal())
    macd_hist = _last(macd_ind.macd_diff())
    indicators["macd"] = macd_val
    indicators["macd_signal"] = macd_sig
    indicators["macd_histogram"] = macd_hist

    # Determine MACD crossover state
    macd_series = macd_ind.macd_diff()
    if len(macd_series) >= 2 and pd.notna(macd_series.iloc[-1]) and pd.notna(macd_series.iloc[-2]):
        prev_hist = macd_series.iloc[-2]
        curr_hist = macd_series.iloc[-1]
        if prev_hist <= 0 < curr_hist:
            indicators["macd_crossover"] = "bullish_crossover"
        elif prev_hist >= 0 > curr_hist:
            indicators["macd_crossover"] = "bearish_crossover"
        else:
            indicators["macd_crossover"] = "none"
    else:
        indicators["macd_crossover"] = "unknown"

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    indicators["bollinger_upper"] = _last(bb.bollinger_hband())
    indicators["bollinger_lower"] = _last(bb.bollinger_lband())
    indicators["bollinger_mid"] = _last(bb.bollinger_mavg())

    # ATR
    atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
    indicators["atr"] = _last(atr.average_true_range())

    # Volume SMA
    indicators["volume_sma_20"] = _last(ta.trend.SMAIndicator(volume, window=20).sma_indicator())

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close, volume)
    indicators["obv"] = _last(obv.on_balance_volume())

    # Current price for context
    indicators["current_price"] = round(float(close.iloc[-1]), 2)

    return indicators


def get_macro_data() -> dict:
    """Fetch SPY daily change and VIX level for macro regime assessment."""
    result = {"spy_change_pct": 0.0, "vix_level": 0.0, "fetched": False}
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="5d", interval="1d")
        if len(spy_hist) >= 2:
            prev = float(spy_hist["Close"].iloc[-2])
            curr = float(spy_hist["Close"].iloc[-1])
            result["spy_change_pct"] = round((curr - prev) / prev, 4)
            result["spy_price"] = round(curr, 2)
    except Exception as e:
        logger.warning(f"SPY data fetch failed: {e}")

    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d", interval="1d")
        if not vix_hist.empty:
            result["vix_level"] = round(float(vix_hist["Close"].iloc[-1]), 2)
    except Exception as e:
        logger.warning(f"VIX data fetch failed: {e}")

    result["fetched"] = True
    return result


def classify_regime(macro_data: dict = None, daily_df: pd.DataFrame = None) -> dict:
    """Classify market regime using 50-day SPY slope + VIX terciles.

    Returns: {trend: bull|bear|sideways, volatility: low|normal|high, vix: float}
    """
    regime = {"trend": "sideways", "volatility": "normal", "vix": 0.0}

    if macro_data is None:
        macro_data = get_macro_data()

    vix = macro_data.get("vix_level", 0)
    regime["vix"] = vix

    # VIX terciles
    if vix < 20:
        regime["volatility"] = "low"
    elif vix < 30:
        regime["volatility"] = "normal"
    else:
        regime["volatility"] = "high"

    # SPY 50-day slope for trend
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="3mo", interval="1d")
        if len(spy_hist) >= 50:
            close_50 = spy_hist["Close"].tail(50)
            x = np.arange(len(close_50))
            slope = np.polyfit(x, close_50.values, 1)[0]
            # Normalize slope as % of price per day
            avg_price = close_50.mean()
            slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0

            if slope_pct > 0.05:
                regime["trend"] = "bull"
            elif slope_pct < -0.05:
                regime["trend"] = "bear"
            else:
                regime["trend"] = "sideways"

            regime["spy_slope_pct_per_day"] = round(slope_pct, 4)
    except Exception as e:
        logger.warning(f"SPY trend classification failed: {e}")

    logger.info(f"Regime: trend={regime['trend']}, volatility={regime['volatility']}, VIX={regime['vix']}")
    return regime


def check_macro_gate(config: dict = None) -> dict:
    """Check if macro conditions require elevated conviction thresholds.

    Returns: {gate_active: bool, reason: str, confidence_threshold_override: float|None}
    """
    if config is None:
        config = load_config()

    gate_config = config.get("risk_params", {}).get("macro_gate", {})
    spy_threshold = gate_config.get("spy_daily_drop_threshold", -0.02)
    vix_threshold = gate_config.get("vix_threshold", 30)
    elevated_conf = gate_config.get("elevated_confidence_required", 0.80)

    macro = get_macro_data()
    reasons = []

    if macro["spy_change_pct"] <= spy_threshold:
        reasons.append(f"SPY down {macro['spy_change_pct']*100:.1f}% (threshold: {spy_threshold*100:.0f}%)")

    if macro["vix_level"] >= vix_threshold:
        reasons.append(f"VIX at {macro['vix_level']:.1f} (threshold: {vix_threshold})")

    gate_active = len(reasons) > 0
    result = {
        "gate_active": gate_active,
        "reason": "; ".join(reasons) if reasons else "Macro conditions normal",
        "confidence_threshold_override": elevated_conf if gate_active else None,
        "spy_change_pct": macro["spy_change_pct"],
        "vix_level": macro["vix_level"],
    }

    if gate_active:
        logger.warning(f"MACRO GATE ACTIVE: {result['reason']}")

    return result


def get_market_summary(ticker: str) -> dict:
    """Bundle current price + indicators + macro regime into one payload."""
    current = get_current_price(ticker)

    # Daily data for longer-term indicators
    daily = get_price_history(ticker, period="3mo", interval="1d")
    daily_indicators = calculate_indicators(daily)

    # Intraday for short-term context
    try:
        intraday = get_intraday(ticker, interval="5m")
        intraday_indicators = calculate_indicators(intraday)
        has_intraday = True
    except (ValueError, Exception) as e:
        logger.info(f"Intraday data unavailable: {e}")
        intraday_indicators = {}
        has_intraday = False

    # Recent daily candles (last 5 days) as context
    recent_days = []
    for i in range(-min(5, len(daily)), 0):
        row = daily.iloc[i]
        recent_days.append({
            "date": str(daily.index[i].date()),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    # Macro regime data
    try:
        macro = get_macro_data()
        regime = classify_regime(macro, daily)
    except Exception as e:
        logger.warning(f"Macro/regime classification failed: {e}")
        macro = {"spy_change_pct": 0, "vix_level": 0, "fetched": False}
        regime = {"trend": "unknown", "volatility": "unknown", "vix": 0}

    return {
        "ticker": ticker,
        "current": current,
        "daily_indicators": daily_indicators,
        "intraday_indicators": intraday_indicators if has_intraday else None,
        "recent_days": recent_days,
        "macro": macro,
        "regime": regime,
        "fetched_at": str(datetime.utcnow()),
    }


def get_bsm_signals(signal_path: str = None) -> list[dict]:
    """Read latest BSM (Billionaire Signal Monitor) signals if available.

    Returns empty list if BSM not built yet or no signals present.
    Gracefully no-ops when BSM directory doesn't exist.
    """
    if signal_path is None:
        config = load_config()
        bsm_config = config.get("bsm", {})
        if not bsm_config.get("enabled", False):
            return []
        signal_path = bsm_config.get("signal_path", "data/bsm_signals/latest_signals.json")

    from pathlib import Path
    path = Path(signal_path)
    if not path.exists():
        return []

    try:
        import json
        with open(path) as f:
            signals = json.load(f)

        if not isinstance(signals, list):
            signals = [signals]

        # Filter expired signals
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        active = []
        for sig in signals:
            expires = sig.get("expires_at")
            if expires:
                try:
                    exp_dt = datetime.fromisoformat(expires)
                    if exp_dt < now:
                        continue
                except (ValueError, TypeError):
                    pass
            active.append(sig)

        if active:
            logger.info(f"BSM: {len(active)} active signals loaded")
        return active
    except Exception as e:
        logger.warning(f"BSM signal read failed: {e}")
        return []


def _last(series: pd.Series) -> float | None:
    """Get the last non-NaN value from a series, rounded."""
    val = series.dropna()
    if val.empty:
        return None
    return round(float(val.iloc[-1]), 4)
