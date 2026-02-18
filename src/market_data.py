"""Market data service — price fetching and technical indicators."""

import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
from .utils import setup_logging

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


def get_market_summary(ticker: str) -> dict:
    """Bundle current price + indicators + recent history into one payload."""
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

    return {
        "ticker": ticker,
        "current": current,
        "daily_indicators": daily_indicators,
        "intraday_indicators": intraday_indicators if has_intraday else None,
        "recent_days": recent_days,
        "fetched_at": str(datetime.utcnow()),
    }


def _last(series: pd.Series) -> float | None:
    """Get the last non-NaN value from a series, rounded."""
    val = series.dropna()
    if val.empty:
        return None
    return round(float(val.iloc[-1]), 4)
