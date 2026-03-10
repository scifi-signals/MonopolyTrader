"""Market data service — price fetching, technical indicators, macro regime.

v4: Added get_world_snapshot() for macro + EV peer context.
    Removed: check_macro_gate, get_bsm_signals, check_volume_spike, check_vix_change.
    Claude evaluates macro conditions directly from raw data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime
from .utils import load_config, setup_logging

logger = setup_logging("market_data")


def get_current_price(ticker: str) -> dict:
    """Fetch current/latest price data for a ticker."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="2d", interval="1m")
    using_daily = False
    if hist.empty:
        hist = stock.history(period="5d")
        using_daily = True
        if hist.empty:
            raise ValueError(f"No price data available for {ticker}")

    latest = hist.iloc[-1]
    prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest["Close"]
    change = latest["Close"] - prev_close
    change_pct = (change / prev_close) * 100 if prev_close else 0

    daily_volume = int(latest["Volume"])
    if not using_daily:
        try:
            daily_hist = stock.history(period="5d", interval="1d")
            if not daily_hist.empty:
                daily_volume = int(daily_hist["Volume"].iloc[-1])
        except Exception:
            pass

    return {
        "price": round(float(latest["Close"]), 2),
        "open": round(float(latest["Open"]), 2),
        "high": round(float(latest["High"]), 2),
        "low": round(float(latest["Low"]), 2),
        "volume": daily_volume,
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
    """Calculate technical indicators from an OHLCV DataFrame."""
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

    # MACD crossover state
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

    # ADX
    if len(df) >= 28:
        adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
        indicators["adx"] = _last(adx_ind.adx())
        indicators["adx_pos"] = _last(adx_ind.adx_pos())
        indicators["adx_neg"] = _last(adx_ind.adx_neg())
    else:
        indicators["adx"] = None
        indicators["adx_pos"] = None
        indicators["adx_neg"] = None

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
    """Classify market regime using TSLA ADX + SPY slope + VIX.

    v4: Simplified — no hysteresis file saving. Returns regime dict directly.
    """
    regime = {
        "trend": "sideways", "directional": "range_bound",
        "volatility": "normal", "vix": 0.0, "adx": 0.0,
        "strategy_mode": "mean_reversion",
    }

    if macro_data is None:
        macro_data = get_macro_data()

    vix = macro_data.get("vix_level", 0)
    regime["vix"] = vix

    if vix < 20:
        regime["volatility"] = "low"
    elif vix < 30:
        regime["volatility"] = "normal"
    else:
        regime["volatility"] = "high"

    # ADX-based directional classification
    if daily_df is not None and len(daily_df) >= 30:
        try:
            adx_ind = ta.trend.ADXIndicator(
                daily_df["High"], daily_df["Low"], daily_df["Close"], window=14
            )
            adx_series = adx_ind.adx().dropna()
            if len(adx_series) >= 3:
                adx_value = float(adx_series.iloc[-1])
                regime["adx"] = round(adx_value, 2)

                adx_avg_3 = float(adx_series.tail(3).mean())
                regime["adx_avg_3d"] = round(adx_avg_3, 2)

                if adx_avg_3 >= 25:
                    regime["directional"] = "trending"
                elif adx_avg_3 <= 20:
                    regime["directional"] = "range_bound"
                # 20-25 dead zone: default to range_bound

                # +DI vs -DI
                di_pos = adx_ind.adx_pos().dropna()
                di_neg = adx_ind.adx_neg().dropna()
                if len(di_pos) > 0 and len(di_neg) > 0:
                    regime["di_positive"] = round(float(di_pos.iloc[-1]), 2)
                    regime["di_negative"] = round(float(di_neg.iloc[-1]), 2)
        except Exception as e:
            logger.warning(f"ADX calculation failed: {e}")

    # SPY 50-day slope for trend classification
    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="3mo", interval="1d")
        if len(spy_hist) >= 50:
            close_50 = spy_hist["Close"].tail(50)
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

            regime["spy_slope_pct_per_day"] = round(slope_pct, 4)
    except Exception as e:
        logger.warning(f"SPY trend classification failed: {e}")

    # Strategy mode recommendation
    if regime["directional"] == "trending":
        regime["strategy_mode"] = "momentum"
    else:
        regime["strategy_mode"] = "mean_reversion"

    logger.info(
        f"Regime: {regime['directional']} (ADX={regime['adx']:.1f}) "
        f"trend={regime['trend']}, vol={regime['volatility']}, VIX={regime['vix']}"
    )
    return regime


def get_market_summary(ticker: str) -> dict:
    """Bundle current price + indicators + macro regime into one payload."""
    current = get_current_price(ticker)

    daily = get_price_history(ticker, period="3mo", interval="1d")
    daily_indicators = calculate_indicators(daily)

    try:
        intraday = get_intraday(ticker, interval="5m")
        intraday_indicators = calculate_indicators(intraday)
        has_intraday = True
    except (ValueError, Exception) as e:
        logger.info(f"Intraday data unavailable: {e}")
        intraday_indicators = {}
        has_intraday = False

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


def get_world_snapshot(config: dict = None) -> dict:
    """Fetch macro instruments + EV peers for world context.

    v4 addition: gives Claude full awareness of macro environment
    and competitor movements every cycle.
    """
    if config is None:
        config = load_config()

    world_tickers = config.get("world_tickers", {})
    macro_tickers = world_tickers.get("macro", [])
    peer_tickers = world_tickers.get("ev_peers", [])

    NAMES = {
        "^TNX": "10Y Treasury Yield",
        "CL=F": "Crude Oil WTI",
        "BTC-USD": "Bitcoin",
        "^IXIC": "NASDAQ Composite",
        "DX-Y.NYB": "US Dollar Index",
        "SPY": "S&P 500 ETF",
        "^VIX": "VIX Volatility",
    }

    result = {"macro": {}, "ev_peers": {}, "fetched_at": str(datetime.utcnow())}

    for ticker in macro_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d", interval="1d")
            if len(hist) >= 2:
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                change_pct = round((curr - prev) / prev * 100, 2)
                result["macro"][ticker] = {
                    "price": round(curr, 2),
                    "change_pct": change_pct,
                    "name": NAMES.get(ticker, ticker),
                }
        except Exception as e:
            logger.warning(f"World snapshot {ticker}: {e}")

    for ticker in peer_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d", interval="1d")
            if len(hist) >= 2:
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                change_pct = round((curr - prev) / prev * 100, 2)
                result["ev_peers"][ticker] = {
                    "price": round(curr, 2),
                    "change_pct": change_pct,
                }
        except Exception as e:
            logger.warning(f"World snapshot {ticker}: {e}")

    return result


def get_options_snapshot(ticker: str) -> dict:
    """Fetch options market data: put/call ratio, max pain, unusual volume."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        # Use nearest expiration
        nearest = expirations[0]
        chain = stock.option_chain(nearest)
        calls = chain.calls
        puts = chain.puts

        # Put/call ratio by open interest
        total_call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
        total_put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0
        pc_ratio = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0

        # Max pain: strike with highest combined OI
        all_strikes = set(calls["strike"].tolist() + puts["strike"].tolist())
        max_pain_strike = 0
        max_oi = 0
        for strike in all_strikes:
            call_oi = int(calls.loc[calls["strike"] == strike, "openInterest"].sum()) if not calls.empty else 0
            put_oi = int(puts.loc[puts["strike"] == strike, "openInterest"].sum()) if not puts.empty else 0
            total_oi = call_oi + put_oi
            if total_oi > max_oi:
                max_oi = total_oi
                max_pain_strike = strike

        # Unusual volume: any contract with volume > 5x its OI
        unusual = False
        for df in [calls, puts]:
            if "volume" in df.columns and "openInterest" in df.columns:
                mask = (df["volume"] > 5 * df["openInterest"]) & (df["volume"] > 100)
                if mask.any():
                    unusual = True
                    break

        return {
            "put_call_ratio": pc_ratio,
            "max_pain": round(float(max_pain_strike), 2),
            "unusual_volume": unusual,
            "nearest_expiry": nearest,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
        }
    except Exception as e:
        logger.warning(f"Options snapshot failed: {e}")
        return {}


def get_analyst_consensus(ticker: str) -> dict:
    """Fetch analyst recommendations and price targets."""
    try:
        stock = yf.Ticker(ticker)
        result = {}

        # Recommendations
        try:
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                latest = recs.iloc[-1]
                result["strong_buy"] = int(latest.get("strongBuy", 0))
                result["buy"] = int(latest.get("buy", 0))
                result["hold"] = int(latest.get("hold", 0))
                result["sell"] = int(latest.get("sell", 0))
                result["strong_sell"] = int(latest.get("strongSell", 0))
        except Exception:
            pass

        # Price targets
        try:
            targets = stock.analyst_price_targets
            if targets is not None:
                result["target_mean"] = round(float(targets.get("mean", 0)), 2)
                result["target_high"] = round(float(targets.get("high", 0)), 2)
                result["target_low"] = round(float(targets.get("low", 0)), 2)
                result["target_current"] = round(float(targets.get("current", 0)), 2)
        except Exception:
            pass

        return result
    except Exception as e:
        logger.warning(f"Analyst consensus failed: {e}")
        return {}


def get_institutional_data(ticker: str) -> dict:
    """Fetch institutional ownership and short interest."""
    try:
        stock = yf.Ticker(ticker)
        result = {}

        # Institutional holders
        try:
            holders = stock.institutional_holders
            if holders is not None and not holders.empty:
                result["top_holders"] = holders["Holder"].head(5).tolist()
                total_pct = holders["pctHeld"].sum() if "pctHeld" in holders.columns else 0
                result["institutional_pct"] = round(float(total_pct) * 100, 1)
        except Exception:
            pass

        # Short interest from info
        try:
            info = stock.info
            if info:
                short_pct = info.get("shortPercentOfFloat", 0)
                if short_pct:
                    result["short_interest_pct"] = round(float(short_pct) * 100, 2)
                shares_short = info.get("sharesShort", 0)
                if shares_short:
                    result["shares_short"] = int(shares_short)
        except Exception:
            pass

        return result
    except Exception as e:
        logger.warning(f"Institutional data failed: {e}")
        return {}


def _last(series: pd.Series) -> float | None:
    """Get the last non-NaN value from a series, rounded."""
    val = series.dropna()
    if val.empty:
        return None
    return round(float(val.iloc[-1]), 4)
