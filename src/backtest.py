"""Walk-forward backtest engine — validate strategies have any edge before trading live."""

import random
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from .market_data import calculate_indicators
from .utils import setup_logging

logger = setup_logging("backtest")


@dataclass
class BacktestResult:
    name: str
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown_pct: float
    sharpe_ratio: float
    final_value: float
    trades: list = field(default_factory=list, repr=False)

    @property
    def win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return (self.winning_trades / total * 100) if total > 0 else 0.0


class WalkForwardBacktest:
    """Download historical OHLCV and walk forward through daily bars,
    applying signals, simulating trades with ATR stops + slippage."""

    def __init__(
        self,
        ticker: str = "TSLA",
        start: str = "2022-01-01",
        end: str = "2025-12-31",
        starting_balance: float = 1000.0,
        slippage_pct: float = 0.0005,
        volatile_slippage_pct: float = 0.0015,
        max_position_pct: float = 0.65,
        max_risk_per_trade_pct: float = 0.02,
        atr_stop_multiplier: float = 2.5,
    ):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.starting_balance = starting_balance
        self.slippage_pct = slippage_pct
        self.volatile_slippage_pct = volatile_slippage_pct
        self.max_position_pct = max_position_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.atr_stop_multiplier = atr_stop_multiplier
        self.df = None
        self.spy_df = None
        self.vix_df = None

    def load_data(self):
        """Download historical daily OHLCV data for TSLA, SPY, and VIX."""
        logger.info(f"Downloading {self.ticker} {self.start} to {self.end}...")
        stock = yf.Ticker(self.ticker)
        self.df = stock.history(start=self.start, end=self.end, interval="1d")
        if self.df.empty:
            raise ValueError(f"No data for {self.ticker} in range {self.start}-{self.end}")
        logger.info(f"Loaded {len(self.df)} daily bars for {self.ticker}")

        # Download SPY for regime classification
        logger.info("Downloading SPY data for regime classification...")
        try:
            self.spy_df = yf.Ticker("SPY").history(start=self.start, end=self.end, interval="1d")
            logger.info(f"Loaded {len(self.spy_df)} daily bars for SPY")
        except Exception as e:
            logger.warning(f"SPY download failed: {e}")
            self.spy_df = pd.DataFrame()

        # Download VIX for volatility regime
        logger.info("Downloading VIX data...")
        try:
            self.vix_df = yf.Ticker("^VIX").history(start=self.start, end=self.end, interval="1d")
            logger.info(f"Loaded {len(self.vix_df)} daily bars for VIX")
        except Exception as e:
            logger.warning(f"VIX download failed: {e}")
            self.vix_df = pd.DataFrame()

        return self.df

    def _simulate(self, signals: list[dict], name: str) -> BacktestResult:
        """Walk through bars with pre-computed signals, apply ATR stops + VIX-aware slippage."""
        cash = self.starting_balance
        shares = 0.0
        entry_price = 0.0
        stop_price = 0.0
        peak_value = self.starting_balance
        max_dd = 0.0
        trades = []
        daily_returns = []
        prev_value = self.starting_balance

        for i, sig in enumerate(signals):
            price = sig["close"]
            atr = sig.get("atr", price * 0.03)
            vix = sig.get("vix", 20)
            action = sig.get("action", "HOLD")

            # VIX-aware slippage: volatile rate when VIX > 25
            slippage = self.volatile_slippage_pct if vix > 25 else self.slippage_pct

            # Check stop loss first
            if shares > 0 and price <= stop_price:
                sell_price = price * (1 - slippage)
                pnl = (sell_price - entry_price) * shares
                cash += shares * sell_price
                trades.append({"action": "SELL", "price": sell_price, "pnl": pnl, "reason": "stop_loss"})
                shares = 0.0

            # Execute signal
            if action == "BUY" and shares == 0 and cash > 0:
                buy_price = price * (1 + slippage)
                stop_dist = atr * self.atr_stop_multiplier
                risk_amount = (cash + shares * price) * self.max_risk_per_trade_pct
                risk_shares = risk_amount / stop_dist if stop_dist > 0 else 0
                max_trade = (cash + shares * price) * self.max_position_pct
                trade_shares = max_trade / buy_price if buy_price > 0 else 0
                buy_shares = min(risk_shares, trade_shares, cash / buy_price)

                if buy_shares > 0.001:
                    cost = buy_shares * buy_price
                    cash -= cost
                    shares = buy_shares
                    entry_price = buy_price
                    stop_price = buy_price - stop_dist
                    trades.append({"action": "BUY", "price": buy_price, "shares": buy_shares})

            elif action == "SELL" and shares > 0:
                sell_price = price * (1 - slippage)
                pnl = (sell_price - entry_price) * shares
                cash += shares * sell_price
                trades.append({"action": "SELL", "price": sell_price, "pnl": pnl, "reason": "signal"})
                shares = 0.0

            # Track portfolio value
            total_value = cash + shares * price
            daily_ret = (total_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_ret)
            prev_value = total_value
            peak_value = max(peak_value, total_value)
            dd = (peak_value - total_value) / peak_value if peak_value > 0 else 0
            max_dd = max(max_dd, dd)

        # Final value
        final_price = signals[-1]["close"] if signals else 0
        final_value = cash + shares * final_price

        # Sharpe ratio (annualized)
        if daily_returns:
            arr = np.array(daily_returns)
            mean_ret = np.mean(arr)
            std_ret = np.std(arr)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        total_return = ((final_value - self.starting_balance) / self.starting_balance) * 100
        wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losses = sum(1 for t in trades if t.get("pnl", 0) < 0)

        return BacktestResult(
            name=name,
            total_return_pct=round(total_return, 2),
            total_trades=len([t for t in trades if t["action"] == "BUY"]),
            winning_trades=wins,
            losing_trades=losses,
            max_drawdown_pct=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            final_value=round(final_value, 2),
            trades=trades,
        )

    def _prepare_indicators(self) -> list[dict]:
        """Pre-compute indicators for all bars, including VIX and SPY regime."""
        # Build VIX lookup by date
        vix_lookup = {}
        if self.vix_df is not None and not self.vix_df.empty:
            for idx, row in self.vix_df.iterrows():
                vix_lookup[str(idx.date())] = float(row["Close"])

        # Build SPY regime lookup (50-day slope)
        spy_lookup = {}
        if self.spy_df is not None and not self.spy_df.empty:
            spy_close = self.spy_df["Close"]
            for i in range(50, len(self.spy_df)):
                date_str = str(self.spy_df.index[i].date())
                sma50 = float(spy_close.iloc[i - 49:i + 1].mean())
                current = float(spy_close.iloc[i])
                prev = float(spy_close.iloc[i - 1]) if i > 0 else current
                spy_lookup[date_str] = {
                    "spy_daily_change": (current - prev) / prev if prev > 0 else 0,
                    "spy_trend": "bull" if current > sma50 else "bear",
                }

        bars = []
        for i in range(50, len(self.df)):
            window = self.df.iloc[max(0, i - 100):i + 1]
            ind = calculate_indicators(window)
            row = self.df.iloc[i]
            date_str = str(self.df.index[i].date())

            # Add VIX
            vix = vix_lookup.get(date_str, 20.0)

            # Add SPY regime
            spy = spy_lookup.get(date_str, {"spy_daily_change": 0, "spy_trend": "sideways"})

            # Classify regime: trend from SPY slope, volatility from VIX
            if vix < 20:
                vol_regime = "low"
            elif vix < 30:
                vol_regime = "normal"
            else:
                vol_regime = "high"

            bars.append({
                "date": date_str,
                "close": float(row["Close"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "volume": int(row["Volume"]),
                "vix": vix,
                "spy_trend": spy["spy_trend"],
                "spy_daily_change": spy["spy_daily_change"],
                "vol_regime": vol_regime,
                **ind,
            })
        return bars

    def run_momentum_strategy(self, bars: list[dict]) -> BacktestResult:
        """Momentum: Buy when price > SMA20 > SMA50, RSI 50-70."""
        signals = []
        for b in bars:
            price = b["close"]
            sma20 = b.get("sma_20")
            sma50 = b.get("sma_50")
            rsi = b.get("rsi_14")
            action = "HOLD"
            if sma20 and sma50 and rsi:
                if price > sma20 > sma50 and 50 <= rsi <= 70:
                    action = "BUY"
                elif price < sma20 and (rsi > 75 or price < sma50):
                    action = "SELL"
            signals.append({**b, "action": action})
        return self._simulate(signals, "momentum")

    def run_mean_reversion_strategy(self, bars: list[dict]) -> BacktestResult:
        """Mean reversion: Buy at lower BB + low RSI."""
        signals = []
        for b in bars:
            price = b["close"]
            bb_lower = b.get("bollinger_lower")
            bb_upper = b.get("bollinger_upper")
            rsi = b.get("rsi_14")
            action = "HOLD"
            if bb_lower and bb_upper and rsi:
                if price <= bb_lower and rsi < 35:
                    action = "BUY"
                elif price >= bb_upper and rsi > 70:
                    action = "SELL"
            signals.append({**b, "action": action})
        return self._simulate(signals, "mean_reversion")

    def run_technical_strategy(self, bars: list[dict]) -> BacktestResult:
        """Technical: MACD crossover based."""
        signals = []
        for b in bars:
            cross = b.get("macd_crossover", "none")
            action = "HOLD"
            if cross == "bullish_crossover":
                action = "BUY"
            elif cross == "bearish_crossover":
                action = "SELL"
            signals.append({**b, "action": action})
        return self._simulate(signals, "technical_macd")

    def run_sentiment_proxy_strategy(self, bars: list[dict]) -> BacktestResult:
        """Sentiment proxy using volume + price patterns (no LLM needed in backtest).
        Buy on volume breakouts with positive close; sell on volume spikes with negative close."""
        signals = []
        for i, b in enumerate(bars):
            action = "HOLD"
            vol = b.get("volume", 0)
            vol_sma = b.get("volume_sma_20")
            close = b["close"]
            open_p = b["open"]
            rsi = b.get("rsi_14")

            if vol_sma and vol_sma > 0:
                vol_ratio = vol / vol_sma

                # High volume + positive close = bullish sentiment proxy
                if vol_ratio > 1.5 and close > open_p and (rsi is None or rsi < 70):
                    action = "BUY"
                # High volume + negative close = bearish sentiment proxy
                elif vol_ratio > 1.5 and close < open_p and (rsi is None or rsi > 30):
                    action = "SELL"
                # Gap up with follow-through
                elif i > 0 and close > bars[i - 1]["close"] * 1.02 and close > open_p:
                    action = "BUY"
                # Gap down with follow-through
                elif i > 0 and close < bars[i - 1]["close"] * 0.98 and close < open_p:
                    action = "SELL"

            signals.append({**b, "action": action})
        return self._simulate(signals, "sentiment_proxy")

    def run_dca_strategy(self, bars: list[dict]) -> BacktestResult:
        """DCA as a strategy through the simulation engine — buys every 5 bars (weekly)."""
        signals = []
        for i, b in enumerate(bars):
            # Buy every ~5 bars (approximately weekly)
            action = "BUY" if i % 5 == 0 else "HOLD"
            signals.append({**b, "action": action})
        return self._simulate(signals, "dca")

    def run_buy_and_hold(self) -> BacktestResult:
        """Buy on day 1, hold forever."""
        if self.df is None or self.df.empty:
            return BacktestResult("buy_and_hold", 0, 0, 0, 0, 0, 0, self.starting_balance)

        start_price = float(self.df["Close"].iloc[0])
        end_price = float(self.df["Close"].iloc[-1])
        shares = self.starting_balance / (start_price * (1 + self.slippage_pct))
        final = shares * end_price
        ret = ((final - self.starting_balance) / self.starting_balance) * 100

        # Drawdown
        closes = self.df["Close"].values
        values = shares * closes
        peak = np.maximum.accumulate(values)
        dd = np.max((peak - values) / peak) * 100

        # Sharpe
        daily_ret = np.diff(closes) / closes[:-1]
        sharpe = (np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if np.std(daily_ret) > 0 else 0

        return BacktestResult("buy_and_hold", round(ret, 2), 1, 0, 0, round(dd, 2), round(sharpe, 3), round(final, 2))

    def run_dca(self, weekly_amount: float = 50.0) -> BacktestResult:
        """Dollar cost average — buy fixed amount weekly."""
        if self.df is None or self.df.empty:
            return BacktestResult("dca", 0, 0, 0, 0, 0, 0, self.starting_balance)

        cash = self.starting_balance
        shares = 0.0
        trades_count = 0
        peak_value = self.starting_balance
        max_dd = 0.0
        daily_returns = []
        prev_value = self.starting_balance
        last_buy_week = -1

        for i in range(len(self.df)):
            price = float(self.df["Close"].iloc[i])
            week_num = self.df.index[i].isocalendar()[1]

            if week_num != last_buy_week and cash >= weekly_amount:
                buy_price = price * (1 + self.slippage_pct)
                buy_shares = weekly_amount / buy_price
                cash -= weekly_amount
                shares += buy_shares
                trades_count += 1
                last_buy_week = week_num

            total = cash + shares * price
            daily_ret = (total - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_ret)
            prev_value = total
            peak_value = max(peak_value, total)
            dd = (peak_value - total) / peak_value if peak_value > 0 else 0
            max_dd = max(max_dd, dd)

        final = cash + shares * float(self.df["Close"].iloc[-1])
        ret = ((final - self.starting_balance) / self.starting_balance) * 100
        arr = np.array(daily_returns)
        sharpe = (np.mean(arr) / np.std(arr) * np.sqrt(252)) if np.std(arr) > 0 else 0

        return BacktestResult("dca_weekly", round(ret, 2), trades_count, 0, 0, round(max_dd * 100, 2), round(sharpe, 3), round(final, 2))

    def run_random_traders(self, n: int = 100, bars: list[dict] = None) -> list[BacktestResult]:
        """Run N random traders with same rules — the null hypothesis."""
        results = []
        for i in range(n):
            signals = []
            for b in bars:
                r = random.random()
                if r < 0.15:
                    action = "BUY"
                elif r < 0.25:
                    action = "SELL"
                else:
                    action = "HOLD"
                signals.append({**b, "action": action})
            result = self._simulate(signals, f"random_{i+1:03d}")
            results.append(result)
        return results

    def run_all(self) -> dict:
        """Run all strategies + benchmarks + random traders. Print comparison."""
        if self.df is None:
            self.load_data()

        logger.info("Pre-computing indicators...")
        bars = self._prepare_indicators()
        logger.info(f"Prepared {len(bars)} bars with indicators")

        # Run strategies (all 5)
        logger.info("Running strategies...")
        momentum = self.run_momentum_strategy(bars)
        mean_rev = self.run_mean_reversion_strategy(bars)
        technical = self.run_technical_strategy(bars)
        sentiment = self.run_sentiment_proxy_strategy(bars)
        dca_strat = self.run_dca_strategy(bars)

        # Benchmarks
        logger.info("Running benchmarks...")
        buy_hold = self.run_buy_and_hold()
        dca = self.run_dca()

        # Random traders
        logger.info("Running 100 random traders...")
        randoms = self.run_random_traders(100, bars)
        random_returns = sorted([r.total_return_pct for r in randoms])
        random_median = np.median(random_returns)
        random_p25 = np.percentile(random_returns, 25)
        random_p75 = np.percentile(random_returns, 75)

        # Results
        strategies = [momentum, mean_rev, technical, sentiment, dca_strat]
        benchmarks = [buy_hold, dca]
        all_results = strategies + benchmarks

        print("\n" + "=" * 80)
        print("WALK-FORWARD BACKTEST RESULTS")
        print(f"Ticker: {self.ticker} | Period: {self.start} to {self.end} | Bars: {len(bars)}")
        print("=" * 80)

        print(f"\n{'Strategy':<20} {'Return':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>8} {'Sharpe':>8} {'Final':>10}")
        print("-" * 80)
        for r in all_results:
            print(f"{r.name:<20} {r.total_return_pct:>9.1f}% {r.total_trades:>8} {r.win_rate:>7.1f}% {r.max_drawdown_pct:>7.1f}% {r.sharpe_ratio:>8.3f} {r.final_value:>10.2f}")

        print(f"\n{'Random Traders (n=100)':<20}")
        print(f"  Median:  {random_median:>8.1f}%")
        print(f"  25th:    {random_p25:>8.1f}%")
        print(f"  75th:    {random_p75:>8.1f}%")
        print(f"  Best:    {random_returns[-1]:>8.1f}%")
        print(f"  Worst:   {random_returns[0]:>8.1f}%")

        # Compare each strategy to random
        print(f"\n{'Strategy vs Random':}")
        print("-" * 50)
        any_beats_random = False
        for r in strategies:
            pctile = sum(1 for rr in random_returns if rr < r.total_return_pct) / len(random_returns) * 100
            beats = r.total_return_pct > random_median
            marker = "BEATS" if beats else "LOSES TO"
            print(f"  {r.name:<20} {marker} random median ({pctile:.0f}th percentile)")
            if beats:
                any_beats_random = True

        print("\n" + "=" * 80)
        if any_beats_random:
            print("RESULT: At least one strategy shows edge over random. Proceed with live trading.")
        else:
            print("RESULT: PHASE_0_FAIL — No strategy beats random. Review before proceeding.")
        print("=" * 80 + "\n")

        return {
            "strategies": {r.name: r.__dict__ for r in strategies},
            "benchmarks": {r.name: r.__dict__ for r in benchmarks},
            "random_stats": {
                "median": random_median,
                "p25": random_p25,
                "p75": random_p75,
                "best": random_returns[-1],
                "worst": random_returns[0],
            },
            "any_beats_random": any_beats_random,
            "bars_count": len(bars),
        }
