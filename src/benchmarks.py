"""Benchmark tracking — 4 benchmarks, graduation criteria, verdict system."""

import random
from datetime import datetime, timezone

import numpy as np
import yfinance as yf

from .utils import load_config, load_json, save_json, iso_now, DATA_DIR, setup_logging

logger = setup_logging("benchmarks")

BENCHMARKS_PATH = DATA_DIR / "benchmarks.json"


def _default_benchmarks(start_date: str, start_balance: float) -> dict:
    return {
        "start_date": start_date,
        "start_balance": start_balance,
        "last_updated": iso_now(),
        "buy_hold_tsla": {"start_price": 0, "shares": 0, "values": []},
        "buy_hold_spy": {"start_price": 0, "shares": 0, "values": []},
        "dca_tsla": {"weekly_amount": 50, "shares": 0, "cash_remaining": start_balance, "total_invested": 0, "values": [], "last_buy_week": ""},
        "random_traders": {"count": 100, "results": [], "last_run_date": ""},
    }


class BenchmarkTracker:
    """Track 4 benchmarks: Buy&Hold TSLA, Buy&Hold SPY, DCA TSLA, Random Traders."""

    def __init__(self, start_date: str = None, start_balance: float = None):
        config = load_config()
        self.start_balance = start_balance or config["starting_balance"]
        self.data = load_json(BENCHMARKS_PATH)
        if not self.data or "start_date" not in self.data:
            sd = start_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.data = _default_benchmarks(sd, self.start_balance)
            self._init_benchmarks()
            self.save()

    def save(self):
        self.data["last_updated"] = iso_now()
        save_json(BENCHMARKS_PATH, self.data)

    def _init_benchmarks(self):
        """Initialize buy-and-hold benchmarks with start prices."""
        try:
            tsla = yf.Ticker("TSLA")
            tsla_hist = tsla.history(period="5d", interval="1d")
            if not tsla_hist.empty:
                price = float(tsla_hist["Close"].iloc[-1])
                self.data["buy_hold_tsla"]["start_price"] = price
                self.data["buy_hold_tsla"]["shares"] = self.start_balance / price
        except Exception as e:
            logger.warning(f"TSLA init failed: {e}")

        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="5d", interval="1d")
            if not spy_hist.empty:
                price = float(spy_hist["Close"].iloc[-1])
                self.data["buy_hold_spy"]["start_price"] = price
                self.data["buy_hold_spy"]["shares"] = self.start_balance / price
        except Exception as e:
            logger.warning(f"SPY init failed: {e}")

    def update_daily(self, date: str = None, tsla_price: float = None, spy_price: float = None):
        """Update all benchmarks with today's closing prices."""
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Fetch prices if not provided
        if tsla_price is None:
            try:
                t = yf.Ticker("TSLA")
                h = t.history(period="2d", interval="1d")
                tsla_price = float(h["Close"].iloc[-1]) if not h.empty else 0
            except Exception:
                tsla_price = 0

        if spy_price is None:
            try:
                s = yf.Ticker("SPY")
                h = s.history(period="2d", interval="1d")
                spy_price = float(h["Close"].iloc[-1]) if not h.empty else 0
            except Exception:
                spy_price = 0

        # Buy & Hold TSLA
        bh_tsla = self.data["buy_hold_tsla"]
        if bh_tsla["shares"] > 0 and tsla_price > 0:
            value = round(bh_tsla["shares"] * tsla_price, 2)
            bh_tsla["values"].append({"date": date, "value": value, "price": tsla_price})

        # Buy & Hold SPY
        bh_spy = self.data["buy_hold_spy"]
        if bh_spy["shares"] > 0 and spy_price > 0:
            value = round(bh_spy["shares"] * spy_price, 2)
            bh_spy["values"].append({"date": date, "value": value, "price": spy_price})

        # DCA TSLA — buy $50 weekly
        dca = self.data["dca_tsla"]
        week_key = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-W%W")
        if week_key != dca.get("last_buy_week", "") and dca["cash_remaining"] >= dca["weekly_amount"] and tsla_price > 0:
            buy_shares = dca["weekly_amount"] / tsla_price
            dca["shares"] += buy_shares
            dca["cash_remaining"] -= dca["weekly_amount"]
            dca["total_invested"] += dca["weekly_amount"]
            dca["last_buy_week"] = week_key

        if tsla_price > 0:
            dca_value = round(dca["shares"] * tsla_price + dca["cash_remaining"], 2)
            dca["values"].append({"date": date, "value": dca_value})

        # Random traders — run monthly
        rt = self.data["random_traders"]
        if not rt.get("last_run_date") or date[:7] != rt["last_run_date"][:7]:
            self._update_random_traders(date)

        self.save()
        logger.info(f"Benchmarks updated for {date}")

    def _update_random_traders(self, date: str):
        """Run 100 random trader simulations using actual price history."""
        try:
            tsla = yf.Ticker("TSLA")
            start = self.data["start_date"]
            hist = tsla.history(start=start, end=date, interval="1d")
            if len(hist) < 5:
                return

            closes = hist["Close"].values
            results = []
            for i in range(100):
                cash = self.start_balance
                shares = 0.0
                for j, price in enumerate(closes):
                    r = random.random()
                    if r < 0.10 and cash > 50:
                        buy_amt = min(cash * 0.20, cash)
                        buy_shares = buy_amt / price
                        cash -= buy_amt
                        shares += buy_shares
                    elif r < 0.15 and shares > 0:
                        sell_shares = shares * 0.5
                        cash += sell_shares * price
                        shares -= sell_shares
                final = cash + shares * closes[-1]
                results.append(round(final, 2))

            self.data["random_traders"]["results"] = sorted(results)
            self.data["random_traders"]["last_run_date"] = date
        except Exception as e:
            logger.warning(f"Random traders update failed: {e}")

    def get_comparison(self, agent_value: float, window_days: int = 30) -> dict:
        """Compare agent vs all 4 benchmarks."""
        bh_tsla_vals = self.data["buy_hold_tsla"]["values"]
        bh_spy_vals = self.data["buy_hold_spy"]["values"]
        dca_vals = self.data["dca_tsla"]["values"]
        random_results = self.data["random_traders"]["results"]

        bh_tsla_value = bh_tsla_vals[-1]["value"] if bh_tsla_vals else self.start_balance
        bh_spy_value = bh_spy_vals[-1]["value"] if bh_spy_vals else self.start_balance
        dca_value = dca_vals[-1]["value"] if dca_vals else self.start_balance
        random_median = float(np.median(random_results)) if random_results else self.start_balance

        agent_return = ((agent_value - self.start_balance) / self.start_balance) * 100
        bh_tsla_return = ((bh_tsla_value - self.start_balance) / self.start_balance) * 100
        bh_spy_return = ((bh_spy_value - self.start_balance) / self.start_balance) * 100
        dca_return = ((dca_value - self.start_balance) / self.start_balance) * 100
        random_return = ((random_median - self.start_balance) / self.start_balance) * 100

        # Agent percentile vs random
        if random_results:
            pctile = sum(1 for r in random_results if r < agent_value) / len(random_results) * 100
        else:
            pctile = 50.0

        return {
            "agent": {"value": agent_value, "return_pct": round(agent_return, 2)},
            "buy_hold_tsla": {"value": bh_tsla_value, "return_pct": round(bh_tsla_return, 2)},
            "buy_hold_spy": {"value": bh_spy_value, "return_pct": round(bh_spy_return, 2)},
            "dca_tsla": {"value": dca_value, "return_pct": round(dca_return, 2)},
            "random_median": {"value": random_median, "return_pct": round(random_return, 2)},
            "alpha_vs_tsla": round(agent_return - bh_tsla_return, 2),
            "alpha_vs_spy": round(agent_return - bh_spy_return, 2),
            "percentile_vs_random": round(pctile, 1),
            "beats_buy_hold_tsla": agent_value > bh_tsla_value,
            "beats_buy_hold_spy": agent_value > bh_spy_value,
            "beats_dca": agent_value > dca_value,
            "beats_random_median": agent_value > random_median,
        }

    def check_graduation_criteria(self, agent_metrics: dict) -> dict:
        """Check all 12 graduation criteria. Returns pass/fail for each."""
        config = load_config()
        grad = config.get("graduation", {})

        criteria = {}

        # 1. Min trading days
        trading_days = agent_metrics.get("trading_days", 0)
        criteria["min_trading_days"] = {
            "required": grad.get("min_trading_days", 90),
            "actual": trading_days,
            "passed": trading_days >= grad.get("min_trading_days", 90),
        }

        # 2. Min trades
        total_trades = agent_metrics.get("total_trades", 0)
        criteria["min_trades"] = {
            "required": grad.get("min_trades", 50),
            "actual": total_trades,
            "passed": total_trades >= grad.get("min_trades", 50),
        }

        # 3. Percentile vs random
        pctile = agent_metrics.get("percentile_vs_random", 0)
        criteria["percentile_vs_random"] = {
            "required": grad.get("min_percentile_vs_random", 75),
            "actual": pctile,
            "passed": pctile >= grad.get("min_percentile_vs_random", 75),
        }

        # 4. Sharpe ratio
        sharpe = agent_metrics.get("sharpe_ratio", 0)
        criteria["sharpe_ratio"] = {
            "required": grad.get("min_sharpe_ratio", 0.5),
            "actual": sharpe,
            "passed": sharpe >= grad.get("min_sharpe_ratio", 0.5),
        }

        # 5. Max drawdown
        drawdown = agent_metrics.get("max_drawdown_pct", 100)
        criteria["max_drawdown"] = {
            "required": grad.get("max_drawdown_pct", 15),
            "actual": drawdown,
            "passed": drawdown <= grad.get("max_drawdown_pct", 15),
        }

        # 6. Prediction accuracy
        pred_acc = agent_metrics.get("prediction_accuracy_pct", 0)
        criteria["prediction_accuracy"] = {
            "required": grad.get("min_prediction_accuracy_pct", 55),
            "actual": pred_acc,
            "passed": pred_acc >= grad.get("min_prediction_accuracy_pct", 55),
        }

        # 7. Beats buy-and-hold TSLA
        criteria["beats_buy_hold_tsla"] = {
            "required": True,
            "actual": agent_metrics.get("beats_buy_hold_tsla", False),
            "passed": agent_metrics.get("beats_buy_hold_tsla", False),
        }

        # 8. Beats buy-and-hold SPY
        criteria["beats_buy_hold_spy"] = {
            "required": True,
            "actual": agent_metrics.get("beats_buy_hold_spy", False),
            "passed": agent_metrics.get("beats_buy_hold_spy", False),
        }

        # 9. Beats DCA
        criteria["beats_dca"] = {
            "required": True,
            "actual": agent_metrics.get("beats_dca", False),
            "passed": agent_metrics.get("beats_dca", False),
        }

        # 10. Beats random median
        criteria["beats_random_median"] = {
            "required": True,
            "actual": agent_metrics.get("beats_random_median", False),
            "passed": agent_metrics.get("beats_random_median", False),
        }

        # 11. Regime diversity
        regimes = agent_metrics.get("regime_count", 0)
        criteria["regime_diversity"] = {
            "required": grad.get("min_regime_diversity", 2),
            "actual": regimes,
            "passed": regimes >= grad.get("min_regime_diversity", 2),
        }

        # 12. Positive total return
        total_return = agent_metrics.get("total_return_pct", -100)
        criteria["positive_return"] = {
            "required": 0,
            "actual": total_return,
            "passed": total_return > 0,
        }

        passed = sum(1 for c in criteria.values() if c["passed"])
        total = len(criteria)

        return {
            "criteria": criteria,
            "passed": passed,
            "total": total,
            "all_passed": passed == total,
        }

    def calculate_verdict(self, comparison: dict, grad_result: dict) -> str:
        """Determine overall verdict based on benchmarks and graduation."""
        pctile = comparison.get("percentile_vs_random", 50)
        beats_tsla = comparison.get("beats_buy_hold_tsla", False)
        beats_spy = comparison.get("beats_buy_hold_spy", False)
        all_passed = grad_result.get("all_passed", False)
        passed = grad_result.get("passed", 0)
        total = grad_result.get("total", 12)

        if all_passed:
            return "graduating"
        if beats_tsla and beats_spy and pctile >= 75:
            return "outperforming"
        if pctile >= 60 and (beats_tsla or beats_spy):
            return "promising"
        if passed < 3:
            return "too_early"
        if pctile < 40:
            return "underperforming"
        return "inconclusive"
