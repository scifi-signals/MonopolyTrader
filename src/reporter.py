"""Dashboard reporter — generates JSON data for the dashboard.

v5: Added Market Intelligence Document and daily briefing to dashboard data.
"""

import math
from datetime import datetime
from pathlib import Path

from .utils import (
    load_config, load_json, save_json, iso_now, format_currency,
    is_market_open, now_et, DATA_DIR, DASHBOARD_DIR, setup_logging,
)
from .market_data import get_current_price, get_price_history
from .portfolio import (
    load_portfolio, load_transactions, get_portfolio_summary,
    SNAPSHOTS_DIR
)
from .journal import load_journal, get_journal_stats
from .thesis_builder import get_learning_metrics

logger = setup_logging("reporter")

DATA_JSON_PATH = DASHBOARD_DIR / "data.json"


def generate_dashboard_data(full: bool = False) -> dict:
    """Generate the complete data payload for the dashboard.

    Args:
        full: If True, recalculates benchmark from scratch (slow — yfinance fetch).
    """
    config = load_config()
    ticker = config["ticker"]

    # Current price
    try:
        current = get_current_price(ticker)
    except Exception:
        current = {"price": 0, "change": 0, "change_pct": 0, "volume": 0}

    # Portfolio
    summary = get_portfolio_summary()

    # Transactions
    transactions = load_transactions()

    # Snapshots for portfolio value chart
    snapshots = _load_all_snapshots()

    # Benchmark (buy-and-hold)
    if full:
        benchmark = _calculate_benchmark(ticker, snapshots)
    else:
        existing = load_json(DATA_JSON_PATH, default=None)
        benchmark = existing.get("benchmark", []) if existing else []
        if not benchmark:
            benchmark = _calculate_benchmark(ticker, snapshots)

    # Trade journal
    journal_entries = load_journal()
    journal_stats = get_journal_stats()

    # Latest decision cycle for Agent's Mind card
    latest_cycle = load_json(DATA_DIR / "latest_cycle.json", default=None)

    # Market Intelligence Document
    market_intelligence = load_json(DATA_DIR / "market_intelligence.json", default={})

    # Daily briefing
    daily_briefing = load_json(DATA_DIR / "daily_briefing.json", default={})

    # Performance analytics
    performance_analytics = _build_performance_analytics(snapshots)

    # Thesis ledger / playbook
    thesis_ledger = load_json(DATA_DIR / "thesis_ledger.json", default={})
    learning_metrics = {}
    try:
        if thesis_ledger:
            learning_metrics = get_learning_metrics(thesis_ledger)
    except Exception as e:
        logger.warning(f"Learning metrics failed: {e}")

    data = {
        "generated_at": iso_now(),
        "ticker": ticker,
        "current_price": current,
        "latest_cycle": latest_cycle,
        "portfolio": summary,
        "transactions": transactions,
        "snapshots": snapshots,
        "benchmark": benchmark,
        "trade_journal": journal_entries,
        "journal_stats": journal_stats,
        "thesis_ledger": thesis_ledger,
        "learning_metrics": learning_metrics,
        "market_intelligence": market_intelligence,
        "daily_briefing": daily_briefing,
        "performance_analytics": performance_analytics,
        "market_open": is_market_open(),
        "time_et": now_et().strftime("%Y-%m-%d %H:%M ET"),
        "config": {
            "starting_balance": config["starting_balance"],
            "risk_params": config["risk_params"],
        },
    }

    # Save to dashboard directory
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    save_json(DATA_JSON_PATH, data)
    logger.info(f"Dashboard data generated: {DATA_JSON_PATH}")
    return data


def _load_all_snapshots() -> list:
    """Load all daily snapshots sorted by date."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for path in sorted(SNAPSHOTS_DIR.glob("*.json")):
        data = load_json(path)
        if data:
            snapshots.append(data)
    return snapshots


def _calculate_benchmark(ticker: str, snapshots: list) -> list:
    """Calculate buy-and-hold benchmark values aligned to snapshot dates."""
    if not snapshots:
        return []

    try:
        hist = get_price_history(ticker, period="6mo", interval="1d")
        first_date = snapshots[0].get("date", "")
        start_rows = hist.loc[hist.index.strftime("%Y-%m-%d") == first_date]
        if start_rows.empty:
            start_price = float(hist["Close"].iloc[0])
        else:
            start_price = float(start_rows["Close"].iloc[0])

        starting_balance = snapshots[0].get("total_value", 1000)
        shares_buyhold = starting_balance / start_price

        benchmark = []
        for snap in snapshots:
            date = snap.get("date", "")
            date_rows = hist.loc[hist.index.strftime("%Y-%m-%d") == date]
            if not date_rows.empty:
                close = float(date_rows["Close"].iloc[-1])
                benchmark.append({
                    "date": date,
                    "value": round(shares_buyhold * close, 2),
                })
            else:
                if benchmark:
                    benchmark.append({"date": date, "value": benchmark[-1]["value"]})

        return benchmark
    except Exception as e:
        logger.warning(f"Benchmark calculation failed: {e}")
        return []


def _build_performance_analytics(snapshots: list) -> dict:
    """Compute diagnostic analytics for the dashboard charts."""

    # Drawdown series
    drawdown_series = []
    peak = 0.0
    for snap in snapshots:
        val = snap.get("total_value", 0)
        if val > peak:
            peak = val
        dd_pct = ((val - peak) / peak * 100) if peak > 0 else 0.0
        drawdown_series.append({
            "date": snap.get("date", ""),
            "drawdown_pct": round(dd_pct, 2),
            "peak": round(peak, 2),
        })

    # Rolling Sharpe (20-day window, annualized)
    rolling_sharpe = []
    if len(snapshots) >= 2:
        values = [s.get("total_value", 0) for s in snapshots]
        daily_returns = []
        for i in range(1, len(values)):
            if values[i - 1] > 0:
                daily_returns.append((values[i] - values[i - 1]) / values[i - 1])
            else:
                daily_returns.append(0.0)

        window = 20
        for i in range(window - 1, len(daily_returns)):
            w = daily_returns[i - window + 1 : i + 1]
            mean_r = sum(w) / len(w)
            var_r = sum((r - mean_r) ** 2 for r in w) / len(w)
            std_r = math.sqrt(var_r) if var_r > 0 else 0
            sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0
            rolling_sharpe.append({
                "date": snapshots[i + 1].get("date", ""),
                "sharpe": round(sharpe, 2),
            })

    return {
        "drawdown_series": drawdown_series,
        "rolling_sharpe": rolling_sharpe,
    }
