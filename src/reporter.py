"""Dashboard reporter — generates JSON data and HTML dashboard."""

from datetime import datetime
from pathlib import Path

from .utils import (
    load_config, load_json, save_json, iso_now, format_currency,
    is_market_open, now_et, DATA_DIR, DASHBOARD_DIR, setup_logging,
    get_cost_summary,
)
from .market_data import get_current_price, get_price_history
from .portfolio import (
    load_portfolio, load_transactions, get_portfolio_summary,
    SNAPSHOTS_DIR
)
from .knowledge_base import (
    get_lessons, get_patterns, get_predictions, get_strategy_scores,
    get_journal, get_prediction_accuracy, get_knowledge_summary,
    get_tsla_profile,
)

logger = setup_logging("reporter")

DATA_JSON_PATH = DASHBOARD_DIR / "data.json"


def generate_dashboard_data(full: bool = False) -> dict:
    """Generate the complete data payload for the dashboard.

    Args:
        full: If True, recalculates benchmark from scratch (slow — yfinance fetch).
              If False, reuses cached benchmark from existing data.json.
              Use full=True for daily research cycle, False for intra-day refreshes.
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

    # Benchmark (buy-and-hold) — cache for intra-day refreshes
    if full:
        benchmark = _calculate_benchmark(ticker, snapshots)
    else:
        existing = load_json(DATA_JSON_PATH, default=None)
        benchmark = existing.get("benchmark", []) if existing else []
        if not benchmark:
            benchmark = _calculate_benchmark(ticker, snapshots)

    # Knowledge
    lessons = get_lessons()
    patterns = get_patterns()
    predictions = get_predictions()
    scores = get_strategy_scores()
    accuracy = get_prediction_accuracy()
    journal = get_journal()
    profile = get_tsla_profile()

    # Strategy weight history from rebalance records
    weight_history = _build_weight_history(scores)

    # Prediction scoreboard
    scoreboard = _build_scoreboard(predictions)

    # Benchmarks comparison + graduation
    benchmarks_comparison = {}
    graduation_criteria = {}
    verdict = "too_early"
    try:
        from .benchmarks import BenchmarkTracker
        bt = BenchmarkTracker()
        benchmarks_comparison = bt.get_comparison(summary["total_value"])

        # Build agent metrics for graduation check
        agent_metrics = {
            "trading_days": len(snapshots),
            "total_trades": summary.get("total_trades", 0),
            "percentile_vs_random": benchmarks_comparison.get("percentile_vs_random", 0),
            "sharpe_ratio": 0,  # TODO: calculate from snapshots
            "max_drawdown_pct": 0,  # TODO: calculate from snapshots
            "prediction_accuracy_pct": 0,
            "beats_buy_hold_tsla": benchmarks_comparison.get("beats_buy_hold_tsla", False),
            "beats_buy_hold_spy": benchmarks_comparison.get("beats_buy_hold_spy", False),
            "beats_dca": benchmarks_comparison.get("beats_dca", False),
            "beats_random_median": benchmarks_comparison.get("beats_random_median", False),
            "regime_count": 0,
            "total_return_pct": summary.get("total_pnl_pct", 0),
        }
        # Prediction accuracy
        if accuracy.get("direction_accuracy"):
            accs = [v["accuracy_pct"] for v in accuracy["direction_accuracy"].values()]
            agent_metrics["prediction_accuracy_pct"] = sum(accs) / len(accs) if accs else 0

        graduation_result = bt.check_graduation_criteria(agent_metrics)
        graduation_criteria = graduation_result
        verdict = bt.calculate_verdict(benchmarks_comparison, graduation_result)
    except Exception as e:
        logger.warning(f"Benchmark comparison failed: {e}")

    # Health & alerts
    health = {}
    active_alerts = []
    try:
        from .observability import HealthChecker
        health = HealthChecker().check()
    except Exception:
        pass
    try:
        alerts_data = load_json(DATA_DIR.parent / "logs" / "alerts.json", default=[])
        active_alerts = [a for a in alerts_data if a.get("status") == "active"]
    except Exception:
        pass

    # Milestones
    milestones = load_json(DATA_DIR / "milestones.json", default=[])

    # API costs
    cost_summary = {}
    try:
        cost_summary = get_cost_summary(days=30)
    except Exception as e:
        logger.warning(f"Cost summary failed: {e}")

    # Hold log summary with counterfactual stats
    hold_log = load_json(DATA_DIR / "hold_log.json", default=[])
    cf_stats = _calculate_counterfactual_stats(hold_log)
    hold_summary = {
        "total_holds": len(hold_log),
        "recent": hold_log[-10:] if hold_log else [],
        "counterfactual_stats": cf_stats,
    }

    # Ensemble data
    ensemble_data = {}
    try:
        from .ensemble import list_agents, get_ensemble_summary
        agents = list_agents()
        if agents:
            ensemble_data = {
                "agents": get_ensemble_summary(),
                "leaderboard": load_json(DATA_DIR / "ensemble" / "leaderboard.json", default={}),
                "harmony": load_json(DATA_DIR / "ensemble" / "harmony_analysis.json", default={}),
            }
    except Exception as e:
        logger.warning(f"Ensemble data load failed: {e}")

    data = {
        "generated_at": iso_now(),
        "ticker": ticker,
        "current_price": current,
        "portfolio": summary,
        "transactions": transactions,
        "snapshots": snapshots,
        "benchmark": benchmark,
        "benchmarks_comparison": benchmarks_comparison,
        "graduation_criteria": graduation_criteria,
        "verdict": verdict,
        "health": health,
        "active_alerts": active_alerts,
        "milestones": milestones,
        "hold_log_summary": hold_summary,
        "lessons": lessons,
        "patterns": patterns,
        "predictions": scoreboard,
        "prediction_accuracy": accuracy,
        "strategy_scores": scores,
        "weight_history": weight_history,
        "journal": journal,
        "tsla_profile": profile,
        "knowledge_summary": get_knowledge_summary(),
        "market_open": is_market_open(),
        "time_et": now_et().strftime("%Y-%m-%d %H:%M ET"),
        "ensemble": ensemble_data,
        "api_costs": cost_summary,
        "config": {
            "starting_balance": config["starting_balance"],
            "strategies_enabled": config["strategies_enabled"],
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
                # Use last known value
                if benchmark:
                    benchmark.append({"date": date, "value": benchmark[-1]["value"]})

        return benchmark
    except Exception as e:
        logger.warning(f"Benchmark calculation failed: {e}")
        return []


def _build_weight_history(scores: dict) -> list:
    """Build strategy weight history from rebalance events."""
    history = []
    strategies = scores.get("strategies", {})
    rebalances = scores.get("rebalance_history", [])

    # Start with initial weights
    initial = {name: s.get("initial_weight", 0.2) for name, s in strategies.items()}
    if rebalances:
        history.append({"timestamp": rebalances[0].get("timestamp", ""), "weights": dict(initial)})
        current = dict(initial)
        for r in rebalances:
            for name, change in r.get("changes", {}).items():
                current[name] = round(current.get(name, 0.2) + change, 4)
            history.append({"timestamp": r["timestamp"], "weights": dict(current)})
    else:
        history.append({"timestamp": iso_now(), "weights": initial})

    # Add current weights
    current_weights = {name: s["weight"] for name, s in strategies.items()}
    history.append({"timestamp": iso_now(), "weights": current_weights})

    return history


def _calculate_counterfactual_stats(hold_log: list) -> dict:
    """Calculate aggregate stats on HOLD counterfactual outcomes."""
    scored = [h for h in hold_log if h.get("counterfactual_scored")]
    if not scored:
        return {"total_scored": 0}

    correct_holds = 0
    missed_gains = 0
    total_missed_pnl = 0.0

    for h in scored:
        cf = h.get("counterfactual_outcome", {})
        if not isinstance(cf, dict):
            continue
        verdict = cf.get("verdict", "")
        if verdict == "correct_hold":
            correct_holds += 1
        elif verdict == "missed_gain":
            missed_gains += 1
            total_missed_pnl += abs(cf.get("hypothetical_pnl", 0))

    return {
        "total_scored": len(scored),
        "correct_holds": correct_holds,
        "missed_gains": missed_gains,
        "total_missed_pnl": round(total_missed_pnl, 2),
        "correct_hold_pct": round(correct_holds / len(scored) * 100, 1) if scored else 0,
    }


def _build_scoreboard(predictions: list) -> list:
    """Build prediction scoreboard with outcomes."""
    scoreboard = []
    for p in predictions:
        entry = {
            "id": p["id"],
            "timestamp": p["timestamp"],
            "price_at_prediction": p.get("price_at_prediction", 0),
            "predictions": p.get("predictions", {}),
            "outcomes": p.get("outcomes", {}),
            "reasoning": p.get("reasoning", "")[:150],
        }
        # Score summary
        correct = 0
        total = 0
        for horizon, outcome in entry["outcomes"].items():
            if outcome and outcome.get("direction_correct") is not None:
                total += 1
                if outcome["direction_correct"]:
                    correct += 1
        entry["score"] = f"{correct}/{total}" if total > 0 else "pending"
        scoreboard.append(entry)
    return scoreboard
