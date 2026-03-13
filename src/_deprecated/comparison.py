"""Agent comparison engine — leaderboard, correlations, harmony detection."""

import numpy as np
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR
from .ensemble import list_agents, load_agent_portfolio, load_agent_transactions, load_agent_config

logger = setup_logging("comparison")

ENSEMBLE_DIR = DATA_DIR / "ensemble"


def calculate_agent_metrics(agent_name: str) -> dict:
    """Calculate comprehensive metrics for a single agent."""
    config_data = load_agent_config(agent_name)
    portfolio = load_agent_portfolio(agent_name)
    transactions = load_agent_transactions(agent_name)

    total_value = portfolio.get("total_value", 1000)
    starting = 1000.0
    total_pnl = total_value - starting
    total_pnl_pct = (total_pnl / starting) * 100

    # Trade stats
    sells = [t for t in transactions if t["action"] == "SELL"]
    buys = [t for t in transactions if t["action"] == "BUY"]
    realized_pnl = sum(t.get("realized_pnl", 0) or 0 for t in sells)
    winning = sum(1 for t in sells if (t.get("realized_pnl", 0) or 0) > 0)
    losing = sum(1 for t in sells if (t.get("realized_pnl", 0) or 0) < 0)
    win_rate = (winning / len(sells) * 100) if sells else 0.0

    # Average trade return
    trade_returns = [(t.get("realized_pnl", 0) or 0) for t in sells]
    avg_return = np.mean(trade_returns) if trade_returns else 0.0

    # Max drawdown from transaction history
    peak = starting
    max_dd = 0
    for t in transactions:
        val = t.get("portfolio_value_after", starting)
        peak = max(peak, val)
        dd = (peak - val) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Daily returns for Sharpe ratio (approximate from transactions)
    # Use portfolio value snapshots if available
    sharpe = 0.0
    if len(transactions) > 2:
        values = [starting] + [t.get("portfolio_value_after", starting) for t in transactions]
        returns = np.diff(values) / values[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    return {
        "agent": agent_name,
        "display_name": config_data.get("display_name", agent_name),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "realized_pnl": round(realized_pnl, 2),
        "total_trades": len(transactions),
        "total_buys": len(buys),
        "total_sells": len(sells),
        "winning_trades": winning,
        "losing_trades": losing,
        "win_rate": round(win_rate, 1),
        "avg_trade_return": round(avg_return, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "learning_enabled": config_data.get("learning_enabled", False),
    }


def generate_leaderboard() -> list[dict]:
    """Generate ranked leaderboard of all agents.

    Ranks by total return %, with tiebreakers on Sharpe ratio and win rate.
    """
    agent_names = list_agents()
    if not agent_names:
        return []

    metrics = []
    for name in agent_names:
        try:
            m = calculate_agent_metrics(name)
            metrics.append(m)
        except Exception as e:
            logger.warning(f"Failed to calculate metrics for {name}: {e}")

    # Sort by total return (primary), Sharpe (secondary), win rate (tertiary)
    metrics.sort(
        key=lambda m: (m["total_pnl_pct"], m["sharpe_ratio"], m["win_rate"]),
        reverse=True,
    )

    # Add rank
    for i, m in enumerate(metrics):
        m["rank"] = i + 1

    # Save
    leaderboard = {
        "timestamp": iso_now(),
        "agents": metrics,
    }
    save_json(ENSEMBLE_DIR / "leaderboard.json", leaderboard)
    return metrics


def calculate_correlation_matrix() -> dict:
    """Calculate correlation between agent trade decisions.

    Measures: do agents agree or disagree? High correlation = redundant agents.
    Low/negative correlation = diversifying agents.
    """
    agent_names = list_agents()
    if len(agent_names) < 2:
        return {"agents": agent_names, "matrix": {}, "note": "Need 2+ agents"}

    # Build action timeseries per agent (BUY=1, SELL=-1, HOLD=0)
    agent_actions = {}
    all_timestamps = set()

    for name in agent_names:
        txns = load_agent_transactions(name)
        actions = {}
        for t in txns:
            ts = t["timestamp"][:16]  # Truncate to minute
            actions[ts] = 1 if t["action"] == "BUY" else -1
            all_timestamps.add(ts)
        agent_actions[name] = actions

    if not all_timestamps:
        return {"agents": agent_names, "matrix": {}, "note": "No transactions yet"}

    timestamps = sorted(all_timestamps)

    # Build numeric arrays
    arrays = {}
    for name in agent_names:
        arrays[name] = [agent_actions.get(name, {}).get(ts, 0) for ts in timestamps]

    # Calculate pairwise correlations
    matrix = {}
    for a in agent_names:
        matrix[a] = {}
        for b in agent_names:
            if a == b:
                matrix[a][b] = 1.0
            else:
                arr_a = np.array(arrays[a])
                arr_b = np.array(arrays[b])
                if np.std(arr_a) > 0 and np.std(arr_b) > 0:
                    corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
                else:
                    corr = 0.0
                matrix[a][b] = round(corr, 3)

    result = {
        "timestamp": iso_now(),
        "agents": agent_names,
        "matrix": matrix,
        "data_points": len(timestamps),
    }
    save_json(ENSEMBLE_DIR / "correlation_matrix.json", result)
    return result


def detect_harmony() -> dict:
    """Detect which agent combinations work better than individuals.

    Harmony = when agents with different strategies produce uncorrelated returns,
    the combined portfolio has lower drawdown than any individual.
    """
    agent_names = list_agents()
    if len(agent_names) < 2:
        return {"note": "Need 2+ agents for harmony analysis"}

    # Calculate each agent's return and max drawdown
    agent_metrics = {}
    for name in agent_names:
        try:
            agent_metrics[name] = calculate_agent_metrics(name)
        except Exception:
            continue

    if len(agent_metrics) < 2:
        return {"note": "Not enough agent data"}

    # Simulate equal-weighted blend
    avg_return = np.mean([m["total_pnl_pct"] for m in agent_metrics.values()])
    avg_drawdown = np.mean([m["max_drawdown_pct"] for m in agent_metrics.values()])
    min_drawdown = min(m["max_drawdown_pct"] for m in agent_metrics.values())

    # Check if blend has lower drawdown (diversification benefit)
    # True diversification benefit requires uncorrelated strategies
    corr = calculate_correlation_matrix()
    avg_correlation = 0.0
    corr_count = 0
    for a in agent_names:
        for b in agent_names:
            if a != b and a in corr.get("matrix", {}) and b in corr["matrix"].get(a, {}):
                avg_correlation += corr["matrix"][a][b]
                corr_count += 1
    if corr_count > 0:
        avg_correlation /= corr_count

    # Harmony score: higher when returns are good and correlation is low
    harmony_score = avg_return * (1 - abs(avg_correlation))

    result = {
        "timestamp": iso_now(),
        "blend_return_pct": round(avg_return, 2),
        "blend_avg_drawdown_pct": round(avg_drawdown, 2),
        "best_individual_drawdown_pct": round(min_drawdown, 2),
        "avg_correlation": round(avg_correlation, 3),
        "harmony_score": round(harmony_score, 2),
        "diversification_benefit": avg_correlation < 0.5,
        "agents": {
            name: {
                "return_pct": m["total_pnl_pct"],
                "drawdown_pct": m["max_drawdown_pct"],
            }
            for name, m in agent_metrics.items()
        },
    }
    save_json(ENSEMBLE_DIR / "harmony_analysis.json", result)
    return result


def get_ensemble_comparison() -> dict:
    """Full ensemble comparison — leaderboard + correlation + harmony.

    Called by reporter.py for dashboard data.
    """
    leaderboard = generate_leaderboard()
    correlation = calculate_correlation_matrix()
    harmony = detect_harmony()

    return {
        "leaderboard": leaderboard,
        "correlation": correlation,
        "harmony": harmony,
    }
