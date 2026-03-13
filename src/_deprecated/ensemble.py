"""Multi-agent ensemble orchestrator — runs agents in parallel with independent portfolios."""

import json
import copy
from pathlib import Path
from .utils import (
    load_config, load_json, save_json, iso_now, setup_logging,
    ROOT_DIR, DATA_DIR,
)
from .market_data import get_market_summary, check_macro_gate, classify_regime
from .portfolio import (
    update_market_price, calculate_position_size,
    apply_gap_risk_reduction, check_stop_losses as _check_stop_losses,
)
from .strategies import evaluate_all_strategies, aggregate_signals
from .agent import make_decision
from .knowledge_base import get_relevant_knowledge, get_strategy_scores

logger = setup_logging("ensemble")

AGENTS_DIR = ROOT_DIR / "agents"
AGENT_DATA_DIR = DATA_DIR / "agents"


def list_agents() -> list[str]:
    """Return list of available agent names from agents/ directory."""
    if not AGENTS_DIR.exists():
        return []
    return [
        p.stem for p in sorted(AGENTS_DIR.glob("*.json"))
        if p.stem not in ("__template",)
    ]


def load_agent_config(agent_name: str) -> dict:
    """Load an agent's personality config."""
    path = AGENTS_DIR / f"{agent_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {path}")
    return load_json(path)


def _agent_data_dir(agent_name: str) -> Path:
    """Get or create the data directory for a specific agent."""
    d = AGENT_DATA_DIR / agent_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_agent_portfolio(agent_name: str) -> dict:
    """Load an agent's portfolio (separate from other agents)."""
    path = _agent_data_dir(agent_name) / "portfolio.json"
    portfolio = load_json(path)
    if not portfolio:
        config = load_config()
        portfolio = {
            "cash": config["starting_balance"],
            "holdings": {
                config["ticker"]: {
                    "shares": 0.0,
                    "avg_cost_basis": 0.0,
                    "current_price": 0.0,
                    "unrealized_pnl": 0.0,
                }
            },
            "total_value": config["starting_balance"],
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "created_at": iso_now(),
            "last_updated": iso_now(),
        }
        save_json(path, portfolio)
    return portfolio


def save_agent_portfolio(agent_name: str, portfolio: dict):
    """Save an agent's portfolio."""
    portfolio["last_updated"] = iso_now()
    save_json(_agent_data_dir(agent_name) / "portfolio.json", portfolio)


def load_agent_transactions(agent_name: str) -> list:
    """Load an agent's transaction history."""
    return load_json(_agent_data_dir(agent_name) / "transactions.json", default=[])


def save_agent_transactions(agent_name: str, transactions: list):
    """Save an agent's transactions."""
    save_json(_agent_data_dir(agent_name) / "transactions.json", transactions)


def _generate_txn_id(transactions: list) -> str:
    """Generate next transaction ID."""
    n = len(transactions) + 1
    return f"txn_{n:03d}"


def execute_agent_trade(
    agent_name: str, action: str, shares: float, price: float,
    decision: dict = None, vix: float = 0,
) -> dict:
    """Execute a simulated trade for a specific agent."""
    config = load_config()
    ticker = config["ticker"]
    portfolio = load_agent_portfolio(agent_name)
    transactions = load_agent_transactions(agent_name)

    # Slippage
    risk = config["risk_params"]
    base_slippage = risk.get("slippage_per_side_pct", 0.0005)
    volatile_slippage = risk.get("slippage_volatile_per_side_pct", 0.0015)
    slippage = volatile_slippage if vix > 25 else base_slippage

    exec_price = round(price * (1 + slippage) if action == "BUY" else price * (1 - slippage), 2)
    total_cost = round(shares * exec_price, 2)

    h = portfolio["holdings"].setdefault(ticker, {
        "shares": 0.0, "avg_cost_basis": 0.0,
        "current_price": 0.0, "unrealized_pnl": 0.0,
    })

    # Validate
    if action == "BUY":
        if total_cost > portfolio["cash"]:
            return {"status": "rejected", "reason": "insufficient cash"}
        max_trade = portfolio["total_value"] * risk["max_single_trade_pct"]
        if total_cost > max_trade:
            return {"status": "rejected", "reason": "exceeds max trade size"}
    elif action == "SELL":
        if shares > h.get("shares", 0):
            return {"status": "rejected", "reason": "insufficient shares"}

    realized_pnl = 0.0
    if action == "BUY":
        old_total = h["shares"] * h["avg_cost_basis"]
        new_total = old_total + total_cost
        h["shares"] = round(h["shares"] + shares, 6)
        h["avg_cost_basis"] = round(new_total / h["shares"], 2) if h["shares"] > 0 else 0.0
        portfolio["cash"] = round(portfolio["cash"] - total_cost, 2)
    elif action == "SELL":
        realized_pnl = round((exec_price - h["avg_cost_basis"]) * shares, 2)
        h["shares"] = round(h["shares"] - shares, 6)
        if h["shares"] < 0.0001:
            h["shares"] = 0.0
            h["avg_cost_basis"] = 0.0
        portfolio["cash"] = round(portfolio["cash"] + total_cost, 2)

    # Update market price
    h["current_price"] = round(exec_price, 2)
    if h["shares"] > 0:
        h["unrealized_pnl"] = round((exec_price - h["avg_cost_basis"]) * h["shares"], 2)
    else:
        h["unrealized_pnl"] = 0.0

    holdings_value = sum(hld["shares"] * hld["current_price"] for hld in portfolio["holdings"].values())
    portfolio["total_value"] = round(portfolio["cash"] + holdings_value, 2)
    portfolio["total_pnl"] = round(portfolio["total_value"] - config["starting_balance"], 2)
    portfolio["total_pnl_pct"] = round((portfolio["total_pnl"] / config["starting_balance"]) * 100, 2)
    portfolio["total_trades"] += 1
    if action == "SELL":
        if realized_pnl >= 0:
            portfolio["winning_trades"] += 1
        else:
            portfolio["losing_trades"] += 1

    # Build transaction record
    txn = {
        "id": _generate_txn_id(transactions),
        "timestamp": iso_now(),
        "agent": agent_name,
        "action": action,
        "ticker": ticker,
        "shares": shares,
        "price": exec_price,
        "total_cost": total_cost,
        "realized_pnl": realized_pnl if action == "SELL" else None,
        "cash_after": portfolio["cash"],
        "portfolio_value_after": portfolio["total_value"],
        "strategy": (decision or {}).get("strategy", "manual"),
        "confidence": (decision or {}).get("confidence", 0),
        "hypothesis": (decision or {}).get("hypothesis", ""),
        "reasoning": (decision or {}).get("reasoning", ""),
    }

    transactions.append(txn)
    save_agent_transactions(agent_name, transactions)
    save_agent_portfolio(agent_name, portfolio)

    logger.info(f"[{agent_name}] {action} {shares:.4f} @ ${exec_price:.2f} (value: ${portfolio['total_value']:.2f})")
    return {"status": "executed", "transaction": txn, "portfolio": portfolio}


def run_agent_cycle(agent_name: str, market_data: dict, macro_gate: dict, regime: dict) -> dict:
    """Run a single decision cycle for one agent.

    Returns a summary dict of what the agent decided.
    """
    config = load_config()
    ticker = config["ticker"]
    agent_config = load_agent_config(agent_name)
    current_price = market_data["current"]["price"]
    atr = market_data.get("daily_indicators", {}).get("atr", 0) or 0
    vix = regime.get("vix", 0)

    logger.info(f"[{agent_name}] Running cycle...")

    # Load agent-specific portfolio
    portfolio = load_agent_portfolio(agent_name)
    portfolio = update_market_price(portfolio, ticker, current_price)
    save_agent_portfolio(agent_name, portfolio)

    # Build strategy scores with agent's weight overrides
    scores = get_strategy_scores()
    agent_weights = agent_config.get("strategy_weights", {})
    for strat_name, weight in agent_weights.items():
        if strat_name in scores.get("strategies", {}):
            scores["strategies"][strat_name] = {
                **scores["strategies"][strat_name],
                "weight": weight,
            }

    # Run strategies with regime awareness
    signals = evaluate_all_strategies(market_data, portfolio, scores, regime=regime)
    aggregate = aggregate_signals(signals)

    # Position sizing
    max_shares = calculate_position_size(
        portfolio["total_value"], current_price, atr, vix, config
    )
    max_shares = apply_gap_risk_reduction(max_shares, config)

    # Knowledge context (shared across agents)
    knowledge = get_relevant_knowledge(market_data)

    # Call Claude for decision (with agent's custom prompt addon)
    decision = make_decision(
        market_data, signals, aggregate, portfolio, knowledge,
        macro_gate=macro_gate, regime=regime,
        agent_config=agent_config,
    )

    action = decision.get("action", "HOLD")
    shares = decision.get("shares", 0)

    # Macro gate override
    if macro_gate.get("gate_active") and action == "BUY":
        conf_threshold = macro_gate.get("confidence_threshold_override", 0.80)
        if decision.get("confidence", 0) < conf_threshold:
            logger.info(f"[{agent_name}] Macro gate override: BUY blocked")
            action = "HOLD"

    # Cap shares
    if action == "BUY" and shares > max_shares:
        shares = max_shares

    # Execute
    result_summary = {
        "agent": agent_name,
        "action": action,
        "shares": shares,
        "confidence": decision.get("confidence", 0),
        "strategy": decision.get("strategy", ""),
        "hypothesis": decision.get("hypothesis", ""),
        "portfolio_value": portfolio["total_value"],
    }

    if action in ("BUY", "SELL") and shares > 0:
        result = execute_agent_trade(agent_name, action, shares, current_price, decision, vix)
        result_summary["execution"] = result["status"]
        if result["status"] == "executed":
            result_summary["portfolio_value"] = result["portfolio"]["total_value"]
    else:
        result_summary["execution"] = "hold"

    return result_summary


def run_ensemble_cycle(agent_names: list[str] = None) -> list[dict]:
    """Run all agents through a decision cycle on the same market data.

    Args:
        agent_names: Specific agents to run. If None, runs all available agents.

    Returns:
        List of result summaries, one per agent.
    """
    if agent_names is None:
        agent_names = list_agents()

    if not agent_names:
        logger.warning("No agents configured")
        return []

    config = load_config()
    ticker = config["ticker"]

    # Shared market data — fetched once for all agents
    logger.info(f"=== Ensemble Cycle: {', '.join(agent_names)} ===")
    market_data = get_market_summary(ticker)
    regime = market_data.get("regime", {})
    macro_gate = check_macro_gate(config)

    if macro_gate.get("gate_active"):
        logger.warning(f"Macro gate active: {macro_gate['reason']}")

    # Run each agent sequentially (they share the Claude API, can't truly parallelize)
    results = []
    for name in agent_names:
        try:
            result = run_agent_cycle(name, market_data, macro_gate, regime)
            results.append(result)
        except Exception as e:
            logger.error(f"[{name}] Cycle failed: {e}", exc_info=True)
            results.append({
                "agent": name,
                "action": "ERROR",
                "error": str(e),
            })

    # Log summary
    logger.info("=== Ensemble Results ===")
    for r in results:
        logger.info(
            f"  [{r['agent']}] {r.get('action', '?')} "
            f"conf={r.get('confidence', 0):.2f} "
            f"value=${r.get('portfolio_value', 0):.2f}"
        )

    # Save ensemble snapshot
    snapshot = {
        "timestamp": iso_now(),
        "regime": regime,
        "macro_gate": macro_gate.get("gate_active", False),
        "agents": results,
    }
    save_json(DATA_DIR / "ensemble" / "latest_cycle.json", snapshot)

    return results


def get_ensemble_summary() -> dict:
    """Get current state of all agents for dashboard/comparison."""
    agent_names = list_agents()
    config = load_config()
    summaries = {}

    for name in agent_names:
        try:
            portfolio = load_agent_portfolio(name)
            transactions = load_agent_transactions(name)
            agent_config = load_agent_config(name)

            sells = [t for t in transactions if t["action"] == "SELL"]
            realized_pnl = sum(t.get("realized_pnl", 0) or 0 for t in sells)

            summaries[name] = {
                "display_name": agent_config.get("display_name", name),
                "description": agent_config.get("description", ""),
                "total_value": portfolio.get("total_value", config["starting_balance"]),
                "total_pnl": portfolio.get("total_pnl", 0),
                "total_pnl_pct": portfolio.get("total_pnl_pct", 0),
                "total_trades": portfolio.get("total_trades", 0),
                "winning_trades": portfolio.get("winning_trades", 0),
                "losing_trades": portfolio.get("losing_trades", 0),
                "realized_pnl": round(realized_pnl, 2),
                "cash": portfolio.get("cash", 0),
                "holdings": portfolio.get("holdings", {}),
                "last_updated": portfolio.get("last_updated", ""),
                "learning_enabled": agent_config.get("learning_enabled", False),
            }
        except Exception as e:
            logger.warning(f"Failed to load summary for {name}: {e}")

    return summaries
