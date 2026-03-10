"""Portfolio management — trade execution, risk rules, P&L tracking.

v4: Simplified. No stop losses (Claude decides exits), no ATR sizing (Claude decides),
    no daily loss limit, no EOD close, no gap risk reduction.
    Two rules: max 50% position, $100 cash minimum.
"""

from datetime import datetime, timezone
from pathlib import Path
from .utils import (
    load_config, load_json, save_json, iso_now, generate_id,
    DATA_DIR, setup_logging
)

logger = setup_logging("portfolio")

PORTFOLIO_PATH = DATA_DIR / "portfolio.json"
TRANSACTIONS_PATH = DATA_DIR / "transactions.json"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"


def _default_portfolio(config: dict) -> dict:
    now = iso_now()
    return {
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
        "created_at": now,
        "last_updated": now,
    }


def load_portfolio() -> dict:
    config = load_config()
    portfolio = load_json(PORTFOLIO_PATH)
    if not portfolio:
        portfolio = _default_portfolio(config)
        save_portfolio(portfolio)
        logger.info(f"Initialized portfolio with ${config['starting_balance']:.2f}")
    return portfolio


def save_portfolio(portfolio: dict):
    portfolio["last_updated"] = iso_now()
    save_json(PORTFOLIO_PATH, portfolio)


def load_transactions() -> list:
    return load_json(TRANSACTIONS_PATH, default=[])


def save_transactions(transactions: list):
    save_json(TRANSACTIONS_PATH, transactions)


def update_market_price(portfolio: dict, ticker: str, current_price: float) -> dict:
    """Update holdings with current market price and recalculate P&L."""
    if ticker not in portfolio["holdings"]:
        return portfolio

    h = portfolio["holdings"][ticker]
    h["current_price"] = round(current_price, 2)

    if h["shares"] > 0:
        h["unrealized_pnl"] = round(
            (current_price - h["avg_cost_basis"]) * h["shares"], 2
        )
    else:
        h["unrealized_pnl"] = 0.0

    config = load_config()
    holdings_value = sum(
        hld["shares"] * hld["current_price"]
        for hld in portfolio["holdings"].values()
    )
    portfolio["total_value"] = round(portfolio["cash"] + holdings_value, 2)
    portfolio["total_pnl"] = round(
        portfolio["total_value"] - config["starting_balance"], 2
    )
    portfolio["total_pnl_pct"] = round(
        (portfolio["total_pnl"] / config["starting_balance"]) * 100, 2
    )
    return portfolio


def validate_trade(action: str, shares: float, price: float, portfolio: dict, config: dict = None) -> tuple[bool, str]:
    """Check if a trade passes v4 risk rules. Returns (ok, reason).

    v4 rules (simple):
    - Max 50% of portfolio value in any position
    - Keep $100 cash minimum
    - Shares must be positive
    """
    if config is None:
        config = load_config()
    risk = config["risk_params"]
    ticker = config["ticker"]

    if shares <= 0:
        return False, "Shares must be positive"

    if not risk.get("enable_fractional_shares", True) and shares != int(shares):
        return False, "Fractional shares disabled"

    total_cost = shares * price

    if action == "BUY":
        if total_cost > portfolio["cash"]:
            return False, f"Insufficient cash: need ${total_cost:.2f}, have ${portfolio['cash']:.2f}"

        # Max position size: 50% of portfolio
        max_position_pct = risk.get("max_position_pct", 0.50)
        current_holding_value = portfolio["holdings"].get(ticker, {}).get("shares", 0) * price
        new_position_value = current_holding_value + total_cost
        max_position = portfolio["total_value"] * max_position_pct
        if new_position_value > max_position:
            return False, f"Position would be ${new_position_value:.2f}, exceeds max ${max_position:.2f} ({max_position_pct*100:.0f}%)"

        # Cash reserve: keep $100 minimum
        min_cash = risk.get("min_cash_reserve", 100.00)
        cash_after = portfolio["cash"] - total_cost
        if cash_after < min_cash:
            return False, f"Would leave ${cash_after:.2f} cash, below minimum ${min_cash:.2f}"

    elif action == "SELL":
        held = portfolio["holdings"].get(ticker, {}).get("shares", 0)
        if shares > held:
            return False, f"Can't sell {shares:.4f} shares, only hold {held:.4f}"
    else:
        return False, f"Unknown action: {action}"

    return True, "OK"


def execute_trade(action: str, shares: float, price: float, decision: dict = None) -> dict:
    """Execute a simulated trade. Returns the transaction record.

    v4: simplified. No trade_context, no constraints. Claude decided, we execute.
    """
    config = load_config()
    ticker = config["ticker"]
    portfolio = load_portfolio()
    transactions = load_transactions()

    # Apply flat slippage
    risk = config["risk_params"]
    slippage = risk.get("slippage_per_side_pct", 0.0005)
    if action == "BUY":
        exec_price = round(price * (1 + slippage), 2)
    else:
        exec_price = round(price * (1 - slippage), 2)

    # Validate
    ok, reason = validate_trade(action, shares, exec_price, portfolio)
    if not ok:
        logger.warning(f"Trade rejected: {reason}")
        return {"status": "rejected", "reason": reason}

    total_cost = round(shares * exec_price, 2)
    h = portfolio["holdings"].setdefault(ticker, {
        "shares": 0.0, "avg_cost_basis": 0.0,
        "current_price": 0.0, "unrealized_pnl": 0.0,
    })

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

    portfolio = update_market_price(portfolio, ticker, exec_price)
    portfolio["total_trades"] += 1
    if action == "SELL":
        if realized_pnl >= 0:
            portfolio["winning_trades"] += 1
        else:
            portfolio["losing_trades"] += 1

    # Build transaction record (simplified for v4)
    txn_id = generate_id("txn", [t["id"] for t in transactions])
    txn = {
        "id": txn_id,
        "timestamp": iso_now(),
        "action": action,
        "ticker": ticker,
        "shares": shares,
        "price": exec_price,
        "total_cost": total_cost,
        "realized_pnl": realized_pnl if action == "SELL" else None,
        "cash_after": portfolio["cash"],
        "portfolio_value_after": portfolio["total_value"],
        "confidence": decision.get("confidence", 0) if decision else 0,
        "strategy": decision.get("strategy", "") if decision else "",
        "reasoning": decision.get("reasoning", "") if decision else "",
    }

    transactions.append(txn)
    save_transactions(transactions)
    save_portfolio(portfolio)

    logger.info(
        f"{action} {shares:.4f} {ticker} @ ${exec_price:.2f} "
        f"(total: ${total_cost:.2f}, cash: ${portfolio['cash']:.2f}, "
        f"value: ${portfolio['total_value']:.2f})"
    )

    return {"status": "executed", "transaction": txn, "portfolio": portfolio}


def check_cooldown(ticker: str) -> bool:
    """Returns True if cooldown period has passed since last trade.

    v4: flat 15-minute cooldown, no ATR scaling.
    """
    config = load_config()
    cooldown = config["risk_params"].get("cooldown_minutes", 15)

    transactions = load_transactions()
    ticker_trades = [t for t in transactions if t["ticker"] == ticker]
    if not ticker_trades:
        return True

    last_trade_time = datetime.fromisoformat(ticker_trades[-1]["timestamp"])
    now = datetime.now(timezone.utc)
    minutes_since = (now - last_trade_time).total_seconds() / 60

    return minutes_since >= cooldown


def get_portfolio_summary() -> dict:
    """Current state with all P&L calculations."""
    config = load_config()
    portfolio = load_portfolio()
    transactions = load_transactions()

    realized_pnl = sum(
        t.get("realized_pnl", 0) or 0
        for t in transactions if t["action"] == "SELL"
    )

    unrealized_pnl = sum(
        h["unrealized_pnl"] for h in portfolio["holdings"].values()
    )

    win_rate = 0.0
    if portfolio["total_trades"] > 0:
        sells = portfolio["winning_trades"] + portfolio["losing_trades"]
        if sells > 0:
            win_rate = round(portfolio["winning_trades"] / sells * 100, 1)

    return {
        **portfolio,
        "realized_pnl": round(realized_pnl, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "win_rate": win_rate,
        "starting_balance": config["starting_balance"],
    }


def save_snapshot():
    """Save current portfolio state to daily snapshot."""
    portfolio = load_portfolio()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot_path = SNAPSHOTS_DIR / f"{today}.json"

    snapshot = {
        "date": today,
        "total_value": portfolio["total_value"],
        "cash": portfolio["cash"],
        "holdings": portfolio["holdings"],
        "total_pnl": portfolio["total_pnl"],
        "total_pnl_pct": portfolio["total_pnl_pct"],
        "total_trades": portfolio["total_trades"],
        "timestamp": iso_now(),
    }
    save_json(snapshot_path, snapshot)
    logger.info(f"Saved snapshot: {today} — value ${portfolio['total_value']:.2f}")
