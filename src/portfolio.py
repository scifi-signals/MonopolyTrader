"""Portfolio management — trade execution, risk rules, P&L tracking."""

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

    # Recalculate total value
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
    """Check if a trade passes all risk rules. Returns (ok, reason)."""
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
        # Check cash available
        if total_cost > portfolio["cash"]:
            return False, f"Insufficient cash: need ${total_cost:.2f}, have ${portfolio['cash']:.2f}"

        # Max single trade size
        max_trade = portfolio["total_value"] * risk["max_single_trade_pct"]
        if total_cost > max_trade:
            return False, f"Trade ${total_cost:.2f} exceeds max single trade ${max_trade:.2f} ({risk['max_single_trade_pct']*100:.0f}% of portfolio)"

        # Max position size
        current_holding_value = portfolio["holdings"].get(ticker, {}).get("shares", 0) * price
        new_position_value = current_holding_value + total_cost
        max_position = portfolio["total_value"] * risk["max_position_pct"]
        if new_position_value > max_position:
            return False, f"Position would be ${new_position_value:.2f}, exceeds max ${max_position:.2f} ({risk['max_position_pct']*100:.0f}%)"

        # Cash reserve
        cash_after = portfolio["cash"] - total_cost
        min_cash = portfolio["total_value"] * risk["min_cash_reserve_pct"]
        if cash_after < min_cash:
            return False, f"Would leave ${cash_after:.2f} cash, below minimum reserve ${min_cash:.2f}"

    elif action == "SELL":
        held = portfolio["holdings"].get(ticker, {}).get("shares", 0)
        if shares > held:
            return False, f"Can't sell {shares:.4f} shares, only hold {held:.4f}"
    else:
        return False, f"Unknown action: {action}"

    return True, "OK"


def execute_trade(action: str, shares: float, price: float, decision: dict = None) -> dict:
    """Execute a simulated trade. Returns the transaction record."""
    config = load_config()
    ticker = config["ticker"]
    portfolio = load_portfolio()
    transactions = load_transactions()

    # Apply slippage — use volatile rate when VIX > 25
    risk = config["risk_params"]
    base_slippage = risk.get("slippage_per_side_pct", config.get("slippage_pct", 0.001))
    volatile_slippage = risk.get("slippage_volatile_per_side_pct", base_slippage)
    # Check if we're in volatile conditions (decision dict may carry vix)
    vix = (decision or {}).get("_vix", 0)
    slippage = volatile_slippage if vix > 25 else base_slippage
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
        # Update average cost basis
        old_total = h["shares"] * h["avg_cost_basis"]
        new_total = old_total + total_cost
        h["shares"] = round(h["shares"] + shares, 6)
        h["avg_cost_basis"] = round(new_total / h["shares"], 2) if h["shares"] > 0 else 0.0
        portfolio["cash"] = round(portfolio["cash"] - total_cost, 2)
    elif action == "SELL":
        realized_pnl = round((exec_price - h["avg_cost_basis"]) * shares, 2)
        h["shares"] = round(h["shares"] - shares, 6)
        if h["shares"] < 0.0001:  # Clean up dust
            h["shares"] = 0.0
            h["avg_cost_basis"] = 0.0
        portfolio["cash"] = round(portfolio["cash"] + total_cost, 2)

    # Update market price and recalculate
    portfolio = update_market_price(portfolio, ticker, exec_price)
    portfolio["total_trades"] += 1
    if action == "SELL":
        if realized_pnl >= 0:
            portfolio["winning_trades"] += 1
        else:
            portfolio["losing_trades"] += 1

    # Build transaction record
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
        "strategy": decision.get("strategy", "manual") if decision else "manual",
        "confidence": decision.get("confidence", 0) if decision else 0,
        "hypothesis": decision.get("hypothesis", "") if decision else "",
        "reasoning": decision.get("reasoning", "") if decision else "",
        "signals": decision.get("signals", {}) if decision else {},
        "knowledge_applied": decision.get("knowledge_applied", []) if decision else [],
        "regime": decision.get("_regime", {}) if decision else {},
        "streak_breaker": decision.get("_streak_breaker") if decision else None,
        "review": None,
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


def check_stop_losses(current_price: float, atr: float = None, vix: float = None) -> dict | None:
    """Check if any position triggers a stop loss.

    Uses dynamic ATR-based stops when atr is provided, falls back to
    a fixed 8% emergency stop otherwise.
    """
    config = load_config()
    ticker = config["ticker"]
    portfolio = load_portfolio()
    h = portfolio["holdings"].get(ticker, {})

    if h.get("shares", 0) <= 0:
        return None

    risk = config["risk_params"]
    entry = h["avg_cost_basis"]

    if risk.get("stop_loss_method") == "dynamic_atr" and atr and atr > 0:
        # ATR-based dynamic stop
        multipliers = risk.get("stop_loss_atr_multipliers", {})
        vix_thresholds = multipliers.get("vix_thresholds", [20, 30])
        vix_val = vix or 0

        if vix_val < vix_thresholds[0]:
            mult = multipliers.get("low_vix", 2.0)
        elif vix_val < vix_thresholds[1]:
            mult = multipliers.get("normal_vix", 2.5)
        else:
            mult = multipliers.get("high_vix", 3.0)

        stop_price = entry - (atr * mult)
        if current_price <= stop_price:
            loss_pct = (current_price - entry) / entry * 100
            logger.warning(
                f"ATR STOP LOSS triggered: {ticker} at ${current_price:.2f} "
                f"(entry ${entry:.2f}, ATR stop ${stop_price:.2f}, "
                f"mult={mult}, ATR={atr:.2f}, VIX={vix_val:.1f}, loss {loss_pct:.1f}%)"
            )
            return {
                "action": "SELL",
                "shares": h["shares"],
                "reason": f"ATR stop loss at ${stop_price:.2f} (ATR={atr:.2f}x{mult}, loss {loss_pct:.1f}%)",
                "strategy": "stop_loss",
            }
    else:
        # Emergency fixed stop at 8%
        loss_pct = (current_price - entry) / entry
        if loss_pct <= -0.08:
            logger.warning(
                f"EMERGENCY STOP LOSS triggered: {ticker} at ${current_price:.2f} "
                f"(cost basis ${entry:.2f}, loss {loss_pct*100:.1f}%)"
            )
            return {
                "action": "SELL",
                "shares": h["shares"],
                "reason": f"Emergency stop loss at {loss_pct*100:.1f}% loss",
                "strategy": "stop_loss",
            }
    return None


def calculate_position_size(
    portfolio_value: float,
    entry_price: float,
    atr: float,
    vix: float = 0,
    config: dict = None,
) -> float:
    """Calculate position size using inverse ATR method.

    Wider stop (higher ATR) = smaller position. Risk max 2% of portfolio per trade.
    Returns max shares to buy.
    """
    if config is None:
        config = load_config()

    risk = config["risk_params"]
    max_risk_pct = risk.get("max_risk_per_trade_pct", 0.02)
    max_trade_pct = risk.get("max_single_trade_pct", 0.20)

    # ATR stop distance
    multipliers = risk.get("stop_loss_atr_multipliers", {})
    vix_thresholds = multipliers.get("vix_thresholds", [20, 30])

    if vix < vix_thresholds[0]:
        mult = multipliers.get("low_vix", 2.0)
    elif vix < vix_thresholds[1]:
        mult = multipliers.get("normal_vix", 2.5)
    else:
        mult = multipliers.get("high_vix", 3.0)

    stop_distance = atr * mult if atr > 0 else entry_price * 0.05

    # Risk-based sizing: risk_amount / stop_distance = max shares
    risk_amount = portfolio_value * max_risk_pct
    risk_shares = risk_amount / stop_distance if stop_distance > 0 else 0

    # Also cap by max single trade %
    max_trade_value = portfolio_value * max_trade_pct
    trade_shares = max_trade_value / entry_price if entry_price > 0 else 0

    max_shares = min(risk_shares, trade_shares)

    logger.info(
        f"Position sizing: ATR={atr:.2f}, mult={mult}, stop_dist=${stop_distance:.2f}, "
        f"risk_shares={risk_shares:.4f}, trade_shares={trade_shares:.4f}, max={max_shares:.4f}"
    )
    return round(max_shares, 4)


def apply_gap_risk_reduction(max_shares: float, config: dict = None) -> float:
    """Reduce position size before known high-risk events (earnings, FOMC, etc.).

    Returns the reduced max_shares. Checks for upcoming events that create
    overnight gap risk.
    """
    if config is None:
        config = load_config()

    risk = config.get("risk_params", {})
    reduction = risk.get("gap_risk_size_reduction_pct", 0.50)

    # Check for known upcoming events that create gap risk
    from .utils import load_json, KNOWLEDGE_DIR
    earnings = load_json(KNOWLEDGE_DIR / "research" / "earnings_history.json", default={})

    # If we have any indication of upcoming earnings within 48h, apply reduction
    has_gap_risk = False
    if earnings.get("upcoming_earnings"):
        has_gap_risk = True

    # Also check if VIX is elevated (>25) which indicates potential gap risk
    # This is a softer trigger — reduce by half the gap reduction
    try:
        from .market_data import get_macro_data
        macro = get_macro_data()
        if macro.get("vix", 0) > 25:
            # Apply partial reduction for elevated VIX
            reduced = max_shares * (1 - reduction * 0.5)
            logger.info(f"Elevated VIX gap risk: reducing position from {max_shares:.4f} to {reduced:.4f}")
            return round(reduced, 4)
    except Exception:
        pass

    if has_gap_risk:
        reduced = max_shares * (1 - reduction)
        logger.info(f"Gap risk reduction: {max_shares:.4f} -> {reduced:.4f} ({reduction*100:.0f}% reduction)")
        return round(reduced, 4)

    return max_shares


def check_daily_loss_limit() -> bool:
    """Returns True if daily loss limit has been breached (should stop trading)."""
    config = load_config()
    portfolio = load_portfolio()
    limit = config["risk_params"]["daily_loss_limit_pct"]

    # Compare current value to start-of-day snapshot
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    snapshot_path = SNAPSHOTS_DIR / f"{today}.json"
    snapshot = load_json(snapshot_path)

    if not snapshot:
        return False  # No snapshot yet today, can't check

    start_value = snapshot.get("total_value", config["starting_balance"])
    current_loss = (portfolio["total_value"] - start_value) / start_value

    if current_loss <= -limit:
        logger.warning(f"Daily loss limit breached: {current_loss*100:.1f}% (limit: {limit*100:.0f}%)")
        return True
    return False


def check_cooldown(ticker: str) -> bool:
    """Returns True if cooldown period has passed since last trade on this ticker."""
    config = load_config()
    cooldown = config["risk_params"]["cooldown_minutes"]
    transactions = load_transactions()

    ticker_trades = [t for t in transactions if t["ticker"] == ticker]
    if not ticker_trades:
        return True  # No trades yet, no cooldown

    last_trade_time = datetime.fromisoformat(ticker_trades[-1]["timestamp"])
    now = datetime.now(timezone.utc)
    minutes_since = (now - last_trade_time).total_seconds() / 60

    return minutes_since >= cooldown


def get_portfolio_summary() -> dict:
    """Current state with all P&L calculations."""
    config = load_config()
    portfolio = load_portfolio()
    transactions = load_transactions()

    # Calculate realized P&L from all sells
    realized_pnl = sum(
        t.get("realized_pnl", 0) or 0
        for t in transactions if t["action"] == "SELL"
    )

    # Unrealized from current holdings
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

    # Only save if we don't already have one, or update existing
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


