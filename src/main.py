"""MonopolyTrader — main entry point with scheduler and orchestration."""

import argparse
import asyncio
import signal
import sys
import time
import schedule
from datetime import datetime

from .utils import (
    load_config, is_market_open, now_et, iso_now,
    format_currency, format_pct, setup_logging, DATA_DIR
)
from .market_data import get_current_price, get_market_summary
from .portfolio import (
    load_portfolio, execute_trade, save_portfolio, save_snapshot,
    check_stop_losses, check_daily_loss_limit, check_cooldown,
    update_market_price, get_portfolio_summary,
)
from .strategies import evaluate_all_strategies, aggregate_signals
from .agent import make_decision
from .knowledge_base import (
    initialize as init_kb, get_strategy_scores, get_relevant_knowledge,
    get_knowledge_summary,
)
from .learner import (
    review_trade, review_predictions, discover_patterns,
    evolve_strategy_weights, write_journal_entry, run_learning_cycle,
)
from .researcher import run_full_research

logger = setup_logging("main")

# Global flag for clean shutdown
_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received, finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# --- Decision Cycle (every poll interval during market hours) ---

def run_cycle():
    """The core trading cycle — fetch data, evaluate, decide, execute."""
    config = load_config()
    ticker = config["ticker"]

    if not is_market_open(config):
        return

    if check_daily_loss_limit():
        logger.warning("Daily loss limit reached — skipping cycle")
        return

    try:
        # 1. Fetch market data
        logger.info("--- Decision Cycle ---")
        market_data = get_market_summary(ticker)
        current_price = market_data["current"]["price"]
        logger.info(
            f"{ticker}: ${current_price} "
            f"({market_data['current'].get('change_pct', 0):+.2f}%)"
        )

        # 2. Update portfolio with current price
        portfolio = load_portfolio()
        portfolio = update_market_price(portfolio, ticker, current_price)
        save_portfolio(portfolio)

        # 3. Check stop losses first
        stop_signal = check_stop_losses(current_price)
        if stop_signal:
            logger.warning(f"STOP LOSS: selling {stop_signal['shares']:.4f} shares")
            result = execute_trade(
                "SELL", stop_signal["shares"], current_price,
                {"strategy": "stop_loss", "confidence": 1.0,
                 "hypothesis": stop_signal["reason"],
                 "reasoning": "Automatic stop loss execution"}
            )
            if result["status"] == "executed":
                logger.info(f"Stop loss executed: P&L ${result['transaction'].get('realized_pnl', 0):.2f}")
            return  # Don't make another decision this cycle

        # 4. Check cooldown
        if not check_cooldown(ticker):
            logger.info("Cooldown active — skipping decision")
            return

        # 5. Run strategy signals
        scores = get_strategy_scores()
        signals = evaluate_all_strategies(market_data, portfolio, scores)
        aggregate = aggregate_signals(signals)
        logger.info(f"Aggregate signal: {aggregate.action} (conf={aggregate.confidence:.2f})")

        # 6. Get knowledge context
        knowledge = get_relevant_knowledge(market_data)

        # 7. Call agent for decision
        decision = make_decision(market_data, signals, aggregate, portfolio, knowledge)
        action = decision.get("action", "HOLD")
        shares = decision.get("shares", 0)

        # 8. Execute if not HOLD
        if action in ("BUY", "SELL") and shares > 0:
            result = execute_trade(action, shares, current_price, decision)
            if result["status"] == "executed":
                txn = result["transaction"]
                logger.info(
                    f"EXECUTED: {action} {shares:.4f} @ ${txn['price']:.2f} "
                    f"| Cash: {format_currency(result['portfolio']['cash'])} "
                    f"| Value: {format_currency(result['portfolio']['total_value'])}"
                )
            elif result["status"] == "rejected":
                logger.warning(f"Trade rejected: {result['reason']}")
        else:
            logger.info(f"HOLD — {decision.get('reasoning', 'N/A')[:100]}")

        # 9. Score any pending predictions whose horizon has passed
        scored = review_predictions(current_price)
        if scored:
            logger.info(f"Scored {len(scored)} predictions")

        # 10. Review recent unreviewed trades (if we have outcomes)
        asyncio.run(_review_recent_trades(current_price))

        # 11. Handle research requests from agent
        research_req = decision.get("research_request")
        if research_req:
            logger.info(f"Agent requested research: {research_req}")
            # Queue it — will be handled in the research cycle
            _save_research_request(research_req)

        # 12. Refresh dashboard data (lightweight — cached benchmark)
        try:
            from .reporter import generate_dashboard_data
            generate_dashboard_data()
        except Exception as e:
            logger.warning(f"Dashboard update failed: {e}")

    except Exception as e:
        logger.error(f"Decision cycle error: {e}", exc_info=True)


async def _review_recent_trades(current_price: float):
    """Review any trades that haven't been reviewed yet."""
    from .portfolio import load_transactions
    transactions = load_transactions()
    unreviewed = [
        t for t in transactions
        if t.get("review") is None and t.get("hypothesis")
    ]

    # Only review trades older than 30 minutes (give the hypothesis time to play out)
    from datetime import timezone, timedelta
    now = datetime.now(timezone.utc)
    for trade in unreviewed:
        trade_time = datetime.fromisoformat(trade["timestamp"])
        if (now - trade_time).total_seconds() > 1800:  # 30 min
            logger.info(f"Reviewing trade {trade['id']}...")
            await review_trade(trade, current_price)


def _save_research_request(topic: str):
    """Save a research request for the next research cycle."""
    from .utils import load_json, save_json
    path = DATA_DIR / "research_requests.json"
    requests = load_json(path, default=[])
    requests.append({"topic": topic, "requested_at": iso_now()})
    save_json(path, requests)


# --- Research Cycle (daily, after market close) ---

async def run_daily_research():
    """Daily learning + research cycle. Runs after market close."""
    config = load_config()
    ticker = config["ticker"]
    logger.info("=== Daily Research & Learning Cycle ===")

    try:
        # Get current price for reviews
        current = get_current_price(ticker)
        current_price = current["price"]
        portfolio = load_portfolio()

        # 1. Run full learning cycle
        await run_learning_cycle(current_price, portfolio)

        # 2. Run research tasks
        logger.info("Running research tasks...")
        await run_full_research(ticker)

        # 3. Handle queued research requests
        from .utils import load_json, save_json
        req_path = DATA_DIR / "research_requests.json"
        requests = load_json(req_path, default=[])
        if requests:
            from .researcher import research_on_demand
            for req in requests[-3:]:  # Max 3 on-demand requests per day
                logger.info(f"On-demand research: {req['topic']}")
                await research_on_demand(ticker, req["topic"])
            save_json(req_path, [])  # Clear queue

        # 4. Save daily snapshot
        save_snapshot()

        # 5. Generate dashboard (full=True for daily benchmark recalculation)
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)

        # 6. Summary
        summary = get_portfolio_summary()
        logger.info(
            f"Daily summary: Value={format_currency(summary['total_value'])} "
            f"P&L={format_currency(summary['total_pnl'])} ({format_pct(summary['total_pnl_pct'])}) "
            f"Trades={summary['total_trades']} WinRate={summary['win_rate']}%"
        )
        logger.info(get_knowledge_summary())

    except Exception as e:
        logger.error(f"Daily research cycle error: {e}", exc_info=True)


# --- Bootstrap (first run) ---

async def bootstrap():
    """First-time setup: initialize everything, research before trading."""
    config = load_config()
    ticker = config["ticker"]
    logger.info("=== BOOTSTRAP: First Run Setup ===")

    # 1. Initialize knowledge base
    init_kb()
    logger.info("Knowledge base initialized")

    # 2. Initialize portfolio
    portfolio = load_portfolio()
    logger.info(f"Portfolio initialized: {format_currency(portfolio['total_value'])}")

    # 3. Run all research
    if config["learning"]["research_on_startup"]:
        logger.info("Running initial research (this may take a minute)...")
        await run_full_research(ticker)
        logger.info("Initial research complete")

    # 4. Save initial snapshot
    save_snapshot()

    # 5. Write initial journal entry
    await write_journal_entry(portfolio)

    logger.info("=== Bootstrap complete — ready to trade ===")


def _is_bootstrapped() -> bool:
    """Check if we've already bootstrapped."""
    from .utils import KNOWLEDGE_DIR
    journal = KNOWLEDGE_DIR / "journal.md"
    if not journal.exists():
        return False
    content = journal.read_text()
    return len(content) > 50  # More than just the header


# --- Scheduler ---

def run_scheduler():
    """Main scheduler loop — polls during market hours, researches after close."""
    config = load_config()
    interval = config["poll_interval_minutes"]

    logger.info(f"Scheduler started — polling every {interval} min during market hours")
    logger.info(f"Market hours: {config['market_hours']['open']}-{config['market_hours']['close']} ET")

    # Schedule decision cycles
    schedule.every(interval).minutes.do(run_cycle)

    # Schedule daily research at 4:30 PM ET (30 min after close)
    schedule.every().day.at("16:30").do(lambda: asyncio.run(run_daily_research()))

    # Schedule daily snapshot at market open
    schedule.every().day.at("09:35").do(save_snapshot)

    # Run initial cycle immediately if market is open
    if is_market_open(config):
        logger.info("Market is open — running initial cycle")
        run_cycle()
    else:
        et = now_et()
        logger.info(f"Market closed (ET: {et.strftime('%H:%M %A')}). Waiting for market hours...")

    while _running:
        schedule.run_pending()
        time.sleep(10)

    logger.info("Scheduler stopped")


# --- Single Cycle Mode ---

def run_once():
    """Run a single decision cycle (for testing)."""
    config = load_config()
    ticker = config["ticker"]

    init_kb()

    # Always run even if market is closed (for testing)
    logger.info("--- Single Cycle Mode ---")

    try:
        market_data = get_market_summary(ticker)
        current_price = market_data["current"]["price"]
        logger.info(f"{ticker}: ${current_price}")

        portfolio = load_portfolio()
        portfolio = update_market_price(portfolio, ticker, current_price)
        save_portfolio(portfolio)

        # Check stop losses
        stop_signal = check_stop_losses(current_price)
        if stop_signal:
            logger.warning(f"STOP LOSS triggered")
            execute_trade("SELL", stop_signal["shares"], current_price,
                         {"strategy": "stop_loss", "confidence": 1.0,
                          "hypothesis": stop_signal["reason"],
                          "reasoning": "Automatic stop loss"})
            return

        scores = get_strategy_scores()
        signals = evaluate_all_strategies(market_data, portfolio, scores)
        aggregate = aggregate_signals(signals)

        knowledge = get_relevant_knowledge(market_data)
        decision = make_decision(market_data, signals, aggregate, portfolio, knowledge)

        action = decision.get("action", "HOLD")
        shares = decision.get("shares", 0)

        if action in ("BUY", "SELL") and shares > 0:
            result = execute_trade(action, shares, current_price, decision)
            status = result["status"]
            if status == "executed":
                txn = result["transaction"]
                print(f"\nEXECUTED: {action} {shares:.4f} {ticker} @ ${txn['price']:.2f}")
                print(f"Cash: {format_currency(result['portfolio']['cash'])}")
                print(f"Value: {format_currency(result['portfolio']['total_value'])}")
            else:
                print(f"\nREJECTED: {result.get('reason', 'unknown')}")
        else:
            print(f"\nHOLD: {decision.get('reasoning', 'N/A')[:150]}")

        # Score predictions
        review_predictions(current_price)

        # Print portfolio
        summary = get_portfolio_summary()
        print(f"\nPortfolio: {format_currency(summary['total_value'])} "
              f"(P&L: {format_currency(summary['total_pnl'])} / {format_pct(summary['total_pnl_pct'])})")

    except Exception as e:
        logger.error(f"Single cycle error: {e}", exc_info=True)


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="MonopolyTrader - AI Stock Trading Agent")
    parser.add_argument("--once", action="store_true", help="Run a single decision cycle")
    parser.add_argument("--bootstrap", action="store_true", help="Run first-time setup (research before trading)")
    parser.add_argument("--research", action="store_true", help="Run daily research cycle")
    parser.add_argument("--learn", action="store_true", help="Run learning cycle (review trades, discover patterns)")
    parser.add_argument("--report", action="store_true", help="Print portfolio report")
    parser.add_argument("--status", action="store_true", help="Print current status")
    args = parser.parse_args()

    init_kb()

    if args.bootstrap:
        asyncio.run(bootstrap())
    elif args.once:
        run_once()
    elif args.research:
        asyncio.run(run_daily_research())
    elif args.learn:
        config = load_config()
        current = get_current_price(config["ticker"])
        portfolio = load_portfolio()
        asyncio.run(run_learning_cycle(current["price"], portfolio))
    elif args.report:
        from .reporter import generate_dashboard_data
        generate_dashboard_data()
        _print_status()
        print(f"  Dashboard: dashboard/index.html")
    elif args.status:
        _print_status()
    else:
        # Full scheduler mode
        if not _is_bootstrapped():
            logger.info("First run detected — running bootstrap...")
            asyncio.run(bootstrap())
        run_scheduler()


def _print_status():
    """Print a comprehensive status report."""
    config = load_config()
    ticker = config["ticker"]

    try:
        current = get_current_price(ticker)
        price = current["price"]
    except Exception:
        price = 0
        print("(Could not fetch live price)")

    portfolio = load_portfolio()
    if price > 0:
        portfolio = update_market_price(portfolio, ticker, price)
    summary = get_portfolio_summary()

    print()
    print("MonopolyTrader Status")
    print("=" * 50)
    print(f"  {ticker}: {format_currency(price) if price else 'N/A'}")
    print(f"  Market: {'OPEN' if is_market_open() else 'CLOSED'}")
    print()
    print("Portfolio")
    print("-" * 50)
    print(f"  Cash:       {format_currency(summary['cash'])}")
    holdings = summary.get("holdings", {}).get(ticker, {})
    if holdings.get("shares", 0) > 0:
        print(f"  {ticker}:      {holdings['shares']:.4f} shares @ avg ${holdings['avg_cost_basis']:.2f}")
        print(f"  Position:   {format_currency(holdings['shares'] * holdings['current_price'])}")
        print(f"  Unrealized: {format_currency(holdings['unrealized_pnl'])}")
    else:
        print(f"  {ticker}:      No position")
    print(f"  Total:      {format_currency(summary['total_value'])}")
    print(f"  P&L:        {format_currency(summary['total_pnl'])} ({format_pct(summary['total_pnl_pct'])})")
    print(f"  Trades:     {summary['total_trades']} (W:{summary['winning_trades']} L:{summary['losing_trades']})")
    print(f"  Win Rate:   {summary['win_rate']}%")
    print()
    print("Knowledge")
    print("-" * 50)
    print(f"  {get_knowledge_summary()}")

    scores = get_strategy_scores()
    print()
    print("Strategies")
    print("-" * 50)
    for name, s in scores.get("strategies", {}).items():
        bar = "#" * int(s["weight"] * 50)
        print(f"  {name:18s} {s['weight']:.2f} {bar}")

    print()


if __name__ == "__main__":
    main()
