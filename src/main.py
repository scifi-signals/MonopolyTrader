"""MonopolyTrader v4 — main entry point.

Every 15 minutes during market hours:
  1. Gather data (TSLA + world + news + web search)
  2. Build brief (raw data + portfolio + journal)
  3. Claude decides (BUY / SELL / HOLD)
  4. Execute trade if any
  5. Record in journal
  6. Update dashboard
"""

import argparse
import signal
import sys
import time
import schedule

from .utils import (
    load_config, is_market_open, now_et, iso_now,
    format_currency, setup_logging, DATA_DIR, save_json,
)
from .market_data import get_market_summary, get_world_snapshot
from .portfolio import (
    load_portfolio, execute_trade, save_portfolio, save_snapshot,
    update_market_price, get_portfolio_summary, check_cooldown,
)
from .agent import make_decision
from .news_feed import fetch_news_feed
from .web_search import search_tsla_news
from .events import get_upcoming_events
from .journal import (
    add_entry as journal_add_entry,
    close_entry as journal_close_entry,
    get_recent_entries,
    load_journal,
)

logger = setup_logging("main")

_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received, finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_cycle():
    """The v4 trading cycle. Runs every 15 minutes during market hours."""
    config = load_config()
    ticker = config["ticker"]

    if not is_market_open(config):
        return

    try:
        logger.info("--- v4 Decision Cycle ---")

        # 1. Gather data
        market_data = get_market_summary(ticker)
        current_price = market_data["current"]["price"]
        regime = market_data.get("regime", {})
        vix = regime.get("vix", 0)

        logger.info(
            f"{ticker}: ${current_price} "
            f"({market_data['current'].get('change_pct', 0):+.2f}%) "
            f"| VIX: {vix:.1f}"
        )

        # World snapshot (macro + EV peers)
        world = {}
        try:
            world = get_world_snapshot(config)
        except Exception as e:
            logger.warning(f"World snapshot failed: {e}")

        # News feed
        news_feed = None
        try:
            news_feed = fetch_news_feed(ticker)
        except Exception as e:
            logger.warning(f"News feed failed: {e}")

        # Web search
        web_results = []
        try:
            if config.get("brave_search", {}).get("enabled", False):
                web_results = search_tsla_news()
        except Exception as e:
            logger.warning(f"Web search failed: {e}")

        # Upcoming events (FOMC, CPI, earnings)
        events = {}
        try:
            events = get_upcoming_events(hours=72)
            if events.get("macro_events"):
                for ev in events["macro_events"]:
                    logger.info(f"Upcoming event: {ev['event']} in {ev['hours_until']:.0f}h")
            if events.get("tsla_earnings"):
                logger.info(f"TSLA earnings: {events['tsla_earnings']['date']} ({events['tsla_earnings']['days_until']}d away)")
        except Exception as e:
            logger.warning(f"Events calendar failed: {e}")

        # 2. Update portfolio with current price
        portfolio = load_portfolio()
        portfolio = update_market_price(portfolio, ticker, current_price)
        save_portfolio(portfolio)

        # 3. Get journal entries
        journal_entries = get_recent_entries(
            config.get("journal", {}).get("max_entries_in_brief", 10)
        )

        # 4. Check cooldown
        if not check_cooldown(ticker):
            logger.info("Cooldown active — skipping Claude call")
            _update_dashboard(market_data, portfolio)
            return

        # 5. Claude decides
        decision = make_decision(
            market_data=market_data,
            world=world,
            portfolio=portfolio,
            news_feed=news_feed,
            web_results=web_results,
            journal_entries=journal_entries,
            config=config,
            events=events,
        )

        action = decision.get("action", "HOLD")
        shares = decision.get("shares", 0)

        # 6. Execute
        if action in ("BUY", "SELL") and shares > 0:
            result = execute_trade(action, shares, current_price, decision)

            if result["status"] == "executed":
                txn = result["transaction"]
                logger.info(
                    f"EXECUTED: {action} {shares:.4f} @ ${txn['price']:.2f} "
                    f"| Cash: {format_currency(result['portfolio']['cash'])} "
                    f"| Value: {format_currency(result['portfolio']['total_value'])}"
                )

                # Build market snapshot for journal
                market_snapshot = (
                    f"{ticker} ${current_price} "
                    f"({market_data['current'].get('change_pct', 0):+.2f}%), "
                    f"VIX {vix:.1f}"
                )
                spy_data = world.get("macro", {}).get("SPY", {})
                if spy_data:
                    market_snapshot += f", SPY {spy_data.get('change_pct', 0):+.2f}%"

                # Record in journal
                journal_add_entry(
                    trade_id=txn["id"],
                    action=action,
                    ticker=ticker,
                    shares=txn["shares"],
                    price=txn["price"],
                    reasoning=decision.get("reasoning", ""),
                    confidence=decision.get("confidence", 0),
                    portfolio_value=result["portfolio"]["total_value"],
                    market_snapshot=market_snapshot,
                )

                # If this was a SELL, close the corresponding BUY journal entry
                if action == "SELL" and txn.get("realized_pnl") is not None:
                    _close_journal_for_sell(txn)

            elif result["status"] == "rejected":
                logger.warning(f"Trade rejected: {result['reason']}")
        else:
            logger.info(f"HOLD — {decision.get('reasoning', 'N/A')[:120]}")

        # 7. Save latest cycle for dashboard
        _save_latest_cycle(decision, market_data, regime, world)

        # 8. Update dashboard
        _update_dashboard(market_data, portfolio)

    except Exception as e:
        logger.error(f"v4 decision cycle error: {e}", exc_info=True)


def _close_journal_for_sell(sell_txn: dict):
    """Find the most recent open BUY journal entry and close it."""
    journal = load_journal()
    for entry in reversed(journal):
        if (entry["action"] == "BUY"
            and entry["ticker"] == sell_txn["ticker"]
            and entry.get("lesson") is None):
            journal_close_entry(
                open_trade_id=entry["trade_id"],
                close_trade_id=sell_txn["id"],
                close_price=sell_txn["price"],
                realized_pnl=sell_txn["realized_pnl"],
            )
            return
    logger.warning(f"No open BUY journal entry found to close for {sell_txn['id']}")


def _save_latest_cycle(decision: dict, market_data: dict, regime: dict, world: dict):
    """Save the latest cycle data for the dashboard Agent's Mind card."""
    cycle_data = {
        "timestamp": iso_now(),
        "action": decision.get("action", "HOLD"),
        "shares": decision.get("shares", 0),
        "confidence": decision.get("confidence", 0),
        "reasoning": decision.get("reasoning", ""),
        "risk_note": decision.get("risk_note", ""),
        "price": market_data.get("current", {}).get("price", 0),
        "regime": regime,
        "vix": regime.get("vix", 0),
    }
    save_json(DATA_DIR / "latest_cycle.json", cycle_data)


def _update_dashboard(market_data: dict, portfolio: dict):
    """Refresh dashboard data."""
    try:
        from .reporter import generate_dashboard_data
        generate_dashboard_data()
    except Exception as e:
        logger.warning(f"Dashboard update failed: {e}")


def run_daily_tasks():
    """Run after market close: snapshot."""
    try:
        save_snapshot()
    except Exception as e:
        logger.warning(f"Daily tasks failed: {e}")

    try:
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)
    except Exception as e:
        logger.warning(f"Dashboard generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="MonopolyTrader v4")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--report", action="store_true", help="Generate dashboard and exit")
    args = parser.parse_args()

    config = load_config()
    logger.info("MonopolyTrader v4 starting")

    # Ensure data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "snapshots").mkdir(parents=True, exist_ok=True)

    if args.report:
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)
        logger.info("Dashboard generated")
        return

    if args.once:
        run_cycle()
        return

    # Schedule
    interval = config.get("poll_interval_minutes", 15)
    schedule.every(interval).minutes.do(run_cycle)
    schedule.every().day.at("16:15").do(run_daily_tasks)

    logger.info(f"Scheduler started: every {interval}min during market hours")
    run_cycle()  # Run immediately

    while _running:
        schedule.run_pending()
        time.sleep(10)

    logger.info("MonopolyTrader v4 stopped")


if __name__ == "__main__":
    main()
