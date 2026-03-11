"""MonopolyTrader v6 — main entry point.

Every 15 minutes during market hours (with anti-churn cooldowns):
  1. Gather data (TSLA + world + news + web search)
  2. Build brief (raw data + portfolio + journal + spread cost + shadow journal)
  3. Claude decides (BUY / SELL / HOLD)
  4. Execute trade if any
  5. Record in journal (with strategy field)
  6. Log HOLD decisions to shadow journal
  7. Update shadow prices for recent HOLDs
  8. Update dashboard
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
from .tags import compute_tags
from .analyst import run_nightly_update, run_pre_market
from .shadow_journal import log_hold_decision, update_shadow_prices
from .prediction_tracker import log_prediction, update_predictions
from .journal import (
    add_entry as journal_add_entry,
    close_entry as journal_close_entry,
    get_recent_entries,
    load_journal,
    update_intra_trade_prices,
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
    """The v6 research cycle. Runs every 15 minutes during market hours."""
    config = load_config()
    ticker = config["ticker"]

    if not is_market_open(config):
        return

    try:
        logger.info("--- v6 Research Cycle ---")

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
            config.get("journal", {}).get("max_entries_in_brief", 5)
        )

        # 4. Intra-trade price tracking — runs every cycle including cooldown (v6.1 Blindspot #3)
        try:
            holdings = portfolio.get("holdings", {}).get(ticker, {})
            if holdings.get("shares", 0) > 0:
                update_intra_trade_prices(current_price, ticker)
        except Exception as e:
            logger.debug(f"Intra-trade price update failed: {e}")

        # 4a. Check cooldown
        if not check_cooldown(ticker):
            logger.info("Cooldown active -- skipping Claude call")
            # Still update shadow prices and predictions during cooldown
            try:
                update_shadow_prices(current_price)
            except Exception as e:
                logger.warning(f"Shadow price update failed: {e}")
            try:
                update_predictions(current_price)
            except Exception as e:
                logger.warning(f"Prediction update during cooldown failed: {e}")
            _update_dashboard(market_data, portfolio)
            return

        # 4c. Fetch options data for tags + brief (v6.1 Blindspot #4)
        options_data = None
        try:
            from .market_data import get_options_snapshot
            options_data = get_options_snapshot(ticker)
        except Exception as e:
            logger.debug(f"Options data for tags failed: {e}")

        # 4d. Compute pre-decision tags for pattern matching
        #     (action="HOLD" since we don't know the decision yet)
        from .thesis_builder import find_matching_patterns, format_matching_patterns_for_brief
        pre_decision_tags = compute_tags(
            market_data=market_data,
            world=world,
            portfolio=portfolio,
            config=config,
            events=events,
            action="HOLD",
            news_feed=news_feed,
            options_data=options_data,
        )
        matched_patterns = find_matching_patterns(pre_decision_tags)

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
            matched_patterns=matched_patterns,
            options_data=options_data,
        )

        action = decision.get("action", "HOLD")
        shares = decision.get("shares", 0)

        # Compute tags for this cycle (used for trades AND hold logging)
        cycle_tags = compute_tags(
            market_data=market_data,
            world=world,
            portfolio=portfolio,
            config=config,
            events=events,
            action=action,
            news_feed=news_feed,
            options_data=options_data,
        )

        # 5b. Log prediction (every cycle, BUY/SELL/HOLD)
        try:
            log_prediction(
                decision=decision,
                market_data=market_data,
                tags=cycle_tags,
                current_price=current_price,
            )
        except Exception as e:
            logger.warning(f"Prediction logging failed: {e}")

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

                # v6.1 Blindspot #9: Check thesis consistency
                thesis_consistent = _check_thesis_consistency(action, portfolio, ticker)

                # Record in journal (with strategy field preserved)
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
                    tags=cycle_tags,
                    strategy=decision.get("strategy", ""),
                    hypothesis=decision.get("hypothesis", ""),
                    expected_learning=decision.get("expected_learning", ""),
                    thesis_consistent=thesis_consistent,
                )

                # If this was a SELL, close the corresponding BUY journal entry
                if action == "SELL" and txn.get("realized_pnl") is not None:
                    _close_journal_for_sell(txn)

                    # Rebuild playbook immediately on trade close
                    try:
                        from .thesis_builder import build_ledger
                        build_ledger()
                        logger.info("Playbook rebuilt after trade close")
                    except Exception as e:
                        logger.warning(f"Playbook rebuild failed: {e}")

            elif result["status"] == "rejected":
                logger.warning(f"Trade rejected: {result['reason']}")
        else:
            # HOLD decision
            strategy = decision.get("strategy", "")
            hypothesis = decision.get("hypothesis", "")
            reasoning = decision.get("reasoning", "N/A")[:120]
            logger.info(f"HOLD [{strategy}] — {reasoning}")

            # Log HOLD to shadow journal
            try:
                log_hold_decision(
                    decision=decision,
                    market_data=market_data,
                    tags=cycle_tags,
                )
            except Exception as e:
                logger.warning(f"Shadow journal logging failed: {e}")

        # 7. Update shadow prices for all recent HOLDs (every cycle)
        try:
            updated_count = update_shadow_prices(current_price)
            if updated_count > 0:
                logger.debug(f"Shadow journal: updated {updated_count} HOLD entries")
        except Exception as e:
            logger.warning(f"Shadow price update failed: {e}")

        # 7b. Update and resolve pending predictions (every cycle)
        try:
            resolved_count = update_predictions(current_price)
            if resolved_count > 0:
                logger.info(f"Predictions: resolved {resolved_count}")
        except Exception as e:
            logger.warning(f"Prediction update failed: {e}")

        # 8. Save latest cycle for dashboard
        _save_latest_cycle(decision, market_data, regime, world)

        # 9. Monitor portfolio health
        _check_portfolio_health(portfolio)

        # 10. Update dashboard
        _update_dashboard(market_data, portfolio)

    except Exception as e:
        logger.error(f"v6 research cycle error: {e}", exc_info=True)


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


def _check_thesis_consistency(action: str, portfolio: dict, ticker: str) -> bool | None:
    """Check if a trade action is consistent with the current MID thesis.

    v6.1 Blindspot #9: Thesis consistency scoring.

    Returns:
        True if consistent, False if inconsistent, None if unknown.
    """
    try:
        from .analyst import load_mid
        mid = load_mid()
        if not mid or not mid.get("thesis"):
            return None

        direction = mid["thesis"].get("direction", "neutral")

        if direction == "neutral":
            return True  # Neutral thesis is consistent with any action

        # Check if this is a new position vs closing existing
        holdings = portfolio.get("holdings", {}).get(ticker, {})
        has_position = holdings.get("shares", 0) > 0.0001

        if action == "BUY":
            if direction == "bearish":
                # Buying when thesis is bearish = inconsistent
                # Unless this is a very specific contrarian play
                logger.info("Thesis inconsistency: BUY while MID is bearish")
                return False
            return True

        elif action == "SELL":
            if direction == "bullish" and has_position:
                # Could be taking profit on an existing position — that's fine
                # But opening a new short-side bet is inconsistent
                # Since we don't short, SELL always means closing a long
                # Selling when bullish = potentially inconsistent
                logger.info("Thesis inconsistency: SELL while MID is bullish")
                return False
            return True

        return None
    except Exception:
        return None


def _save_latest_cycle(decision: dict, market_data: dict, regime: dict, world: dict):
    """Save the latest cycle data for the dashboard Agent's Mind card."""
    cycle_data = {
        "timestamp": iso_now(),
        "action": decision.get("action", "HOLD"),
        "shares": decision.get("shares", 0),
        "confidence": decision.get("confidence", 0),
        "strategy": decision.get("strategy", ""),
        "hypothesis": decision.get("hypothesis", ""),
        "expected_learning": decision.get("expected_learning", ""),
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


def _check_portfolio_health(portfolio: dict):
    """Monitor portfolio for concerning patterns."""
    value = portfolio.get("total_value", 1000)
    config = load_config()
    starting = config.get("starting_balance", 1000)

    if value < starting * 0.7:
        logger.error(f"ALERT: Portfolio at ${value:.2f} -- 30%+ drawdown from ${starting}")
    elif value < starting * 0.8:
        logger.warning(f"WARNING: Portfolio at ${value:.2f} -- 20%+ drawdown from ${starting}")

    # Check for consecutive losses
    transactions = load_portfolio().get("total_trades", 0)
    if transactions >= 5:
        from .portfolio import load_transactions
        txns = load_transactions()
        recent_sells = [t for t in txns if t["action"] == "SELL"][-5:]
        if len(recent_sells) >= 5 and all(t.get("realized_pnl", 0) < 0 for t in recent_sells):
            logger.warning("WARNING: 5 consecutive losing trades")


def run_pre_market_task():
    """Run pre-market analysis at 9:00 AM ET on weekdays."""
    if now_et().weekday() > 4:
        return
    try:
        logger.info("--- Pre-Market Research ---")
        briefing = run_pre_market()
        logger.info(
            f"Pre-market complete: status={briefing.get('thesis_status')}, "
            f"posture={briefing.get('recommended_posture')}"
        )
    except Exception as e:
        logger.error(f"Pre-market research failed: {e}")


def run_daily_tasks():
    """Run after market close: snapshot + thesis ledger rebuild + analyst update."""
    try:
        save_snapshot()
    except Exception as e:
        logger.warning(f"Snapshot failed: {e}")

    # Rebuild thesis ledger from all trades
    try:
        from .thesis_builder import build_ledger
        ledger = build_ledger()
        logger.info(
            f"Thesis ledger rebuilt: {ledger.get('active_trades', 0)} active trades, "
            f"{len(ledger.get('theses', {}))} theses, "
            f"{len(ledger.get('multi_tag_patterns', {}))} multi-tag patterns"
        )
    except Exception as e:
        logger.warning(f"Thesis ledger build failed: {e}")

    # v6.1 Blindspot #10: Measure event impacts from recent days
    try:
        from .events import measure_recent_event_impacts
        measure_recent_event_impacts()
        logger.info("Event impact measurement complete")
    except Exception as e:
        logger.debug(f"Event impact measurement failed: {e}")

    # Prune old predictions
    try:
        from .prediction_tracker import prune_predictions
        prune_predictions(days=30)
    except Exception as e:
        logger.debug(f"Prediction pruning failed: {e}")

    # Run nightly analyst — update Market Intelligence Document
    try:
        mid = run_nightly_update()
        logger.info(
            f"Nightly analyst complete: thesis={mid.get('thesis', {}).get('direction')}, "
            f"confidence={mid.get('thesis', {}).get('confidence')}"
        )
    except Exception as e:
        logger.error(f"Nightly analyst failed: {e}")

    try:
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)
    except Exception as e:
        logger.warning(f"Dashboard generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="MonopolyTrader v6")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--report", action="store_true", help="Generate dashboard and exit")
    parser.add_argument("--analyst", action="store_true", help="Run nightly analyst and exit")
    parser.add_argument("--premarket", action="store_true", help="Run pre-market briefing and exit")
    args = parser.parse_args()

    config = load_config()
    logger.info("MonopolyTrader v6 starting")

    # Ensure data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "snapshots").mkdir(parents=True, exist_ok=True)

    if args.report:
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)
        logger.info("Dashboard generated")
        return

    if args.analyst:
        mid = run_nightly_update()
        logger.info(f"Analyst complete: {mid.get('thesis', {}).get('direction')}")
        return

    if args.premarket:
        briefing = run_pre_market()
        logger.info(f"Pre-market complete: {briefing.get('thesis_status')}")
        return

    if args.once:
        run_cycle()
        return

    # Schedule
    interval = config.get("poll_interval_minutes", 15)
    schedule.every(interval).minutes.do(run_cycle)
    schedule.every().day.at("09:00").do(run_pre_market_task)
    schedule.every().day.at("16:15").do(run_daily_tasks)

    logger.info(f"Scheduler started: every {interval}min + 9:00 pre-market + 16:15 daily tasks")
    run_cycle()  # Run immediately

    while _running:
        schedule.run_pending()
        time.sleep(10)

    logger.info("MonopolyTrader v6 stopped")


if __name__ == "__main__":
    main()
