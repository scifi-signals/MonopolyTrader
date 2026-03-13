"""MonopolyTrader v8 — main entry point.

Code computes, AI judges. Every cycle:
  1. Collect market data, news, compute tags
  2. Record cycle observation + resolve past outcomes
  3. Rebuild signal weights, compute composite score
  4. Risk checks — trailing stop, time stop, daily limit
  5. If signal actionable or position held: call AI for judgment
  6. Validate AI decision against risk rules
  7. Execute trade or HOLD
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
    update_market_price, get_portfolio_summary, execute_stop_exit,
)
from .agent import make_decision
from .news_feed import fetch_news_feed
from .events import get_upcoming_events
from .tags import compute_tags
from .analyst import run_nightly_update
from .outcome_tracker import (
    log_cycle, resolve_outcomes, update_action, prune_outcomes,
)
from .signal_engine import (
    rebuild_signal_registry, compute_composite_score,
    compute_position_size, get_signal_summary,
)
from .risk_manager import (
    RiskManager, save_active_position, clear_active_position,
    update_peak_price, has_active_position, record_trade_pnl,
    load_active_position,
)
from .journal import (
    add_entry as journal_add_entry,
    close_entry as journal_close_entry,
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
    """The v8 cycle: collect -> record -> resolve -> signal -> risk -> decide -> execute."""
    config = load_config()
    ticker = config["ticker"]

    if not is_market_open(config):
        return

    try:
        logger.info("--- v8 Cycle ---")
        risk = RiskManager(config)

        # 1. Collect market data
        market_data = get_market_summary(ticker)
        current_price = market_data["current"]["price"]
        regime = market_data.get("regime", {})
        vix = regime.get("vix", 0)

        logger.info(
            f"{ticker}: ${current_price} "
            f"({market_data['current'].get('change_pct', 0):+.2f}%) "
            f"| VIX: {vix:.1f}"
        )

        # World snapshot
        world = {}
        try:
            world = get_world_snapshot(config)
        except Exception as e:
            logger.warning(f"World snapshot failed: {e}")

        # News
        news_feed = None
        try:
            news_feed = fetch_news_feed(ticker)
        except Exception as e:
            logger.warning(f"News feed failed: {e}")

        # Events
        events = {}
        try:
            events = get_upcoming_events(hours=72)
        except Exception as e:
            logger.warning(f"Events failed: {e}")

        # Options
        options_data = None
        try:
            from .market_data import get_options_snapshot
            options_data = get_options_snapshot(ticker)
        except Exception as e:
            logger.debug(f"Options data failed: {e}")

        # Update portfolio with current price
        portfolio = load_portfolio()
        portfolio = update_market_price(portfolio, ticker, current_price)
        save_portfolio(portfolio)

        # Intra-trade price tracking (runs even during stops)
        try:
            holdings = portfolio.get("holdings", {}).get(ticker, {})
            if holdings.get("shares", 0) > 0:
                update_intra_trade_prices(current_price, ticker)
        except Exception as e:
            logger.debug(f"Intra-trade price update failed: {e}")

        # Compute tags
        tags = compute_tags(
            market_data=market_data,
            world=world,
            portfolio=portfolio,
            config=config,
            events=events,
            action="HOLD",
            news_feed=news_feed,
            options_data=options_data,
        )

        # 2. Record cycle observation
        cycle_record = log_cycle(current_price, tags, action="pending")

        # 3. Resolve past cycle outcomes
        resolved = resolve_outcomes(current_price)
        if resolved > 0:
            logger.debug(f"Resolved {resolved} outcome slots")

        # 4. Rebuild signal weights + compute composite score
        registry = rebuild_signal_registry()
        sig = compute_composite_score(tags, registry)
        sizing = compute_position_size(
            sig, portfolio["total_value"], current_price, config
        )

        logger.info(
            f"Signal: {sig['score']:+.4f} ({sig['direction']}) "
            f"| Confidence: {sig['confidence']} (n={sig['n']}) "
            f"| Size: {sizing['tier']}"
        )

        # 5. Risk checks — code-enforced, AI cannot override
        # Update peak price for trailing stop
        update_peak_price(current_price)

        # Check trailing stop
        stop_triggered, stop_reason = risk.check_trailing_stop(current_price)
        if stop_triggered:
            logger.warning(f"TRAILING STOP: {stop_reason}")
            result = execute_stop_exit(ticker, current_price, stop_reason)
            if result["status"] == "executed":
                record_trade_pnl(result["transaction"]["realized_pnl"])
                clear_active_position()
                _close_journal_for_sell(result["transaction"])
            update_action(cycle_record["id"], "SELL_STOP")
            _save_latest_cycle(
                {"action": "SELL_STOP", "reasoning": stop_reason},
                market_data, regime,
            )
            _update_dashboard(market_data, load_portfolio())
            return

        # Check time stop
        time_triggered, time_reason = risk.check_time_stop()
        if time_triggered:
            logger.warning(f"TIME STOP: {time_reason}")
            result = execute_stop_exit(ticker, current_price, time_reason)
            if result["status"] == "executed":
                record_trade_pnl(result["transaction"]["realized_pnl"])
                clear_active_position()
                _close_journal_for_sell(result["transaction"])
            update_action(cycle_record["id"], "SELL_TIME_STOP")
            _save_latest_cycle(
                {"action": "SELL_TIME_STOP", "reasoning": time_reason},
                market_data, regime,
            )
            _update_dashboard(market_data, load_portfolio())
            return

        # Check daily loss limit
        loss_hit, loss_reason = risk.check_daily_loss_limit()
        if loss_hit:
            logger.warning(f"DAILY LOSS LIMIT: {loss_reason}")
            update_action(cycle_record["id"], "HOLD_LOSS_LIMIT")
            _save_latest_cycle(
                {"action": "HOLD", "reasoning": loss_reason},
                market_data, regime,
            )
            _update_dashboard(market_data, portfolio)
            return

        # 6. Decide — call AI only if signal is actionable or we have a position
        has_pos = has_active_position()
        signal_actionable = abs(sig["score"]) >= risk.min_edge_threshold

        if not signal_actionable and not has_pos:
            logger.info(
                f"Skip AI: signal {sig['score']:+.4f} below threshold "
                f"{risk.min_edge_threshold}, no position"
            )
            update_action(cycle_record["id"], "HOLD_NO_SIGNAL")
            _save_latest_cycle(
                {
                    "action": "HOLD",
                    "reasoning": f"Signal {sig['score']:+.4f} below threshold",
                },
                market_data, regime,
            )
            _update_dashboard(market_data, portfolio)
            return

        # Call AI for judgment
        decision = make_decision(
            market_data=market_data,
            world=world,
            portfolio=portfolio,
            news_feed=news_feed,
            config=config,
            events=events,
            signal=sig,
            sizing=sizing,
            current_tags=tags,
            options_data=options_data,
        )

        action = decision.get("action", "HOLD")
        shares = 0

        # 7. Validate against risk rules
        if action == "BUY":
            shares = sizing.get("shares", 0)
            if shares <= 0:
                logger.info(f"Signal says no trade: {sizing['reasoning']}")
                action = "HOLD"
            else:
                ok, reason = risk.validate_trade(
                    action, sig, portfolio, current_price
                )
                if not ok:
                    logger.warning(f"Risk rejected BUY: {reason}")
                    action = "HOLD"
                    shares = 0

        elif action == "SELL":
            held = portfolio.get("holdings", {}).get(ticker, {}).get("shares", 0)
            if held <= 0:
                logger.info("AI suggested SELL but no position — HOLD")
                action = "HOLD"
            else:
                shares = held  # sell entire position

        # Record action in outcome tracker
        update_action(cycle_record["id"], action)

        # 8. Execute
        if action == "BUY" and shares > 0:
            result = execute_trade(action, shares, current_price, decision)

            if result["status"] == "executed":
                txn = result["transaction"]
                logger.info(
                    f"EXECUTED: BUY {shares:.4f} @ ${txn['price']:.2f} "
                    f"| Cash: {format_currency(result['portfolio']['cash'])} "
                    f"| Value: {format_currency(result['portfolio']['total_value'])}"
                )

                # Set up trailing stop tracking
                stop_pct = risk.clamp_stop_pct(
                    decision.get("stop_pct", risk.trailing_stop_default)
                )
                save_active_position(txn["price"], stop_pct, decision)

                # Journal entry
                market_snap = (
                    f"{ticker} ${current_price} "
                    f"({market_data['current'].get('change_pct', 0):+.2f}%), "
                    f"VIX {vix:.1f}"
                )
                journal_add_entry(
                    trade_id=txn["id"],
                    action="BUY",
                    ticker=ticker,
                    shares=txn["shares"],
                    price=txn["price"],
                    reasoning=decision.get("reasoning", ""),
                    confidence=decision.get("confidence", 0),
                    portfolio_value=result["portfolio"]["total_value"],
                    market_snapshot=market_snap,
                    tags=tags,
                    strategy=decision.get("strategy", "signal_driven"),
                    hypothesis=decision.get("hypothesis", ""),
                    expected_learning=decision.get("expected_learning", ""),
                )
            else:
                logger.warning(f"Trade rejected: {result.get('reason', 'unknown')}")

        elif action == "SELL" and shares > 0:
            result = execute_trade(action, shares, current_price, decision)

            if result["status"] == "executed":
                txn = result["transaction"]
                logger.info(
                    f"EXECUTED: SELL {shares:.4f} @ ${txn['price']:.2f} "
                    f"| P&L: {format_currency(txn['realized_pnl'])}"
                )
                record_trade_pnl(txn["realized_pnl"])
                clear_active_position()
                _close_journal_for_sell(txn)
        else:
            reasoning = decision.get("reasoning", "N/A")[:120]
            logger.info(f"HOLD — {reasoning}")

        # 9. Save cycle + update dashboard
        _save_latest_cycle(decision, market_data, regime)
        _update_dashboard(market_data, load_portfolio())

    except Exception as e:
        logger.error(f"v8 cycle error: {e}", exc_info=True)


def _close_journal_for_sell(sell_txn: dict):
    """Close the most recent open BUY journal entry."""
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
    logger.warning(f"No open BUY entry to close for {sell_txn['id']}")


def _save_latest_cycle(decision: dict, market_data: dict, regime: dict):
    """Save latest cycle data for dashboard."""
    save_json(DATA_DIR / "latest_cycle.json", {
        "timestamp": iso_now(),
        "action": decision.get("action", "HOLD"),
        "reasoning": decision.get("reasoning", ""),
        "confidence": decision.get("confidence", 0),
        "override": decision.get("override", False),
        "price": market_data.get("current", {}).get("price", 0),
        "regime": regime,
        "vix": regime.get("vix", 0),
    })


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
        logger.error(
            f"ALERT: Portfolio at ${value:.2f} — 30%+ drawdown from ${starting}"
        )
    elif value < starting * 0.8:
        logger.warning(
            f"WARNING: Portfolio at ${value:.2f} — 20%+ drawdown from ${starting}"
        )


# --- Daily Tasks ---

_daily_tasks_ran_today = None


def _check_daily_tasks_time():
    """Check if it's time for daily tasks (4:15-4:30 PM ET)."""
    global _daily_tasks_ran_today
    et = now_et()
    today = et.strftime("%Y-%m-%d")
    if _daily_tasks_ran_today == today:
        return
    if et.weekday() > 4:
        return
    if et.hour == 16 and 15 <= et.minute < 30:
        _daily_tasks_ran_today = today
        run_daily_tasks()


def run_daily_tasks():
    """Run after market close: snapshot + prune + nightly narrative."""
    try:
        save_snapshot()
    except Exception as e:
        logger.warning(f"Snapshot failed: {e}")

    # Prune old outcome records
    try:
        prune_outcomes(days=90)
    except Exception as e:
        logger.debug(f"Outcome pruning failed: {e}")

    # Measure event impacts
    try:
        from .events import measure_recent_event_impacts
        measure_recent_event_impacts()
    except Exception as e:
        logger.debug(f"Event impact measurement failed: {e}")

    # Nightly analyst — generate narrative
    try:
        run_nightly_update()
        logger.info("Nightly analyst complete")
    except Exception as e:
        logger.error(f"Nightly analyst failed: {e}")

    # Update dashboard
    try:
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)
    except Exception as e:
        logger.warning(f"Dashboard generation failed: {e}")


# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="MonopolyTrader v8")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--report", action="store_true", help="Generate dashboard and exit")
    parser.add_argument("--analyst", action="store_true", help="Run nightly analyst and exit")
    parser.add_argument("--migrate", action="store_true", help="Migrate v7 data to v8 format")
    args = parser.parse_args()

    config = load_config()
    logger.info("MonopolyTrader v8 starting")

    # Ensure data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "snapshots").mkdir(parents=True, exist_ok=True)

    if args.report:
        from .reporter import generate_dashboard_data
        generate_dashboard_data(full=True)
        logger.info("Dashboard generated")
        return

    if args.analyst:
        run_nightly_update()
        logger.info("Analyst complete")
        return

    if args.migrate:
        from .outcome_tracker import migrate_v7_data
        count = migrate_v7_data()
        logger.info(f"Migration complete: {count} records")
        reg = rebuild_signal_registry()
        logger.info(
            f"Signal registry built: {len(reg.get('signals', {}))} single-tag, "
            f"{len(reg.get('combos', {}))} combo signals"
        )
        return

    if args.once:
        run_cycle()
        return

    # Schedule
    interval = config.get("poll_interval_minutes", 15)
    schedule.every(interval).minutes.do(run_cycle)
    schedule.every(5).minutes.do(_check_daily_tasks_time)

    logger.info(
        f"Scheduler started: every {interval}min + 16:15 ET daily tasks"
    )
    run_cycle()  # Run immediately

    while _running:
        schedule.run_pending()
        time.sleep(10)

    logger.info("MonopolyTrader v8 stopped")


if __name__ == "__main__":
    main()
