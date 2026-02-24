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
    format_currency, format_pct, setup_logging, DATA_DIR,
    load_json,
)
from .market_data import get_current_price, get_market_summary, check_macro_gate, classify_regime
from .portfolio import (
    load_portfolio, execute_trade, save_portfolio, save_snapshot,
    check_stop_losses, check_daily_loss_limit, check_cooldown,
    update_market_price, get_portfolio_summary, calculate_position_size,
    apply_gap_risk_reduction,
)
from .strategies import evaluate_all_strategies, aggregate_signals, calculate_signal_balance
from .agent import make_decision, make_decision_multi_step
from .knowledge_base import (
    initialize as init_kb, get_strategy_scores, get_relevant_knowledge,
    get_knowledge_summary,
)
from .learner import (
    review_trade, review_predictions, discover_patterns,
    evolve_strategy_weights, write_journal_entry, run_learning_cycle,
    review_hold_outcomes,
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
    """The core v3 trading cycle — regime-aware, macro-gated, ATR-sized."""
    config = load_config()
    ticker = config["ticker"]

    if not is_market_open(config):
        return

    if check_daily_loss_limit():
        logger.warning("Daily loss limit reached — skipping cycle")
        return

    try:
        # 1. Fetch market data (now includes macro + regime)
        logger.info("--- Decision Cycle ---")
        market_data = get_market_summary(ticker)
        current_price = market_data["current"]["price"]
        regime = market_data.get("regime", {})
        macro = market_data.get("macro", {})
        atr = market_data.get("daily_indicators", {}).get("atr", 0) or 0
        vix = regime.get("vix", 0)

        logger.info(
            f"{ticker}: ${current_price} "
            f"({market_data['current'].get('change_pct', 0):+.2f}%) "
            f"| Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')} "
            f"| VIX: {vix:.1f}"
        )

        # 2. Check macro gate
        macro_gate = check_macro_gate(config)
        if macro_gate["gate_active"]:
            logger.warning(f"Macro gate: {macro_gate['reason']}")

        # 3. Check earnings blackout
        earnings_blocked = _check_earnings_blackout(ticker, config)
        if earnings_blocked:
            logger.info("Earnings blackout active — only SELL allowed")

        # 4. Update portfolio with current price
        portfolio = load_portfolio()
        portfolio = update_market_price(portfolio, ticker, current_price)
        save_portfolio(portfolio)

        # 5. Check stop losses (dynamic ATR-based)
        stop_signal = check_stop_losses(current_price, atr=atr, vix=vix)
        if stop_signal:
            logger.warning(f"STOP LOSS: selling {stop_signal['shares']:.4f} shares")
            result = execute_trade(
                "SELL", stop_signal["shares"], current_price,
                {"strategy": "stop_loss", "confidence": 1.0,
                 "hypothesis": stop_signal["reason"],
                 "reasoning": "Automatic stop loss execution",
                 "_vix": vix}
            )
            if result["status"] == "executed":
                logger.info(f"Stop loss executed: P&L ${result['transaction'].get('realized_pnl', 0):.2f}")
            return

        # 6. Check cooldown
        if not check_cooldown(ticker):
            logger.info("Cooldown active — skipping decision")
            return

        # 7. Run strategy signals (regime-aware)
        scores = get_strategy_scores()
        signals = evaluate_all_strategies(market_data, portfolio, scores, regime=regime)
        aggregate = aggregate_signals(signals)
        logger.info(f"Aggregate signal: {aggregate.action} (conf={aggregate.confidence:.2f})")

        # 8. Calculate position size (inverse ATR + gap risk reduction)
        max_shares = calculate_position_size(
            portfolio["total_value"], current_price, atr, vix, config
        )
        max_shares = apply_gap_risk_reduction(max_shares, config)

        # 9. Get knowledge context
        knowledge = get_relevant_knowledge(market_data)

        # 10. Call agent for decision (with macro/regime context)
        if config.get("multi_step_reasoning"):
            logger.info("Using multi-step reasoning mode")
            decision = make_decision_multi_step(
                market_data, signals, aggregate, portfolio, knowledge,
                macro_gate=macro_gate, regime=regime,
            )
        else:
            decision = make_decision(
                market_data, signals, aggregate, portfolio, knowledge,
                macro_gate=macro_gate, regime=regime,
            )
        action = decision.get("action", "HOLD")
        shares = decision.get("shares", 0)

        # 11. Apply macro gate confidence override
        if macro_gate["gate_active"] and action == "BUY":
            conf_threshold = macro_gate.get("confidence_threshold_override", 0.80)
            if decision.get("confidence", 0) < conf_threshold:
                logger.warning(
                    f"Macro gate override: BUY confidence {decision.get('confidence', 0):.2f} "
                    f"< required {conf_threshold:.2f}. Forcing HOLD."
                )
                action = "HOLD"
                _log_hold_decision(decision, market_data, signals, "macro_gate_override")

        # 12. Apply earnings blackout (block BUY only)
        if earnings_blocked and action == "BUY":
            logger.info("Earnings blackout: blocking BUY, forcing HOLD")
            action = "HOLD"
            _log_hold_decision(decision, market_data, signals, "earnings_blackout")

        # 13. Cap shares by ATR-sized max
        if action == "BUY" and shares > max_shares:
            logger.info(f"ATR sizing: capping {shares:.4f} to {max_shares:.4f} shares")
            shares = max_shares

        # 14. Execute or log HOLD
        if action in ("BUY", "SELL") and shares > 0:
            decision["_vix"] = vix  # Pass VIX for slippage calculation
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
            _log_hold_decision(decision, market_data, signals, "agent_decision")

        # 15. Score pending predictions
        scored = review_predictions(current_price)
        if scored:
            logger.info(f"Scored {len(scored)} predictions")

        # 15b. Score pending HOLD counterfactuals
        try:
            evaluate_hold_counterfactuals(current_price)
        except Exception as e:
            logger.warning(f"Hold counterfactual scoring failed: {e}")

        # 16. Review recent unreviewed trades
        asyncio.run(_review_recent_trades(current_price))

        # 17. Handle research requests
        research_req = decision.get("research_request")
        if research_req:
            logger.info(f"Agent requested research: {research_req}")
            _save_research_request(research_req)

        # 18. Refresh dashboard
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


def _log_hold_decision(decision: dict, market_data: dict, signals, reason: str):
    """Log a HOLD decision with counterfactual tracking data and signal balance."""
    from .utils import load_json, save_json
    path = DATA_DIR / "hold_log.json"
    holds = load_json(path, default=[])

    strongest = None
    if signals:
        non_hold = [s for s in signals if s.action != "HOLD"]
        if non_hold:
            strongest = max(non_hold, key=lambda s: s.confidence)

    # Calculate signal balance at time of HOLD
    signal_balance = calculate_signal_balance(signals) if signals else None

    # Extract hold_analysis from agent decision
    hold_analysis = decision.get("hold_analysis")

    entry = {
        "timestamp": iso_now(),
        "reason": reason,
        "original_action": decision.get("action", "HOLD"),
        "original_confidence": decision.get("confidence", 0),
        "original_strategy": decision.get("strategy", ""),
        "price_at_hold": market_data.get("current", {}).get("price", 0),
        "strongest_signal_ignored": {
            "action": strongest.action,
            "strategy": strongest.strategy,
            "confidence": strongest.confidence,
        } if strongest else None,
        "signal_balance": signal_balance,
        "hold_analysis": hold_analysis,
        "regime": market_data.get("regime", {}),
        "counterfactual_scored": False,
        "counterfactual_outcome": None,
    }
    holds.append(entry)
    # Keep last 200
    if len(holds) > 200:
        holds = holds[-200:]
    save_json(path, holds)


def evaluate_hold_counterfactuals(current_price: float):
    """Score past HOLD decisions against actual price movement.

    For each unscored HOLD older than 2 hours, calculate what would have
    happened if the strongest ignored signal had been followed:
    - If strongest was BUY: did price go up? (missed opportunity)
    - If strongest was SELL: did price go down? (missed opportunity)
    - Calculate hypothetical P&L based on a standard position size.
    """
    from .utils import load_json, save_json
    from datetime import datetime, timezone

    path = DATA_DIR / "hold_log.json"
    holds = load_json(path, default=[])
    if not holds:
        return

    now = datetime.now(timezone.utc)
    updated = False
    config = load_config()
    portfolio_value = load_portfolio().get("total_value", 1000)
    standard_trade_pct = config.get("risk_params", {}).get("max_single_trade_pct", 0.20)
    hypothetical_trade_value = portfolio_value * standard_trade_pct * 0.5  # Half of max

    for hold in holds:
        if hold.get("counterfactual_scored"):
            continue

        hold_time = datetime.fromisoformat(hold["timestamp"])
        elapsed_hours = (now - hold_time).total_seconds() / 3600

        # Score after 2 hours
        if elapsed_hours < 2:
            continue

        price_at_hold = hold.get("price_at_hold", 0)
        if price_at_hold <= 0:
            hold["counterfactual_scored"] = True
            updated = True
            continue

        strongest = hold.get("strongest_signal_ignored")
        if not strongest:
            hold["counterfactual_scored"] = True
            hold["counterfactual_outcome"] = {
                "verdict": "no_signal",
                "note": "No non-HOLD signal was present.",
            }
            updated = True
            continue

        price_change = current_price - price_at_hold
        price_change_pct = (price_change / price_at_hold) * 100

        # Calculate hypothetical shares
        hypothetical_shares = hypothetical_trade_value / price_at_hold if price_at_hold > 0 else 0

        ignored_action = strongest.get("action", "HOLD")
        if ignored_action == "BUY":
            # Would buying have been profitable?
            hypothetical_pnl = hypothetical_shares * price_change
            was_right = price_change > 0
            verdict = "missed_gain" if was_right else "correct_hold"
        elif ignored_action == "SELL":
            # Would selling have been profitable?
            hypothetical_pnl = hypothetical_shares * (-price_change)
            was_right = price_change < 0
            verdict = "missed_gain" if was_right else "correct_hold"
        else:
            hypothetical_pnl = 0
            was_right = False
            verdict = "neutral"

        hold["counterfactual_scored"] = True
        hold["counterfactual_outcome"] = {
            "price_after_2hr": round(current_price, 2),
            "price_change_pct": round(price_change_pct, 3),
            "ignored_action": ignored_action,
            "ignored_confidence": strongest.get("confidence", 0),
            "hypothetical_pnl": round(hypothetical_pnl, 2),
            "was_signal_right": was_right,
            "verdict": verdict,
        }
        updated = True

        if was_right:
            logger.info(
                f"HOLD counterfactual: {ignored_action} signal was RIGHT — "
                f"missed ${abs(hypothetical_pnl):.2f} "
                f"(price moved {price_change_pct:+.2f}%)"
            )
        else:
            logger.info(
                f"HOLD counterfactual: {ignored_action} signal was WRONG — "
                f"HOLD was correct (price moved {price_change_pct:+.2f}%)"
            )

    if updated:
        save_json(path, holds)


def _check_earnings_blackout(ticker: str, config: dict = None) -> bool:
    """Check if we're within earnings blackout window."""
    if config is None:
        config = load_config()
    blackout_hours = config.get("risk_params", {}).get("earnings_blackout_hours", 48)
    if blackout_hours <= 0:
        return False

    from .utils import load_json, KNOWLEDGE_DIR
    earnings = load_json(KNOWLEDGE_DIR / "research" / "earnings_history.json", default={})
    # Look for upcoming earnings dates in the findings
    # Simple heuristic: check if any finding mentions "upcoming" or a near date
    # For now, this is a stub that returns False — earnings dates need to be
    # explicitly tracked. Will be populated by researcher.py.
    return False


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

        # 4. Update benchmarks
        try:
            from .benchmarks import BenchmarkTracker
            bt = BenchmarkTracker()
            bt.update_daily(tsla_price=current_price)
            logger.info("Benchmarks updated")
        except Exception as e:
            logger.warning(f"Benchmark update failed: {e}")

        # 5. Check milestones
        try:
            from .observability import MilestoneChecker
            from .benchmarks import BenchmarkTracker
            bt_check = BenchmarkTracker()
            comparison = bt_check.get_comparison(portfolio["total_value"])
            milestone_metrics = {
                "trading_days": len(list((DATA_DIR / "snapshots").glob("*.json"))),
                "total_trades": portfolio.get("total_trades", 0),
                "percentile_vs_random": comparison.get("percentile_vs_random", 0),
                "beats_buy_hold_tsla": comparison.get("beats_buy_hold_tsla", False),
                "prediction_accuracy_pct": 0,
                "total_return_pct": portfolio.get("total_pnl_pct", 0),
                "max_drawdown_pct": 0,
            }
            mc = MilestoneChecker()
            new_milestones = mc.check_all(milestone_metrics)
            if new_milestones:
                for m in new_milestones:
                    logger.info(f"MILESTONE: {m['milestone_id']} — {m['message']}")
        except Exception as e:
            logger.warning(f"Milestone check failed: {e}")

        # 6. Save daily snapshot
        save_snapshot()

        # 7. Generate dashboard (full=True for daily benchmark recalculation)
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

    # Schedule weekly lesson decay (Sundays at 6 PM ET)
    def _weekly_decay():
        try:
            from .knowledge_base import apply_lesson_decay
            archived = apply_lesson_decay()
            logger.info(f"Weekly lesson decay complete: {archived} archived")
        except Exception as e:
            logger.warning(f"Lesson decay failed: {e}")
    schedule.every().sunday.at("18:00").do(_weekly_decay)

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
    """Run a single v3 decision cycle (for testing)."""
    config = load_config()
    ticker = config["ticker"]

    init_kb()

    logger.info("--- Single Cycle Mode (v3) ---")

    try:
        market_data = get_market_summary(ticker)
        current_price = market_data["current"]["price"]
        regime = market_data.get("regime", {})
        atr = market_data.get("daily_indicators", {}).get("atr", 0) or 0
        vix = regime.get("vix", 0)

        print(f"\n{ticker}: ${current_price}")
        print(f"Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')} | VIX: {vix:.1f} | ATR: {atr:.2f}")

        # Macro gate
        macro_gate = check_macro_gate(config)
        if macro_gate["gate_active"]:
            print(f"MACRO GATE ACTIVE: {macro_gate['reason']}")

        portfolio = load_portfolio()
        portfolio = update_market_price(portfolio, ticker, current_price)
        save_portfolio(portfolio)

        # Dynamic ATR stop losses
        stop_signal = check_stop_losses(current_price, atr=atr, vix=vix)
        if stop_signal:
            logger.warning(f"STOP LOSS triggered")
            execute_trade("SELL", stop_signal["shares"], current_price,
                         {"strategy": "stop_loss", "confidence": 1.0,
                          "hypothesis": stop_signal["reason"],
                          "reasoning": "Automatic stop loss",
                          "_vix": vix})
            return

        scores = get_strategy_scores()
        signals = evaluate_all_strategies(market_data, portfolio, scores, regime=regime)
        aggregate = aggregate_signals(signals)

        # Position sizing with gap risk reduction
        max_shares = calculate_position_size(
            portfolio["total_value"], current_price, atr, vix, config
        )
        max_shares = apply_gap_risk_reduction(max_shares, config)
        print(f"ATR position cap: {max_shares:.4f} shares")

        knowledge = get_relevant_knowledge(market_data)
        decision = make_decision(
            market_data, signals, aggregate, portfolio, knowledge,
            macro_gate=macro_gate, regime=regime,
        )

        action = decision.get("action", "HOLD")
        shares = decision.get("shares", 0)

        # Cap by ATR sizing
        if action == "BUY" and shares > max_shares:
            print(f"ATR sizing: capping {shares:.4f} to {max_shares:.4f}")
            shares = max_shares

        if action in ("BUY", "SELL") and shares > 0:
            decision["_vix"] = vix
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

        review_predictions(current_price)

        summary = get_portfolio_summary()
        print(f"\nPortfolio: {format_currency(summary['total_value'])} "
              f"(P&L: {format_currency(summary['total_pnl'])} / {format_pct(summary['total_pnl_pct'])})")

    except Exception as e:
        logger.error(f"Single cycle error: {e}", exc_info=True)


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="MonopolyTrader - AI Stock Trading Agent")
    parser.add_argument("--once", action="store_true", help="Run a single decision cycle")
    parser.add_argument("--ensemble", action="store_true", help="Run ensemble cycle (all agents)")
    parser.add_argument("--bootstrap", action="store_true", help="Run first-time setup (research before trading)")
    parser.add_argument("--research", action="store_true", help="Run daily research cycle")
    parser.add_argument("--learn", action="store_true", help="Run learning cycle (review trades, discover patterns)")
    parser.add_argument("--report", action="store_true", help="Print portfolio report")
    parser.add_argument("--status", action="store_true", help="Print current status")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtest vs random traders")
    parser.add_argument("--health", action="store_true", help="Check system health status")
    parser.add_argument("--alerts", action="store_true", help="Show active anomaly alerts")
    parser.add_argument("--leaderboard", action="store_true", help="Show agent ensemble leaderboard")
    parser.add_argument("--explain-trade", type=str, metavar="TXN_ID", help="Explain a specific trade decision")
    parser.add_argument("--milestones", action="store_true", help="Show triggered milestones")
    parser.add_argument("--graduation-report", action="store_true", help="Show graduation criteria status")
    parser.add_argument("--debug-traces", type=int, nargs="?", const=5, metavar="N", help="Show last N decision traces")
    parser.add_argument("--influence-report", action="store_true", help="Show strategy influence breakdown")
    args = parser.parse_args()

    init_kb()

    if args.ensemble:
        from .ensemble import run_ensemble_cycle
        if not is_market_open():
            print("Market is closed. Running ensemble cycle anyway for testing...")
        results = run_ensemble_cycle()
        print(f"\nEnsemble Results ({len(results)} agents)")
        print("=" * 60)
        for r in results:
            print(
                f"  [{r.get('agent', '?')}] {r.get('action', '?')} "
                f"conf={r.get('confidence', 0):.2f} "
                f"value=${r.get('portfolio_value', 0):.2f} "
                f"({r.get('execution', '?')})"
            )
        print()
        return
    elif args.leaderboard:
        from .comparison import generate_leaderboard
        lb = generate_leaderboard()
        if not lb:
            print("No agent data yet. Run --ensemble first.")
        else:
            print(f"\nAgent Leaderboard")
            print("=" * 70)
            print(f"{'#':<4} {'Agent':<25} {'Return':>8} {'Trades':>7} {'Win%':>6} {'Sharpe':>7} {'MaxDD':>7}")
            print("-" * 70)
            for m in lb:
                print(
                    f"{m['rank']:<4} {m['display_name']:<25} "
                    f"{m['total_pnl_pct']:>+7.2f}% {m['total_trades']:>7} "
                    f"{m['win_rate']:>5.0f}% {m['sharpe_ratio']:>7.3f} "
                    f"{m['max_drawdown_pct']:>6.1f}%"
                )
        print()
        return
    elif args.explain_trade:
        _explain_trade(args.explain_trade)
        return
    elif args.milestones:
        milestones = load_json(DATA_DIR / "milestones.json", default=[])
        if milestones:
            print(f"\nTriggered Milestones ({len(milestones)})")
            print("=" * 60)
            for m in milestones:
                print(f"  [{m.get('severity', 'info')}] {m.get('milestone_id', '?')}")
                print(f"    {m.get('message', '')[:100]}")
                print(f"    Triggered: {m.get('triggered_at', '?')}")
        else:
            print("\nNo milestones triggered yet.")
        print()
        return
    elif args.graduation_report:
        try:
            from .benchmarks import BenchmarkTracker
            portfolio = load_portfolio()
            bt = BenchmarkTracker()
            comparison = bt.get_comparison(portfolio["total_value"])
            grad = comparison.get("graduation_criteria", {})
            print(f"\nGraduation Report")
            print("=" * 60)
            passed = 0
            total = 0
            for name, criteria in grad.items():
                total += 1
                icon = "PASS" if criteria.get("passed") else "FAIL"
                if criteria.get("passed"):
                    passed += 1
                print(f"  [{icon}] {name}: {criteria.get('actual', '?')} (required: {criteria.get('required', '?')})")
            print(f"\n  {passed}/{total} criteria met")
        except Exception as e:
            print(f"Graduation report error: {e}")
        print()
        return
    elif args.debug_traces:
        from .utils import LOGS_DIR
        traces_dir = LOGS_DIR / "traces"
        if not traces_dir.exists():
            print("No traces directory yet. Run a trading cycle first.")
            return
        # Find the latest trace files
        trace_files = sorted(traces_dir.rglob("trace_*.json"), reverse=True)
        n = args.debug_traces
        if not trace_files:
            print("No traces found.")
        else:
            print(f"\nLatest {min(n, len(trace_files))} Decision Traces")
            print("=" * 60)
            for tf in trace_files[:n]:
                trace = load_json(tf)
                print(f"\n  File: {tf.name}")
                print(f"  Time: {trace.get('timestamp', '?')}")
                print(f"  Action: {trace.get('decision', {}).get('action', '?')}")
                print(f"  Confidence: {trace.get('decision', {}).get('confidence', 0):.2f}")
                regime = trace.get('regime', {})
                print(f"  Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')}")
                anomalies = trace.get('anomalies', [])
                if anomalies:
                    print(f"  Anomalies: {', '.join(str(a) for a in anomalies)}")
        print()
        return
    elif args.influence_report:
        scores = get_strategy_scores()
        transactions = load_portfolio()
        print(f"\nStrategy Influence Report")
        print("=" * 60)
        for name, s in scores.get("strategies", {}).items():
            pnl = s.get("total_pnl", 0)
            trades = s.get("total_trades", 0)
            wr = s.get("win_rate", 0) * 100
            trend = s.get("trend", "neutral")
            print(f"  {name:20s} weight={s['weight']:.3f} trades={trades:3d} "
                  f"win_rate={wr:5.1f}% pnl=${pnl:+.2f} trend={trend}")
        print()
        # Show current portfolio allocation
        portfolio = load_portfolio()
        print("Portfolio Allocation")
        print("-" * 40)
        for ticker, h in portfolio.get("holdings", {}).items():
            if h.get("shares", 0) > 0:
                value = h["shares"] * h.get("current_price", 0)
                pct = (value / portfolio["total_value"] * 100) if portfolio["total_value"] > 0 else 0
                print(f"  {ticker}: {h['shares']:.4f} shares (${value:.2f}, {pct:.1f}%)")
        cash_pct = (portfolio["cash"] / portfolio["total_value"] * 100) if portfolio["total_value"] > 0 else 0
        print(f"  Cash: ${portfolio['cash']:.2f} ({cash_pct:.1f}%)")
        print()
        return
    elif args.backtest:
        from .backtest import WalkForwardBacktest
        bt = WalkForwardBacktest()
        bt.run_all()
        return
    elif args.health:
        try:
            from .observability import HealthChecker
            health = HealthChecker().check()
            print("\nSystem Health")
            print("=" * 40)
            for component, status in health.get("components", {}).items():
                icon = "OK" if status.get("healthy") else "FAIL"
                print(f"  [{icon}] {component}: {status.get('detail', '')}")
            print()
        except ImportError:
            print("Observability module not yet available")
        return
    elif args.alerts:
        from .utils import load_json, LOGS_DIR
        alerts = load_json(LOGS_DIR / "alerts.json", default=[])
        active = [a for a in alerts if a.get("status") == "active"]
        if active:
            print(f"\nActive Alerts ({len(active)})")
            print("=" * 50)
            for a in active:
                print(f"  [{a.get('severity', '?')}] {a.get('type', '?')}: {a.get('message', '')}")
        else:
            print("\nNo active alerts.")
        print()
        return
    elif args.bootstrap:
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


def _explain_trade(txn_id: str):
    """Print detailed explanation of a specific trade."""
    from .portfolio import load_transactions
    transactions = load_transactions()
    txn = next((t for t in transactions if t["id"] == txn_id), None)

    if not txn:
        # Also search agent transactions
        from .ensemble import list_agents, load_agent_transactions
        for name in list_agents():
            agent_txns = load_agent_transactions(name)
            txn = next((t for t in agent_txns if t["id"] == txn_id), None)
            if txn:
                print(f"(Found in agent: {name})")
                break

    if not txn:
        print(f"Trade {txn_id} not found.")
        return

    print(f"\nTrade Explanation: {txn_id}")
    print("=" * 60)
    print(f"  Time:       {txn.get('timestamp', '?')}")
    print(f"  Action:     {txn.get('action', '?')} {txn.get('shares', 0):.4f} shares @ ${txn.get('price', 0):.2f}")
    print(f"  Strategy:   {txn.get('strategy', '?')}")
    print(f"  Confidence: {txn.get('confidence', 0):.2f}")
    print(f"  Hypothesis: {txn.get('hypothesis', 'N/A')}")
    print(f"  Reasoning:  {txn.get('reasoning', 'N/A')}")
    if txn.get("realized_pnl") is not None:
        print(f"  P&L:        ${txn['realized_pnl']:.2f}")
    if txn.get("review"):
        print(f"  Review:     {txn['review']}")
    print()


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
