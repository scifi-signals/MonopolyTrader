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
    get_position_direction,
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
    load_active_position, record_exit_time, check_reentry_cooldown,
    record_stop_out,
)
from .journal import (
    add_entry as journal_add_entry,
    close_entry as journal_close_entry,
    load_journal,
    update_intra_trade_prices,
)

logger = setup_logging("main")

_running = True
_last_cycle_time = None


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received, finishing current cycle...")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def run_cycle():
    """The v8 cycle: collect -> record -> resolve -> signal -> risk -> decide -> execute."""
    global _last_cycle_time
    config = load_config()
    ticker = config["ticker"]

    if not is_market_open(config):
        return

    # Adaptive interval: 5 min when holding a position, 15 min when flat
    adaptive = config.get("v8_risk", {}).get("adaptive_cycle", {})
    normal_interval = adaptive.get("normal_interval_minutes", 15)

    from datetime import datetime, timezone

    if not has_active_position() and _last_cycle_time is not None:
        elapsed = (datetime.now(timezone.utc) - _last_cycle_time).total_seconds() / 60
        if elapsed < normal_interval:
            return  # Flat and checked recently — skip

    _last_cycle_time = datetime.now(timezone.utc)

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
            if abs(holdings.get("shares", 0)) > 0.0001:
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
        half_life = config.get("v8_risk", {}).get("recency_half_life_days", 14)
        registry = rebuild_signal_registry(half_life_days=half_life)
        sig = compute_composite_score(tags, registry)
        sizing = compute_position_size(
            sig, portfolio["total_value"], current_price, config
        )

        logger.info(
            f"Signal: {sig['score']:+.3f} ({sig['direction']}) "
            f"| Edge: {sig.get('edge', 0):+.6f} "
            f"| Confidence: {sig['confidence']} (n={sig['n']}) "
            f"| Size: {sizing['tier']}"
        )

        # 5. Risk checks — code-enforced, AI cannot override
        # Update peak price for trailing stop
        update_peak_price(current_price)

        # Tighten trailing stop on profitable trades
        risk.tighten_trailing_stop(current_price)

        # Check trailing stop
        stop_triggered, stop_reason = risk.check_trailing_stop(current_price)
        if stop_triggered:
            logger.warning(f"TRAILING STOP: {stop_reason}")
            result = execute_stop_exit(ticker, current_price, stop_reason)
            if result["status"] == "executed":
                txn = result["transaction"]
                record_trade_pnl(txn["realized_pnl"])
                record_stop_out()
                clear_active_position()
                _close_journal_for_exit(txn)
                stop_action = f"{txn['action']}_STOP"
            else:
                stop_action = "STOP_FAILED"
            update_action(cycle_record["id"], stop_action, {
                "source": "stop_trailing", "signal_score": sig["score"],
                "signal_direction": sig["direction"], "ai_called": False,
                "final_action": stop_action,
            })
            _save_latest_cycle(
                {"action": stop_action, "reasoning": stop_reason},
                market_data, regime,
            )
            _update_dashboard(market_data, load_portfolio())
            return

        # Check time stop (only on losing/flat trades — winners run)
        time_triggered, time_reason = risk.check_time_stop(current_price)
        if time_triggered:
            logger.warning(f"TIME STOP: {time_reason}")
            result = execute_stop_exit(ticker, current_price, time_reason)
            if result["status"] == "executed":
                txn = result["transaction"]
                record_trade_pnl(txn["realized_pnl"])
                record_stop_out()
                clear_active_position()
                _close_journal_for_exit(txn)
                stop_action = f"{txn['action']}_TIME_STOP"
            else:
                stop_action = "TIME_STOP_FAILED"
            update_action(cycle_record["id"], stop_action, {
                "source": "stop_time", "signal_score": sig["score"],
                "signal_direction": sig["direction"], "ai_called": False,
                "final_action": stop_action,
            })
            _save_latest_cycle(
                {"action": stop_action, "reasoning": time_reason},
                market_data, regime,
            )
            _update_dashboard(market_data, load_portfolio())
            return

        # Check end-of-day close (avoid overnight gap risk)
        eod_triggered, eod_reason = risk.check_eod_close()
        if eod_triggered:
            logger.warning(f"EOD CLOSE: {eod_reason}")
            result = execute_stop_exit(ticker, current_price, eod_reason)
            if result["status"] == "executed":
                txn = result["transaction"]
                record_trade_pnl(txn["realized_pnl"])
                clear_active_position()
                _close_journal_for_exit(txn)
                stop_action = f"{txn['action']}_EOD"
            else:
                stop_action = "EOD_FAILED"
            update_action(cycle_record["id"], stop_action, {
                "source": "stop_eod", "signal_score": sig["score"],
                "signal_direction": sig["direction"], "ai_called": False,
                "final_action": stop_action,
            })
            _save_latest_cycle(
                {"action": stop_action, "reasoning": eod_reason},
                market_data, regime,
            )
            _update_dashboard(market_data, load_portfolio())
            return

        # Check daily loss limit
        loss_hit, loss_reason = risk.check_daily_loss_limit()
        if loss_hit:
            logger.warning(f"DAILY LOSS LIMIT: {loss_reason}")
            update_action(cycle_record["id"], "HOLD_LOSS_LIMIT", {
                "source": "hold_loss_limit", "signal_score": sig["score"],
                "signal_direction": sig["direction"], "ai_called": False,
                "final_action": "HOLD",
            })
            _save_latest_cycle(
                {"action": "HOLD", "reasoning": loss_reason},
                market_data, regime,
            )
            _update_dashboard(market_data, portfolio)
            return

        # Check stop-out circuit breaker (2 stops = done for the day)
        # Only blocks new entries when flat — doesn't force exit on open positions
        if not has_active_position():
            stops_hit, stops_reason = risk.check_stop_out_limit()
            if stops_hit:
                logger.warning(f"CIRCUIT BREAKER: {stops_reason}")
                update_action(cycle_record["id"], "HOLD_CIRCUIT_BREAKER", {
                    "source": "stop_out_limit", "signal_score": sig["score"],
                    "signal_direction": sig["direction"], "ai_called": False,
                    "final_action": "HOLD",
                })
                _save_latest_cycle(
                    {"action": "HOLD", "reasoning": stops_reason},
                    market_data, regime,
                )
                _update_dashboard(market_data, portfolio)
                return

        # Earnings blackout — observe only, no new entries
        # (Exits still allowed — trailing/time/EOD stops already ran above)
        if not has_active_position():
            blackout, blackout_reason = risk.check_earnings_blackout(events)
            if blackout:
                logger.warning(f"EARNINGS BLACKOUT: {blackout_reason}")
                logger.info(
                    f"Observe-only: signal={sig['score']:+.3f} "
                    f"({sig['direction']}), sizing={sizing['tier']} "
                    f"— would have {'traded' if sizing['tier'] != 'no_trade' else 'skipped'}"
                )
                update_action(cycle_record["id"], "HOLD_EARNINGS_BLACKOUT", {
                    "source": "earnings_blackout",
                    "signal_score": sig["score"],
                    "signal_direction": sig["direction"],
                    "sizing_tier": sizing["tier"],
                    "sizing_shares": sizing.get("shares", 0),
                    "ai_called": False,
                    "final_action": "HOLD",
                    "observe_only": True,
                })
                _save_latest_cycle(
                    {"action": "HOLD", "reasoning": blackout_reason},
                    market_data, regime,
                )
                _update_dashboard(market_data, portfolio)
                return

        # 6. Decide — route to AI, contrarian, or skip
        pos_direction = get_position_direction(ticker)
        signal_actionable = abs(sig["score"]) >= risk.min_edge_threshold
        signal_bullish = sig.get("direction") == "bullish" and sig["score"] > 0
        signal_bearish = sig.get("direction") == "bearish" and sig["score"] < 0
        short_enabled = config.get("v8_risk", {}).get("enable_short_selling", True)
        atr = market_data.get("daily_indicators", {}).get("atr", 0)

        # Re-entry cooldown — block new entries after recent AI-initiated exit
        if pos_direction is None and signal_actionable:
            cooldown_blocked, cooldown_reason = check_reentry_cooldown(
                risk.reentry_cooldown_minutes
            )
            if cooldown_blocked:
                logger.info(f"COOLDOWN: {cooldown_reason}")
                signal_actionable = False
                update_action(cycle_record["id"], "HOLD_COOLDOWN", {
                    "source": "reentry_cooldown",
                    "signal_score": sig["score"],
                    "signal_direction": sig["direction"],
                    "ai_called": False,
                    "final_action": "HOLD",
                })

        if pos_direction == "long":
            # Holding long — always call AI to evaluate exit
            decision = make_decision(
                market_data=market_data, world=world, portfolio=portfolio,
                news_feed=news_feed, config=config, events=events,
                signal=sig, sizing=sizing, current_tags=tags,
                options_data=options_data,
            )
            action = decision.get("action", "HOLD")
            shares = 0

            if action == "SELL":
                held = portfolio.get("holdings", {}).get(ticker, {}).get("shares", 0)
                if held > 0:
                    shares = held
                else:
                    action = "HOLD"
            elif action in ("BUY", "SHORT", "COVER"):
                # Can't open new positions while holding — treat as HOLD
                action = "HOLD"

            update_action(cycle_record["id"], action, {
                "source": "ai_exit_long", "signal_score": sig["score"],
                "signal_direction": sig["direction"], "ai_called": True,
                "ai_action": decision.get("action", "HOLD"),
                "ai_confidence": decision.get("confidence", 0),
                "final_action": action,
            })
            _execute_action(
                action, shares, current_price, decision, ticker,
                market_data, regime, vix, tags, risk, atr, config,
            )

        elif pos_direction == "short":
            # Holding short — always call AI to evaluate cover
            decision = make_decision(
                market_data=market_data, world=world, portfolio=portfolio,
                news_feed=news_feed, config=config, events=events,
                signal=sig, sizing=sizing, current_tags=tags,
                options_data=options_data,
            )
            action = decision.get("action", "HOLD")
            shares = 0

            if action == "COVER":
                held = abs(portfolio.get("holdings", {}).get(ticker, {}).get("shares", 0))
                if held > 0.0001:
                    shares = held
                else:
                    action = "HOLD"
            elif action in ("BUY", "SELL", "SHORT"):
                # Can't open new positions while holding short — treat as HOLD
                action = "HOLD"

            update_action(cycle_record["id"], action, {
                "source": "ai_exit_short", "signal_score": sig["score"],
                "signal_direction": sig["direction"], "ai_called": True,
                "ai_action": decision.get("action", "HOLD"),
                "ai_confidence": decision.get("confidence", 0),
                "final_action": action,
            })
            _execute_action(
                action, shares, current_price, decision, ticker,
                market_data, regime, vix, tags, risk, atr, config,
            )

        elif signal_bullish and signal_actionable:
            # Signal says buy — call AI to confirm
            decision = make_decision(
                market_data=market_data, world=world, portfolio=portfolio,
                news_feed=news_feed, config=config, events=events,
                signal=sig, sizing=sizing, current_tags=tags,
                options_data=options_data,
            )
            action = decision.get("action", "HOLD")
            shares = 0

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

            elif action in ("SELL", "SHORT", "COVER"):
                # Not valid when flat — treat as HOLD
                action = "HOLD"

            update_action(cycle_record["id"], action, {
                "source": "ai_confirm", "signal_score": sig["score"],
                "signal_direction": sig["direction"], "ai_called": True,
                "ai_action": decision.get("action", "HOLD"),
                "ai_confidence": decision.get("confidence", 0),
                "signal_agreed": decision.get("action") == "BUY",
                "final_action": action,
            })
            _execute_action(
                action, shares, current_price, decision, ticker,
                market_data, regime, vix, tags, risk, atr, config,
            )

        elif signal_bearish and signal_actionable and short_enabled:
            # Signal says short — call AI to confirm
            decision = make_decision(
                market_data=market_data, world=world, portfolio=portfolio,
                news_feed=news_feed, config=config, events=events,
                signal=sig, sizing=sizing, current_tags=tags,
                options_data=options_data,
            )
            action = decision.get("action", "HOLD")
            shares = 0

            if action == "SHORT":
                shares = sizing.get("shares", 0)
                if shares <= 0:
                    logger.info(f"Signal says no trade: {sizing['reasoning']}")
                    action = "HOLD"
                else:
                    ok, reason = risk.validate_trade(
                        action, sig, portfolio, current_price
                    )
                    if not ok:
                        logger.warning(f"Risk rejected SHORT: {reason}")
                        action = "HOLD"
                        shares = 0

            elif action in ("BUY", "SELL", "COVER"):
                # Not valid — treat as HOLD
                action = "HOLD"

            update_action(cycle_record["id"], action, {
                "source": "ai_confirm_short", "signal_score": sig["score"],
                "signal_direction": sig["direction"], "ai_called": True,
                "ai_action": decision.get("action", "HOLD"),
                "ai_confidence": decision.get("confidence", 0),
                "signal_agreed": decision.get("action") == "SHORT",
                "final_action": action,
            })
            _execute_action(
                action, shares, current_price, decision, ticker,
                market_data, regime, vix, tags, risk, atr, config,
            )

        else:
            # Flat + weak/no actionable signal — check for contrarian entry
            is_contrarian, reason = _detect_contrarian(
                market_data, sig, config, tags
            )

            if is_contrarian:
                # Code-driven contrarian entry — no AI call
                ok, risk_reason = risk.validate_trade(
                    "BUY", sig, portfolio, current_price, contrarian=True
                )
                if not ok:
                    logger.info(f"Contrarian blocked by risk: {risk_reason}")
                    update_action(cycle_record["id"], "HOLD_RISK_LIMIT", {
                        "source": "code_contrarian_blocked",
                        "signal_score": sig["score"],
                        "signal_direction": sig["direction"],
                        "ai_called": False, "final_action": "HOLD",
                    })
                    _save_latest_cycle(
                        {"action": "HOLD", "reasoning": risk_reason},
                        market_data, regime,
                    )
                    _update_dashboard(market_data, portfolio)
                    return

                c_config = config.get("v8_risk", {}).get("contrarian", {})
                max_pct = c_config.get("max_position_pct", 0.05)
                min_cash = config.get("risk_params", {}).get(
                    "min_cash_reserve", 100.0
                )
                available = portfolio["total_value"] * max_pct
                available = min(available, portfolio.get("cash", 0) - min_cash)
                shares = round(available / current_price, 4) if available > 0 else 0

                if shares > 0:
                    logger.info(f"CONTRARIAN ENTRY: {reason}")
                    decision = {
                        "action": "BUY",
                        "reasoning": f"Code-driven contrarian: {reason}",
                        "confidence": 0.0,
                        "override": True,
                        "strategy": "contrarian",
                    }
                    update_action(cycle_record["id"], "BUY_CONTRARIAN", {
                        "source": "code_contrarian",
                        "signal_score": sig["score"],
                        "signal_direction": sig["direction"],
                        "ai_called": False, "final_action": "BUY",
                    })
                    _execute_action(
                        "BUY", shares, current_price, decision, ticker,
                        market_data, regime, vix, tags, risk, atr, config,
                    )
                else:
                    logger.info("Contrarian: insufficient cash")
                    update_action(cycle_record["id"], "HOLD_NO_CASH", {
                        "source": "code_contrarian_blocked",
                        "signal_score": sig["score"],
                        "signal_direction": sig["direction"],
                        "ai_called": False, "final_action": "HOLD",
                    })
                    _save_latest_cycle(
                        {"action": "HOLD", "reasoning": "Contrarian blocked: insufficient cash"},
                        market_data, regime,
                    )
                    _update_dashboard(market_data, portfolio)
            else:
                # Nothing to do — skip AI entirely
                logger.info(f"Skip AI: flat + no signal ({reason})")
                update_action(cycle_record["id"], "HOLD_NO_SIGNAL", {
                    "source": "code_skip", "signal_score": sig["score"],
                    "signal_direction": sig["direction"],
                    "ai_called": False, "final_action": "HOLD",
                })
                _save_latest_cycle(
                    {
                        "action": "HOLD",
                        "reasoning": f"No signal, no contrarian: {reason}",
                    },
                    market_data, regime,
                )
                _update_dashboard(market_data, portfolio)

    except Exception as e:
        logger.error(f"v8 cycle error: {e}", exc_info=True)


def _detect_contrarian(market_data: dict, signal: dict, config: dict,
                       tags: dict = None) -> tuple[bool, str]:
    """Code-driven contrarian entry detection. No AI involved.

    Triggers when RSI is oversold AND the signal isn't catastrophically
    bearish AND the regime isn't confirmed bearish. Returns (is_contrarian, reason).
    """
    # Regime gate: never auto-buy into a confirmed bear trend
    if tags:
        trend_dir = tags.get("trend_direction", "")
        trend = tags.get("trend", "")
        if trend_dir == "down" or trend == "bear":
            return False, (
                f"Regime gate: trend_direction={trend_dir}, trend={trend} — "
                f"no contrarian BUY in confirmed bear regime"
            )

    indicators = market_data.get("daily_indicators", {})
    rsi = indicators.get("rsi_14", 50)
    score = signal.get("score", 0)

    c = config.get("v8_risk", {}).get("contrarian", {})
    rsi_threshold = c.get("rsi_threshold", 30)
    min_score = c.get("min_signal_score", -0.40)

    if rsi >= rsi_threshold:
        return False, f"RSI {rsi:.0f} above threshold {rsi_threshold}"

    if score < min_score:
        return False, f"Signal {score:+.3f} below floor {min_score}"

    return True, f"RSI {rsi:.0f} oversold, signal {score:+.3f} within range"


def _execute_action(
    action: str, shares: float, current_price: float,
    decision: dict, ticker: str, market_data: dict,
    regime: dict, vix: float, tags: dict,
    risk, atr: float, config: dict,
):
    """Execute a BUY, SELL, SHORT, COVER, or HOLD and update all tracking."""
    if action == "BUY" and shares > 0:
        result = execute_trade(action, shares, current_price, decision)

        if result["status"] == "executed":
            txn = result["transaction"]
            logger.info(
                f"EXECUTED: BUY {shares:.4f} @ ${txn['price']:.2f} "
                f"| Cash: {format_currency(result['portfolio']['cash'])} "
                f"| Value: {format_currency(result['portfolio']['total_value'])}"
            )

            stop_pct = risk.compute_atr_stop_pct(atr, current_price)
            if decision.get("stop_pct"):
                ai_stop = decision["stop_pct"]
                if ai_stop > 0.1:
                    ai_stop = ai_stop / 100
                stop_pct = risk.clamp_stop_pct(max(stop_pct, ai_stop))

            save_active_position(txn["price"], stop_pct, decision, direction="long")
            logger.info(
                f"Stop set: {stop_pct:.1%} "
                f"(ATR ${atr:.2f} x {risk.atr_stop_multiplier})"
            )

            _add_journal_entry(
                txn, "BUY", ticker, current_price, market_data, vix,
                tags, decision, result,
            )
        else:
            logger.warning(f"Trade rejected: {result.get('reason', 'unknown')}")

    elif action == "SHORT" and shares > 0:
        result = execute_trade(action, shares, current_price, decision)

        if result["status"] == "executed":
            txn = result["transaction"]
            logger.info(
                f"EXECUTED: SHORT {shares:.4f} @ ${txn['price']:.2f} "
                f"| Cash: {format_currency(result['portfolio']['cash'])} "
                f"| Value: {format_currency(result['portfolio']['total_value'])}"
            )

            stop_pct = risk.compute_atr_stop_pct(atr, current_price)
            if decision.get("stop_pct"):
                ai_stop = decision["stop_pct"]
                if ai_stop > 0.1:
                    ai_stop = ai_stop / 100
                stop_pct = risk.clamp_stop_pct(max(stop_pct, ai_stop))

            save_active_position(txn["price"], stop_pct, decision, direction="short")
            logger.info(
                f"Short stop set: {stop_pct:.1%} "
                f"(ATR ${atr:.2f} x {risk.atr_stop_multiplier})"
            )

            _add_journal_entry(
                txn, "SHORT", ticker, current_price, market_data, vix,
                tags, decision, result,
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
            _close_journal_for_exit(txn)
            record_exit_time("AI-initiated SELL")

    elif action == "COVER" and shares > 0:
        result = execute_trade(action, shares, current_price, decision)

        if result["status"] == "executed":
            txn = result["transaction"]
            logger.info(
                f"EXECUTED: COVER {shares:.4f} @ ${txn['price']:.2f} "
                f"| P&L: {format_currency(txn['realized_pnl'])}"
            )
            record_trade_pnl(txn["realized_pnl"])
            clear_active_position()
            _close_journal_for_exit(txn)
            record_exit_time("AI-initiated COVER")

    else:
        reasoning = decision.get("reasoning", "N/A")[:120]
        logger.info(f"HOLD — {reasoning}")

    # Save cycle + update dashboard
    _save_latest_cycle(decision, market_data, regime)
    _update_dashboard(market_data, load_portfolio())


def _close_journal_for_exit(exit_txn: dict):
    """Close the most recent open BUY or SHORT journal entry."""
    journal = load_journal()
    # SELL closes BUY; COVER closes SHORT
    open_action = "SHORT" if exit_txn["action"] == "COVER" else "BUY"
    for entry in reversed(journal):
        if (entry["action"] == open_action
            and entry["ticker"] == exit_txn["ticker"]
            and entry.get("lesson") is None):
            journal_close_entry(
                open_trade_id=entry["trade_id"],
                close_trade_id=exit_txn["id"],
                close_price=exit_txn["price"],
                realized_pnl=exit_txn["realized_pnl"],
            )
            return
    logger.warning(f"No open {open_action} entry to close for {exit_txn['id']}")


def _add_journal_entry(
    txn: dict, action: str, ticker: str, current_price: float,
    market_data: dict, vix: float, tags: dict, decision: dict,
    result: dict,
):
    """Add a journal entry for a BUY or SHORT trade."""
    market_snap = (
        f"{ticker} ${current_price} "
        f"({market_data['current'].get('change_pct', 0):+.2f}%), "
        f"VIX {vix:.1f}"
    )
    journal_add_entry(
        trade_id=txn["id"],
        action=action,
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

    # Judgment scorecard — measure AI value-add
    try:
        from .judgment_scorecard import build_scorecard
        sc = build_scorecard()
        logger.info(
            f"Judgment scorecard: {sc.get('outcomes_with_meta', 0)} tracked, "
            f"{len(sc.get('sources', {}))} sources"
        )
    except Exception as e:
        logger.debug(f"Judgment scorecard failed: {e}")

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

    # Schedule — runs every 5 min, but run_cycle() skips when flat + checked recently
    adaptive = config.get("v8_risk", {}).get("adaptive_cycle", {})
    fast_interval = adaptive.get("position_interval_minutes", 5)
    normal_interval = adaptive.get("normal_interval_minutes", 15)
    schedule.every(fast_interval).minutes.do(run_cycle)
    schedule.every(5).minutes.do(_check_daily_tasks_time)

    logger.info(
        f"Scheduler started: every {fast_interval}min (flat: {normal_interval}min) "
        f"+ 16:15 ET daily tasks"
    )
    run_cycle()  # Run immediately

    while _running:
        schedule.run_pending()
        time.sleep(10)

    logger.info("MonopolyTrader v8 stopped")


if __name__ == "__main__":
    main()
