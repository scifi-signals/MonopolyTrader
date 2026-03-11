"""AI decision engine — Claude IS the research brain.

v6: Researcher identity. Trades are experiments, not bets. The playbook is the
primary output, not P&L. HOLD is the default. Don't repeat failed experiments.
"""

import json
from datetime import datetime
from .utils import (
    load_config, format_currency, iso_now, setup_logging,
    call_ai_with_fallback,
)
from .journal import get_recent_entries, format_journal_for_brief
from .news_feed import NewsFeed, format_news_for_prompt
from .web_search import format_search_results
from .events import format_events_for_brief
from .thesis_builder import format_playbook_for_brief, format_matching_patterns_for_brief
from .analyst import load_mid, format_mid_for_system_prompt, format_briefing_for_prompt

logger = setup_logging("agent")


SYSTEM_PROMPT = """You are MonopolyTrader v6, a RESEARCHER studying TSLA stock patterns. You manage $1,000 of Monopoly dollars as your experimental budget.

Your PRIMARY OUTPUT is a playbook of tested patterns — which market conditions produce reliable moves and which don't. Trades are EXPERIMENTS that generate data for this playbook. P&L is a secondary signal about pattern quality, not a score to maximize.

## CORE PRINCIPLES

1. HOLD is the default. You need a clear, testable hypothesis with expected NEW learning to justify opening an experiment. "I think the price will go up" is not a hypothesis. "RSI divergence from daily trend in range-bound markets predicts a reversal within 2 hours" is a hypothesis.

2. Don't repeat failed experiments. Check your PLAYBOOK before every decision. If it shows N>=5 trades at <25% win rate for conditions matching the current market, trading here teaches you NOTHING new. That pattern has been tested. Move on.

3. Confidence must reflect your playbook's empirical win rate for these conditions, NOT your narrative conviction about this specific setup. If your playbook shows 20% wins in similar conditions, your confidence should be near 0.2 regardless of how compelling today's chart looks.

4. When you HOLD, explain what hypothesis you WOULD test and what conditions would need to change. A good HOLD decision is as valuable as a good trade — it means you correctly identified that the current setup has no new learning to offer.

5. The money is imaginary. Losses are FINE if they generate genuinely new knowledge — a pattern tested for the first time, a new market condition explored. Redundant losses (repeating a known-losing pattern) are WASTE. They add no information to the playbook.

6. Track what you're learning. Every trade should have a specific hypothesis and expected learning outcome. After the trade closes, the lesson should confirm or refute the hypothesis, not just say "should have waited."

## POSITION LIMITS (enforced by code)
1. Max 50% of portfolio value in any position
2. Keep $100 cash minimum
3. Fractional shares OK
4. No stop losses — you evaluate every position with context each cycle

## TRADING COSTS
Every trade has a real cost in slippage. The SPREAD COST section shows your round-trip breakeven. Factor this into every decision — an experiment that targets moves smaller than the spread cost cannot generate useful data because noise dominates the signal.

## SHADOW JOURNAL
Your HOLD decisions are tracked. The system records what WOULD have happened if you'd traded during each HOLD. Use this data to calibrate: are you holding too much (missing profitable setups) or not enough (correctly avoiding losses)?

## RESEARCH METRICS
Your brief includes research efficiency metrics. Pay attention to:
- Experiment efficiency: are you exploring NEW conditions or repeating old ones?
- Redundant loss rate: this should be 0%. Any loss in a known-losing pattern is waste.
- Calibration error: does your stated confidence match actual outcomes?

## PREDICTIONS
Every cycle you MUST output a short-term price prediction. This is separate from your trade decision — it's about what you think the price will do regardless of whether you trade. Your predictions are scored against reality and the results appear in your PREDICTION SCORECARD.

This is how you learn from EVERY cycle, not just trades. 26 cycles per day = 26 prediction data points. Pay attention to your scorecard — if you're 80% accurate on direction in trending markets but 30% in range-bound, that tells you where your understanding is strong vs weak.

Magnitude thresholds: "small" = 0.2-0.5% move, "moderate" = 0.5-1.5%, "large" = >1.5%. Direction "flat" = <0.2% move.
Cycles: 1-4 (15min to 1hr). Shorter timeframe = bolder prediction = more learning.

Respond ONLY with valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": <float, 0 if HOLD>,
  "confidence": <float 0-1>,
  "strategy": "<name of the strategy you're using or observing this cycle>",
  "hypothesis": "<specific testable hypothesis for this trade, or 'N/A - observing' for HOLD>",
  "expected_learning": "<what new data this trade will add to the playbook, or what you're watching for during HOLD>",
  "reasoning": "<your analysis: current market conditions, what your playbook says about these conditions, why this experiment is or isn't worth running>",
  "risk_note": "<what could go wrong, and what outcome would refute your hypothesis>",
  "prediction": {
    "direction": "up" | "down" | "flat",
    "magnitude": "small" | "moderate" | "large",
    "cycles": <int 1-4, how many 15-min cycles until evaluation>,
    "basis": "<1 sentence: what signal drives this prediction>"
  }
}"""


def build_market_brief(
    market_data: dict,
    world: dict,
    portfolio: dict,
    news_feed: NewsFeed | None,
    web_results: list[dict],
    journal_entries: list[dict],
    config: dict,
    events: dict | None = None,
    matched_patterns: list[dict] | None = None,
    options_data: dict | None = None,
) -> str:
    """Build the complete market brief for Claude.

    This is the single prompt that contains everything Claude needs
    to make a trading decision.
    """
    current = market_data.get("current", {})
    daily = market_data.get("daily_indicators", {})
    intraday = market_data.get("intraday_indicators") or {}
    recent = market_data.get("recent_days", [])
    regime = market_data.get("regime", {})
    ticker = config["ticker"]

    parts = []

    # === TSLA Price & Indicators ===
    parts.append(f"=== {ticker} ===")
    parts.append(f"Price: ${current.get('price', 'N/A')} ({current.get('change_pct', 0):+.2f}%)")
    parts.append(f"Volume: {current.get('volume', 0):,}")
    parts.append("")

    parts.append("Daily Indicators:")
    indicator_keys = [
        "rsi_14", "sma_20", "sma_50",
        "macd", "macd_signal", "macd_histogram", "macd_crossover",
        "bollinger_upper", "bollinger_lower", "bollinger_mid",
        "atr", "adx", "adx_pos", "adx_neg",
        "ema_12", "ema_26", "volume_sma_20", "obv",
    ]
    for key in indicator_keys:
        val = daily.get(key)
        if val is not None:
            parts.append(f"  {key}: {val}")

    if intraday and intraday.get("current_price"):
        parts.append("")
        parts.append("Intraday (5min) Indicators:")
        for key in ["rsi_14", "sma_20", "macd_crossover", "bollinger_upper", "bollinger_lower"]:
            val = intraday.get(key)
            if val is not None:
                parts.append(f"  {key}: {val}")

    if recent:
        parts.append("")
        parts.append("Last 5 Days:")
        for day in recent:
            parts.append(
                f"  {day['date']}: O={day['open']} H={day['high']} "
                f"L={day['low']} C={day['close']} V={day['volume']:,}"
            )

    # === World Context ===
    parts.append("")
    parts.append("=== WORLD ===")

    macro = world.get("macro", {})
    if macro:
        parts.append("Macro:")
        for sym, data in macro.items():
            name = data.get("name", sym)
            parts.append(f"  {name} ({sym}): {data['price']} ({data['change_pct']:+.2f}%)")

    peers = world.get("ev_peers", {})
    if peers:
        parts.append("EV Peers:")
        for sym, data in peers.items():
            parts.append(f"  {sym}: ${data['price']} ({data['change_pct']:+.2f}%)")

    if regime:
        parts.append("")
        intraday_dir = regime.get("intraday_directional")
        intraday_adx = regime.get("intraday_adx")
        intraday_str = ""
        if intraday_dir:
            intraday_str = f" | Intraday: {intraday_dir}"
            if intraday_adx is not None:
                intraday_str += f" (ADX={intraday_adx:.1f})"
        parts.append(f"Regime: trend={regime.get('trend','?')} directional={regime.get('directional','?')} "
                     f"volatility={regime.get('volatility','?')} VIX={regime.get('vix',0):.1f} "
                     f"ADX={regime.get('adx',0):.1f}{intraday_str}")

    # === News ===
    parts.append("")
    parts.append("=== NEWS ===")
    if news_feed and news_feed.items:
        parts.append(format_news_for_prompt(news_feed, max_items=10))
    else:
        parts.append("No news available.")

    # === Upcoming Events ===
    if events:
        parts.append("")
        parts.append("=== UPCOMING EVENTS ===")
        parts.append(format_events_for_brief(events))

    # === Web Search ===
    if web_results:
        parts.append("")
        parts.append("=== WEB SEARCH (last 24h) ===")
        parts.append(format_search_results(web_results))

    # === Today's Briefing ===
    try:
        from .utils import load_json, DATA_DIR
        briefing = load_json(DATA_DIR / "daily_briefing.json", default={})
        if briefing:
            parts.append("")
            parts.append("=== TODAY'S BRIEFING ===")
            briefing_text = format_briefing_for_prompt(briefing)
            # Check staleness
            import os
            briefing_path = DATA_DIR / "daily_briefing.json"
            if briefing_path.exists():
                mtime = os.path.getmtime(briefing_path)
                age_hours = (datetime.now().timestamp() - mtime) / 3600
                if age_hours > 6:
                    parts.append(f"[STALE — generated {age_hours:.0f}h ago]")
            parts.append(briefing_text)
    except Exception:
        pass

    # === Options Flow ===
    try:
        options = options_data
        if not options:
            from .market_data import get_options_snapshot
            options = get_options_snapshot(ticker)
        if options:
            parts.append("")
            parts.append("=== OPTIONS FLOW ===")
            parts.append(f"  Put/Call Ratio: {options.get('put_call_ratio', 'N/A')}")
            parts.append(f"  Max Pain: ${options.get('max_pain', 'N/A')}")
            parts.append(f"  Unusual Volume: {'YES' if options.get('unusual_volume') else 'No'}")
            parts.append(f"  Nearest Expiry: {options.get('nearest_expiry', 'N/A')}")
    except Exception:
        pass

    # === Analyst Consensus ===
    try:
        from .market_data import get_analyst_consensus
        analysts = get_analyst_consensus(ticker)
        if analysts:
            parts.append("")
            parts.append("=== ANALYST CONSENSUS ===")
            buys = analysts.get('strong_buy', 0) + analysts.get('buy', 0)
            holds = analysts.get('hold', 0)
            sells = analysts.get('sell', 0) + analysts.get('strong_sell', 0)
            parts.append(f"  Buy: {buys}, Hold: {holds}, Sell: {sells}")
            if analysts.get('target_mean'):
                vs_current = ""
                if current.get("price") and analysts["target_mean"] > 0:
                    diff = ((analysts["target_mean"] - current["price"]) / current["price"]) * 100
                    vs_current = f" ({diff:+.1f}% vs current)"
                parts.append(f"  Mean Target: ${analysts['target_mean']}{vs_current}")
                parts.append(f"  Range: ${analysts.get('target_low', '?')} - ${analysts.get('target_high', '?')}")
    except Exception:
        pass

    # === Institutional ===
    try:
        from .market_data import get_institutional_data
        inst = get_institutional_data(ticker)
        if inst:
            parts.append("")
            parts.append("=== INSTITUTIONAL ===")
            if inst.get("institutional_pct"):
                parts.append(f"  Institutional Ownership: {inst['institutional_pct']}%")
            if inst.get("short_interest_pct"):
                parts.append(f"  Short Interest: {inst['short_interest_pct']}%")
            if inst.get("top_holders"):
                parts.append(f"  Top Holders: {', '.join(inst['top_holders'][:3])}")
    except Exception:
        pass

    # === Drawdown Status (v6.1 Blindspot #6) ===
    try:
        drawdown_text = _compute_drawdown_section(portfolio, config)
        if drawdown_text:
            parts.append("")
            parts.append(drawdown_text)
    except Exception:
        pass

    # === Last Cycle (v6.1 Blindspot #11) ===
    try:
        last_cycle_text = _format_last_cycle(current.get("price", 0))
        if last_cycle_text:
            parts.append("")
            parts.append(last_cycle_text)
    except Exception:
        pass

    # === Portfolio ===
    parts.append("")
    parts.append("=== PORTFOLIO ===")
    parts.append(f"Cash: {format_currency(portfolio.get('cash', 0))}")
    parts.append(f"Total Value: {format_currency(portfolio.get('total_value', 0))}")
    parts.append(f"P&L: {format_currency(portfolio.get('total_pnl', 0))} ({portfolio.get('total_pnl_pct', 0):+.2f}%)")
    parts.append(f"Trades: {portfolio.get('total_trades', 0)} (W:{portfolio.get('winning_trades', 0)} L:{portfolio.get('losing_trades', 0)})")

    holdings = portfolio.get("holdings", {})
    h = holdings.get(ticker, {})
    if h.get("shares", 0) > 0:
        parts.append(
            f"Position: {h['shares']:.4f} {ticker} @ avg ${h['avg_cost_basis']:.2f} "
            f"(current ${h['current_price']:.2f}, unrealized {format_currency(h['unrealized_pnl'])})"
        )
        pnl_pct = ((h['current_price'] - h['avg_cost_basis']) / h['avg_cost_basis'] * 100) if h['avg_cost_basis'] > 0 else 0
        if pnl_pct < 0:
            parts.append(f"  >>> UNDERWATER: {pnl_pct:.1f}% loss. Evaluate: exit, hold, or buy more? <<<")
    else:
        parts.append("Position: FLAT (all cash)")

    # === Spread Cost (hurdle rate) ===
    slippage_pct = config["risk_params"].get("slippage_per_side_pct", 0.0005)
    round_trip_pct = slippage_pct * 2 * 100  # both sides, as percentage
    price = current.get("price", 400)
    parts.append("")
    parts.append("=== SPREAD COST ===")
    parts.append(f"Round-trip slippage: {round_trip_pct:.2f}% (${price * slippage_pct * 2:.2f} per share)")
    parts.append(f"Breakeven move: >{round_trip_pct:.2f}% (>${price * slippage_pct * 2:.2f}/share)")
    parts.append(f"Minimum target move (2x breakeven): >{round_trip_pct * 2:.2f}% (>${price * slippage_pct * 4:.2f}/share)")

    # === Trading Stats (self-awareness) ===
    try:
        from .portfolio import load_transactions
        all_txns = load_transactions()
        all_sells = [t for t in all_txns if t["action"] == "SELL"]
        if all_sells:
            parts.append("")
            parts.append("=== YOUR TRADING STATS ===")
            total_sells = len(all_sells)
            winners = sum(1 for t in all_sells if (t.get("realized_pnl", 0) or 0) > 0)
            losers = sum(1 for t in all_sells if (t.get("realized_pnl", 0) or 0) < 0)
            win_rate = (winners / total_sells * 100) if total_sells > 0 else 0
            total_realized = sum(t.get("realized_pnl", 0) or 0 for t in all_sells)
            avg_win = 0
            avg_loss = 0
            winning_pnls = [t.get("realized_pnl", 0) for t in all_sells if (t.get("realized_pnl", 0) or 0) > 0]
            losing_pnls = [t.get("realized_pnl", 0) for t in all_sells if (t.get("realized_pnl", 0) or 0) < 0]
            if winning_pnls:
                avg_win = sum(winning_pnls) / len(winning_pnls)
            if losing_pnls:
                avg_loss = sum(losing_pnls) / len(losing_pnls)
            parts.append(f"Closed trades: {total_sells} (W:{winners} L:{losers})")
            parts.append(f"Win rate: {win_rate:.0f}%")
            parts.append(f"Total realized P&L: ${total_realized:.2f}")
            parts.append(f"Avg win: ${avg_win:.2f} | Avg loss: ${avg_loss:.2f}")

            # Recent streak
            consecutive_losses = 0
            for t in reversed(all_sells):
                if (t.get("realized_pnl", 0) or 0) < 0:
                    consecutive_losses += 1
                else:
                    break
            if consecutive_losses > 0:
                streak_pnl = sum(t.get("realized_pnl", 0) or 0 for t in all_sells[-consecutive_losses:])
                parts.append(f"Current streak: {consecutive_losses} consecutive losses (${streak_pnl:.2f})")

            # Today's activity
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_txns = [t for t in all_txns if t["timestamp"].startswith(today)]
            today_sells = [t for t in today_txns if t["action"] == "SELL"]
            today_buys = [t for t in today_txns if t["action"] == "BUY"]
            if today_txns:
                today_pnl = sum(t.get("realized_pnl", 0) or 0 for t in today_sells)
                parts.append(f"Today: {len(today_buys)} buys, {len(today_sells)} sells, P&L: ${today_pnl:.2f}")
    except Exception:
        pass

    # Position limits for Claude
    max_position = portfolio.get("total_value", 1000) * config["risk_params"].get("max_position_pct", 0.50)
    current_holding_value = h.get("shares", 0) * h.get("current_price", 0)
    available_for_position = max_position - current_holding_value
    max_buy_value = min(available_for_position, portfolio.get("cash", 0) - config["risk_params"].get("min_cash_reserve", 100))
    if max_buy_value > 0 and current.get("price", 0) > 0:
        max_shares = max_buy_value / current["price"]
        parts.append(f"Max BUY: {max_shares:.4f} shares (${max_buy_value:.2f})")
    elif h.get("shares", 0) > 0:
        parts.append(f"Can SELL up to {h['shares']:.4f} shares")

    # === Pattern Matches for Current Conditions ===
    if matched_patterns:
        parts.append("")
        parts.append("=== PATTERN MATCHES (current conditions) ===")
        parts.append(format_matching_patterns_for_brief(matched_patterns))

    # === Playbook (learning stats) ===
    parts.append("")
    parts.append("=== YOUR PLAYBOOK ===")
    try:
        from .utils import load_json, DATA_DIR
        ledger = load_json(DATA_DIR / "thesis_ledger.json", default={})
        parts.append(format_playbook_for_brief(ledger))
    except Exception:
        parts.append("Playbook not yet available.")

    # === Shadow Journal (HOLD tracking) ===
    try:
        from .shadow_journal import format_shadow_for_brief
        shadow_text = format_shadow_for_brief()
        if shadow_text:
            parts.append("")
            parts.append("=== HOLD SHADOW TRACKING ===")
            parts.append(shadow_text)
    except Exception:
        pass

    # === Prediction Scorecard ===
    try:
        from .prediction_tracker import format_prediction_scorecard
        scorecard = format_prediction_scorecard(hours=72)
        if scorecard:
            parts.append("")
            parts.append("=== YOUR PREDICTION SCORECARD (last 3 days) ===")
            parts.append(scorecard)
    except Exception:
        pass

    # === Trade Journal ===
    parts.append("")
    parts.append("=== YOUR TRADE JOURNAL (last 5) ===")
    parts.append(format_journal_for_brief(journal_entries))

    return "\n".join(parts)


def _compute_drawdown_section(portfolio: dict, config: dict) -> str:
    """Compute drawdown status for the agent's brief.

    v6.1 Blindspot #6: Makes drawdown awareness explicit and prominent.
    """
    from .utils import load_json, DATA_DIR
    from .portfolio import load_transactions
    from pathlib import Path
    import os

    current_value = portfolio.get("total_value", 1000)
    starting = config.get("starting_balance", 1000)

    # Find portfolio peak from snapshots
    snapshots_dir = DATA_DIR / "snapshots"
    peak_value = starting
    peak_date = "start"

    if snapshots_dir.exists():
        for snap_file in sorted(snapshots_dir.glob("*.json")):
            try:
                snap = load_json(snap_file, default={})
                snap_value = snap.get("total_value", 0)
                if snap_value > peak_value:
                    peak_value = snap_value
                    peak_date = snap.get("date", snap_file.stem)
            except Exception:
                continue

    # Also consider current value as potential peak
    if current_value > peak_value:
        peak_value = current_value
        peak_date = "now"

    drawdown_pct = ((current_value - peak_value) / peak_value * 100) if peak_value > 0 else 0

    # Count consecutive losses
    consecutive_losses = 0
    try:
        txns = load_transactions()
        all_sells = [t for t in txns if t["action"] == "SELL"]
        for t in reversed(all_sells):
            if (t.get("realized_pnl", 0) or 0) < 0:
                consecutive_losses += 1
            else:
                break
    except Exception:
        pass

    # Days since peak
    days_since_peak = "?"
    if peak_date != "now" and peak_date != "start":
        try:
            from datetime import datetime, timezone
            peak_dt = datetime.strptime(peak_date, "%Y-%m-%d")
            now = datetime.now(timezone.utc)
            days_since_peak = (now.replace(tzinfo=None) - peak_dt).days
        except Exception:
            pass

    # Risk level classification
    if (abs(drawdown_pct) > 15) or consecutive_losses > 5:
        risk_level = "HIGH"
        risk_note = "consider reducing position sizes significantly"
    elif (abs(drawdown_pct) > 5) or consecutive_losses >= 3:
        risk_level = "ELEVATED"
        risk_note = "consider reducing position sizes"
    else:
        risk_level = "LOW"
        risk_note = "normal operations"

    parts = ["=== DRAWDOWN STATUS ==="]
    if drawdown_pct < -0.01:
        parts.append(
            f"Current drawdown: {drawdown_pct:.2f}% from peak "
            f"({format_currency(peak_value)} -> {format_currency(current_value)})"
        )
        parts.append(f"Days since peak: {days_since_peak}")
    else:
        parts.append(f"At or near peak: {format_currency(current_value)}")

    if consecutive_losses > 0:
        parts.append(f"Consecutive losses: {consecutive_losses}")

    parts.append(f"Risk level: {risk_level} -- {risk_note}")

    return "\n".join(parts)


def _format_last_cycle(current_price: float) -> str:
    """Format the last cycle's decision for short-term memory.

    v6.1 Blindspot #11: Makes latest_cycle.json useful by showing
    the agent what it decided last cycle and what happened since.
    """
    from .utils import load_json, DATA_DIR
    from datetime import datetime, timezone

    cycle_data = load_json(DATA_DIR / "latest_cycle.json", default={})
    if not cycle_data or not cycle_data.get("timestamp"):
        return ""

    action = cycle_data.get("action", "HOLD")
    strategy = cycle_data.get("strategy", "")
    confidence = cycle_data.get("confidence", 0)
    prev_price = cycle_data.get("price", 0)

    # Compute time since last cycle
    time_ago = "?"
    try:
        last_time = datetime.fromisoformat(cycle_data["timestamp"])
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        mins_ago = (now - last_time).total_seconds() / 60
        time_ago = f"{mins_ago:.0f} min ago"
    except Exception:
        pass

    # Price change since last cycle
    price_change_str = ""
    if prev_price > 0 and current_price > 0:
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        price_change_str = f"Price then: ${prev_price:.2f} -> Price now: ${current_price:.2f} ({change_pct:+.2f}%)"

    parts = [f"=== LAST CYCLE ({time_ago}) ==="]
    strategy_str = f" | Strategy: {strategy}" if strategy else ""
    parts.append(f"Action: {action}{strategy_str} | Confidence: {confidence:.2f}")
    if price_change_str:
        parts.append(price_change_str)

    return "\n".join(parts)


def make_decision(
    market_data: dict,
    world: dict,
    portfolio: dict,
    news_feed: NewsFeed | None,
    web_results: list[dict],
    journal_entries: list[dict],
    config: dict,
    events: dict | None = None,
    matched_patterns: list[dict] | None = None,
    options_data: dict | None = None,
) -> dict:
    """Call Claude to make a research decision.

    v6: Researcher identity. System prompt includes Market Intelligence Document.
    User prompt includes daily briefing + enhanced market data + shadow journal.
    """
    brief = build_market_brief(
        market_data, world, portfolio, news_feed,
        web_results, journal_entries, config, events,
        matched_patterns, options_data,
    )

    # Build system prompt with Market Intelligence Document
    system = SYSTEM_PROMPT
    try:
        mid = load_mid()
        mid_text = format_mid_for_system_prompt(mid)
        if mid_text:
            system = SYSTEM_PROMPT + "\n\n" + mid_text
    except Exception as e:
        logger.warning(f"Failed to load MID for system prompt: {e}")

    logger.info(f"Calling Claude for decision (brief ~{len(brief)} chars, system ~{len(system)} chars)")

    try:
        raw, model_used = call_ai_with_fallback(
            system=system,
            user=brief,
            max_tokens=1200,
            config=config,
        )

        # Strip code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        decision = json.loads(raw)

        # Validate required fields
        action = decision.get("action", "HOLD")
        if action not in ("BUY", "SELL", "HOLD"):
            logger.warning(f"Invalid action '{action}', defaulting to HOLD")
            decision["action"] = "HOLD"
            decision["shares"] = 0

        if action == "HOLD":
            decision["shares"] = 0

        decision["_model_version"] = model_used

        # Validate prediction field
        pred = decision.get("prediction")
        if not pred or not isinstance(pred, dict):
            decision["prediction"] = {
                "direction": "flat", "magnitude": "small",
                "cycles": 1, "basis": "no prediction provided",
            }
        else:
            if pred.get("direction") not in ("up", "down", "flat"):
                pred["direction"] = "flat"
            if pred.get("magnitude") not in ("small", "moderate", "large"):
                pred["magnitude"] = "small"
            if pred.get("direction") == "flat":
                pred["magnitude"] = "flat"
            cycles = pred.get("cycles", 2)
            pred["cycles"] = max(1, min(4, int(cycles) if isinstance(cycles, (int, float)) else 2))

        logger.info(
            f"Decision: {decision.get('action')} "
            f"{decision.get('shares', 0):.4f} shares, "
            f"confidence={decision.get('confidence', 0):.2f}, "
            f"prediction={decision.get('prediction', {}).get('direction', '?')}/"
            f"{decision.get('prediction', {}).get('magnitude', '?')}"
        )

        return decision

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response: {e}\nRaw: {raw[:500]}")
        return {
            "action": "HOLD", "shares": 0, "confidence": 0,
            "reasoning": f"JSON parse error: {e}",
            "risk_note": "Defaulting to HOLD due to parse error",
        }
    except Exception as e:
        logger.error(f"Agent decision failed: {e}")
        return {
            "action": "HOLD", "shares": 0, "confidence": 0,
            "reasoning": f"Error: {e}",
            "risk_note": f"Agent error: {e}",
        }
