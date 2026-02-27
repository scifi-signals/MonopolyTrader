"""AI decision engine — Claude-powered trading brain.

v2: Two-phase Analyst/Trader architecture for thesis-driven trading.
v1 functions (make_decision, make_decision_multi_step) kept for ensemble/replay compatibility.
"""

import json
import os
import yfinance as yf
from anthropic import Anthropic
from .utils import load_config, format_currency, iso_now, setup_logging, call_ai_with_fallback
from .knowledge_base import get_relevant_knowledge, get_relevant_research, add_prediction
from .market_data import get_bsm_signals
from .strategies import StrategySignal, calculate_signal_balance
from .news_feed import NewsFeed, format_news_for_prompt
from .thesis import Thesis, load_thesis, save_thesis, thesis_changed_meaningfully

logger = setup_logging("agent")

SYSTEM_PROMPT = """You are MonopolyTrader, an AI trading agent managing a virtual portfolio of Monopoly dollars. You are LEARNING — you started with $1,000 and your goal is to grow it by getting better at predicting TSLA's movements over time.

You have a knowledge base of lessons from past trades, patterns you've discovered, and research you've conducted. USE THIS KNOWLEDGE. Reference specific lessons and patterns when making decisions. If you're uncertain, say so — it's better to HOLD than to make a low-conviction trade.

Every trade is a hypothesis. State clearly what you predict will happen and why. You will be reviewed on this prediction later.

RISK RULES (v3 — you MUST follow these):
- Max position: 65% of portfolio value (TSLA is too volatile for more)
- Max single trade: 20% of portfolio value
- Min cash reserve: 15% of portfolio value
- Stop loss: dynamic ATR-based (wider stops in high volatility)
- Position sizing: inverse ATR (wider stop = smaller position, max 2% risk per trade)
- Cooldown: 15 min between trades
- If daily loss exceeds 8%, stop trading
- Macro gate: if SPY down >2% or VIX >30, require confidence >0.80 for BUY
- Earnings blackout: no new BUY within 48 hours of TSLA earnings

When deciding trade size, be conservative early on. The system will cap your shares based on ATR sizing. Small positions let you learn without blowing up.

PREDICTION TRACK RECORD: Your prediction accuracy is shown in the knowledge section. Use it. If a strategy has >60% accuracy, weight your confidence higher on trades using it. If a strategy is below 45%, reduce confidence on those predictions. Don't make high-confidence predictions using strategies where your track record is poor.

Be a scientific instrument, not a storyteller. Base decisions on data, not narratives.

HOLD IS AN ACTIVE DECISION. When you decide to HOLD, you are choosing NOT to act despite available signals. This is a deliberate choice with opportunity cost. You MUST justify every HOLD with the same rigor as a BUY or SELL.

SIGNAL INTERPRETATION: Strategies that return HOLD are ABSTAINING — they have no opinion. They are NOT disagreeing with active signals. If momentum says BUY at 0.75 confidence and three other strategies say HOLD, that means ONE strategy has a strong signal and THREE have nothing to say. That is NOT "only 1 of 5 agrees." Do not use the number of abstaining strategies as a reason to HOLD. Judge the quality and confidence of the ACTIVE signals on their own merits. A single strategy at >0.70 confidence with no opposing signals is a clear trade.

ANTI-PARALYSIS: If a <hold_streak_warning> appears in the data, your excessive caution is proven to be costing money. Respond by:
- Lowering your confidence threshold: act on any active signal with confidence >0.50
- A single strategy at >0.60 confidence with no opposing signals is a CLEAR TRADE
- Your prediction accuracy improves by trading more, not by waiting for perfection
- Prefer a small BUY over another HOLD when the signal balance is positive

Respond ONLY with valid JSON matching this schema:
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": <float, 0 if HOLD>,
  "confidence": <float 0-1>,
  "strategy": "<primary strategy driving this decision>",
  "hypothesis": "<I predict TSLA will [direction] by [amount] over [timeframe] because [reasoning]>",
  "reasoning": "<detailed analysis referencing indicators, signals, and knowledge>",
  "knowledge_applied": ["<lesson_id — MUST list at least one if lessons were provided>"],
  "risk_note": "<any risk concerns>",
  "predictions": {
    "30min": {"direction": "up"|"down"|"flat", "target": <price>, "confidence": <0-1>},
    "2hr": {"direction": "up"|"down"|"flat", "target": <price>, "confidence": <0-1>},
    "1day": {"direction": "up"|"down"|"flat", "target": <price>, "confidence": <0-1>}
  },
  "sentiment_score": <float -1 to 1, your read on current news/sentiment>,
  "hold_analysis": <REQUIRED if action is HOLD, null otherwise> {
    "opportunity_cost": "<what potential gain are you forgoing by not acting?>",
    "downside_protection": "<what risk are you avoiding by not acting?>",
    "signal_balance": "<describe the balance of buy vs sell signals>",
    "decision_boundary": "<what specific change in conditions would flip this to BUY or SELL?>"
  },
  "research_request": "<optional — something you want researched before next decision>"
}"""


def _get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Try loading from file
        for path in ["anthropic_api_key.txt", "../anthropic_api_key.txt"]:
            try:
                with open(path) as f:
                    api_key = f.read().strip()
                    break
            except FileNotFoundError:
                continue
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in env or key file")
    return Anthropic(api_key=api_key)


def _format_market_data(market_data: dict) -> str:
    """Format market data for the agent prompt."""
    current = market_data.get("current", {})
    daily = market_data.get("daily_indicators", {})
    intraday = market_data.get("intraday_indicators")
    recent = market_data.get("recent_days", [])

    parts = [f"Current: ${current.get('price', 'N/A')} ({current.get('change_pct', 0):+.2f}%)"]
    parts.append(f"Volume: {current.get('volume', 0):,}")

    # Position limits (if provided, e.g. from replay engine)
    limits = market_data.get("position_limits")
    if limits:
        parts.append(f">>> MAX TRADE: ${limits['max_trade_value']:.2f} = {limits['max_shares_at_current_price']:.4f} shares at current price. DO NOT exceed this. <<<")

    parts.append("")

    parts.append("Daily Indicators:")
    for key in ["rsi_14", "sma_20", "sma_50", "macd", "macd_signal", "macd_crossover",
                 "bollinger_upper", "bollinger_lower", "atr"]:
        val = daily.get(key)
        if val is not None:
            parts.append(f"  {key}: {val}")

    if intraday and intraday.get("current_price"):
        parts.append("")
        parts.append("Intraday Indicators:")
        for key in ["rsi_14", "sma_20", "macd_crossover"]:
            val = intraday.get(key)
            if val is not None:
                parts.append(f"  {key}: {val}")

    if recent:
        parts.append("")
        parts.append("Recent Days:")
        for day in recent:
            parts.append(f"  {day['date']}: O={day['open']} H={day['high']} L={day['low']} C={day['close']} V={day['volume']:,}")

    return "\n".join(parts)


def _format_signals(signals: list[StrategySignal], aggregate: StrategySignal) -> str:
    """Format strategy signals for the agent prompt, including signal balance."""
    parts = ["Individual Signals:"]
    for s in signals:
        parts.append(f"  [{s.strategy}] {s.action} confidence={s.confidence:.2f} weight={s.weight:.2f}")
        parts.append(f"    {s.reasoning}")

    parts.append("")
    parts.append(f"Aggregate: {aggregate.action} confidence={aggregate.confidence:.2f}")
    parts.append(f"  {aggregate.reasoning}")

    # Add signal balance analysis
    balance = calculate_signal_balance(signals)
    parts.append("")
    parts.append(f"Signal Balance: {balance['balance']:+.4f} "
                 f"(buy_pressure={balance['buy_pressure']:.3f}, sell_pressure={balance['sell_pressure']:.3f})")
    for name, b in balance["breakdown"].items():
        parts.append(f"  {name}: {b['action']} contribution={b['contribution']:+.4f}")

    return "\n".join(parts)


def _format_portfolio(portfolio: dict) -> str:
    """Format portfolio state for the agent prompt."""
    parts = [
        f"Cash: {format_currency(portfolio.get('cash', 0))}",
        f"Total Value: {format_currency(portfolio.get('total_value', 0))}",
        f"P&L: {format_currency(portfolio.get('total_pnl', 0))} ({portfolio.get('total_pnl_pct', 0):+.2f}%)",
        f"Trades: {portfolio.get('total_trades', 0)} (W:{portfolio.get('winning_trades', 0)} L:{portfolio.get('losing_trades', 0)})",
    ]

    holdings = portfolio.get("holdings", {})
    for ticker, h in holdings.items():
        if h.get("shares", 0) > 0:
            parts.append(f"Holding: {h['shares']:.4f} {ticker} @ avg ${h['avg_cost_basis']:.2f} "
                        f"(current ${h['current_price']:.2f}, PnL {format_currency(h['unrealized_pnl'])})")
        else:
            parts.append(f"Holding: No {ticker} position")

    return "\n".join(parts)


def _format_knowledge(knowledge: dict) -> str:
    """Format knowledge bundle for the agent prompt."""
    parts = []

    lessons = knowledge.get("lessons", [])
    if lessons:
        parts.append("Recent Lessons:")
        for l in lessons[-5:]:
            parts.append(f"  [{l['id']}] {l.get('lesson', 'N/A')}")
            if l.get("category"):
                parts.append(f"    Category: {l['category']}")
    else:
        parts.append("No lessons yet — this is early in the learning process.")

    patterns = knowledge.get("patterns", [])
    if patterns:
        parts.append("")
        parts.append("Known Patterns:")
        for p in patterns:
            validated = p.get("times_validated", 0)
            contradicted = p.get("times_contradicted", 0)
            reliability = p.get("reliability", "N/A")
            if validated or contradicted:
                parts.append(f"  [{p['id']}] {p.get('name', 'N/A')} (reliability: {reliability}, validated {validated}x, contradicted {contradicted}x)")
            else:
                parts.append(f"  [{p['id']}] {p.get('name', 'N/A')} (reliability: {reliability}, untested)")
            parts.append(f"    {p.get('description', '')}")

    accuracy = knowledge.get("prediction_accuracy", {})
    if accuracy.get("scored_predictions", 0) > 0:
        parts.append("")
        parts.append("Your Prediction Accuracy:")
        for horizon, data in accuracy.get("direction_accuracy", {}).items():
            parts.append(f"  {horizon}: {data['accuracy_pct']}% ({data['correct']}/{data['total']})")

        # Per-strategy accuracy breakdown
        strat_acc = accuracy.get("strategy_accuracy", {})
        if strat_acc:
            parts.append("Accuracy by Strategy:")
            for strat, data in strat_acc.items():
                pct = data["accuracy_pct"]
                label = ""
                if pct < 45:
                    label = " — poor track record, low confidence warranted"
                elif pct >= 60:
                    label = " — your strongest predictor"
                parts.append(f"  {strat}: {pct}% ({data['correct']}/{data['total']}){label}")
    else:
        parts.append("")
        parts.append("No scored predictions yet.")

    scores = knowledge.get("strategy_scores", {})
    active = {k: v for k, v in scores.items() if v.get("total_trades", 0) > 0}
    if active:
        parts.append("")
        parts.append("Strategy Performance:")
        for name, s in active.items():
            parts.append(f"  {name}: {s['win_rate']*100:.0f}% win rate, {s['total_trades']} trades, weight={s['weight']:.2f}")

    return "\n".join(parts) if parts else "Knowledge base is empty — first trading session."


def _format_bsm_signals(bsm_signals: list[dict]) -> str:
    """Format BSM (Billionaire Signal Monitor) signals for the agent prompt."""
    if not bsm_signals:
        return ""
    parts = ["Active BSM Signals:"]
    for sig in bsm_signals:
        parts.append(
            f"  [{sig.get('source_person', 'unknown')}] {sig.get('ticker', '?')} "
            f"{sig.get('direction', '?')} (strength={sig.get('strength', 0):.2f}, "
            f"conf={sig.get('confidence', 0):.2f})"
        )
        if sig.get("summary"):
            parts.append(f"    {sig['summary'][:150]}")
        if sig.get("time_horizon"):
            parts.append(f"    Horizon: {sig['time_horizon']}")
    return "\n".join(parts)


def _fetch_news(ticker: str) -> str:
    """Fetch recent news headlines for sentiment context."""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return "No recent news available."
        headlines = []
        for item in news[:8]:
            content = item.get("content", {})
            title = content.get("title", item.get("title", ""))
            provider = content.get("provider", {}).get("displayName", "")
            if title:
                headlines.append(f"- {title}" + (f" ({provider})" if provider else ""))
        return "\n".join(headlines) if headlines else "No recent news available."
    except Exception as e:
        logger.warning(f"News fetch failed: {e}")
        return "News unavailable."


def build_hold_streak_warning(hold_streak: dict) -> str:
    """Build the <hold_streak_warning> prompt section from hold streak stats.

    Returns empty string if streak < 5 or hold_streak is None.
    Used by both single-step and multi-step decision paths.
    """
    if not hold_streak or hold_streak.get("consecutive_holds", 0) < 5:
        return ""
    hs = hold_streak
    section = "\n<hold_streak_warning>\n"
    section += f"ATTENTION: You have chosen HOLD for {hs['consecutive_holds']} consecutive decisions.\n"
    if hs.get("last_trade_hours_ago") is not None:
        section += f"Last trade was {hs['last_trade_hours_ago']:.0f} hours ago.\n"
    section += f"Your recent counterfactual scorecard (last 20 scored holds):\n"
    section += f"  Missed gains (should have traded): {hs['recent_missed_gains']} ({hs['missed_gain_pct']}%)\n"
    section += f"  Correct holds (right to wait): {hs['recent_correct_holds']}\n"
    if hs["missed_gain_pct"] > 55:
        section += "\nYour holds are WRONG more often than right. You are leaving money on the table. Lower your conviction threshold and act on signals with confidence >0.50.\n"
    if hs.get("last_trade_hours_ago") and hs["last_trade_hours_ago"] > 24:
        section += "\nYou have not traded in over 24 hours. This is excessive caution. Find a trade.\n"
    section += "</hold_streak_warning>\n"
    return section


def make_decision(
    market_data: dict,
    signals: list[StrategySignal],
    aggregate: StrategySignal,
    portfolio: dict,
    knowledge: dict = None,
    macro_gate: dict = None,
    regime: dict = None,
    agent_config: dict = None,
    hold_streak: dict = None,
) -> dict:
    """Call Claude to make a trading decision.

    Args:
        agent_config: Optional per-agent personality config (from ensemble).
            If provided, appends the agent's system_prompt_addon.

    Returns the parsed decision dict with action, shares, hypothesis, etc.
    """
    config = load_config()
    ticker = config["ticker"]

    if knowledge is None:
        knowledge = get_relevant_knowledge(market_data)

    # Fetch news for sentiment context
    news = _fetch_news(ticker)

    # Fetch BSM signals
    bsm_signals = get_bsm_signals()
    bsm_section = ""
    if bsm_signals:
        bsm_section = f"\n<bsm_signals>\n{_format_bsm_signals(bsm_signals)}\n\nNOTE: If BSM signals are bearish but your technical analysis is bullish, cap position to 10% of portfolio value.\n</bsm_signals>\n"

    # Build macro/regime section
    macro_section = ""
    if regime or macro_gate:
        r = regime or market_data.get("regime", {})
        mg = macro_gate or {}
        macro_section = f"""
<macro_regime>
Trend: {r.get('trend', 'unknown')} | Volatility: {r.get('volatility', 'unknown')} | VIX: {r.get('vix', 0):.1f}
SPY Daily Change: {mg.get('spy_change_pct', 0)*100:.2f}%
Macro Gate: {'ACTIVE — ' + mg.get('reason', '') if mg.get('gate_active') else 'Normal'}
{('Required confidence for BUY: ' + str(mg.get('confidence_threshold_override', 0.80))) if mg.get('gate_active') else ''}
</macro_regime>
"""

    # Build hold streak warning
    hold_streak_section = build_hold_streak_warning(hold_streak)

    # Build the user prompt
    max_trade = portfolio.get('total_value', 1000) * config["risk_params"]["max_single_trade_pct"]
    user_prompt = f"""<market_data>
{_format_market_data(market_data)}
</market_data>
{macro_section}
<strategy_signals>
{_format_signals(signals, aggregate)}
</strategy_signals>

<portfolio>
{_format_portfolio(portfolio)}
</portfolio>

<relevant_knowledge>
{_format_knowledge(knowledge)}
</relevant_knowledge>

<recent_news>
{news}
</recent_news>
{bsm_section}
{hold_streak_section}Analyze all data and decide: BUY, SELL, or HOLD.
If BUY or SELL, specify how many shares (fractional OK, max trade = 20% of portfolio value = {format_currency(max_trade)}).
Current price: ${market_data.get('current', {}).get('price', 'N/A')}.
Respond with JSON only."""

    # Build system prompt (with optional agent personality addon)
    system_prompt = SYSTEM_PROMPT
    if agent_config and agent_config.get("system_prompt_addon"):
        agent_name = agent_config.get("display_name", agent_config.get("name", ""))
        system_prompt += f"\n\nAGENT IDENTITY: {agent_name}\n{agent_config['system_prompt_addon']}"

    logger.info(f"Calling Claude for decision (prompt ~{len(user_prompt)} chars)")

    try:
        raw, model_used = call_ai_with_fallback(
            system=system_prompt,
            user=user_prompt,
            max_tokens=1500,
            config=config,
        )

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        decision = json.loads(raw)

        logger.info(
            f"Decision: {decision.get('action')} "
            f"{decision.get('shares', 0):.4f} shares, "
            f"confidence={decision.get('confidence', 0):.2f}, "
            f"strategy={decision.get('strategy', 'N/A')}"
        )
        logger.info(f"Hypothesis: {decision.get('hypothesis', 'N/A')}")

        # Record predictions
        predictions = decision.get("predictions", {})
        if predictions and any(predictions.values()):
            pred_record = {
                "timestamp": iso_now(),
                "price_at_prediction": market_data.get("current", {}).get("price", 0),
                "predictions": predictions,
                "reasoning": decision.get("hypothesis", ""),
                "outcomes": {k: None for k in predictions},
                "linked_trade": None,
                "model_version": config["anthropic_model"],
            }
            saved = add_prediction(pred_record)
            decision["_prediction_id"] = saved["id"]

        # Validate HOLD decisions include hold_analysis
        if decision.get("action") == "HOLD":
            hold_analysis = decision.get("hold_analysis")
            if not hold_analysis or not isinstance(hold_analysis, dict):
                logger.warning("HOLD decision missing hold_analysis — adding placeholder")
                decision["hold_analysis"] = {
                    "opportunity_cost": decision.get("reasoning", "Not specified"),
                    "downside_protection": decision.get("risk_note", "Not specified"),
                    "signal_balance": "Not analyzed",
                    "decision_boundary": "Not specified",
                }

        # Pass through sentiment score for strategy tracking
        decision["_sentiment_score"] = decision.get("sentiment_score")
        decision["_model_version"] = config["anthropic_model"]

        # BSM conviction cap: if BSM bearish + agent bullish, cap at 10%
        if bsm_signals and decision.get("action") == "BUY":
            bearish_bsm = [s for s in bsm_signals if s.get("direction") == "bearish"]
            if bearish_bsm:
                current_price = market_data.get("current", {}).get("price", 0)
                bsm_cap_pct = config.get("risk_params", {}).get("bsm_conviction_cap_pct", 0.10)
                bsm_max_value = portfolio.get("total_value", 1000) * bsm_cap_pct
                bsm_max_shares = bsm_max_value / current_price if current_price > 0 else 0
                if decision.get("shares", 0) > bsm_max_shares:
                    logger.info(
                        f"BSM cap: bearish signal conflicts with BUY — "
                        f"capping {decision['shares']:.4f} to {bsm_max_shares:.4f} shares"
                    )
                    decision["shares"] = round(bsm_max_shares, 4)
                    decision["risk_note"] = (
                        decision.get("risk_note", "") +
                        f" [BSM conviction cap applied: {bsm_cap_pct*100:.0f}%]"
                    )

        return decision

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response: {e}\nRaw: {raw[:500]}")
        return {
            "action": "HOLD",
            "shares": 0,
            "confidence": 0,
            "strategy": "error",
            "hypothesis": "Failed to parse AI response",
            "reasoning": f"JSON parse error: {e}",
            "knowledge_applied": [],
            "risk_note": "Defaulting to HOLD due to parse error",
        }
    except Exception as e:
        logger.error(f"Agent decision failed: {e}")
        return {
            "action": "HOLD",
            "shares": 0,
            "confidence": 0,
            "strategy": "error",
            "hypothesis": "Agent error — defaulting to HOLD",
            "reasoning": str(e),
            "knowledge_applied": [],
            "risk_note": f"Error: {e}",
        }


# ─── Multi-Step Reasoning ──────────────────────────────────────────────

ANALYSIS_SYSTEM = """You are the Analysis module of MonopolyTrader. Your job is ONLY to analyze the current market state objectively. Do NOT recommend any action.

Examine the data and produce a structured analysis:
1. TREND: What is the current price trend? (bullish/bearish/sideways)
2. MOMENTUM: Is momentum building or fading? Cite specific indicators.
3. VOLATILITY: How volatile is the current environment? What does ATR say?
4. SUPPORT/RESISTANCE: Where are the key levels relative to current price?
5. DIVERGENCES: Any divergences between indicators? (e.g., price up but RSI down)
6. REGIME: What market regime are we in? How does this affect strategy reliability?
7. NEWS CONTEXT: What's the news saying? Is it likely priced in already?

Respond with JSON:
{
  "trend": "bullish"|"bearish"|"sideways",
  "trend_strength": <float 0-1>,
  "momentum": "building"|"fading"|"neutral",
  "volatility": "low"|"normal"|"high",
  "key_levels": {"support": <price>, "resistance": <price>},
  "divergences": ["<list of any divergences found>"],
  "regime_assessment": "<brief regime description>",
  "news_impact": "positive"|"negative"|"neutral"|"priced_in",
  "notable_observations": ["<list of anything unusual or important>"]
}"""

STRATEGY_SYSTEM = """You are the Strategy module of MonopolyTrader. Given an objective market analysis and the current strategy signals, your job is to determine the optimal action.

Consider:
1. Do the strategy signals agree with the analysis? If not, which do you trust more?
2. What is the signal balance telling you? Strong one-sided signals are more convincing.
3. Is this a high-conviction setup or are you gambling? Be honest.
4. If HOLD: what is the opportunity cost? What would flip your decision?
5. What does the knowledge base say about similar setups?

Respond with JSON:
{
  "recommended_action": "BUY"|"SELL"|"HOLD",
  "conviction": <float 0-1>,
  "primary_strategy": "<which strategy is driving this>",
  "supporting_strategies": ["<strategies that agree>"],
  "opposing_strategies": ["<strategies that disagree>"],
  "key_reasoning": "<2-3 sentence core argument>",
  "risk_factors": ["<list of risks>"],
  "hold_justification": <null if not HOLD, otherwise {
    "opportunity_cost": "<what you're giving up>",
    "downside_protection": "<what you're avoiding>",
    "decision_boundary": "<what would change your mind>"
  }>
}"""

EXECUTION_SYSTEM = """You are the Execution module of MonopolyTrader. Given a strategy recommendation, market data, and portfolio state, produce the final trade decision with precise sizing and risk management.

Your job:
1. Validate the strategy recommendation against risk rules
2. Calculate appropriate position size (conservative, respect ATR sizing)
3. Set predictions with honest confidence levels
4. Produce the final JSON decision

RISK RULES (v3 — you MUST follow these):
- Max position: 65% of portfolio value
- Max single trade: 20% of portfolio value
- Min cash reserve: 15% of portfolio value
- Position sizing: inverse ATR (wider stop = smaller position, max 2% risk per trade)
- If daily loss exceeds 8%, force HOLD

Respond with the final decision JSON matching the standard schema."""


def make_decision_multi_step(
    market_data: dict,
    signals: list[StrategySignal],
    aggregate: StrategySignal,
    portfolio: dict,
    knowledge: dict = None,
    macro_gate: dict = None,
    regime: dict = None,
    agent_config: dict = None,
    hold_streak: dict = None,
) -> dict:
    """Three-step reasoning: Analysis → Strategy → Execution.

    Uses separate prompts for each step so reasoning is structured and
    each step can be independently audited. Falls back to single-step
    make_decision() on error.
    """
    config = load_config()
    ticker = config["ticker"]

    if knowledge is None:
        knowledge = get_relevant_knowledge(market_data)

    news = _fetch_news(ticker)
    bsm_signals = get_bsm_signals()

    # --- Step 1: Analysis ---
    analysis_prompt = f"""<market_data>
{_format_market_data(market_data)}
</market_data>

<recent_news>
{news}
</recent_news>

<regime>
Trend: {(regime or {}).get('trend', 'unknown')} | Volatility: {(regime or {}).get('volatility', 'unknown')} | VIX: {(regime or {}).get('vix', 0):.1f}
</regime>

Analyze the current market state for {ticker}. Be objective — no action recommendation."""

    try:
        logger.info("Multi-step: running Analysis phase...")
        raw1, _ = call_ai_with_fallback(
            system=ANALYSIS_SYSTEM,
            user=analysis_prompt,
            max_tokens=800,
            config=config,
        )
        if raw1.startswith("```"):
            raw1 = raw1.split("\n", 1)[1] if "\n" in raw1 else raw1[3:]
            if raw1.endswith("```"):
                raw1 = raw1[:-3]
            raw1 = raw1.strip()
        analysis = json.loads(raw1)
        logger.info(f"Analysis: trend={analysis.get('trend')}, momentum={analysis.get('momentum')}, vol={analysis.get('volatility')}")
    except Exception as e:
        logger.warning(f"Multi-step Analysis failed: {e}. Falling back to single-step.")
        return make_decision(market_data, signals, aggregate, portfolio, knowledge, macro_gate, regime, agent_config, hold_streak)

    # --- Step 2: Strategy ---
    ms_hold_streak_section = build_hold_streak_warning(hold_streak)

    strategy_prompt = f"""<analysis>
{json.dumps(analysis, indent=2)}
</analysis>

<strategy_signals>
{_format_signals(signals, aggregate)}
</strategy_signals>

<relevant_knowledge>
{_format_knowledge(knowledge)}
</relevant_knowledge>

<portfolio_summary>
{_format_portfolio(portfolio)}
</portfolio_summary>
{ms_hold_streak_section}
Based on the analysis and signals, what action should we take? Justify thoroughly."""

    try:
        logger.info("Multi-step: running Strategy phase...")
        raw2, _ = call_ai_with_fallback(
            system=STRATEGY_SYSTEM,
            user=strategy_prompt,
            max_tokens=800,
            config=config,
        )
        if raw2.startswith("```"):
            raw2 = raw2.split("\n", 1)[1] if "\n" in raw2 else raw2[3:]
            if raw2.endswith("```"):
                raw2 = raw2[:-3]
            raw2 = raw2.strip()
        strategy_rec = json.loads(raw2)
        logger.info(f"Strategy: {strategy_rec.get('recommended_action')} conviction={strategy_rec.get('conviction', 0):.2f}")
    except Exception as e:
        logger.warning(f"Multi-step Strategy failed: {e}. Falling back to single-step.")
        return make_decision(market_data, signals, aggregate, portfolio, knowledge, macro_gate, regime, agent_config, hold_streak)

    # --- Step 3: Execution ---
    max_trade = portfolio.get('total_value', 1000) * config["risk_params"]["max_single_trade_pct"]
    bsm_section = ""
    if bsm_signals:
        bsm_section = f"\n<bsm_signals>\n{_format_bsm_signals(bsm_signals)}\n</bsm_signals>\n"

    execution_prompt = f"""<analysis>
{json.dumps(analysis, indent=2)}
</analysis>

<strategy_recommendation>
{json.dumps(strategy_rec, indent=2)}
</strategy_recommendation>

<portfolio>
{_format_portfolio(portfolio)}
</portfolio>

<market_data>
Current price: ${market_data.get('current', {}).get('price', 'N/A')}
ATR: {market_data.get('daily_indicators', {}).get('atr', 'N/A')}
</market_data>
{bsm_section}
Max trade size: {format_currency(max_trade)}. Current price: ${market_data.get('current', {}).get('price', 'N/A')}.

Produce the final trade decision JSON. Include hold_analysis if action is HOLD.
If BUY or SELL, specify exact shares (fractional OK). Be conservative on sizing."""

    # Build execution system prompt with agent addon
    exec_system = EXECUTION_SYSTEM + "\n\n" + SYSTEM_PROMPT.split("Respond ONLY")[0].strip()
    if agent_config and agent_config.get("system_prompt_addon"):
        agent_name = agent_config.get("display_name", agent_config.get("name", ""))
        exec_system += f"\n\nAGENT IDENTITY: {agent_name}\n{agent_config['system_prompt_addon']}"

    try:
        logger.info("Multi-step: running Execution phase...")
        raw3, _ = call_ai_with_fallback(
            system=exec_system,
            user=execution_prompt,
            max_tokens=1500,
            config=config,
        )
        if raw3.startswith("```"):
            raw3 = raw3.split("\n", 1)[1] if "\n" in raw3 else raw3[3:]
            if raw3.endswith("```"):
                raw3 = raw3[:-3]
            raw3 = raw3.strip()
        decision = json.loads(raw3)
    except Exception as e:
        logger.warning(f"Multi-step Execution failed: {e}. Falling back to single-step.")
        return make_decision(market_data, signals, aggregate, portfolio, knowledge, macro_gate, regime, agent_config, hold_streak)

    # Attach multi-step trace metadata
    decision["_multi_step"] = {
        "analysis": analysis,
        "strategy_recommendation": strategy_rec,
    }

    logger.info(
        f"Multi-step Decision: {decision.get('action')} "
        f"{decision.get('shares', 0):.4f} shares, "
        f"confidence={decision.get('confidence', 0):.2f}"
    )

    # Record predictions
    predictions = decision.get("predictions", {})
    if predictions and any(predictions.values()):
        pred_record = {
            "timestamp": iso_now(),
            "price_at_prediction": market_data.get("current", {}).get("price", 0),
            "predictions": predictions,
            "reasoning": decision.get("hypothesis", ""),
            "outcomes": {k: None for k in predictions},
            "linked_trade": None,
            "model_version": config["anthropic_model"],
        }
        saved = add_prediction(pred_record)
        decision["_prediction_id"] = saved["id"]

    # Validate HOLD decisions
    if decision.get("action") == "HOLD":
        hold_analysis = decision.get("hold_analysis")
        if not hold_analysis or not isinstance(hold_analysis, dict):
            # Pull from strategy phase
            hold_just = strategy_rec.get("hold_justification", {})
            decision["hold_analysis"] = {
                "opportunity_cost": (hold_just or {}).get("opportunity_cost", decision.get("reasoning", "Not specified")),
                "downside_protection": (hold_just or {}).get("downside_protection", decision.get("risk_note", "Not specified")),
                "signal_balance": f"Balance: {aggregate.confidence:.2f} toward {aggregate.action}",
                "decision_boundary": (hold_just or {}).get("decision_boundary", "Not specified"),
            }

    decision["_sentiment_score"] = decision.get("sentiment_score")
    decision["_model_version"] = config["anthropic_model"]

    # BSM conviction cap
    if bsm_signals and decision.get("action") == "BUY":
        bearish_bsm = [s for s in bsm_signals if s.get("direction") == "bearish"]
        if bearish_bsm:
            current_price = market_data.get("current", {}).get("price", 0)
            bsm_cap_pct = config.get("risk_params", {}).get("bsm_conviction_cap_pct", 0.10)
            bsm_max_value = portfolio.get("total_value", 1000) * bsm_cap_pct
            bsm_max_shares = bsm_max_value / current_price if current_price > 0 else 0
            if decision.get("shares", 0) > bsm_max_shares:
                decision["shares"] = round(bsm_max_shares, 4)
                decision["risk_note"] = (
                    decision.get("risk_note", "") +
                    f" [BSM conviction cap applied: {bsm_cap_pct*100:.0f}%]"
                )

    return decision


# ─── v2: Thesis-Driven Two-Phase Engine ──────────────────────────────────────

ANALYST_SYSTEM = """You are the ANALYST module of MonopolyTrader v2. Your job is to understand WHY TSLA is moving and maintain a running thesis about the stock.

You focus on CAUSATION, not prediction. Your output is a thesis — a narrative about what is driving TSLA's price and what would change it.

You receive:
- Current thesis (your previous assessment)
- News feed (classified, scored)
- Research findings (earnings, catalysts, correlations, seasonal patterns, sector context)
- Price action and technical data
- Macro regime

Your job:
1. What NEWS or CATALYST is driving TSLA right now?
2. Has the current thesis been confirmed, challenged, or invalidated?
3. What are the key price levels (support/resistance) and what would break them?
4. What is the bull case and bear case?
5. What would invalidate your thesis?

You do NOT recommend trades. You build the thesis that the Trader module uses.

Respond with JSON:
{
  "narrative": "<2-3 sentence summary of what's driving TSLA right now>",
  "direction": "bearish" | "bullish" | "neutral",
  "conviction": <float 0.0-1.0>,
  "key_levels": {"support": [<prices>], "resistance": [<prices>]},
  "bull_case": "<what would make TSLA go up>",
  "bear_case": "<what would make TSLA go down>",
  "invalidation": "<what specific event or price level would prove this thesis wrong>",
  "key_catalysts": ["<list of catalysts currently affecting TSLA>"],
  "thesis_change_reason": "<why you changed (or didn't change) the thesis>",
  "news_citations": ["<specific news items that informed this update>"]
}"""


TRADER_SYSTEM = """You are the TRADER module of MonopolyTrader v2. You receive a thesis about WHY TSLA is moving and use technical indicators to decide WHEN to act.

The thesis determines DIRECTION. Technicals determine TIMING and ENTRY.

Rules:
- If thesis is bearish, do NOT buy. Look for sell signals or hold.
- If thesis is bullish, look for technical confirmation to buy (dip to support, RSI oversold, MACD crossover).
- If thesis is neutral, only act on strong technical signals.
- The thesis conviction weights your confidence. High thesis conviction + technical confirmation = high-confidence trade.
- Low thesis conviction = smaller positions or HOLD.

RISK RULES (v3):
- Max position: 65% of portfolio value
- Max single trade: 20% of portfolio value
- Min cash reserve: 15% of portfolio value
- Position sizing: inverse ATR (wider stop = smaller position, max 2% risk per trade)
- Cooldown: 15 min between trades
- If daily loss exceeds 8%, stop trading
- Macro gate: if SPY down >2% or VIX >30, require confidence >0.80 for BUY

SIGNAL INTERPRETATION: Strategies that return HOLD are ABSTAINING — not disagreeing. A single strategy at >0.70 confidence with no opposing signals is a clear trade.

ANTI-PARALYSIS: If a <hold_streak_warning> appears, lower your threshold and act on signals with confidence >0.50.

Respond ONLY with valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": <float, 0 if HOLD>,
  "confidence": <float 0-1>,
  "strategy": "<primary strategy driving this decision>",
  "hypothesis": "<I predict TSLA will [direction] by [amount] over [timeframe] because [thesis + technical reasoning]>",
  "reasoning": "<how thesis + technicals combine to justify this decision>",
  "knowledge_applied": ["<lesson_id>"],
  "risk_note": "<any risk concerns>",
  "predictions": {
    "1day": {"direction": "up"|"down"|"flat", "target": <price>, "confidence": <0-1>},
    "1week": {"direction": "up"|"down"|"flat", "target": <price>, "confidence": <0-1>}
  },
  "thesis_alignment": "<how this trade aligns with or contradicts the current thesis>",
  "hold_analysis": <REQUIRED if action is HOLD, null otherwise> {
    "opportunity_cost": "<what gain are you forgoing?>",
    "downside_protection": "<what risk are you avoiding?>",
    "signal_balance": "<buy vs sell signal balance>",
    "decision_boundary": "<what would flip this to BUY or SELL?>"
  }
}"""


def _strip_code_fences(raw: str) -> str:
    """Strip markdown code fences from Claude response."""
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
    return raw.strip()


def run_analyst(
    current_thesis: Thesis,
    news_feed: NewsFeed,
    market_data: dict,
    regime: dict = None,
    trigger_reason: str = "",
) -> Thesis:
    """Phase 1: Update the market thesis based on news, research, and price action.

    Does NOT recommend trades. Returns an updated Thesis object.
    """
    config = load_config()
    ticker = config["ticker"]

    # Get research findings
    research_text = get_relevant_research()

    # Format news
    news_text = format_news_for_prompt(news_feed)

    # Format regime
    r = regime or market_data.get("regime", {})
    regime_text = (
        f"Trend: {r.get('trend', 'unknown')} | "
        f"Volatility: {r.get('volatility', 'unknown')} | "
        f"VIX: {r.get('vix', 0):.1f}"
    )

    user_prompt = f"""<current_thesis>
{current_thesis.format_for_prompt()}
</current_thesis>

<trigger>
Thesis update triggered by: {trigger_reason or 'scheduled check'}
</trigger>

<news_feed>
{news_text}
</news_feed>

<research_findings>
{research_text}
</research_findings>

<market_data>
{_format_market_data(market_data)}
</market_data>

<macro_regime>
{regime_text}
</macro_regime>

Update the thesis for {ticker}. What is driving the stock right now? Has anything changed?
If the thesis is still valid, say so and explain why. If it needs updating, explain what changed."""

    try:
        logger.info(f"Analyst: updating thesis (trigger: {trigger_reason})")
        raw, model_used = call_ai_with_fallback(
            system=ANALYST_SYSTEM,
            user=user_prompt,
            max_tokens=1200,
            config=config,
        )
        raw = _strip_code_fences(raw)
        result = json.loads(raw)

        # Build updated thesis
        new_thesis = Thesis(
            narrative=result.get("narrative", current_thesis.narrative),
            direction=result.get("direction", current_thesis.direction),
            conviction=result.get("conviction", current_thesis.conviction),
            key_levels=result.get("key_levels", current_thesis.key_levels),
            bull_case=result.get("bull_case", current_thesis.bull_case),
            bear_case=result.get("bear_case", current_thesis.bear_case),
            invalidation=result.get("invalidation", current_thesis.invalidation),
            key_catalysts=result.get("key_catalysts", current_thesis.key_catalysts),
            updated_by="analyst",
        )

        # Save with reason
        change_reason = result.get("thesis_change_reason", trigger_reason)
        save_thesis(new_thesis, reason=change_reason)

        logger.info(
            f"Analyst: thesis updated — {new_thesis.direction} "
            f"conviction={new_thesis.conviction:.2f}, "
            f"reason={change_reason[:80]}"
        )

        return new_thesis

    except json.JSONDecodeError as e:
        logger.error(f"Analyst: failed to parse response: {e}")
        return current_thesis
    except Exception as e:
        logger.error(f"Analyst: failed: {e}")
        return current_thesis


def run_trader(
    thesis: Thesis,
    market_data: dict,
    signals: list[StrategySignal],
    aggregate: StrategySignal,
    portfolio: dict,
    knowledge: dict = None,
    macro_gate: dict = None,
    regime: dict = None,
    hold_streak: dict = None,
) -> dict:
    """Phase 2: Make a trade decision based on thesis + technical timing.

    The thesis determines direction. Technicals determine timing.
    Returns a decision dict compatible with the v1 format.
    """
    config = load_config()
    ticker = config["ticker"]

    if knowledge is None:
        knowledge = get_relevant_knowledge(market_data)

    # BSM signals
    bsm_signals = get_bsm_signals()
    bsm_section = ""
    if bsm_signals:
        bsm_section = f"\n<bsm_signals>\n{_format_bsm_signals(bsm_signals)}\n</bsm_signals>\n"

    # Macro section
    macro_section = ""
    if regime or macro_gate:
        r = regime or market_data.get("regime", {})
        mg = macro_gate or {}
        macro_section = f"""
<macro_regime>
Trend: {r.get('trend', 'unknown')} | Volatility: {r.get('volatility', 'unknown')} | VIX: {r.get('vix', 0):.1f}
SPY Daily Change: {mg.get('spy_change_pct', 0)*100:.2f}%
Macro Gate: {'ACTIVE — ' + mg.get('reason', '') if mg.get('gate_active') else 'Normal'}
</macro_regime>
"""

    # Hold streak warning
    hold_streak_section = build_hold_streak_warning(hold_streak)

    max_trade = portfolio.get('total_value', 1000) * config["risk_params"]["max_single_trade_pct"]

    user_prompt = f"""<thesis>
{thesis.format_for_prompt()}
</thesis>

<market_data>
{_format_market_data(market_data)}
</market_data>
{macro_section}
<strategy_signals>
{_format_signals(signals, aggregate)}
</strategy_signals>

<portfolio>
{_format_portfolio(portfolio)}
</portfolio>

<relevant_knowledge>
{_format_knowledge(knowledge)}
</relevant_knowledge>
{bsm_section}
{hold_streak_section}The thesis says {thesis.direction.upper()} with conviction {thesis.conviction:.2f}.
Use technical signals to decide IF and WHEN to act on this thesis.
Max trade = 20% of portfolio value = {format_currency(max_trade)}.
Current price: ${market_data.get('current', {}).get('price', 'N/A')}.
Respond with JSON only."""

    try:
        logger.info(f"Trader: making decision (thesis={thesis.direction}, conviction={thesis.conviction:.2f})")
        raw, model_used = call_ai_with_fallback(
            system=TRADER_SYSTEM,
            user=user_prompt,
            max_tokens=1500,
            config=config,
        )
        raw = _strip_code_fences(raw)
        decision = json.loads(raw)

        logger.info(
            f"Trader Decision: {decision.get('action')} "
            f"{decision.get('shares', 0):.4f} shares, "
            f"confidence={decision.get('confidence', 0):.2f}, "
            f"thesis_alignment={decision.get('thesis_alignment', 'N/A')[:60]}"
        )

        # Record predictions (v2 horizons: 1day, 1week)
        predictions = decision.get("predictions", {})
        if predictions and any(predictions.values()):
            pred_record = {
                "timestamp": iso_now(),
                "price_at_prediction": market_data.get("current", {}).get("price", 0),
                "predictions": predictions,
                "reasoning": decision.get("hypothesis", ""),
                "outcomes": {k: None for k in predictions},
                "linked_trade": None,
                "model_version": config["anthropic_model"],
            }
            saved = add_prediction(pred_record)
            decision["_prediction_id"] = saved["id"]

        # Validate HOLD analysis
        if decision.get("action") == "HOLD":
            hold_analysis = decision.get("hold_analysis")
            if not hold_analysis or not isinstance(hold_analysis, dict):
                decision["hold_analysis"] = {
                    "opportunity_cost": decision.get("reasoning", "Not specified"),
                    "downside_protection": decision.get("risk_note", "Not specified"),
                    "signal_balance": f"Thesis: {thesis.direction} conviction={thesis.conviction:.2f}",
                    "decision_boundary": decision.get("thesis_alignment", "Not specified"),
                }

        # Attach metadata
        decision["_sentiment_score"] = decision.get("sentiment_score")
        decision["_model_version"] = config["anthropic_model"]
        decision["_thesis_version"] = thesis.version
        decision["_thesis_direction"] = thesis.direction
        decision["_thesis_conviction"] = thesis.conviction

        # BSM conviction cap
        if bsm_signals and decision.get("action") == "BUY":
            bearish_bsm = [s for s in bsm_signals if s.get("direction") == "bearish"]
            if bearish_bsm:
                current_price = market_data.get("current", {}).get("price", 0)
                bsm_cap_pct = config.get("risk_params", {}).get("bsm_conviction_cap_pct", 0.10)
                bsm_max_value = portfolio.get("total_value", 1000) * bsm_cap_pct
                bsm_max_shares = bsm_max_value / current_price if current_price > 0 else 0
                if decision.get("shares", 0) > bsm_max_shares:
                    decision["shares"] = round(bsm_max_shares, 4)
                    decision["risk_note"] = (
                        decision.get("risk_note", "") +
                        f" [BSM conviction cap applied: {bsm_cap_pct*100:.0f}%]"
                    )

        return decision

    except json.JSONDecodeError as e:
        logger.error(f"Trader: failed to parse response: {e}")
        return _error_hold("Trader JSON parse error")
    except Exception as e:
        logger.error(f"Trader: failed: {e}")
        return _error_hold(f"Trader error: {e}")


def _error_hold(reason: str) -> dict:
    """Return a safe HOLD decision on error."""
    return {
        "action": "HOLD",
        "shares": 0,
        "confidence": 0,
        "strategy": "error",
        "hypothesis": f"Error — defaulting to HOLD: {reason}",
        "reasoning": reason,
        "knowledge_applied": [],
        "risk_note": f"Error: {reason}",
        "predictions": {},
    }
