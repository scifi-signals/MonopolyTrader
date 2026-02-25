"""AI decision engine — Claude-powered trading brain."""

import json
import os
import yfinance as yf
from anthropic import Anthropic
from .utils import load_config, format_currency, iso_now, setup_logging, call_ai_with_fallback
from .knowledge_base import get_relevant_knowledge, add_prediction
from .market_data import get_bsm_signals
from .strategies import StrategySignal, calculate_signal_balance

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

Respond ONLY with valid JSON matching this schema:
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": <float, 0 if HOLD>,
  "confidence": <float 0-1>,
  "strategy": "<primary strategy driving this decision>",
  "hypothesis": "<I predict TSLA will [direction] by [amount] over [timeframe] because [reasoning]>",
  "reasoning": "<detailed analysis referencing indicators, signals, and knowledge>",
  "knowledge_applied": ["<lesson_id or pattern_id>"],
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


def make_decision(
    market_data: dict,
    signals: list[StrategySignal],
    aggregate: StrategySignal,
    portfolio: dict,
    knowledge: dict = None,
    macro_gate: dict = None,
    regime: dict = None,
    agent_config: dict = None,
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
Analyze all data and decide: BUY, SELL, or HOLD.
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
        return make_decision(market_data, signals, aggregate, portfolio, knowledge, macro_gate, regime, agent_config)

    # --- Step 2: Strategy ---
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
        return make_decision(market_data, signals, aggregate, portfolio, knowledge, macro_gate, regime, agent_config)

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
        return make_decision(market_data, signals, aggregate, portfolio, knowledge, macro_gate, regime, agent_config)

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
