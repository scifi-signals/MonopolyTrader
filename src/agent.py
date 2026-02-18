"""AI decision engine — Claude-powered trading brain."""

import json
import os
import yfinance as yf
from anthropic import Anthropic
from .utils import load_config, format_currency, iso_now, setup_logging
from .knowledge_base import get_relevant_knowledge, add_prediction
from .strategies import StrategySignal

logger = setup_logging("agent")

SYSTEM_PROMPT = """You are MonopolyTrader, an AI trading agent managing a virtual portfolio of Monopoly dollars. You are LEARNING — you started with $1,000 and your goal is to grow it by getting better at predicting TSLA's movements over time.

You have a knowledge base of lessons from past trades, patterns you've discovered, and research you've conducted. USE THIS KNOWLEDGE. Reference specific lessons and patterns when making decisions. If you're uncertain, say so — it's better to HOLD than to make a low-conviction trade.

Every trade is a hypothesis. State clearly what you predict will happen and why. You will be reviewed on this prediction later.

RISK RULES (you MUST follow these):
- Max position: 90% of portfolio value
- Max single trade: 25% of portfolio value
- Min cash reserve: 10% of portfolio value
- Stop loss: auto-sell at 5% loss (handled automatically)
- Cooldown: 15 min between trades
- If daily loss exceeds 8%, stop trading

When deciding trade size, be conservative early on. Small positions let you learn without blowing up. As your accuracy improves, you can size up.

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
    """Format strategy signals for the agent prompt."""
    parts = ["Individual Signals:"]
    for s in signals:
        parts.append(f"  [{s.strategy}] {s.action} confidence={s.confidence:.2f} weight={s.weight:.2f}")
        parts.append(f"    {s.reasoning}")

    parts.append("")
    parts.append(f"Aggregate: {aggregate.action} confidence={aggregate.confidence:.2f}")
    parts.append(f"  {aggregate.reasoning}")
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
            parts.append(f"  [{p['id']}] {p.get('name', 'N/A')} (reliability: {p.get('reliability', 'N/A')})")
            parts.append(f"    {p.get('description', '')}")

    accuracy = knowledge.get("prediction_accuracy", {})
    if accuracy.get("scored_predictions", 0) > 0:
        parts.append("")
        parts.append("Your Prediction Accuracy:")
        for horizon, data in accuracy.get("direction_accuracy", {}).items():
            parts.append(f"  {horizon}: {data['accuracy_pct']}% ({data['correct']}/{data['total']})")
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
) -> dict:
    """Call Claude to make a trading decision.

    Returns the parsed decision dict with action, shares, hypothesis, etc.
    """
    config = load_config()
    ticker = config["ticker"]

    if knowledge is None:
        knowledge = get_relevant_knowledge(market_data)

    # Fetch news for sentiment context
    news = _fetch_news(ticker)

    # Build the user prompt
    user_prompt = f"""<market_data>
{_format_market_data(market_data)}
</market_data>

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

Analyze all data and decide: BUY, SELL, or HOLD.
If BUY or SELL, specify how many shares (fractional OK, max trade = 25% of portfolio value = {format_currency(portfolio.get('total_value', 1000) * 0.25)}).
Current price: ${market_data.get('current', {}).get('price', 'N/A')}.
Respond with JSON only."""

    logger.info(f"Calling Claude for decision (prompt ~{len(user_prompt)} chars)")

    try:
        client = _get_client()
        response = client.messages.create(
            model=config["anthropic_model"],
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = response.content[0].text.strip()
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
            }
            saved = add_prediction(pred_record)
            decision["_prediction_id"] = saved["id"]

        # Pass through sentiment score for strategy tracking
        decision["_sentiment_score"] = decision.get("sentiment_score")

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
