"""Learning engine — post-trade reviews, skeptic layer, lesson decay, pattern discovery."""

import json
import os
from anthropic import Anthropic
from .utils import load_config, iso_now, setup_logging
from .knowledge_base import (
    get_lessons, add_lesson, get_patterns, add_pattern,
    get_predictions, update_prediction, get_strategy_scores,
    update_strategy_scores, append_journal, get_knowledge_summary,
    get_prediction_accuracy,
)
from .portfolio import load_transactions, save_transactions

logger = setup_logging("learner")

# Skeptic uses a cheap model separate from the trading model
SKEPTIC_MODEL = "claude-haiku-4-5-20251001"

# v3 structured lesson categories
LESSON_CATEGORIES = [
    "signal_correct", "signal_early", "signal_late", "signal_wrong",
    "risk_sizing_error", "regime_mismatch", "external_shock",
    "stop_loss_whipsaw", "correlated_market_move", "noise_trade",
]


def _get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        for path in ["anthropic_api_key.txt", "../anthropic_api_key.txt"]:
            try:
                with open(path) as f:
                    api_key = f.read().strip()
                    break
            except FileNotFoundError:
                continue
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found")
    return Anthropic(api_key=api_key)


def _call_claude(system: str, user: str, max_tokens: int = 1500) -> tuple[str, str]:
    """Call Claude and return (response_text, model_version)."""
    config = load_config()
    client = _get_client()
    model = config["anthropic_model"]
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text.strip(), model


def _parse_json(raw: str) -> dict | list:
    """Parse JSON from Claude response, stripping markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


# --- Post-Trade Review ---

REVIEW_SYSTEM = """You are the learning module of MonopolyTrader, an AI trading agent. Your job is to analyze completed trades, compare the agent's hypothesis to what actually happened, and extract a specific, actionable lesson.

Be brutally honest. If the trade was based on flawed reasoning, say so. If it was good reasoning but bad luck, distinguish that. The goal is to build a knowledge base that makes the agent smarter over time.

IMPORTANT: Categorize every lesson into EXACTLY ONE of these categories:
- signal_correct: signal was right and trade worked as predicted
- signal_early: right direction but wrong timing — signal fired too soon
- signal_late: right direction but entered too late
- signal_wrong: signal was simply incorrect
- risk_sizing_error: direction was right but position size was wrong
- regime_mismatch: signal would have worked in a different market regime
- external_shock: unpredictable external event overwhelmed the signal
- stop_loss_whipsaw: stop triggered but price recovered
- correlated_market_move: TSLA moved with broader market, not on its own signal
- noise_trade: no real signal — low-conviction trade that shouldn't have been taken

Respond with JSON:
{
  "what_i_predicted": "<summary of the hypothesis>",
  "what_actually_happened": "<what the price actually did>",
  "was_correct": <true/false>,
  "why_right_or_wrong": "<specific analysis>",
  "lesson": "<one clear, actionable lesson>",
  "category": "<exactly one of the 10 categories above>",
  "confidence_adjustment": {
    "<strategy_name>": <float adjustment, e.g. -0.05 or +0.03>
  }
}"""


SKEPTIC_SYSTEM = """You are the Skeptic. Your job is to challenge a lesson the trading agent extracted from a trade. You receive ONLY raw data — no agent narrative.

Ask these questions:
1. SIMPLER EXPLANATION: Did the whole market move the same direction? Check SPY movement.
2. SAMPLE SIZE: How many times has this pattern occurred? If < 5, label "unvalidated".
3. REGIME DEPENDENCY: Would this lesson work in a different rate/volatility environment?
4. FALSIFIABILITY: What would DISPROVE this lesson?

Respond with JSON:
{
  "simpler_explanation": "<alternative explanation if one exists>",
  "sample_size": <estimated occurrences>,
  "validated": <true/false — does the lesson survive scrutiny?>,
  "regime_dependent": <true/false>,
  "falsifiable_test": "<what would disprove this lesson>"
}"""


def _apply_hard_rejections(lesson_category: str, spy_change: float = 0) -> tuple[bool, str]:
    """Auto-reject lessons without LLM when clear confounds exist."""
    if lesson_category == "correlated_market_move" and abs(spy_change) < 0.003:
        return True, "Rejected: labeled correlated_market_move but SPY moved < 0.3%"
    return False, ""


async def skeptic_challenge(lesson: dict, trade: dict, spy_change: float = 0) -> dict:
    """Run skeptic model on a lesson — separate model, raw data only."""
    config = load_config()

    user_prompt = f"""Challenge this lesson from a trading agent:

Trade: {trade['action']} {trade['shares']:.4f} TSLA @ ${trade['price']:.2f}
Category: {lesson.get('category', 'unknown')}
Lesson: {lesson.get('lesson', '')}
What predicted: {lesson.get('what_i_predicted', '')}
What happened: {lesson.get('what_actually_happened', '')}

Raw data:
- SPY movement during same period: {spy_change*100:.2f}%
- Trade P&L: ${trade.get('realized_pnl', 'still open')}

Challenge this lesson. Is there a simpler explanation?"""

    try:
        client = _get_client()
        response = client.messages.create(
            model=SKEPTIC_MODEL,
            max_tokens=800,
            system=SKEPTIC_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        review = _parse_json(raw)
        review["skeptic_model_version"] = SKEPTIC_MODEL
        return review
    except Exception as e:
        logger.warning(f"Skeptic challenge failed: {e}")
        return {"validated": True, "simpler_explanation": "Skeptic unavailable", "skeptic_model_version": SKEPTIC_MODEL}


async def review_trade(trade: dict, current_price: float) -> dict | None:
    """Review a completed trade by comparing hypothesis to outcome.

    v3: includes skeptic challenge, regime tagging, model version tracking.
    """
    if not trade.get("hypothesis"):
        logger.info(f"Skipping review for {trade['id']} — no hypothesis")
        return None

    config = load_config()
    price_change = current_price - trade["price"]
    price_change_pct = (price_change / trade["price"]) * 100

    user_prompt = f"""Review this trade:

Trade: {trade['action']} {trade['shares']:.4f} TSLA @ ${trade['price']:.2f}
Time: {trade['timestamp']}
Strategy: {trade.get('strategy', 'unknown')}
Confidence: {trade.get('confidence', 'N/A')}

Hypothesis: {trade.get('hypothesis', 'None stated')}
Reasoning: {trade.get('reasoning', 'None stated')}

Outcome:
- Current price: ${current_price:.2f}
- Price change since trade: ${price_change:.2f} ({price_change_pct:+.2f}%)
- Trade P&L: ${trade.get('realized_pnl', 'still open')}

What went right or wrong? What should the agent learn?"""

    try:
        raw, model_ver = _call_claude(REVIEW_SYSTEM, user_prompt)
        review = _parse_json(raw)

        # Enforce valid category
        category = review.get("category", "noise_trade")
        if category not in LESSON_CATEGORIES:
            category = "noise_trade"

        # Build lesson record with v3 fields
        lesson = {
            "linked_trade": trade["id"],
            "what_i_predicted": review.get("what_i_predicted", ""),
            "what_actually_happened": review.get("what_actually_happened", ""),
            "why_i_was_wrong": review.get("why_right_or_wrong", ""),
            "lesson": review.get("lesson", ""),
            "category": category,
            "confidence_adjustment": review.get("confidence_adjustment", {}),
            "weight": 1.0,
            "decay_rate": 0.95,
            "times_validated": 0,
            "times_contradicted": 0,
            "model_version": model_ver,
        }

        # Get SPY movement for skeptic
        spy_change = 0
        try:
            from .market_data import get_macro_data
            macro = get_macro_data()
            spy_change = macro.get("spy_change_pct", 0)
        except Exception:
            pass

        # Hard rejection check
        rejected, reason = _apply_hard_rejections(category, spy_change)
        if rejected:
            lesson["skeptic_review"] = {"validated": False, "reason": reason}
            lesson["weight"] = 0.3
            logger.info(f"Hard rejection: {reason}")
        else:
            # Run skeptic challenge
            skeptic = await skeptic_challenge(lesson, trade, spy_change)
            lesson["skeptic_review"] = skeptic
            if not skeptic.get("validated", True):
                lesson["weight"] = 0.5  # Downweight unvalidated lessons
                logger.info(f"Skeptic downweighted lesson: {skeptic.get('simpler_explanation', '')[:80]}")

        # Add regime tag
        try:
            from .market_data import classify_regime
            regime = classify_regime()
            lesson["regime"] = {
                "trend": regime.get("trend", "unknown"),
                "volatility": regime.get("volatility", "unknown"),
                "vix": regime.get("vix", 0),
            }
        except Exception:
            lesson["regime"] = {"trend": "unknown", "volatility": "unknown", "vix": 0}

        saved_lesson = add_lesson(lesson)

        # Mark trade as reviewed
        transactions = load_transactions()
        for t in transactions:
            if t["id"] == trade["id"]:
                t["review"] = {
                    "lesson_id": saved_lesson["id"],
                    "was_correct": review.get("was_correct", False),
                    "reviewed_at": iso_now(),
                }
                break
        save_transactions(transactions)

        # Apply confidence adjustments
        adjustments = review.get("confidence_adjustment", {})
        if adjustments:
            _apply_confidence_adjustments(adjustments, trade)

        logger.info(
            f"Reviewed trade {trade['id']}: lesson={saved_lesson['id']}, "
            f"correct={review.get('was_correct')}, weight={lesson['weight']}"
        )
        return saved_lesson

    except Exception as e:
        logger.error(f"Trade review failed for {trade['id']}: {e}")
        return None


def _apply_confidence_adjustments(adjustments: dict, trade: dict):
    """Apply strategy confidence adjustments from a trade review."""
    scores = get_strategy_scores()
    strategies = scores.get("strategies", {})

    for strategy_name, adj in adjustments.items():
        if strategy_name in strategies:
            s = strategies[strategy_name]
            old_weight = s["weight"]
            # Clamp between 0.05 and 0.40 — never zero out a strategy
            s["weight"] = round(max(0.05, min(0.40, s["weight"] + adj)), 4)
            logger.info(f"Strategy {strategy_name}: weight {old_weight:.3f} -> {s['weight']:.3f} ({adj:+.3f})")

    # Normalize weights to sum to 1.0
    total = sum(s["weight"] for s in strategies.values())
    if total > 0:
        for s in strategies.values():
            s["weight"] = round(s["weight"] / total, 4)

    # Update trade counts
    trade_strategy = trade.get("strategy", "")
    if trade_strategy in strategies:
        strategies[trade_strategy]["total_trades"] += 1
        pnl = trade.get("realized_pnl")
        if pnl is not None:
            if pnl >= 0:
                strategies[trade_strategy]["winning_trades"] += 1
            strategies[trade_strategy]["total_pnl"] = round(
                strategies[trade_strategy]["total_pnl"] + pnl, 2
            )
            total_trades = strategies[trade_strategy]["total_trades"]
            if total_trades > 0:
                strategies[trade_strategy]["win_rate"] = round(
                    strategies[trade_strategy]["winning_trades"] / total_trades, 4
                )

    update_strategy_scores(scores)


# --- Prediction Scoring ---

def review_predictions(current_price: float) -> list[dict]:
    """Check all pending predictions whose time horizon has passed. Score accuracy."""
    from datetime import datetime, timezone, timedelta

    predictions = get_predictions()
    now = datetime.now(timezone.utc)
    scored = []

    horizon_minutes = {"30min": 30, "2hr": 120, "1day": 1440}

    for pred in predictions:
        pred_time = datetime.fromisoformat(pred["timestamp"])
        pred_price = pred.get("price_at_prediction", 0)
        outcomes = pred.get("outcomes", {})
        updated = False

        for horizon, minutes in horizon_minutes.items():
            # Skip if already scored or no prediction made
            if outcomes.get(horizon) is not None:
                continue
            if horizon not in pred.get("predictions", {}):
                continue

            elapsed = (now - pred_time).total_seconds() / 60
            if elapsed < minutes:
                continue

            # Score this prediction
            prediction = pred["predictions"][horizon]
            direction = prediction.get("direction", "flat")
            target = prediction.get("target", pred_price)

            actual_direction = "up" if current_price > pred_price else ("down" if current_price < pred_price else "flat")
            direction_correct = direction == actual_direction
            error_pct = abs(current_price - target) / target * 100 if target > 0 else 0

            outcomes[horizon] = {
                "actual": round(current_price, 2),
                "direction_correct": direction_correct,
                "error_pct": round(error_pct, 2),
                "predicted_direction": direction,
                "actual_direction": actual_direction,
                "predicted_target": target,
            }
            updated = True

            logger.info(
                f"Scored {pred['id']} {horizon}: predicted {direction} -> ${target:.2f}, "
                f"actual {actual_direction} -> ${current_price:.2f}, "
                f"correct={direction_correct}, error={error_pct:.1f}%"
            )

        if updated:
            pred["outcomes"] = outcomes
            update_prediction(pred["id"], {"outcomes": outcomes})
            scored.append(pred)

    return scored


# --- Pattern Discovery ---

PATTERN_SYSTEM = """You are the pattern discovery module of MonopolyTrader. Analyze the agent's trade history and lessons to identify recurring patterns in TSLA's behavior or in the agent's own decision-making.

Look for:
1. TSLA behavior patterns (e.g., "drops after X event", "bounces from Y level")
2. Agent behavior patterns (e.g., "overconfident on momentum trades in choppy markets")
3. Timing patterns (e.g., "better accuracy in afternoon vs morning")
4. Strategy patterns (e.g., "mean reversion works best when RSI < 25")

Return a JSON array of discovered patterns (empty array if none found):
[{
  "name": "<short descriptive name>",
  "description": "<2-3 sentence description with specific evidence>",
  "evidence": ["<trade_id or lesson_id>"],
  "reliability": <0.0 to 1.0 estimate>,
  "tags": ["<category tags>"]
}]"""


async def discover_patterns() -> list[dict]:
    """Analyze trade history + lessons looking for recurring patterns."""
    transactions = load_transactions()
    lessons = get_lessons()

    if len(transactions) < 3 and len(lessons) < 2:
        logger.info("Not enough data for pattern discovery yet")
        return []

    # Summarize recent trades for the prompt
    trade_summary = []
    for t in transactions[-20:]:
        trade_summary.append(
            f"  {t['id']}: {t['action']} {t['shares']:.2f} @ ${t['price']:.2f} "
            f"strategy={t.get('strategy', 'N/A')} "
            f"P&L=${t.get('realized_pnl', 'open')} "
            f"hypothesis: {t.get('hypothesis', 'N/A')[:100]}"
        )

    lesson_summary = []
    for l in lessons[-15:]:
        lesson_summary.append(
            f"  {l['id']}: [{l.get('category', 'other')}] {l.get('lesson', 'N/A')[:120]}"
        )

    user_prompt = f"""Analyze this trading history for patterns:

TRADES ({len(transactions)} total, showing last {min(20, len(transactions))}):
{chr(10).join(trade_summary)}

LESSONS ({len(lessons)} total, showing last {min(15, len(lessons))}):
{chr(10).join(lesson_summary)}

PREDICTION ACCURACY:
{json.dumps(get_prediction_accuracy(), indent=2)}

Find recurring patterns. Be specific — cite trade IDs and lessons as evidence."""

    try:
        raw, model_ver = _call_claude(PATTERN_SYSTEM, user_prompt, max_tokens=2000)
        patterns = _parse_json(raw)

        saved = []
        existing = get_patterns()
        existing_names = {p.get("name", "").lower() for p in existing}

        for p in patterns:
            # Skip if we already have a pattern with a very similar name
            if p.get("name", "").lower() in existing_names:
                continue
            p["sample_size"] = len(transactions)
            p["last_tested"] = iso_now()
            p["model_version"] = model_ver
            saved_pattern = add_pattern(p)
            saved.append(saved_pattern)

        logger.info(f"Pattern discovery: found {len(saved)} new patterns")
        return saved

    except Exception as e:
        logger.error(f"Pattern discovery failed: {e}")
        return []


# --- Strategy Evolution ---

async def evolve_strategy_weights() -> dict:
    """Recalculate strategy weights based on recent performance.

    Uses exponential decay — recent trades matter more than old ones.
    Ensures no strategy drops below 0.05 weight.
    """
    scores = get_strategy_scores()
    strategies = scores.get("strategies", {})
    transactions = load_transactions()

    if not transactions:
        return scores

    # Calculate recent performance per strategy (last 20 trades)
    recent = transactions[-20:]
    strategy_perf = {}

    for t in recent:
        strat = t.get("strategy", "unknown")
        if strat not in strategies:
            continue
        if strat not in strategy_perf:
            strategy_perf[strat] = {"wins": 0, "losses": 0, "pnl": 0}

        pnl = t.get("realized_pnl")
        if pnl is not None:
            strategy_perf[strat]["pnl"] += pnl
            if pnl >= 0:
                strategy_perf[strat]["wins"] += 1
            else:
                strategy_perf[strat]["losses"] += 1

    # Adjust weights based on performance
    old_weights = {k: v["weight"] for k, v in strategies.items()}
    changes = {}

    for name, perf in strategy_perf.items():
        total = perf["wins"] + perf["losses"]
        if total == 0:
            continue
        win_rate = perf["wins"] / total
        # Reward/punish based on win rate vs 50%
        adj = (win_rate - 0.5) * 0.1  # +-5% max per rebalance
        strategies[name]["weight"] = max(0.05, min(0.40, strategies[name]["weight"] + adj))
        changes[name] = round(adj, 4)

    # Normalize
    total_weight = sum(s["weight"] for s in strategies.values())
    if total_weight > 0:
        for s in strategies.values():
            s["weight"] = round(s["weight"] / total_weight, 4)

    # Determine trend for each strategy
    for name, s in strategies.items():
        if s["weight"] > old_weights.get(name, 0.2) + 0.02:
            s["trend"] = "improving"
        elif s["weight"] < old_weights.get(name, 0.2) - 0.02:
            s["trend"] = "declining"
        else:
            s["trend"] = "stable"

    # Record rebalance
    if changes:
        scores.setdefault("rebalance_history", []).append({
            "timestamp": iso_now(),
            "changes": changes,
            "reason": f"Performance-based rebalance from last {len(recent)} trades",
        })

    update_strategy_scores(scores)
    logger.info(f"Strategy evolution: {changes}")
    return scores


# --- Journal Entry ---

JOURNAL_SYSTEM = """You are MonopolyTrader writing a journal entry reflecting on your trading performance and learning progress. Write in first person, be honest about mistakes, and identify what you need to improve.

Keep it to 2-3 paragraphs. Be specific — reference actual trades, predictions, and lessons. End with a clear goal or experiment for the next trading session."""


async def write_journal_entry(portfolio: dict) -> str:
    """Write a reflective journal entry about recent performance."""
    transactions = load_transactions()
    lessons = get_lessons()
    accuracy = get_prediction_accuracy()
    scores = get_strategy_scores()
    summary = get_knowledge_summary()

    recent_trades = transactions[-10:] if transactions else []
    trade_lines = []
    for t in recent_trades:
        review = t.get("review", {})
        trade_lines.append(
            f"  {t['action']} {t['shares']:.2f} @ ${t['price']:.2f} "
            f"(strategy: {t.get('strategy', 'N/A')}, "
            f"P&L: ${t.get('realized_pnl', 'open')}, "
            f"correct: {review.get('was_correct', 'unreviewed')})"
        )

    user_prompt = f"""Write a journal entry based on this data:

Portfolio: ${portfolio.get('total_value', 0):.2f} (P&L: ${portfolio.get('total_pnl', 0):.2f}, {portfolio.get('total_pnl_pct', 0):+.2f}%)
Total trades: {len(transactions)}
Knowledge: {summary}

Recent trades:
{chr(10).join(trade_lines) if trade_lines else "  No trades yet."}

Recent lessons:
{chr(10).join(f"  {l.get('lesson', '')[:100]}" for l in lessons[-5:]) if lessons else "  None yet."}

Prediction accuracy: {json.dumps(accuracy, indent=2)}

Strategy weights: {json.dumps({k: f"{v['weight']:.2f} ({v['trend']})" for k, v in scores.get('strategies', {}).items()}, indent=2)}

Reflect on your performance. What's working? What's not? What will you try next?"""

    try:
        entry, model_ver = _call_claude(JOURNAL_SYSTEM, user_prompt, max_tokens=1000)
        append_journal(f"[Model: {model_ver}]\n\n{entry}")
        logger.info("Journal entry written")
        return entry
    except Exception as e:
        logger.error(f"Journal entry failed: {e}")
        return ""


# --- Convenience: Run All Learning Tasks ---

async def run_learning_cycle(current_price: float, portfolio: dict):
    """Run the full post-session learning cycle."""
    logger.info("Starting learning cycle...")

    # 1. Review unreviewed trades
    transactions = load_transactions()
    unreviewed = [t for t in transactions if t.get("review") is None and t.get("hypothesis")]
    for trade in unreviewed:
        await review_trade(trade, current_price)

    # 2. Score pending predictions
    scored = review_predictions(current_price)
    logger.info(f"Scored {len(scored)} predictions")

    # 3. Discover patterns
    patterns = await discover_patterns()
    logger.info(f"Discovered {len(patterns)} new patterns")

    # 4. Evolve strategy weights
    await evolve_strategy_weights()

    # 5. Write journal
    await write_journal_entry(portfolio)

    logger.info("Learning cycle complete")
