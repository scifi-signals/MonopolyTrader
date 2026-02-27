"""Learning engine — post-trade reviews, skeptic layer, lesson decay, pattern discovery."""

import json
import os
from anthropic import Anthropic
from .utils import load_config, load_json, save_json, iso_now, setup_logging, DATA_DIR
from .knowledge_base import (
    get_lessons, add_lesson, get_patterns, add_pattern,
    get_predictions, update_prediction, get_strategy_scores,
    update_strategy_scores, append_journal, get_knowledge_summary,
    get_prediction_accuracy, PATTERNS_PATH, promote_replay_lesson,
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
    """Parse JSON from Claude response, stripping markdown fences.

    Handles common issues: markdown code blocks, trailing text after
    valid JSON (the 'Extra data' error), and leading non-JSON text.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # First try standard parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Handle trailing text after valid JSON using raw_decode
    # (e.g., Claude appends explanation after the JSON object)
    decoder = json.JSONDecoder()
    # Find the first { or [ to skip any leading text
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                continue

    # Last resort: try the original text (will raise a clear error)
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


def _extract_lesson_tags(trade: dict, lesson: dict) -> list[str]:
    """Extract searchable tags from trade context for tag-based knowledge retrieval.

    Tags encode the market conditions under which the lesson was learned,
    enabling better matching when similar conditions recur.
    """
    tags = []

    # Regime tags
    regime = trade.get("regime", {}) or lesson.get("regime", {})
    if regime.get("trend"):
        tags.append(f"regime_{regime['trend']}")
    if regime.get("volatility"):
        tags.append(f"vol_{regime['volatility']}")

    # VIX level
    vix = regime.get("vix", 0)
    if vix > 25:
        tags.append("vix_high")
    elif vix < 15:
        tags.append("vix_low")

    # RSI from trade signals
    signals = trade.get("signals", {})
    rsi = signals.get("rsi_14") or signals.get("rsi")
    if rsi is not None:
        if rsi < 30:
            tags.append("rsi_oversold")
        elif rsi > 70:
            tags.append("rsi_overbought")

    # Category-derived tags
    category = lesson.get("category", "")
    if category:
        tags.append(f"cat_{category}")

    # Strategy tag
    strategy = trade.get("strategy", "")
    if strategy and strategy != "unknown":
        tags.append(f"strat_{strategy}")

    # Signal correctness
    was_correct = lesson.get("skeptic_review", {}).get("validated")
    if was_correct is True:
        tags.append("skeptic_validated")
    elif was_correct is False:
        tags.append("skeptic_rejected")

    return tags


async def skeptic_challenge(lesson: dict, trade: dict, spy_change: float = 0, *, is_replay: bool = False) -> dict:
    """Run skeptic model on a lesson — separate model, raw data only."""
    config = load_config()

    replay_context = ""
    if is_replay:
        replay_context = """
NOTE: This trade was executed in HISTORICAL REPLAY mode (practicing on past data).
You have the real SPY data for this period. The agent did NOT have hindsight — it
saw obfuscated prices and had no knowledge of future bars. Evaluate the lesson on
its trading logic merits. Be slightly more lenient on sample size since the agent
is actively building experience through practice. A replay lesson that captures
a real pattern is valuable even with limited occurrences — mark validated=true if
the core trading logic is sound, even if sample size is small.
"""

    user_prompt = f"""Challenge this lesson from a trading agent:

Trade: {trade['action']} {trade['shares']:.4f} TSLA @ ${trade['price']:.2f}
Category: {lesson.get('category', 'unknown')}
Lesson: {lesson.get('lesson', '')}
What predicted: {lesson.get('what_i_predicted', '')}
What happened: {lesson.get('what_actually_happened', '')}

Raw data:
- SPY movement during same period: {spy_change*100:.2f}%
- Trade P&L: ${trade.get('realized_pnl', 'still open')}
{replay_context}
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

    # Include strategy signals if stored in the transaction
    signals_text = ""
    signals = trade.get("signals", {})
    if signals:
        sig_lines = [f"  {k}: {v}" for k, v in signals.items()]
        signals_text = "\nStrategy signals at time of trade:\n" + "\n".join(sig_lines)

    user_prompt = f"""Review this trade:

Trade: {trade['action']} {trade['shares']:.4f} TSLA @ ${trade['price']:.2f}
Time: {trade['timestamp']}
Strategy: {trade.get('strategy', 'unknown')}
Confidence: {trade.get('confidence', 'N/A')}
{signals_text}
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

        # Replay source propagation: if this trade came from replay,
        # tag the lesson with lower weight and faster decay
        is_replay = bool(trade.get("_replay_source"))
        base_weight = 0.7 if is_replay else 1.0
        base_decay = 0.90 if is_replay else 0.95

        # Build lesson record with v3+ fields
        lesson = {
            "linked_trade": trade["id"],
            "what_i_predicted": review.get("what_i_predicted", ""),
            "what_actually_happened": review.get("what_actually_happened", ""),
            "why_i_was_wrong": review.get("why_right_or_wrong", ""),
            "lesson": review.get("lesson", ""),
            "category": category,
            "confidence_adjustment": review.get("confidence_adjustment", {}),
            "weight": base_weight,
            "decay_rate": base_decay,
            "times_validated": 0,
            "times_contradicted": 0,
            "model_version": model_ver,
        }

        # Add replay metadata
        if is_replay:
            lesson["source"] = "replay"
            lesson["_replay_source"] = trade["_replay_source"]

        # Get SPY movement for skeptic — use historical data for replay trades
        spy_change = 0
        if is_replay and "_replay_spy_change" in trade:
            spy_change = trade["_replay_spy_change"]
        else:
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
            # Run skeptic challenge (replay-aware)
            skeptic = await skeptic_challenge(lesson, trade, spy_change, is_replay=is_replay)
            lesson["skeptic_review"] = skeptic
            if not skeptic.get("validated", True):
                # Replay lessons get a lighter penalty since the skeptic has
                # limited context (no intraday correlation data, no live news)
                penalty_weight = 0.55 if is_replay else 0.5
                lesson["weight"] = min(lesson["weight"], penalty_weight)
                logger.info(f"Skeptic downweighted lesson: {skeptic.get('simpler_explanation', '')[:80]}")

        # Add regime tag — use regime stored at trade time if available,
        # otherwise fall back to current regime (less accurate for old trades)
        trade_regime = trade.get("regime", {})
        if trade_regime and trade_regime.get("trend"):
            lesson["regime"] = {
                "trend": trade_regime.get("trend", "unknown"),
                "volatility": trade_regime.get("volatility", "unknown"),
                "vix": trade_regime.get("vix", 0),
            }
        else:
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

        # Extract searchable tags (Proposal 11)
        lesson["tags"] = _extract_lesson_tags(trade, lesson)

        saved_lesson = add_lesson(lesson)

        # Promote cited replay lessons if this live trade succeeded (Proposal 6)
        if not is_replay and review.get("was_correct"):
            _promote_cited_replay_lessons(trade)

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

        # Apply confidence adjustments — but ONLY if the skeptic validated the lesson.
        # If the skeptic found a simpler explanation (e.g., SPY correlation), the
        # confidence adjustment is based on a flawed causal story and should be skipped.
        skeptic_validated = lesson.get("skeptic_review", {}).get("validated", True)
        adjustments = review.get("confidence_adjustment", {})
        if adjustments and skeptic_validated:
            _apply_confidence_adjustments(adjustments, trade)
        elif adjustments and not skeptic_validated:
            logger.info(f"Skipping weight adjustments — skeptic invalidated lesson")

        # Update reliability of any patterns this trade referenced
        _update_pattern_reliability(trade, review.get("was_correct", False))

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

    # No normalization — strategies earn/lose trust independently.
    # aggregate_signals() divides by total_weight, so non-normalized is fine.
    # Normalizing rewarded untested strategies (they drifted up when one got punished).

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


# --- Pattern Reliability Updates ---

def _update_pattern_reliability(trade: dict, was_correct: bool):
    """Update reliability of patterns referenced by a trade.

    Reads knowledge_applied from the trade to find pattern IDs.
    Bumps reliability up for correct trades, down for incorrect ones.
    """
    knowledge_applied = trade.get("knowledge_applied", [])
    if not knowledge_applied:
        return

    pattern_ids = [k for k in knowledge_applied if k.startswith("pattern_")]
    if not pattern_ids:
        return

    patterns = get_patterns()
    updated = False
    for pattern in patterns:
        if pattern.get("id") in pattern_ids:
            old_reliability = pattern.get("reliability", 0.5)
            if was_correct:
                pattern["reliability"] = min(1.0, round(old_reliability + 0.05, 4))
                pattern["times_validated"] = pattern.get("times_validated", 0) + 1
            else:
                pattern["reliability"] = max(0.0, round(old_reliability - 0.08, 4))
                pattern["times_contradicted"] = pattern.get("times_contradicted", 0) + 1
            pattern["last_tested"] = iso_now()
            updated = True
            logger.info(
                f"Pattern {pattern['id']} reliability: {old_reliability:.2f} -> {pattern['reliability']:.2f} "
                f"({'validated' if was_correct else 'contradicted'})"
            )

    if updated:
        save_json(PATTERNS_PATH, patterns)


# --- Replay Lesson Promotion ---

def _promote_cited_replay_lessons(trade: dict):
    """When a live trade cites replay lessons and succeeds, promote them.

    This is the bridge between replay practice and live trading: if the agent
    applied a lesson learned during replay and it worked in live conditions,
    boost that lesson's weight toward 1.0 (matching live-sourced lessons).
    """
    knowledge_applied = trade.get("knowledge_applied", [])
    if not knowledge_applied:
        return

    lessons = get_lessons()
    replay_lesson_ids = {
        l["id"] for l in lessons
        if l.get("source") == "replay" and l.get("id") in knowledge_applied
    }

    for lesson_id in replay_lesson_ids:
        promoted = promote_replay_lesson(lesson_id)
        if promoted:
            logger.info(
                f"Promoted replay lesson {lesson_id} — "
                f"cited in successful live trade {trade.get('id', '?')}"
            )


# --- HOLD Counterfactual Learning ---

HOLD_REVIEW_SYSTEM = """You are the learning module of MonopolyTrader. Your job is to analyze HOLD decisions where the agent chose NOT to act, and determine what should be learned.

You receive a HOLD decision with its counterfactual outcome — what WOULD have happened if the agent had followed the strongest ignored signal instead of holding.

Be specific. If the agent kept holding through a clear signal and missed money, identify exactly what the agent should watch for next time. If the hold was correct, identify what made it a good decision so the agent can recognize similar situations.

IMPORTANT: Categorize every lesson into EXACTLY ONE of these categories:
- signal_correct: The ignored signal was right — agent should have acted
- signal_wrong: The ignored signal was wrong — HOLD was the right call
- regime_mismatch: The signal would have worked in a different regime but not this one
- noise_trade: The ignored signal was noise — there was no real opportunity
- correlated_market_move: Price moved with the market, not on the signal
- risk_sizing_error: Signal was right but agent was right to hold due to position/risk limits

Respond with JSON:
{
  "what_signal_said": "<summary of the ignored signal>",
  "what_actually_happened": "<what the price actually did>",
  "was_hold_correct": <true/false>,
  "why": "<specific analysis of why the hold was right or wrong>",
  "lesson": "<one clear, actionable lesson about when to hold vs act>",
  "category": "<exactly one of the categories above>",
  "confidence_adjustment": {
    "<strategy_name>": <float adjustment, e.g. +0.03 if signal was right and ignored, -0.02 if signal was wrong>
  }
}"""


async def review_hold_outcomes() -> list[dict]:
    """Review scored hold counterfactuals and extract lessons.

    This is the critical missing piece — without this, the agent collects
    data about whether its HOLDs were right or wrong but never LEARNS from it.
    """
    hold_log = load_json(DATA_DIR / "hold_log.json", default=[])
    if not hold_log:
        return []

    # Find scored holds that haven't been reviewed yet
    unreviewed = [
        h for h in hold_log
        if h.get("counterfactual_scored")
        and h.get("counterfactual_outcome")
        and not h.get("lesson_extracted")
    ]

    if not unreviewed:
        return []

    # Only review missed gains and a sample of correct holds
    # (we learn more from mistakes than from being right)
    missed = [h for h in unreviewed if h.get("counterfactual_outcome", {}).get("verdict") == "missed_gain"]
    correct = [h for h in unreviewed if h.get("counterfactual_outcome", {}).get("verdict") == "correct_hold"]

    # Review all missed gains, but only 1 in 3 correct holds (avoid noise)
    to_review = missed + correct[::3]

    # Cap per-cycle reviews to prevent batch dumps from polluting the knowledge base.
    # If a backlog builds up (e.g., first deploy), spread reviews across multiple cycles.
    MAX_HOLD_REVIEWS_PER_CYCLE = 10
    if len(to_review) > MAX_HOLD_REVIEWS_PER_CYCLE:
        logger.info(f"Hold review backlog: {len(to_review)} pending, capping at {MAX_HOLD_REVIEWS_PER_CYCLE}")
        to_review = to_review[:MAX_HOLD_REVIEWS_PER_CYCLE]

    lessons_created = []
    for hold in to_review:
        lesson = await _review_single_hold(hold)
        if lesson:
            lessons_created.append(lesson)
            # Mark as reviewed
            hold["lesson_extracted"] = True
            hold["linked_lesson"] = lesson["id"]

    # Mark skipped correct holds as reviewed (only those not in to_review).
    # Don't mark holds that are still in the backlog waiting for the next cycle.
    reviewed_timestamps = {h.get("timestamp") for h in to_review}
    for h in correct:
        if not h.get("lesson_extracted") and h.get("timestamp") not in reviewed_timestamps:
            h["lesson_extracted"] = True

    # Save updated hold log
    save_json(DATA_DIR / "hold_log.json", hold_log)
    logger.info(f"Hold review: {len(missed)} missed gains, {len(correct)} correct holds, {len(lessons_created)} lessons created")
    return lessons_created


async def _review_single_hold(hold: dict) -> dict | None:
    """Review a single hold decision and extract a lesson."""
    cf = hold.get("counterfactual_outcome", {})
    strongest = hold.get("strongest_signal_ignored", {})
    balance = hold.get("signal_balance", {})
    analysis = hold.get("hold_analysis", {})

    user_prompt = f"""Review this HOLD decision and its counterfactual outcome:

HOLD Decision:
- Time: {hold.get('timestamp', 'unknown')}
- Price at hold: ${hold.get('price_at_hold', 0):.2f}
- Reason: {hold.get('reason', 'agent_decision')}
- Regime: {json.dumps(hold.get('regime', {}), default=str)}

Strongest Ignored Signal:
- Action: {strongest.get('action', 'N/A')}
- Strategy: {strongest.get('strategy', 'N/A')}
- Confidence: {strongest.get('confidence', 0):.2f}

Signal Balance: {json.dumps(balance, default=str) if balance else 'N/A'}

Agent's Hold Justification:
- Opportunity cost: {analysis.get('opportunity_cost', 'Not stated') if analysis else 'Not stated'}
- Downside protection: {analysis.get('downside_protection', 'Not stated') if analysis else 'Not stated'}
- Decision boundary: {analysis.get('decision_boundary', 'Not stated') if analysis else 'Not stated'}

COUNTERFACTUAL OUTCOME (what actually happened):
- Price after 2hr: ${cf.get('price_after_2hr', 0):.2f}
- Price change: {cf.get('price_change_pct', 0):+.3f}%
- Ignored signal was: {'RIGHT' if cf.get('was_signal_right') else 'WRONG'}
- Hypothetical P&L if acted: ${cf.get('hypothetical_pnl', 0):.2f}
- Verdict: {cf.get('verdict', 'unknown')}

What should the agent learn about WHEN to hold vs WHEN to act?"""

    try:
        raw, model_ver = _call_claude(HOLD_REVIEW_SYSTEM, user_prompt)
        review = _parse_json(raw)

        category = review.get("category", "noise_trade")
        if category not in LESSON_CATEGORIES:
            category = "noise_trade"

        lesson = {
            "linked_trade": f"hold_{hold.get('timestamp', 'unknown')[:19]}",
            "source": "hold_counterfactual",
            "what_i_predicted": review.get("what_signal_said", ""),
            "what_actually_happened": review.get("what_actually_happened", ""),
            "why_i_was_wrong": review.get("why", ""),
            "lesson": review.get("lesson", ""),
            "category": category,
            "confidence_adjustment": review.get("confidence_adjustment", {}),
            "weight": 0.8 if cf.get("verdict") == "missed_gain" else 0.6,
            "decay_rate": 0.95,
            "times_validated": 0,
            "times_contradicted": 0,
            "model_version": model_ver,
        }

        # Run skeptic challenge on hold lessons too (was previously missing)
        spy_change = 0
        try:
            from .market_data import get_macro_data
            macro = get_macro_data()
            spy_change = macro.get("spy_change_pct", 0)
        except Exception:
            pass

        # Build a trade-like dict so skeptic_challenge() can work with it
        pseudo_trade = {
            "action": strongest.get("action", "HOLD"),
            "shares": 0,
            "price": hold.get("price_at_hold", 0),
            "realized_pnl": cf.get("hypothetical_pnl", 0),
        }
        rejected, reason = _apply_hard_rejections(category, spy_change)
        if rejected:
            lesson["skeptic_review"] = {"validated": False, "reason": reason}
            lesson["weight"] = 0.3
            logger.info(f"Hold lesson hard rejection: {reason}")
        else:
            skeptic = await skeptic_challenge(lesson, pseudo_trade, spy_change)
            lesson["skeptic_review"] = skeptic
            if not skeptic.get("validated", True):
                lesson["weight"] = min(lesson["weight"], 0.5)
                logger.info(f"Skeptic downweighted hold lesson: {skeptic.get('simpler_explanation', '')[:80]}")

        # Add regime tag
        lesson["regime"] = hold.get("regime", {"trend": "unknown", "volatility": "unknown", "vix": 0})

        saved = add_lesson(lesson)

        # Apply confidence adjustments — but ONLY if the skeptic validated the lesson.
        # A skeptic-rejected hold lesson means the price move had a simpler explanation
        # (e.g., SPY correlation), so adjusting weights based on it would be noise.
        skeptic_validated = lesson.get("skeptic_review", {}).get("validated", True)
        adjustments = review.get("confidence_adjustment", {})
        if adjustments and skeptic_validated:
            _apply_hold_confidence_adjustments(adjustments, hold)
        elif adjustments and not skeptic_validated:
            logger.info(f"Skipping hold weight adjustments — skeptic invalidated lesson")

        logger.info(
            f"Hold lesson: {saved['id']} — {cf.get('verdict', '?')}, "
            f"category={category}, hold_correct={review.get('was_hold_correct')}"
        )
        return saved

    except Exception as e:
        logger.error(f"Hold review failed: {e}")
        return None


def _apply_hold_confidence_adjustments(adjustments: dict, hold: dict):
    """Apply strategy confidence adjustments from a hold review.

    Key difference from trade adjustments: if the agent ignored a correct signal,
    BOOST that strategy's weight (it was right, we should listen to it more).
    """
    scores = get_strategy_scores()
    strategies = scores.get("strategies", {})

    for strategy_name, adj in adjustments.items():
        if strategy_name in strategies:
            s = strategies[strategy_name]
            old_weight = s["weight"]
            s["weight"] = round(max(0.05, min(0.40, s["weight"] + adj)), 4)
            logger.info(f"Hold adj: {strategy_name} weight {old_weight:.3f} -> {s['weight']:.3f} ({adj:+.3f})")

    # No normalization — see _apply_confidence_adjustments comment.

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

PATTERN_SYSTEM = """You are the pattern discovery module of MonopolyTrader. Analyze the agent's trade history, HOLD decisions, and lessons to identify recurring patterns in TSLA's behavior or in the agent's own decision-making.

Look for:
1. TSLA behavior patterns (e.g., "drops after X event", "bounces from Y level")
2. Agent behavior patterns (e.g., "overconfident on momentum trades in choppy markets")
3. HOLD patterns (e.g., "agent holds too much when momentum signals fire", "HOLDs during high-volatility regime are usually correct")
4. Timing patterns (e.g., "better accuracy in afternoon vs morning")
5. Strategy patterns (e.g., "mean reversion works best when RSI < 25")

Return a JSON array of discovered patterns (empty array if none found):
[{
  "name": "<short descriptive name>",
  "description": "<2-3 sentence description with specific evidence>",
  "evidence": ["<trade_id or lesson_id>"],
  "reliability": <0.0 to 1.0 estimate>,
  "tags": ["<category tags>"]
}]"""


async def discover_patterns() -> list[dict]:
    """Analyze trade history + hold decisions + lessons for recurring patterns."""
    transactions = load_transactions()
    lessons = get_lessons()
    hold_log = load_json(DATA_DIR / "hold_log.json", default=[])
    scored_holds = [h for h in hold_log if h.get("counterfactual_scored")]

    if len(transactions) < 3 and len(lessons) < 2 and len(scored_holds) < 3:
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

    # Summarize scored HOLD decisions
    hold_summary = []
    for h in scored_holds[-15:]:
        cf = h.get("counterfactual_outcome", {})
        sig = h.get("strongest_signal_ignored", {})
        hold_summary.append(
            f"  HOLD @ ${h.get('price_at_hold', 0):.2f} "
            f"| ignored {sig.get('action', '?')} from {sig.get('strategy', '?')} "
            f"(conf={sig.get('confidence', 0):.2f}) "
            f"| verdict: {cf.get('verdict', '?')} "
            f"| hypo P&L: ${cf.get('hypothetical_pnl', 0):.2f} "
            f"| regime: {h.get('regime', {}).get('trend', '?')}/{h.get('regime', {}).get('volatility', '?')}"
        )

    lesson_summary = []
    for l in lessons[-15:]:
        source_tag = " [from HOLD]" if l.get("source") == "hold_counterfactual" else ""
        lesson_summary.append(
            f"  {l['id']}: [{l.get('category', 'other')}]{source_tag} {l.get('lesson', 'N/A')[:120]}"
        )

    user_prompt = f"""Analyze this trading history for patterns:

TRADES ({len(transactions)} total, showing last {min(20, len(transactions))}):
{chr(10).join(trade_summary) if trade_summary else "  No trades yet."}

HOLD DECISIONS ({len(scored_holds)} scored, showing last {min(15, len(scored_holds))}):
{chr(10).join(hold_summary) if hold_summary else "  No scored holds yet."}

LESSONS ({len(lessons)} total, showing last {min(15, len(lessons))}):
{chr(10).join(lesson_summary)}

PREDICTION ACCURACY:
{json.dumps(get_prediction_accuracy(), indent=2)}

Find recurring patterns. Look especially at HOLD vs ACT patterns — is the agent holding too much? Too little? Are there specific conditions where HOLDs are consistently right or wrong?"""

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

    # No normalization — see _apply_confidence_adjustments comment.

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

    # HOLD performance summary
    hold_log = load_json(DATA_DIR / "hold_log.json", default=[])
    scored_holds = [h for h in hold_log if h.get("counterfactual_scored")]
    missed = [h for h in scored_holds if h.get("counterfactual_outcome", {}).get("verdict") == "missed_gain"]
    correct = [h for h in scored_holds if h.get("counterfactual_outcome", {}).get("verdict") == "correct_hold"]
    hold_lessons = [l for l in lessons if l.get("source") == "hold_counterfactual"]

    hold_section = "  No scored holds yet."
    if scored_holds:
        total_missed_pnl = sum(
            abs(h.get("counterfactual_outcome", {}).get("hypothetical_pnl", 0))
            for h in missed
        )
        hold_section = (
            f"  Total scored: {len(scored_holds)}\n"
            f"  Correct holds: {len(correct)} ({len(correct)/len(scored_holds)*100:.0f}%)\n"
            f"  Missed gains: {len(missed)} (est. ${total_missed_pnl:.2f} left on table)\n"
            f"  Lessons from holds: {len(hold_lessons)}"
        )

    user_prompt = f"""Write a journal entry based on this data:

Portfolio: ${portfolio.get('total_value', 0):.2f} (P&L: ${portfolio.get('total_pnl', 0):.2f}, {portfolio.get('total_pnl_pct', 0):+.2f}%)
Total trades: {len(transactions)}
Knowledge: {summary}

Recent trades:
{chr(10).join(trade_lines) if trade_lines else "  No trades yet."}

HOLD Performance (am I holding too much or too little?):
{hold_section}

Recent lessons:
{chr(10).join(f"  {l.get('lesson', '')[:100]}" for l in lessons[-5:]) if lessons else "  None yet."}

Recent HOLD lessons:
{chr(10).join(f"  {l.get('lesson', '')[:100]}" for l in hold_lessons[-3:]) if hold_lessons else "  None yet."}

Prediction accuracy: {json.dumps(accuracy, indent=2)}

Strategy weights: {json.dumps({k: f"{v['weight']:.2f} ({v['trend']})" for k, v in scores.get('strategies', {}).items()}, indent=2)}

Reflect on your performance. Include your HOLD decision quality — are you holding too cautiously? Missing opportunities? Or are your holds mostly correct? What will you try next?"""

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

    # 2. Review HOLD counterfactual outcomes (THE MISSING PIECE)
    hold_lessons = await review_hold_outcomes()
    logger.info(f"Hold reviews: {len(hold_lessons)} lessons from HOLD decisions")

    # 3. Score pending predictions
    scored = review_predictions(current_price)
    logger.info(f"Scored {len(scored)} predictions")

    # 4. Discover patterns (now includes hold data)
    patterns = await discover_patterns()
    logger.info(f"Discovered {len(patterns)} new patterns")

    # 5. Evolve strategy weights
    await evolve_strategy_weights()

    # 6. Write journal (now includes hold performance)
    await write_journal_entry(portfolio)

    logger.info("Learning cycle complete")
