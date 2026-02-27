"""Knowledge base manager — read/write the agent's growing brain."""

from pathlib import Path
from .utils import (
    load_json, save_json, generate_id, iso_now,
    KNOWLEDGE_DIR, setup_logging
)

logger = setup_logging("knowledge_base")

LESSONS_PATH = KNOWLEDGE_DIR / "lessons.json"
PATTERNS_PATH = KNOWLEDGE_DIR / "patterns.json"
PREDICTIONS_PATH = KNOWLEDGE_DIR / "predictions.json"
TSLA_PROFILE_PATH = KNOWLEDGE_DIR / "tsla_profile.json"
JOURNAL_PATH = KNOWLEDGE_DIR / "journal.md"
STRATEGY_SCORES_PATH = KNOWLEDGE_DIR.parent / "data" / "strategy_scores.json"
RESEARCH_DIR = KNOWLEDGE_DIR / "research"


def _default_strategy_scores() -> dict:
    return {
        "last_updated": iso_now(),
        "strategies": {
            "momentum": {
                "weight": 0.20, "initial_weight": 0.20,
                "total_trades": 0, "winning_trades": 0,
                "win_rate": 0.0, "avg_return_pct": 0.0,
                "total_pnl": 0.0, "trend": "neutral", "notes": "",
            },
            "mean_reversion": {
                "weight": 0.20, "initial_weight": 0.20,
                "total_trades": 0, "winning_trades": 0,
                "win_rate": 0.0, "avg_return_pct": 0.0,
                "total_pnl": 0.0, "trend": "neutral", "notes": "",
            },
            "technical_signals": {
                "weight": 0.20, "initial_weight": 0.20,
                "total_trades": 0, "winning_trades": 0,
                "win_rate": 0.0, "avg_return_pct": 0.0,
                "total_pnl": 0.0, "trend": "neutral", "notes": "",
            },
            "thesis_alignment": {
                "weight": 0.25, "initial_weight": 0.25,
                "total_trades": 0, "winning_trades": 0,
                "win_rate": 0.0, "avg_return_pct": 0.0,
                "total_pnl": 0.0, "trend": "neutral", "notes": "v2: thesis-driven signal",
            },
            # Legacy strategies — kept for backward compat with existing scores
            "sentiment": {
                "weight": 0.10, "initial_weight": 0.20,
                "total_trades": 0, "winning_trades": 0,
                "win_rate": 0.0, "avg_return_pct": 0.0,
                "total_pnl": 0.0, "trend": "neutral", "notes": "deprecated in v2",
            },
            "dca": {
                "weight": 0.05, "initial_weight": 0.20,
                "total_trades": 0, "winning_trades": 0,
                "win_rate": 0.0, "avg_return_pct": 0.0,
                "total_pnl": 0.0, "trend": "neutral", "notes": "deprecated in v2",
            },
        },
        "rebalance_history": [],
    }


def _default_tsla_profile() -> dict:
    return {
        "ticker": "TSLA",
        "last_updated": iso_now(),
        "market_regime": "unknown",
        "key_levels": {"support": [], "resistance": []},
        "behavioral_notes": [],
        "catalyst_sensitivity": {},
        "seasonal_notes": [],
    }


# --- Initialization ---

def initialize():
    """Create all knowledge files with empty/default data if they don't exist."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    if not LESSONS_PATH.exists():
        save_json(LESSONS_PATH, [])
    if not PATTERNS_PATH.exists():
        save_json(PATTERNS_PATH, [])
    if not PREDICTIONS_PATH.exists():
        save_json(PREDICTIONS_PATH, [])
    if not TSLA_PROFILE_PATH.exists():
        save_json(TSLA_PROFILE_PATH, _default_tsla_profile())
    if not STRATEGY_SCORES_PATH.exists():
        save_json(STRATEGY_SCORES_PATH, _default_strategy_scores())
    if not JOURNAL_PATH.exists():
        JOURNAL_PATH.write_text("# MonopolyTrader Journal\n\n")

    # Research files
    for name in ["earnings_history", "catalyst_events", "correlation_notes", "sector_context", "seasonal_patterns"]:
        path = RESEARCH_DIR / f"{name}.json"
        if not path.exists():
            save_json(path, {"topic": name, "last_updated": None, "findings": []})

    logger.info("Knowledge base initialized")


# --- Lessons ---

def get_lessons() -> list:
    return load_json(LESSONS_PATH, default=[])


def add_lesson(lesson: dict):
    lessons = get_lessons()
    if "id" not in lesson:
        lesson["id"] = generate_id("lesson", [l["id"] for l in lessons])
    if "timestamp" not in lesson:
        lesson["timestamp"] = iso_now()
    lessons.append(lesson)
    save_json(LESSONS_PATH, lessons)
    logger.info(f"Added lesson: {lesson['id']}")
    return lesson


# --- Patterns ---

def get_patterns() -> list:
    return load_json(PATTERNS_PATH, default=[])


def add_pattern(pattern: dict):
    patterns = get_patterns()
    if "id" not in pattern:
        pattern["id"] = generate_id("pattern", [p["id"] for p in patterns])
    if "discovered" not in pattern:
        pattern["discovered"] = iso_now()
    patterns.append(pattern)
    save_json(PATTERNS_PATH, patterns)
    logger.info(f"Added pattern: {pattern['id']} — {pattern.get('name', '')}")
    return pattern


# --- Predictions ---

def get_predictions() -> list:
    return load_json(PREDICTIONS_PATH, default=[])


def add_prediction(prediction: dict):
    predictions = get_predictions()
    if "id" not in prediction:
        prediction["id"] = generate_id("pred", [p["id"] for p in predictions])
    if "timestamp" not in prediction:
        prediction["timestamp"] = iso_now()
    predictions.append(prediction)
    save_json(PREDICTIONS_PATH, predictions)
    return prediction


def update_prediction(pred_id: str, updates: dict):
    predictions = get_predictions()
    for p in predictions:
        if p["id"] == pred_id:
            p.update(updates)
            break
    save_json(PREDICTIONS_PATH, predictions)


# --- Strategy Scores ---

def get_strategy_scores() -> dict:
    scores = load_json(STRATEGY_SCORES_PATH)
    if not scores or "strategies" not in scores:
        scores = _default_strategy_scores()
        save_json(STRATEGY_SCORES_PATH, scores)
    return scores


def update_strategy_scores(scores: dict):
    scores["last_updated"] = iso_now()
    save_json(STRATEGY_SCORES_PATH, scores)


# --- Research ---

def get_research(topic: str) -> dict:
    path = RESEARCH_DIR / f"{topic}.json"
    return load_json(path, default={"topic": topic, "last_updated": None, "findings": []})


def add_research(topic: str, findings: dict):
    path = RESEARCH_DIR / f"{topic}.json"
    data = get_research(topic)
    data["last_updated"] = iso_now()
    if isinstance(findings, dict):
        data["findings"].append({"timestamp": iso_now(), **findings})
    save_json(path, data)
    logger.info(f"Added research: {topic}")


# --- TSLA Profile ---

def get_tsla_profile() -> dict:
    return load_json(TSLA_PROFILE_PATH, default=_default_tsla_profile())


def update_tsla_profile(updates: dict):
    profile = get_tsla_profile()
    profile.update(updates)
    profile["last_updated"] = iso_now()
    save_json(TSLA_PROFILE_PATH, profile)


# --- Journal ---

def get_journal() -> str:
    if JOURNAL_PATH.exists():
        return JOURNAL_PATH.read_text(encoding="utf-8")
    return ""


def append_journal(entry: str):
    current = get_journal()
    timestamp = iso_now()
    new_entry = f"\n---\n\n## {timestamp}\n\n{entry}\n"
    JOURNAL_PATH.write_text(current + new_entry, encoding="utf-8")
    logger.info("Journal entry added")


# --- Knowledge Retrieval ---

def apply_lesson_decay(decay_rate: float = 0.95) -> int:
    """Apply weekly decay to all lessons. Archive if weight < 0.3.

    Returns number of lessons archived.
    """
    lessons = get_lessons()
    archived = 0
    for lesson in lessons:
        weight = lesson.get("weight", 1.0)
        new_weight = round(weight * decay_rate, 4)
        lesson["weight"] = new_weight
        if new_weight < 0.3 and not lesson.get("archived"):
            lesson["archived"] = True
            archived += 1

    save_json(LESSONS_PATH, lessons)
    if archived > 0:
        logger.info(f"Lesson decay: {archived} lessons archived (weight < 0.3)")
    return archived


def get_relevant_knowledge(market_context: dict = None) -> dict:
    """Retrieve knowledge relevant to current trading decisions.

    v4: tag-based matching alongside regime filtering. Quality-based scoring
    with validation boost and age decay (Proposals 1, 2).
    """
    from datetime import datetime, timezone

    lessons = get_lessons()
    patterns = get_patterns()
    scores = get_strategy_scores()
    profile = get_tsla_profile()

    # Filter out archived and disabled lessons
    active_lessons = [
        l for l in lessons
        if not l.get("archived") and not l.get("disabled")
    ]

    now = datetime.now(timezone.utc)

    # Extract context tags from market_context for tag-based matching
    context_tags = set()
    if market_context:
        regime = market_context.get("regime", {})
        if regime.get("trend"):
            context_tags.add(f"regime_{regime['trend']}")
        if regime.get("volatility"):
            context_tags.add(f"vol_{regime['volatility']}")

        # Parse indicators for tag matching
        daily = market_context.get("daily_indicators", {})
        rsi = daily.get("rsi_14")
        if rsi is not None:
            if rsi < 30:
                context_tags.add("rsi_oversold")
            elif rsi > 70:
                context_tags.add("rsi_overbought")

        macro = market_context.get("macro", {})
        vix = macro.get("vix_level") or regime.get("vix", 0)
        if vix > 25:
            context_tags.add("vix_high")

    def relevance_score(lesson):
        """Quality-based scoring: weight * validation_boost / age_decay."""
        base = lesson.get("weight", 1.0)

        # Validation boost: each validation adds 10%
        times_validated = lesson.get("times_validated", 0)
        validation_factor = 1 + times_validated * 0.1

        # Age decay: penalize older lessons slightly
        age_weeks = 0
        ts = lesson.get("timestamp")
        if ts:
            try:
                created = datetime.fromisoformat(ts)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_weeks = (now - created).days / 7
            except (ValueError, TypeError):
                pass
        age_factor = 1 + age_weeks * 0.05

        score = base * validation_factor / age_factor

        # Regime match boost
        lr = lesson.get("regime", {})
        if market_context and "regime" in market_context:
            current_regime = market_context["regime"]
            if lr.get("trend") == current_regime.get("trend"):
                score *= 1.5
            if lr.get("volatility") == current_regime.get("volatility"):
                score *= 1.3

        # Tag match boost
        lesson_tags = set(lesson.get("tags", []))
        if context_tags and lesson_tags:
            overlap = len(context_tags & lesson_tags)
            score *= (1 + overlap * 0.2)

        return score

    active_lessons.sort(key=relevance_score, reverse=True)

    recent_lessons = active_lessons[:10]
    active_patterns = [p for p in patterns if p.get("reliability", 0) > 0.3]

    return {
        "lessons": recent_lessons,
        "patterns": active_patterns,
        "strategy_scores": scores["strategies"],
        "tsla_profile": profile,
        "prediction_accuracy": get_prediction_accuracy(),
    }


# --- Replay Lesson Management ---

def purge_replay_lessons(session_id: str = None) -> int:
    """Remove all replay-sourced lessons, optionally filtered by session.

    Returns the number of lessons removed.
    """
    lessons = get_lessons()
    original_count = len(lessons)

    if session_id:
        filtered = [l for l in lessons if l.get("_replay_source") != session_id]
    else:
        filtered = [l for l in lessons if l.get("source") != "replay"]

    removed = original_count - len(filtered)
    if removed > 0:
        save_json(LESSONS_PATH, filtered)
        tag = f" (session {session_id})" if session_id else ""
        logger.info(f"Purged {removed} replay lessons{tag}")

    return removed


def promote_replay_lesson(lesson_id: str, boost: float = 0.05) -> bool:
    """Boost a replay lesson's weight toward 1.0 and upgrade its decay rate.

    Called when a live trade cites a replay lesson and the trade succeeds.
    Returns True if lesson was found and promoted.
    """
    lessons = get_lessons()
    for lesson in lessons:
        if lesson.get("id") == lesson_id:
            old_weight = lesson.get("weight", 0.7)
            lesson["weight"] = min(1.0, round(old_weight + boost, 4))
            lesson["decay_rate"] = max(lesson.get("decay_rate", 0.90), 0.95)
            lesson["times_validated"] = lesson.get("times_validated", 0) + 1
            lesson["last_validated"] = iso_now()
            save_json(LESSONS_PATH, lessons)
            logger.info(
                f"Promoted replay lesson {lesson_id}: "
                f"weight {old_weight:.3f} -> {lesson['weight']:.3f}, "
                f"decay_rate -> {lesson['decay_rate']}"
            )
            return True
    return False


def disable_lesson(lesson_id: str) -> bool:
    """Disable a lesson so it's excluded from retrieval."""
    lessons = get_lessons()
    for lesson in lessons:
        if lesson.get("id") == lesson_id:
            lesson["disabled"] = True
            save_json(LESSONS_PATH, lessons)
            logger.info(f"Disabled lesson {lesson_id}")
            return True
    logger.warning(f"Lesson {lesson_id} not found")
    return False


def enable_lesson(lesson_id: str) -> bool:
    """Re-enable a disabled lesson."""
    lessons = get_lessons()
    for lesson in lessons:
        if lesson.get("id") == lesson_id:
            lesson.pop("disabled", None)
            save_json(LESSONS_PATH, lessons)
            logger.info(f"Enabled lesson {lesson_id}")
            return True
    logger.warning(f"Lesson {lesson_id} not found")
    return False


def get_prediction_accuracy() -> dict:
    """Calculate rolling accuracy stats, including per-strategy breakdown."""
    predictions = get_predictions()
    if not predictions:
        return {
            "total_predictions": 0,
            "scored_predictions": 0,
            "direction_accuracy": {},
            "strategy_accuracy": {},
            "note": "No predictions yet",
        }

    horizons = {"30min": [], "2hr": [], "1day": []}
    for p in predictions:
        outcomes = p.get("outcomes", {})
        for horizon, key in [("30min", "30min"), ("2hr", "2hr"), ("1day", "1day")]:
            if outcomes.get(key) and outcomes[key].get("direction_correct") is not None:
                horizons[horizon].append(outcomes[key]["direction_correct"])

    accuracy = {}
    for horizon, results in horizons.items():
        if results:
            accuracy[horizon] = {
                "correct": sum(results),
                "total": len(results),
                "accuracy_pct": round(sum(results) / len(results) * 100, 1),
            }

    # Per-strategy accuracy: link predictions to trades via linked_trade
    strategy_accuracy = _compute_strategy_accuracy(predictions)

    return {
        "total_predictions": len(predictions),
        "scored_predictions": sum(len(v) for v in horizons.values()),
        "direction_accuracy": accuracy,
        "strategy_accuracy": strategy_accuracy,
    }


def _compute_strategy_accuracy(predictions: list) -> dict:
    """Compute prediction accuracy broken down by strategy.

    Links predictions to their trades (via linked_trade field), looks up
    the strategy from the transaction, and aggregates accuracy per strategy.
    """
    from .portfolio import load_transactions

    # Build a map of trade_id -> strategy
    transactions = load_transactions()
    trade_strategy = {t["id"]: t.get("strategy", "unknown") for t in transactions}

    strategy_results: dict[str, list[bool]] = {}
    for pred in predictions:
        trade_id = pred.get("linked_trade")
        if not trade_id:
            continue
        strategy = trade_strategy.get(trade_id, "unknown")
        if strategy == "unknown":
            continue

        # Aggregate all scored horizons for this prediction
        outcomes = pred.get("outcomes", {})
        for horizon in ("30min", "2hr", "1day"):
            outcome = outcomes.get(horizon)
            if outcome and outcome.get("direction_correct") is not None:
                strategy_results.setdefault(strategy, []).append(outcome["direction_correct"])

    result = {}
    for strategy, results in strategy_results.items():
        if results:
            result[strategy] = {
                "correct": sum(results),
                "total": len(results),
                "accuracy_pct": round(sum(results) / len(results) * 100, 1),
            }
    return result


def get_relevant_research() -> str:
    """Retrieve latest findings from all research topics for the Analyst prompt.

    Reads from knowledge/research/ and returns a formatted text summary
    of the most recent findings from each topic.
    """
    topics = [
        "earnings_history",
        "catalyst_events",
        "correlation_notes",
        "sector_context",
        "seasonal_patterns",
    ]

    parts = []
    for topic in topics:
        data = get_research(topic)
        findings = data.get("findings", [])
        if not findings:
            continue

        # Get the most recent finding
        latest = findings[-1]
        last_updated = data.get("last_updated", "unknown")
        parts.append(f"[{topic}] (updated: {last_updated})")

        # Format the finding — handle both dict and string findings
        if isinstance(latest, dict):
            # Remove timestamp from display (already shown in header)
            display = {k: v for k, v in latest.items() if k != "timestamp"}
            # Truncate long values
            for k, v in display.items():
                if isinstance(v, str) and len(v) > 200:
                    display[k] = v[:200] + "..."
                elif isinstance(v, list) and len(v) > 5:
                    display[k] = v[:5] + ["..."]
            import json
            parts.append(f"  {json.dumps(display, default=str)[:500]}")
        else:
            parts.append(f"  {str(latest)[:500]}")

    if not parts:
        return "No research findings available yet."

    return "\n".join(parts)


def get_knowledge_summary() -> str:
    """One-paragraph summary of what the agent has learned so far."""
    lessons = get_lessons()
    patterns = get_patterns()
    accuracy = get_prediction_accuracy()
    scores = get_strategy_scores()

    parts = [f"MonopolyTrader has {len(lessons)} lessons and {len(patterns)} patterns."]

    if accuracy["scored_predictions"] > 0:
        for h, data in accuracy["direction_accuracy"].items():
            parts.append(f"{h} accuracy: {data['accuracy_pct']}% ({data['total']} predictions)")

    # Best/worst strategy
    strats = scores.get("strategies", {})
    if any(s["total_trades"] > 0 for s in strats.values()):
        best = max(strats.items(), key=lambda x: x[1].get("win_rate", 0))
        parts.append(f"Best strategy: {best[0]} ({best[1]['win_rate']*100:.0f}% win rate)")

    return " ".join(parts)
