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
            "sentiment": {
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
            "dca": {
                "weight": 0.20, "initial_weight": 0.20,
                "total_trades": 0, "winning_trades": 0,
                "win_rate": 0.0, "avg_return_pct": 0.0,
                "total_pnl": 0.0, "trend": "neutral", "notes": "",
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
    for name in ["earnings_history", "catalyst_events", "correlation_notes", "sector_context"]:
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

def get_relevant_knowledge(market_context: dict = None) -> dict:
    """Retrieve knowledge relevant to current trading decisions.

    Returns a condensed bundle for the agent prompt.
    """
    lessons = get_lessons()
    patterns = get_patterns()
    scores = get_strategy_scores()
    profile = get_tsla_profile()

    # For now, return the most recent lessons and all patterns.
    # Phase 3 will add smarter relevance filtering.
    recent_lessons = lessons[-10:] if lessons else []
    active_patterns = [p for p in patterns if p.get("reliability", 0) > 0.3]

    return {
        "lessons": recent_lessons,
        "patterns": active_patterns,
        "strategy_scores": scores["strategies"],
        "tsla_profile": profile,
        "prediction_accuracy": get_prediction_accuracy(),
    }


def get_prediction_accuracy() -> dict:
    """Calculate rolling accuracy stats."""
    predictions = get_predictions()
    if not predictions:
        return {
            "total_predictions": 0,
            "scored_predictions": 0,
            "direction_accuracy": {},
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

    return {
        "total_predictions": len(predictions),
        "scored_predictions": sum(len(v) for v in horizons.values()),
        "direction_accuracy": accuracy,
    }


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
