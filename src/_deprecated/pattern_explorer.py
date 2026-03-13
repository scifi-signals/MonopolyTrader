"""Pattern Explorer — active experiment design for the researcher.

Maps the tag space, identifies conditions that have been observed (HOLDs)
but never traded, ranks them by potential signal strength, and designs
micro-experiments with specific hypotheses and small position sizes.

This turns the researcher from passive ("I'll hold until something changes")
to active ("Under these specific conditions, I predict X — let's test with
a small position").
"""

from itertools import combinations
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("pattern_explorer")

EXPLORATION_MAP_PATH = DATA_DIR / "exploration_map.json"
HOLD_JOURNAL_PATH = DATA_DIR / "hold_journal.json"
LEDGER_PATH = DATA_DIR / "thesis_ledger.json"
PREDICTIONS_PATH = DATA_DIR / "predictions.json"
JOURNAL_PATH = DATA_DIR / "trade_journal.json"

# All known tag names and their possible values
TAG_SPACE = {
    "rsi_zone": ["oversold", "neutral", "overbought"],
    "trend": ["above_sma50", "below_sma50"],
    "volatility": ["low_vix", "normal_vix", "high_vix"],
    "regime": ["trending", "range_bound", "unknown"],
    "macd": ["bullish_cross", "bearish_cross", "neutral"],
    "market_context": ["spy_up", "spy_flat", "spy_down"],
    "position_state": [
        "flat", "opening_new", "holding", "adding_to_winner",
        "adding_to_loser", "taking_profit", "cutting_loss",
    ],
    "event_proximity": ["no_event", "pre_event_72h", "pre_event_24h"],
    "time_of_day": ["morning_open", "midday", "afternoon", "power_hour"],
    "intraday_daily_divergence": ["aligned", "mild_divergence", "strong_divergence", "no_data"],
    "news_catalyst": ["none", "earnings", "product", "macro", "analyst", "legal"],
    "options_sentiment": ["call_heavy", "balanced", "put_heavy", "unavailable"],
    "intraday_regime": ["trending", "range_bound", "unavailable"],
    "regime_age": ["new", "established", "mature", "unknown"],
    "hold_duration": ["quick", "medium", "extended", "overnight"],
}

# Minimum observations before we consider a condition "explored"
MIN_TRADES_EXPLORED = 3
MIN_HOLDS_FOR_SIGNAL = 5

# Candidate scoring weights
FREQUENCY_WEIGHT = 0.3
PREDICTION_ACCURACY_WEIGHT = 0.3
SHADOW_SIGNAL_WEIGHT = 0.4


def map_tag_space() -> dict:
    """Enumerate all single-tag values and compute coverage.

    Returns dict with total possible, observed in trades, observed in HOLDs,
    and unobserved conditions.
    """
    ledger = load_json(LEDGER_PATH, default={})
    theses = ledger.get("theses", {})
    hold_journal = load_json(HOLD_JOURNAL_PATH, default=[])

    # All possible single-tag values
    all_conditions = set()
    for tag_name, values in TAG_SPACE.items():
        for value in values:
            all_conditions.add(f"{tag_name}={value}")

    # Conditions observed in trades (from playbook)
    traded_conditions = set()
    for key in theses:
        # Playbook uses "tag:value" format
        traded_conditions.add(key.replace(":", "="))

    # Conditions observed in HOLDs
    held_conditions = set()
    for entry in hold_journal:
        tags = entry.get("tags", {})
        for tag_name, tag_value in tags.items():
            held_conditions.add(f"{tag_name}={tag_value}")

    return {
        "total_possible": len(all_conditions),
        "traded": len(traded_conditions),
        "held_only": len(held_conditions - traded_conditions),
        "unobserved": len(all_conditions - traded_conditions - held_conditions),
        "traded_conditions": sorted(traded_conditions),
        "held_only_conditions": sorted(held_conditions - traded_conditions),
        "unobserved_conditions": sorted(all_conditions - traded_conditions - held_conditions),
    }


def identify_exploration_gaps(min_trades: int = MIN_TRADES_EXPLORED, min_holds: int = MIN_HOLDS_FOR_SIGNAL) -> list:
    """Identify conditions that are frequent in HOLDs but rarely/never traded.

    These are the best exploration candidates — the researcher sees these
    conditions regularly but has no data on what happens when trading them.

    Returns list of gap dicts sorted by hold frequency.
    """
    ledger = load_json(LEDGER_PATH, default={})
    theses = ledger.get("theses", {})
    hold_journal = load_json(HOLD_JOURNAL_PATH, default=[])

    # Count HOLDs per single-tag condition
    hold_counts = {}
    hold_outcomes = {}  # track shadow P&L outcomes per condition

    for entry in hold_journal:
        tags = entry.get("tags", {})
        shadow_pnl = entry.get("shadow_pnl_buy")
        resolved = entry.get("shadow_resolved", False)

        for tag_name, tag_value in tags.items():
            key = f"{tag_name}={tag_value}"
            hold_counts[key] = hold_counts.get(key, 0) + 1

            if resolved and shadow_pnl is not None:
                if key not in hold_outcomes:
                    hold_outcomes[key] = {"positive": 0, "negative": 0, "total": 0}
                hold_outcomes[key]["total"] += 1
                if shadow_pnl > 0:
                    hold_outcomes[key]["positive"] += 1
                else:
                    hold_outcomes[key]["negative"] += 1

    # Also count 2-tag combo HOLDs
    combo_hold_counts = {}
    combo_hold_outcomes = {}
    for entry in hold_journal:
        tags = entry.get("tags", {})
        shadow_pnl = entry.get("shadow_pnl_buy")
        resolved = entry.get("shadow_resolved", False)
        tag_items = sorted(tags.items())

        for pair in combinations(tag_items, 2):
            combo_key = " + ".join(f"{k}={v}" for k, v in pair)
            combo_hold_counts[combo_key] = combo_hold_counts.get(combo_key, 0) + 1

            if resolved and shadow_pnl is not None:
                if combo_key not in combo_hold_outcomes:
                    combo_hold_outcomes[combo_key] = {"positive": 0, "negative": 0, "total": 0}
                combo_hold_outcomes[combo_key]["total"] += 1
                if shadow_pnl > 0:
                    combo_hold_outcomes[combo_key]["positive"] += 1
                else:
                    combo_hold_outcomes[combo_key]["negative"] += 1

    gaps = []

    # Single-tag gaps
    for key, hold_count in hold_counts.items():
        if hold_count < min_holds:
            continue

        # Check if traded enough
        playbook_key = key.replace("=", ":")
        trade_count = theses.get(playbook_key, {}).get("trades", 0)
        if trade_count >= min_trades:
            continue  # Already explored

        outcomes = hold_outcomes.get(key, {})
        shadow_positive_rate = (
            outcomes.get("positive", 0) / outcomes["total"]
            if outcomes.get("total", 0) > 0 else None
        )

        gaps.append({
            "condition": key,
            "combo_size": 1,
            "hold_count": hold_count,
            "trade_count": trade_count,
            "shadow_positive_rate": round(shadow_positive_rate, 3) if shadow_positive_rate is not None else None,
            "shadow_observations": outcomes.get("total", 0),
        })

    # 2-tag combo gaps
    multi = ledger.get("multi_tag_patterns", {})
    for key, hold_count in combo_hold_counts.items():
        if hold_count < min_holds:
            continue

        trade_count = multi.get(key, {}).get("trades", 0)
        if trade_count >= min_trades:
            continue

        outcomes = combo_hold_outcomes.get(key, {})
        shadow_positive_rate = (
            outcomes.get("positive", 0) / outcomes["total"]
            if outcomes.get("total", 0) > 0 else None
        )

        gaps.append({
            "condition": key,
            "combo_size": 2,
            "hold_count": hold_count,
            "trade_count": trade_count,
            "shadow_positive_rate": round(shadow_positive_rate, 3) if shadow_positive_rate is not None else None,
            "shadow_observations": outcomes.get("total", 0),
        })

    # Sort by hold frequency (most observed first)
    gaps.sort(key=lambda g: -g["hold_count"])
    return gaps


def rank_exploration_candidates(gaps: list = None) -> list:
    """Score and rank exploration gaps by potential value.

    Scoring:
    - Frequency (30%): how often this condition appears
    - Prediction accuracy (30%): how well predictions work here
    - Shadow signal (40%): whether shadow P&L suggests profitability

    Returns top 10 ranked candidates.
    """
    if gaps is None:
        gaps = identify_exploration_gaps()

    if not gaps:
        return []

    # Get prediction accuracy by condition
    predictions = load_json(PREDICTIONS_PATH, default=[])
    pred_accuracy = _compute_prediction_accuracy_by_condition(predictions)

    # Normalize frequency scores
    max_hold_count = max(g["hold_count"] for g in gaps) if gaps else 1

    scored = []
    for gap in gaps:
        # Frequency score (0-1)
        freq_score = gap["hold_count"] / max_hold_count

        # Prediction accuracy score (0-1, defaults to 0.5 if no data)
        cond_parts = [p.strip() for p in gap["condition"].split("+")]
        pred_scores = []
        for part in cond_parts:
            if part in pred_accuracy and pred_accuracy[part]["total"] >= 5:
                pred_scores.append(pred_accuracy[part]["accuracy"])
        pred_score = sum(pred_scores) / len(pred_scores) if pred_scores else 0.5

        # Shadow signal score (0-1): higher if shadow P&L is positive
        shadow_rate = gap.get("shadow_positive_rate")
        if shadow_rate is not None and gap.get("shadow_observations", 0) >= 3:
            shadow_score = shadow_rate
        else:
            shadow_score = 0.5  # No data — neutral

        # Combined score
        total_score = (
            FREQUENCY_WEIGHT * freq_score
            + PREDICTION_ACCURACY_WEIGHT * pred_score
            + SHADOW_SIGNAL_WEIGHT * shadow_score
        )

        scored.append({
            **gap,
            "frequency_score": round(freq_score, 3),
            "prediction_accuracy_score": round(pred_score, 3),
            "shadow_signal_score": round(shadow_score, 3),
            "exploration_score": round(total_score, 3),
        })

    scored.sort(key=lambda s: -s["exploration_score"])
    return scored[:10]


def design_micro_experiment(candidate: dict) -> dict:
    """Design a micro-experiment for a given exploration candidate.

    Returns a structured experiment with:
    - hypothesis, position size, success criteria, max duration
    """
    condition = candidate["condition"]
    shadow_rate = candidate.get("shadow_positive_rate")
    score = candidate.get("exploration_score", 0)

    # Determine direction from shadow signal
    if shadow_rate is not None and shadow_rate > 0.55:
        direction = "BUY"
        hypothesis = (
            f"Under {condition}, shadow P&L is positive {shadow_rate:.0%} of the time — "
            f"test with a small position"
        )
    elif shadow_rate is not None and shadow_rate < 0.45:
        direction = "cautious"
        hypothesis = (
            f"Under {condition}, shadow P&L is negative {1 - shadow_rate:.0%} of the time — "
            f"but untested in real trades. Small exploratory position to gather data."
        )
    else:
        direction = "exploratory"
        hypothesis = (
            f"Under {condition}, no clear shadow signal — "
            f"pure exploration to gather first data point"
        )

    # Position size: always small for exploration
    position_size = 0.25 if score > 0.6 else 0.15

    return {
        "condition": condition,
        "hypothesis": hypothesis,
        "direction": direction,
        "position_size_pct": position_size,
        "max_duration_cycles": 8,  # 2 hours max
        "success_criteria": (
            f"Close profitable within {8 * 15} minutes, OR "
            f"gather data on how this condition resolves"
        ),
        "expected_learning": (
            f"First real data on trading under {condition} — "
            f"currently {candidate.get('hold_count', 0)} HOLDs, "
            f"{candidate.get('trade_count', 0)} trades"
        ),
        "exploration_score": score,
    }


def format_explorer_for_brief(current_tags: dict = None) -> str:
    """Format exploration opportunities for the agent's brief.

    If current conditions match a top candidate, show the experiment.
    Returns compact text (2-4 lines) or empty string.
    """
    data = load_json(EXPLORATION_MAP_PATH, default={})
    candidates = data.get("ranked_candidates", [])

    if not candidates:
        # Show coverage stats if available
        coverage = data.get("tag_space_coverage", {})
        if coverage:
            traded = coverage.get("traded", 0)
            total = coverage.get("total_possible", 0)
            held = coverage.get("held_only", 0)
            if total > 0:
                return (
                    f"Exploration: {traded}/{total} conditions traded, "
                    f"{held} observed in HOLDs but never traded"
                )
        return ""

    parts = []

    # Check if current conditions match any candidate
    matching = []
    if current_tags:
        for candidate in candidates:
            if _candidate_matches_tags(candidate, current_tags):
                matching.append(candidate)

    if matching:
        top = matching[0]
        experiment = design_micro_experiment(top)
        parts.append(
            f"EXPERIMENT OPPORTUNITY (score {top['exploration_score']:.2f}):"
        )
        parts.append(f"  {experiment['hypothesis']}")
        parts.append(
            f"  Size: {experiment['position_size_pct']:.0%} of max | "
            f"Max {experiment['max_duration_cycles']} cycles | "
            f"Data: {top['hold_count']} HOLDs, {top['trade_count']} trades"
        )
    else:
        # Show top candidate even if not matching current conditions
        top = candidates[0]
        parts.append(
            f"Top exploration target: {top['condition']} "
            f"(score {top['exploration_score']:.2f}, "
            f"{top['hold_count']} HOLDs, {top['trade_count']} trades)"
        )

    # Show coverage summary
    coverage = data.get("tag_space_coverage", {})
    if coverage:
        traded = coverage.get("traded", 0)
        total = coverage.get("total_possible", 0)
        gaps = len(data.get("exploration_gaps", []))
        if total > 0:
            parts.append(
                f"  Coverage: {traded}/{total} conditions traded, "
                f"{gaps} gaps identified"
            )

    return "\n".join(parts) if parts else ""


def rebuild_exploration_map() -> dict:
    """Rebuild the full exploration map and save to disk. Called nightly."""
    tag_space = map_tag_space()
    gaps = identify_exploration_gaps()
    candidates = rank_exploration_candidates(gaps)

    result = {
        "last_updated": iso_now(),
        "tag_space_coverage": tag_space,
        "exploration_gaps": gaps,
        "ranked_candidates": candidates,
        "experiments": [design_micro_experiment(c) for c in candidates[:5]],
    }

    save_json(EXPLORATION_MAP_PATH, result)
    logger.info(
        f"Exploration map rebuilt: {tag_space['traded']}/{tag_space['total_possible']} "
        f"conditions traded, {len(gaps)} gaps, {len(candidates)} candidates ranked"
    )
    return result


def _compute_prediction_accuracy_by_condition(predictions: list) -> dict:
    """Compute prediction direction accuracy grouped by single-tag conditions."""
    accuracy = {}

    for p in predictions:
        if not p.get("resolved") or not p.get("result"):
            continue
        if p["result"].get("error"):
            continue

        tags = p.get("tags", {})
        dir_correct = p["result"].get("direction_correct", False)

        for tag_name, tag_value in tags.items():
            key = f"{tag_name}={tag_value}"
            if key not in accuracy:
                accuracy[key] = {"total": 0, "correct": 0}
            accuracy[key]["total"] += 1
            if dir_correct:
                accuracy[key]["correct"] += 1

    for key, stats in accuracy.items():
        stats["accuracy"] = (
            round(stats["correct"] / stats["total"], 3)
            if stats["total"] > 0 else 0
        )

    return accuracy


def _candidate_matches_tags(candidate: dict, current_tags: dict) -> bool:
    """Check if a candidate's condition matches current tag values."""
    condition = candidate.get("condition", "")
    parts = [p.strip() for p in condition.split("+")]
    for part in parts:
        if "=" in part:
            tag_name, tag_value = part.split("=", 1)
            tag_name = tag_name.strip()
            tag_value = tag_value.strip()
            if current_tags.get(tag_name) != tag_value:
                return False
    return True
