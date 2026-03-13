"""Prediction tracker — learn from every cycle, not just trades.

Every 15-min cycle, Claude predicts TSLA direction + magnitude over the next
1-4 cycles. Predictions are scored against reality and accuracy data feeds
back into the agent's brief, telling it where its understanding is strong vs weak.

This turns 26 daily cycles into 26 learning opportunities instead of ~1 from trades.
"""

from datetime import datetime, timezone, timedelta
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("prediction_tracker")

PREDICTIONS_PATH = DATA_DIR / "predictions.json"

# Magnitude thresholds (absolute % move)
FLAT_THRESHOLD_PCT = 0.2
SMALL_THRESHOLD_PCT = 0.5
MODERATE_THRESHOLD_PCT = 1.5
# Above moderate = large

# Scoring weights
DIRECTION_WEIGHT = 0.6
MAGNITUDE_WEIGHT = 0.3
BONUS_WEIGHT = 0.1


def _classify_magnitude(change_pct: float) -> str:
    """Classify a percent change into magnitude bucket."""
    abs_change = abs(change_pct)
    if abs_change < FLAT_THRESHOLD_PCT:
        return "flat"
    elif abs_change < SMALL_THRESHOLD_PCT:
        return "small"
    elif abs_change < MODERATE_THRESHOLD_PCT:
        return "moderate"
    else:
        return "large"


def _classify_direction(change_pct: float) -> str:
    """Classify a percent change into direction."""
    if change_pct > FLAT_THRESHOLD_PCT:
        return "up"
    elif change_pct < -FLAT_THRESHOLD_PCT:
        return "down"
    else:
        return "flat"


def _score_prediction(predicted: dict, actual_change_pct: float) -> dict:
    """Score a prediction against the actual outcome.

    Returns result dict with actual values and scores.
    """
    actual_dir = _classify_direction(actual_change_pct)
    actual_mag = _classify_magnitude(actual_change_pct)

    pred_dir = predicted.get("direction", "flat")
    pred_mag = predicted.get("magnitude", "small")

    # Direction scoring
    if pred_dir == "flat" and actual_dir == "flat":
        direction_correct = True
    elif pred_dir == actual_dir:
        direction_correct = True
    elif pred_dir == "flat" and abs(actual_change_pct) < SMALL_THRESHOLD_PCT:
        # "Flat" prediction and actual was small — partially correct
        direction_correct = True
    else:
        direction_correct = False

    # Magnitude scoring (only if direction is correct)
    magnitude_correct = False
    if direction_correct:
        if pred_dir == "flat":
            magnitude_correct = actual_mag in ("flat", "small")
        else:
            magnitude_correct = pred_mag == actual_mag

    # Compute score
    score = 0.0
    if direction_correct:
        score += DIRECTION_WEIGHT
        if magnitude_correct:
            score += MAGNITUDE_WEIGHT + BONUS_WEIGHT
        else:
            # Partial credit: right direction, adjacent magnitude
            mag_order = ["flat", "small", "moderate", "large"]
            if pred_mag in mag_order and actual_mag in mag_order:
                diff = abs(mag_order.index(pred_mag) - mag_order.index(actual_mag))
                if diff == 1:
                    score += MAGNITUDE_WEIGHT * 0.5  # Partial magnitude credit

    return {
        "actual_change_pct": round(actual_change_pct, 3),
        "actual_direction": actual_dir,
        "actual_magnitude": actual_mag,
        "direction_correct": direction_correct,
        "magnitude_correct": magnitude_correct,
        "score": round(score, 3),
    }


def log_prediction(decision: dict, market_data: dict, tags: dict, current_price: float):
    """Log a prediction from Claude's decision. Called every cycle.

    Args:
        decision: Claude's full decision dict (must include 'prediction' key)
        market_data: Current market data
        tags: Current cycle tags
        current_price: TSLA price at prediction time
    """
    predictions = load_json(PREDICTIONS_PATH, default=[])

    pred = decision.get("prediction")
    if not pred or not isinstance(pred, dict):
        # Claude didn't provide a prediction — log a default
        pred = {
            "direction": "flat",
            "magnitude": "small",
            "cycles": 1,
            "basis": "no prediction provided",
        }
        logger.debug("No prediction in decision, using default")

    # Validate and clamp
    direction = pred.get("direction", "flat")
    if direction not in ("up", "down", "flat"):
        direction = "flat"

    magnitude = pred.get("magnitude", "small")
    if magnitude not in ("small", "moderate", "large"):
        magnitude = "small"
    # If direction is flat, magnitude is irrelevant
    if direction == "flat":
        magnitude = "flat"

    cycles = pred.get("cycles", 2)
    if not isinstance(cycles, (int, float)):
        cycles = 2
    cycles = max(1, min(4, int(cycles)))

    basis = str(pred.get("basis", ""))[:200]

    now = datetime.now(timezone.utc)
    resolve_after = now + timedelta(minutes=15 * cycles)

    # Generate ID
    existing_ids = {p.get("id", "") for p in predictions}
    pred_id = f"pred_{len(predictions) + 1:04d}"
    while pred_id in existing_ids:
        pred_id = f"pred_{int(pred_id.split('_')[1]) + 1:04d}"

    entry = {
        "id": pred_id,
        "timestamp": iso_now(),
        "price_at_prediction": round(current_price, 2),
        "direction": direction,
        "magnitude": magnitude,
        "cycles": cycles,
        "basis": basis,
        "resolve_after": resolve_after.isoformat(),
        "tags": tags,
        "action_this_cycle": decision.get("action", "HOLD"),
        "price_updates": [],
        "resolved": False,
        "result": None,
    }

    predictions.append(entry)
    save_json(PREDICTIONS_PATH, predictions)

    logger.info(
        f"Prediction logged: {direction}/{magnitude} in {cycles} cycles "
        f"(basis: {basis[:60]})"
    )


def update_predictions(current_price: float) -> int:
    """Update pending predictions with current price. Resolve expired ones.

    Called every cycle (including during cooldown).
    Returns the number of predictions resolved this cycle.
    """
    predictions = load_json(PREDICTIONS_PATH, default=[])
    if not predictions:
        return 0

    now = datetime.now(timezone.utc)
    resolved_count = 0
    changed = False

    for entry in predictions:
        if entry.get("resolved"):
            continue

        # Update price tracking (cap at 10 updates)
        if len(entry.get("price_updates", [])) < 10:
            entry.setdefault("price_updates", []).append(round(current_price, 2))
            changed = True

        # Check if it's time to resolve
        resolve_time = datetime.fromisoformat(entry["resolve_after"])
        if resolve_time.tzinfo is None:
            resolve_time = resolve_time.replace(tzinfo=timezone.utc)

        if now >= resolve_time:
            _resolve_prediction(entry, current_price)
            resolved_count += 1
            changed = True

    if changed:
        save_json(PREDICTIONS_PATH, predictions)

    return resolved_count


def _resolve_prediction(entry: dict, final_price: float):
    """Score and resolve a single prediction."""
    entry_price = entry["price_at_prediction"]
    if entry_price <= 0:
        entry["resolved"] = True
        entry["result"] = {"error": "invalid entry price"}
        return

    actual_change_pct = ((final_price - entry_price) / entry_price) * 100

    result = _score_prediction(
        {"direction": entry["direction"], "magnitude": entry["magnitude"]},
        actual_change_pct,
    )
    result["final_price"] = round(final_price, 2)

    entry["resolved"] = True
    entry["result"] = result

    logger.info(
        f"Prediction {entry['id']} resolved: "
        f"predicted {entry['direction']}/{entry['magnitude']}, "
        f"actual {result['actual_direction']}/{result['actual_magnitude']} "
        f"({actual_change_pct:+.2f}%), "
        f"score={result['score']:.2f}, "
        f"direction={'correct' if result['direction_correct'] else 'WRONG'}"
    )


def get_prediction_summary(hours: int = 72) -> dict:
    """Aggregate prediction stats from recent resolved predictions.

    Args:
        hours: Look back this many hours.
    """
    predictions = load_json(PREDICTIONS_PATH, default=[])
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    resolved = []
    for p in predictions:
        if not p.get("resolved") or not p.get("result"):
            continue
        if p["result"].get("error"):
            continue
        ts = datetime.fromisoformat(p["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            resolved.append(p)

    if not resolved:
        return {"resolved": 0, "total_predictions": len(predictions)}

    direction_correct = sum(1 for p in resolved if p["result"]["direction_correct"])
    magnitude_correct = sum(1 for p in resolved if p["result"]["magnitude_correct"])
    scores = [p["result"]["score"] for p in resolved]

    # Check for systematic magnitude bias
    pred_mags = [p["magnitude"] for p in resolved if p["direction"] != "flat"]
    actual_mags = [p["result"]["actual_magnitude"] for p in resolved if p["direction"] != "flat"]
    mag_order = {"flat": 0, "small": 1, "moderate": 2, "large": 3}
    overpredict = False
    if pred_mags and actual_mags:
        avg_pred = sum(mag_order.get(m, 1) for m in pred_mags) / len(pred_mags)
        avg_actual = sum(mag_order.get(m, 1) for m in actual_mags) / len(actual_mags)
        overpredict = avg_pred > avg_actual + 0.3

    # By direction breakdown
    by_direction = {}
    for d in ("up", "down", "flat"):
        subset = [p for p in resolved if p["direction"] == d]
        if subset:
            correct = sum(1 for p in subset if p["result"]["direction_correct"])
            by_direction[d] = {
                "count": len(subset),
                "correct": correct,
                "accuracy": round(correct / len(subset), 3),
            }

    return {
        "resolved": len(resolved),
        "total_predictions": len(predictions),
        "direction_accuracy": round(direction_correct / len(resolved), 3),
        "magnitude_accuracy": round(magnitude_correct / len(resolved), 3),
        "avg_score": round(sum(scores) / len(scores), 3),
        "overpredict_magnitude": overpredict,
        "by_direction": by_direction,
    }


def get_prediction_accuracy_by_tags(min_count: int = 3) -> dict:
    """Group resolved predictions by tag values, compute accuracy per group.

    Only includes single-tag groupings (12 tag types).
    Returns dict keyed by "tag_name=tag_value".
    """
    predictions = load_json(PREDICTIONS_PATH, default=[])

    # Only use resolved predictions from last 30 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    resolved = []
    for p in predictions:
        if not p.get("resolved") or not p.get("result"):
            continue
        if p["result"].get("error"):
            continue
        ts = datetime.fromisoformat(p["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            resolved.append(p)

    if not resolved:
        return {}

    # Group by single tag values
    tag_groups = {}
    for p in resolved:
        tags = p.get("tags", {})
        for tag_name, tag_value in tags.items():
            key = f"{tag_name}={tag_value}"
            if key not in tag_groups:
                tag_groups[key] = []
            tag_groups[key].append(p)

    # Compute accuracy per group (filter by min_count)
    result = {}
    for key, preds in tag_groups.items():
        if len(preds) < min_count:
            continue

        dir_correct = sum(1 for p in preds if p["result"]["direction_correct"])
        scores = [p["result"]["score"] for p in preds]

        result[key] = {
            "predictions": len(preds),
            "direction_accuracy": round(dir_correct / len(preds), 3),
            "avg_score": round(sum(scores) / len(scores), 3),
        }

    return result


def format_prediction_scorecard(hours: int = 72) -> str:
    """Format prediction accuracy as compact text for the agent's brief.

    Returns empty string if insufficient data (< 5 resolved predictions).
    """
    summary = get_prediction_summary(hours)
    if summary.get("resolved", 0) < 5:
        return ""

    parts = []
    n = summary["resolved"]
    dir_acc = summary["direction_accuracy"]
    mag_acc = summary["magnitude_accuracy"]
    avg_score = summary["avg_score"]

    parts.append(
        f"Direction accuracy: {dir_acc:.0%} ({int(dir_acc * n)}/{n} correct) | "
        f"Magnitude accuracy: {mag_acc:.0%} | "
        f"Avg score: {avg_score:.2f}/1.0"
    )

    # Bias detection
    if summary.get("overpredict_magnitude"):
        parts.append("Bias: You over-predict magnitude — actual moves are smaller than you expect")

    # By-direction breakdown
    by_dir = summary.get("by_direction", {})
    dir_parts = []
    for d in ("up", "down", "flat"):
        if d in by_dir:
            info = by_dir[d]
            dir_parts.append(f"{d}: {info['accuracy']:.0%} ({info['count']})")
    if dir_parts:
        parts.append(f"By direction: {', '.join(dir_parts)}")

    # Best/worst tag conditions
    tag_acc = get_prediction_accuracy_by_tags(min_count=3)
    if tag_acc:
        sorted_tags = sorted(tag_acc.items(), key=lambda x: x[1]["direction_accuracy"], reverse=True)
        best = [(k, v) for k, v in sorted_tags if v["direction_accuracy"] >= 0.60][:3]
        worst = [(k, v) for k, v in sorted_tags if v["direction_accuracy"] <= 0.45][:3]

        if best:
            best_str = ", ".join(f"{k} ({v['direction_accuracy']:.0%})" for k, v in best)
            parts.append(f"Strong reads: {best_str}")
        if worst:
            worst_str = ", ".join(f"{k} ({v['direction_accuracy']:.0%})" for k, v in worst)
            parts.append(f"Weak reads: {worst_str}")

    return "\n".join(parts)


def prune_predictions(days: int = 30):
    """Remove resolved predictions older than N days."""
    predictions = load_json(PREDICTIONS_PATH, default=[])
    if not predictions:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    original_count = len(predictions)

    kept = []
    for p in predictions:
        if not p.get("resolved"):
            kept.append(p)
            continue
        ts = datetime.fromisoformat(p["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            kept.append(p)

    if len(kept) < original_count:
        save_json(PREDICTIONS_PATH, kept)
        logger.info(f"Pruned {original_count - len(kept)} old predictions (kept {len(kept)})")
