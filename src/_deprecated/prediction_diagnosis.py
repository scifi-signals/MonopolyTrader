"""Prediction Diagnosis — learn WHY predictions are right or wrong.

Goes beyond the scorecard (which just reports accuracy) to diagnose
systematic biases, track accuracy by condition, and generate prescriptive
insights that tell the researcher what to change.

Replaces the prediction scorecard in the agent's brief with richer feedback.
"""

from datetime import datetime, timezone, timedelta
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("prediction_diagnosis")

PREDICTIONS_PATH = DATA_DIR / "predictions.json"
DIAGNOSIS_PATH = DATA_DIR / "prediction_diagnosis.json"


def diagnose_predictions(lookback_days: int = 30) -> dict:
    """Full prediction diagnosis with bias detection and prescriptive insights.

    Returns structured diagnosis dict.
    """
    predictions = load_json(PREDICTIONS_PATH, default=[])
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Filter to resolved predictions within window
    resolved = []
    for p in predictions:
        if not p.get("resolved") or not p.get("result"):
            continue
        if p["result"].get("error"):
            continue
        try:
            ts = datetime.fromisoformat(p["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                resolved.append(p)
        except (ValueError, KeyError):
            continue

    if len(resolved) < 5:
        return {
            "last_updated": iso_now(),
            "total_analyzed": len(resolved),
            "insufficient_data": True,
        }

    # === Direction Bias ===
    pred_directions = [p["direction"] for p in resolved]
    actual_directions = [p["result"]["actual_direction"] for p in resolved]

    direction_counts = {"up": 0, "down": 0, "flat": 0}
    actual_counts = {"up": 0, "down": 0, "flat": 0}
    for d in pred_directions:
        direction_counts[d] = direction_counts.get(d, 0) + 1
    for d in actual_directions:
        actual_counts[d] = actual_counts.get(d, 0) + 1

    n = len(resolved)
    direction_bias = {}
    for d in ("up", "down", "flat"):
        pred_pct = direction_counts.get(d, 0) / n
        actual_pct = actual_counts.get(d, 0) / n
        ratio = pred_pct / actual_pct if actual_pct > 0 else float("inf")
        direction_bias[d] = {
            "predicted_pct": round(pred_pct, 3),
            "actual_pct": round(actual_pct, 3),
            "bias_ratio": round(ratio, 2) if ratio != float("inf") else None,
        }

    # Identify the strongest bias
    bias_verdict = None
    max_bias_d = max(
        [d for d in ("up", "down") if direction_bias[d]["bias_ratio"] is not None],
        key=lambda d: direction_bias[d]["bias_ratio"],
        default=None,
    )
    if max_bias_d and direction_bias[max_bias_d]["bias_ratio"] and direction_bias[max_bias_d]["bias_ratio"] > 1.5:
        pred_pct = direction_bias[max_bias_d]["predicted_pct"]
        actual_pct = direction_bias[max_bias_d]["actual_pct"]
        ratio = direction_bias[max_bias_d]["bias_ratio"]
        bias_name = "bearish" if max_bias_d == "down" else "bullish"
        bias_verdict = (
            f"{ratio:.1f}x {bias_name} bias — you predict {max_bias_d.upper()} "
            f"{pred_pct:.0%} but market is {max_bias_d.upper()} only {actual_pct:.0%}"
        )

    # === Confidence Calibration ===
    dir_correct = sum(1 for p in resolved if p["result"]["direction_correct"])
    actual_accuracy = dir_correct / n

    # Get average stated confidence from the predictions
    # Predictions don't store confidence directly — use action_this_cycle as proxy
    # and compute from overall direction accuracy
    confidence_calibration = {
        "actual_direction_accuracy": round(actual_accuracy, 3),
    }

    overconfidence_verdict = None
    if actual_accuracy < 0.45:
        overconfidence_verdict = (
            f"Direction accuracy {actual_accuracy:.0%} is below random (50%). "
            f"Your market reads are currently unreliable."
        )

    # === Magnitude Bias ===
    mag_order = {"flat": 0, "small": 1, "moderate": 2, "large": 3}
    pred_mags = [p["magnitude"] for p in resolved if p["direction"] != "flat"]
    actual_mags = [p["result"]["actual_magnitude"] for p in resolved if p["direction"] != "flat"]

    magnitude_bias = {}
    if pred_mags and actual_mags:
        avg_pred_mag = sum(mag_order.get(m, 1) for m in pred_mags) / len(pred_mags)
        avg_actual_mag = sum(mag_order.get(m, 1) for m in actual_mags) / len(actual_mags)
        magnitude_bias = {
            "avg_predicted": round(avg_pred_mag, 2),
            "avg_actual": round(avg_actual_mag, 2),
            "over_predicting": avg_pred_mag > avg_actual_mag + 0.3,
        }

    # === Accuracy by Tag Condition ===
    accuracy_by_condition = _compute_accuracy_by_tags(resolved, min_count=5)

    # === Accuracy by Time of Day ===
    accuracy_by_time = {}
    for p in resolved:
        tod = p.get("tags", {}).get("time_of_day", "unknown")
        if tod not in accuracy_by_time:
            accuracy_by_time[tod] = {"total": 0, "correct": 0}
        accuracy_by_time[tod]["total"] += 1
        if p["result"]["direction_correct"]:
            accuracy_by_time[tod]["correct"] += 1

    for tod, stats in accuracy_by_time.items():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 3) if stats["total"] > 0 else 0

    # === Error Classification ===
    errors = {"direction_error": 0, "magnitude_only_error": 0, "correct": 0}
    for p in resolved:
        r = p["result"]
        if r["direction_correct"] and r["magnitude_correct"]:
            errors["correct"] += 1
        elif r["direction_correct"]:
            errors["magnitude_only_error"] += 1
        else:
            errors["direction_error"] += 1

    # === Prescriptive Insights ===
    insights = []

    # Bias insight
    if bias_verdict:
        insights.append(bias_verdict)

    # Accuracy insight
    if overconfidence_verdict:
        insights.append(overconfidence_verdict)

    # Best/worst condition insights
    if accuracy_by_condition:
        sorted_conds = sorted(
            accuracy_by_condition.items(),
            key=lambda x: x[1]["accuracy"],
        )
        # Worst conditions
        worst = [(k, v) for k, v in sorted_conds if v["accuracy"] < 0.40 and v["total"] >= 5]
        for key, stats in worst[:2]:
            insights.append(
                f"Weak signal: {key} — {stats['accuracy']:.0%} accuracy "
                f"(N={stats['total']}). Don't trust your reads here."
            )
        # Best conditions
        best = [(k, v) for k, v in reversed(sorted_conds) if v["accuracy"] > 0.55 and v["total"] >= 5]
        for key, stats in best[:2]:
            insights.append(
                f"Strong signal: {key} — {stats['accuracy']:.0%} accuracy "
                f"(N={stats['total']}). Your reads are reliable here."
            )

    # Magnitude bias insight
    if magnitude_bias.get("over_predicting"):
        insights.append(
            "You over-predict magnitude — actual moves are smaller than expected. "
            "Target 'small' magnitude more often."
        )

    # Time of day insight
    best_time = max(
        [(t, s) for t, s in accuracy_by_time.items() if s["total"] >= 5],
        key=lambda x: x[1]["accuracy"],
        default=None,
    )
    worst_time = min(
        [(t, s) for t, s in accuracy_by_time.items() if s["total"] >= 5],
        key=lambda x: x[1]["accuracy"],
        default=None,
    )
    if best_time and worst_time and best_time[0] != worst_time[0]:
        if best_time[1]["accuracy"] - worst_time[1]["accuracy"] > 0.15:
            insights.append(
                f"Best reads at {best_time[0]} ({best_time[1]['accuracy']:.0%}), "
                f"worst at {worst_time[0]} ({worst_time[1]['accuracy']:.0%}). "
                f"Trust {best_time[0]} predictions more."
            )

    return {
        "last_updated": iso_now(),
        "total_analyzed": n,
        "insufficient_data": False,
        "direction_bias": direction_bias,
        "bias_verdict": bias_verdict,
        "confidence_calibration": confidence_calibration,
        "overconfidence_verdict": overconfidence_verdict,
        "magnitude_bias": magnitude_bias,
        "accuracy_by_condition": accuracy_by_condition,
        "accuracy_by_time": accuracy_by_time,
        "error_classification": errors,
        "prescriptive_insights": insights,
    }


def _compute_accuracy_by_tags(resolved: list, min_count: int = 5) -> dict:
    """Group predictions by single-tag values, compute accuracy per group."""
    tag_groups = {}
    for p in resolved:
        tags = p.get("tags", {})
        for tag_name, tag_value in tags.items():
            key = f"{tag_name}={tag_value}"
            if key not in tag_groups:
                tag_groups[key] = {"total": 0, "correct": 0}
            tag_groups[key]["total"] += 1
            if p["result"]["direction_correct"]:
                tag_groups[key]["correct"] += 1

    result = {}
    for key, stats in tag_groups.items():
        if stats["total"] >= min_count:
            stats["accuracy"] = round(stats["correct"] / stats["total"], 3)
            result[key] = stats

    return result


def format_prediction_diagnosis_for_brief() -> str:
    """Format prediction diagnosis for the agent's brief.

    Replaces the old prediction scorecard with richer, prescriptive feedback.
    Returns 3-5 lines of compact text, or empty string if insufficient data.
    """
    diagnosis = load_json(DIAGNOSIS_PATH, default={})
    if not diagnosis or diagnosis.get("insufficient_data", True):
        # Fall back to basic stats if diagnosis not yet built
        return ""

    parts = []
    n = diagnosis.get("total_analyzed", 0)
    cal = diagnosis.get("confidence_calibration", {})
    accuracy = cal.get("actual_direction_accuracy", 0)

    parts.append(f"Predictions ({n} analyzed): {accuracy:.0%} direction accuracy")

    # Show bias verdict if present
    bias = diagnosis.get("bias_verdict")
    if bias:
        parts.append(f"  BIAS: {bias}")

    # Show top 2 prescriptive insights
    insights = diagnosis.get("prescriptive_insights", [])
    shown = 0
    for insight in insights:
        if insight == bias:
            continue  # Already showed bias
        parts.append(f"  {insight}")
        shown += 1
        if shown >= 2:
            break

    return "\n".join(parts) if parts else ""


def rebuild_prediction_diagnosis():
    """Rebuild prediction diagnosis and save to disk. Called nightly."""
    diagnosis = diagnose_predictions(lookback_days=30)
    save_json(DIAGNOSIS_PATH, diagnosis)
    n = diagnosis.get("total_analyzed", 0)
    insights = len(diagnosis.get("prescriptive_insights", []))
    logger.info(f"Prediction diagnosis rebuilt: {n} analyzed, {insights} insights")
    return diagnosis
