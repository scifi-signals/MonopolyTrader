"""Trade statistics engine — learns from historical trade outcomes.

Computes per-dimension hit rates, conviction calibration, and generates
soft constraints that modify position sizing when the agent is trading
into setups with poor historical performance.
"""

import math
from datetime import datetime, timezone
from .utils import load_json, save_json, KNOWLEDGE_DIR, setup_logging
from .portfolio import load_transactions

logger = setup_logging("trade_stats")

STATS_PATH = KNOWLEDGE_DIR / "trade_stats.json"
CONSTRAINTS_PATH = KNOWLEDGE_DIR / "constraints.json"

# Dimensions we slice trades by
DIMENSIONS = [
    "regime_trend",        # bull / bear / sideways
    "regime_volatility",   # high / normal / low
    "thesis_direction",    # bullish / bearish / neutral
    "trigger_type",        # news_high_impact / price_level_break / volume_spike / etc.
    "time_of_day_session", # OPENING_HOUR / MORNING / MIDDAY / POWER_HOUR
    "action",              # BUY / SELL
]

MIN_SAMPLE_SIZE = 5           # Need at least this many trades before stats matter
MIN_CONSTRAINT_SAMPLE = 5     # Minimum trades before generating constraints
DECAY_LAMBDA = 0.1            # Exponential decay rate (per day)


def _extract_dimension(txn: dict, dim: str) -> str | None:
    """Extract a dimension value from a transaction record."""
    tc = txn.get("trade_context", {})
    regime = txn.get("regime", {})

    if dim == "regime_trend":
        return tc.get("regime_trend") or regime.get("trend")
    elif dim == "regime_volatility":
        return tc.get("regime_volatility") or regime.get("volatility")
    elif dim == "thesis_direction":
        return tc.get("thesis_direction")
    elif dim == "trigger_type":
        return tc.get("trigger_type")
    elif dim == "time_of_day_session":
        return tc.get("time_of_day_session")
    elif dim == "action":
        return txn.get("action")
    return None


def _is_winning_trade(txn: dict) -> bool | None:
    """Determine if a trade was a winner. Returns None if not yet resolved."""
    if txn["action"] == "SELL":
        pnl = txn.get("realized_pnl")
        if pnl is not None:
            return pnl >= 0
    elif txn["action"] == "BUY":
        # A BUY is winning if a subsequent SELL of the same ticker made money,
        # or if unrealized P&L is positive. For simplicity, check if confidence
        # matched outcome via later review.
        review = txn.get("review")
        if review and isinstance(review, dict):
            category = review.get("category", "")
            return category in ("signal_correct",)
        # No review yet — can't score
        return None
    return None


def _time_weight(txn: dict) -> float:
    """Compute exponential time decay weight for a transaction."""
    ts = txn.get("timestamp")
    if not ts:
        return 0.5
    try:
        trade_time = datetime.fromisoformat(ts)
        if trade_time.tzinfo is None:
            trade_time = trade_time.replace(tzinfo=timezone.utc)
        days_old = (datetime.now(timezone.utc) - trade_time).total_seconds() / 86400
        return math.exp(-DECAY_LAMBDA * days_old)
    except (ValueError, TypeError):
        return 0.5


def compute_trade_stats(transactions: list = None) -> dict:
    """Compute per-dimension trade statistics from transaction history.

    Returns a dict with breakdowns per dimension:
    {
        "regime_trend": {
            "sideways": {"wins": 1, "total": 8, "hit_rate": 0.125, "sample_size": 8, ...},
            ...
        },
        ...
        "overall": {"wins": 5, "total": 22, "hit_rate": 0.227, ...},
        "conviction_calibration": {...},
    }
    """
    if transactions is None:
        transactions = load_transactions()

    stats = {"dimensions": {}, "overall": {}, "conviction_calibration": {}}

    # Filter to trades with outcomes we can score
    scorable = []
    for txn in transactions:
        # For SELL trades, we can always score via realized_pnl
        if txn["action"] == "SELL" and txn.get("realized_pnl") is not None:
            scorable.append(txn)
        # For BUY trades with reviews, we can check the category
        elif txn["action"] == "BUY" and txn.get("review"):
            scorable.append(txn)

    if not scorable:
        stats["overall"] = {
            "wins": 0, "total": 0, "hit_rate": 0.0,
            "weighted_wins": 0.0, "weighted_total": 0.0,
            "note": "No scorable trades yet",
        }
        stats["conviction_calibration"] = compute_conviction_calibration(transactions)
        return stats

    # Overall stats
    total_weight = 0.0
    win_weight = 0.0
    total_count = 0
    win_count = 0

    for txn in scorable:
        w = _time_weight(txn)
        won = _is_winning_trade(txn)
        if won is None:
            continue
        total_weight += w
        total_count += 1
        if won:
            win_weight += w
            win_count += 1

    stats["overall"] = {
        "wins": win_count,
        "total": total_count,
        "hit_rate": round(win_count / total_count, 3) if total_count > 0 else 0.0,
        "weighted_wins": round(win_weight, 3),
        "weighted_total": round(total_weight, 3),
        "weighted_hit_rate": round(win_weight / total_weight, 3) if total_weight > 0 else 0.0,
    }

    # Per-dimension stats
    for dim in DIMENSIONS:
        dim_stats = {}
        for txn in scorable:
            value = _extract_dimension(txn, dim)
            if value is None:
                value = "unknown"
            value = str(value).lower()

            if value not in dim_stats:
                dim_stats[value] = {
                    "wins": 0, "total": 0,
                    "weighted_wins": 0.0, "weighted_total": 0.0,
                    "avg_confidence": 0.0, "confidences": [],
                    "pnl_sum": 0.0,
                }

            bucket = dim_stats[value]
            w = _time_weight(txn)
            won = _is_winning_trade(txn)
            if won is None:
                continue

            bucket["total"] += 1
            bucket["weighted_total"] += w
            if won:
                bucket["wins"] += 1
                bucket["weighted_wins"] += w

            conf = txn.get("confidence", 0)
            bucket["confidences"].append(conf)

            if txn["action"] == "SELL" and txn.get("realized_pnl") is not None:
                bucket["pnl_sum"] += txn["realized_pnl"]

        # Finalize per-value stats
        for value, bucket in dim_stats.items():
            n = bucket["total"]
            bucket["hit_rate"] = round(bucket["wins"] / n, 3) if n > 0 else 0.0
            bucket["weighted_hit_rate"] = (
                round(bucket["weighted_wins"] / bucket["weighted_total"], 3)
                if bucket["weighted_total"] > 0 else 0.0
            )
            bucket["avg_confidence"] = (
                round(sum(bucket["confidences"]) / len(bucket["confidences"]), 3)
                if bucket["confidences"] else 0.0
            )
            bucket["sample_size"] = n
            bucket["avg_pnl"] = round(bucket["pnl_sum"] / n, 2) if n > 0 else 0.0
            del bucket["confidences"]  # Don't persist raw list
            del bucket["pnl_sum"]

        stats["dimensions"][dim] = dim_stats

    # Conviction calibration
    stats["conviction_calibration"] = compute_conviction_calibration(transactions)

    return stats


def compute_conviction_calibration(transactions: list = None) -> dict:
    """Bin stated conviction vs actual win rate to detect overconfidence.

    Bins: 0.0-0.4 (low), 0.4-0.6 (medium), 0.6-0.8 (high), 0.8-1.0 (very high)
    """
    if transactions is None:
        transactions = load_transactions()

    bins = {
        "0.0-0.4": {"wins": 0, "total": 0, "avg_stated": 0.0, "confidences": []},
        "0.4-0.6": {"wins": 0, "total": 0, "avg_stated": 0.0, "confidences": []},
        "0.6-0.8": {"wins": 0, "total": 0, "avg_stated": 0.0, "confidences": []},
        "0.8-1.0": {"wins": 0, "total": 0, "avg_stated": 0.0, "confidences": []},
    }

    def _get_bin(conf: float) -> str:
        if conf < 0.4:
            return "0.0-0.4"
        elif conf < 0.6:
            return "0.4-0.6"
        elif conf < 0.8:
            return "0.6-0.8"
        else:
            return "0.8-1.0"

    for txn in transactions:
        won = _is_winning_trade(txn)
        if won is None:
            continue
        conf = txn.get("confidence", 0)
        bin_key = _get_bin(conf)
        bins[bin_key]["total"] += 1
        bins[bin_key]["confidences"].append(conf)
        if won:
            bins[bin_key]["wins"] += 1

    # Finalize
    overconfident = False
    for bin_key, bucket in bins.items():
        n = bucket["total"]
        bucket["actual_win_rate"] = round(bucket["wins"] / n, 3) if n > 0 else 0.0
        bucket["avg_stated"] = (
            round(sum(bucket["confidences"]) / len(bucket["confidences"]), 3)
            if bucket["confidences"] else 0.0
        )
        bucket["sample_size"] = n
        del bucket["confidences"]

        # Check for systematic overconfidence: stated confidence > actual win rate + 20pp
        if n >= MIN_SAMPLE_SIZE and bucket["avg_stated"] > bucket["actual_win_rate"] + 0.2:
            overconfident = True

    return {
        "bins": bins,
        "systematically_overconfident": overconfident,
    }


def generate_constraints(stats: dict = None) -> list[dict]:
    """Generate soft position-sizing constraints from trade statistics.

    Returns a list of constraint rules. Each constraint specifies a condition
    (dimension + value), a position modifier, and the evidence behind it.
    """
    if stats is None:
        stats = compute_trade_stats()

    constraints = []

    # Per-dimension constraints
    for dim, dim_stats in stats.get("dimensions", {}).items():
        for value, bucket in dim_stats.items():
            if value == "unknown":
                continue

            n = bucket.get("sample_size", 0)
            hit_rate = bucket.get("weighted_hit_rate", bucket.get("hit_rate", 0.5))

            # Severe underperformance: hit rate < 20% with enough data
            if hit_rate < 0.20 and n >= MIN_CONSTRAINT_SAMPLE + 3:
                constraints.append({
                    "condition": {dim: value},
                    "modifier": 0.25,
                    "reason": f"{dim}={value}: hit rate {hit_rate:.1%} (N={n}). Quarter position.",
                    "sample_size": n,
                    "hit_rate": hit_rate,
                    "confidence": "high" if n >= 10 else "medium",
                })
            # Moderate underperformance: hit rate < 30% with some data
            elif hit_rate < 0.30 and n >= MIN_CONSTRAINT_SAMPLE:
                constraints.append({
                    "condition": {dim: value},
                    "modifier": 0.5,
                    "reason": f"{dim}={value}: hit rate {hit_rate:.1%} (N={n}). Halve position.",
                    "sample_size": n,
                    "hit_rate": hit_rate,
                    "confidence": "medium" if n >= 8 else "low",
                })

    # Conviction discount if systematically overconfident
    cal = stats.get("conviction_calibration", {})
    if cal.get("systematically_overconfident"):
        constraints.append({
            "condition": {"conviction_calibration": "overconfident"},
            "modifier": 0.8,
            "reason": "Systematic overconfidence detected. Applying 20% conviction discount.",
            "sample_size": sum(
                b.get("sample_size", 0) for b in cal.get("bins", {}).values()
            ),
            "confidence": "high",
        })

    return constraints


def apply_statistical_constraints(
    max_shares: float,
    trade_context: dict,
    constraints: list[dict],
) -> tuple[float, list[dict]]:
    """Apply statistical constraints to position sizing.

    Returns (modified_max_shares, list_of_constraints_applied).
    Multiple constraints can stack multiplicatively.
    """
    if not constraints:
        return max_shares, []

    applied = []
    modified = max_shares

    for constraint in constraints:
        condition = constraint.get("condition", {})
        matches = True

        for dim, required_value in condition.items():
            # Special case: conviction calibration applies globally
            if dim == "conviction_calibration":
                matches = True
                break

            actual_value = trade_context.get(dim, "")
            if str(actual_value).lower() != str(required_value).lower():
                matches = False
                break

        if matches:
            modifier = constraint.get("modifier", 1.0)
            old = modified
            modified = round(modified * modifier, 4)
            applied.append({
                "condition": condition,
                "modifier": modifier,
                "reason": constraint.get("reason", ""),
                "old_shares": old,
                "new_shares": modified,
            })
            logger.info(
                f"CONSTRAINT APPLIED: {constraint.get('reason', '')} "
                f"-- position {old:.4f} -> {modified:.4f} shares"
            )

    # Floor: never reduce below 10% of original (prevent total lockout)
    floor = round(max_shares * 0.1, 4)
    if modified < floor and max_shares > 0:
        modified = floor
        logger.info(f"Constraint floor: clamped to {floor:.4f} (10% of original)")

    return modified, applied


def format_stats_for_prompt(stats: dict, constraints: list[dict] = None) -> str:
    """Format trade statistics as a human-readable summary for the trader prompt."""
    lines = []

    overall = stats.get("overall", {})
    total = overall.get("total", 0)
    if total == 0:
        return "No trade statistics available yet (need scored trades)."

    lines.append(
        f"Overall: {overall.get('wins', 0)}/{total} trades won "
        f"({overall.get('hit_rate', 0):.0%} hit rate, "
        f"time-weighted: {overall.get('weighted_hit_rate', 0):.0%})"
    )

    # Per-dimension summaries (only show dimensions with enough data)
    for dim, dim_stats in stats.get("dimensions", {}).items():
        notable = []
        for value, bucket in dim_stats.items():
            if value == "unknown":
                continue
            n = bucket.get("sample_size", 0)
            if n >= 3:  # Show even with small samples, but note it
                hr = bucket.get("hit_rate", 0)
                flag = ""
                if n >= MIN_SAMPLE_SIZE and hr < 0.30:
                    flag = " [POOR]"
                elif n >= MIN_SAMPLE_SIZE and hr > 0.60:
                    flag = " [STRONG]"
                notable.append(f"  {value}: {hr:.0%} ({n} trades){flag}")

        if notable:
            dim_label = dim.replace("_", " ").title()
            lines.append(f"\n{dim_label}:")
            lines.extend(notable)

    # Conviction calibration
    cal = stats.get("conviction_calibration", {})
    cal_bins = cal.get("bins", {})
    cal_lines = []
    for bin_key, bucket in cal_bins.items():
        n = bucket.get("sample_size", 0)
        if n >= 3:
            stated = bucket.get("avg_stated", 0)
            actual = bucket.get("actual_win_rate", 0)
            gap = stated - actual
            marker = ""
            if gap > 0.2 and n >= MIN_SAMPLE_SIZE:
                marker = " ← OVERCONFIDENT"
            cal_lines.append(
                f"  Conviction {bin_key}: stated avg {stated:.0%}, actual win {actual:.0%} "
                f"(N={n}){marker}"
            )

    if cal_lines:
        lines.append("\nConviction Calibration:")
        lines.extend(cal_lines)
        if cal.get("systematically_overconfident"):
            lines.append("  WARNING: Systematic overconfidence detected — positions auto-reduced 20%")

    # Active constraints
    if constraints:
        lines.append(f"\nActive Constraints ({len(constraints)}):")
        for c in constraints:
            lines.append(f"  • {c.get('reason', 'unknown')}")

    return "\n".join(lines)


def compute_and_save_trade_stats() -> dict:
    """Recompute all stats from transaction history and save."""
    transactions = load_transactions()
    stats = compute_trade_stats(transactions)
    save_json(STATS_PATH, stats)
    logger.info(
        f"Trade stats updated: {stats['overall'].get('total', 0)} scorable trades, "
        f"{stats['overall'].get('hit_rate', 0):.0%} hit rate"
    )
    return stats


def generate_and_save_constraints() -> list[dict]:
    """Generate constraints from current stats and save."""
    stats = load_json(STATS_PATH, default={})
    if not stats:
        stats = compute_and_save_trade_stats()

    constraints = generate_constraints(stats)
    save_json(CONSTRAINTS_PATH, constraints)
    logger.info(f"Constraints updated: {len(constraints)} active rules")
    return constraints


def load_trade_stats() -> dict:
    """Load cached trade stats."""
    return load_json(STATS_PATH, default={})


def load_constraints() -> list[dict]:
    """Load cached constraint rules."""
    return load_json(CONSTRAINTS_PATH, default=[])
