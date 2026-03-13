"""Thesis Builder — nightly aggregation of trade tags into a statistical playbook.

Reads all trades from trade_journal.json, groups by single-tag values AND
multi-tag combinations, computes win rates and P&L stats, writes thesis_ledger.json.

v6: Added multi-factor (2-tag, 3-tag) analysis, strategy aggregation,
find_matching_patterns(), and research metrics.

Runs after market close. No LLM calls. Pure Python math.
"""

from collections import Counter
from datetime import datetime, timezone, timedelta
from itertools import combinations
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("thesis_builder")

JOURNAL_PATH = DATA_DIR / "trade_journal.json"
LEDGER_PATH = DATA_DIR / "thesis_ledger.json"

# Time-based decay weights
DECAY_WINDOWS = [
    (30, 1.0),    # last 30 days: full weight
    (60, 0.7),    # 30-60 days ago: 0.7
    (90, 0.4),    # 60-90 days ago: 0.4
    # older than 90 days: excluded
]


def build_ledger() -> dict:
    """Build the thesis ledger from all closed trades in the journal.

    Groups trades by individual tag values AND multi-tag combinations,
    computes weighted stats. Returns the complete ledger dict.
    """
    journal = load_json(JOURNAL_PATH, default=[])
    now = datetime.now(timezone.utc)

    # Only use closed trades with tags
    closed = [
        e for e in journal
        if e.get("realized_pnl") is not None and e.get("tags")
    ]

    if not closed:
        ledger = {
            "last_updated": iso_now(),
            "total_trades": 0,
            "theses": {},
            "multi_tag_patterns": {},
            "strategy_stats": {},
            "calibration": {},
            "validation_warnings": [],
        }
        save_json(LEDGER_PATH, ledger)
        logger.info("Thesis ledger: no closed trades with tags yet")
        return ledger

    # Compute time-decay weight for each trade
    for trade in closed:
        trade_time = datetime.fromisoformat(trade["closed_at"])
        if trade_time.tzinfo is None:
            trade_time = trade_time.replace(tzinfo=timezone.utc)
        days_ago = (now - trade_time).days
        weight = 0.0
        for max_days, w in DECAY_WINDOWS:
            if days_ago <= max_days:
                weight = w
                break
        trade["_weight"] = weight

    # Filter out trades older than 90 days for stats (but count them)
    weighted_trades = [t for t in closed if t["_weight"] > 0]
    total_all = len(closed)

    # Build single-tag theses
    theses = _build_single_tag_theses(weighted_trades)

    # Build multi-tag patterns (2-tag and 3-tag combos)
    multi_tag_patterns = _build_multi_tag_patterns(weighted_trades)

    # Build strategy stats
    strategy_stats = _build_strategy_stats(weighted_trades)

    # Confidence calibration
    calibration = _compute_calibration(weighted_trades)

    # Validation
    warnings = validate_ledger(theses, len(weighted_trades))

    # v6.1 Blindspot #1: Outcome type aggregation
    outcome_stats = _build_outcome_type_stats(weighted_trades)

    # v6.1 Blindspot #2: Exit quality metrics
    exit_quality_stats = _build_exit_quality_stats(weighted_trades)

    # v6.1 Blindspot #7: Anomaly detection (right-for-wrong-reasons)
    anomaly_flags = _detect_anomalous_trades(weighted_trades, theses, multi_tag_patterns)

    # v6.1 Blindspot #9: Thesis consistency scoring
    thesis_consistency = _build_thesis_consistency_stats(weighted_trades)

    ledger = {
        "last_updated": iso_now(),
        "total_trades": total_all,
        "active_trades": len(weighted_trades),
        "theses": theses,
        "multi_tag_patterns": multi_tag_patterns,
        "strategy_stats": strategy_stats,
        "calibration": calibration,
        "validation_warnings": warnings,
        "outcome_stats": outcome_stats,
        "exit_quality": exit_quality_stats,
        "anomaly_flags": anomaly_flags,
        "thesis_consistency": thesis_consistency,
    }

    save_json(LEDGER_PATH, ledger)
    logger.info(
        f"Thesis ledger built: {len(weighted_trades)} active trades, "
        f"{len(theses)} single-tag theses, {len(multi_tag_patterns)} multi-tag patterns, "
        f"{len(strategy_stats)} strategies, {len(warnings)} warnings, "
        f"{len(anomaly_flags)} anomalies"
    )
    return ledger


def _build_single_tag_theses(weighted_trades: list) -> dict:
    """Build single-tag performance stats from weighted trades."""
    theses = {}
    for trade in weighted_trades:
        tags = trade["tags"]
        pnl = trade["realized_pnl"]
        pnl_pct = (pnl / trade["total_value"] * 100) if trade.get("total_value", 0) > 0 else 0
        is_win = pnl >= 0
        conf = trade.get("confidence", 0.5)
        weight = trade["_weight"]

        for tag_name, tag_value in tags.items():
            key = f"{tag_name}:{tag_value}"
            if key not in theses:
                theses[key] = {
                    "tag": tag_name,
                    "value": tag_value,
                    "trades": 0,
                    "weighted_trades": 0.0,
                    "wins": 0,
                    "weighted_wins": 0.0,
                    "total_pnl": 0.0,
                    "total_pnl_pct": 0.0,
                    "confidences": [],
                    "pnl_values": [],
                    "last_trade": None,
                }

            t = theses[key]
            t["trades"] += 1
            t["weighted_trades"] += weight
            if is_win:
                t["wins"] += 1
                t["weighted_wins"] += weight
            t["total_pnl"] += pnl
            t["total_pnl_pct"] += pnl_pct
            t["confidences"].append(conf)
            t["pnl_values"].append(pnl)
            t["last_trade"] = trade.get("closed_at", trade.get("timestamp"))

    # Compute final stats per thesis
    for key, t in theses.items():
        _finalize_stats(t)

    return theses


def _build_multi_tag_patterns(weighted_trades: list) -> dict:
    """Build 2-tag and 3-tag combination performance stats.

    Only includes combinations that appear in at least 3 trades.
    """
    # Count all 2-tag and 3-tag combos
    combo_counter = Counter()
    for trade in weighted_trades:
        tags = trade["tags"]
        tag_items = sorted(tags.items())  # Consistent ordering

        # 2-tag combos
        for pair in combinations(tag_items, 2):
            combo_key = " + ".join(f"{k}={v}" for k, v in pair)
            combo_counter[combo_key] += 1

        # 3-tag combos
        for triple in combinations(tag_items, 3):
            combo_key = " + ".join(f"{k}={v}" for k, v in triple)
            combo_counter[combo_key] += 1

    # Filter to combos with N>=3
    frequent_combos = {k for k, n in combo_counter.items() if n >= 3}

    if not frequent_combos:
        return {}

    # Build stats for frequent combos
    patterns = {}
    for trade in weighted_trades:
        tags = trade["tags"]
        tag_items = sorted(tags.items())
        pnl = trade["realized_pnl"]
        is_win = pnl >= 0
        weight = trade["_weight"]
        conf = trade.get("confidence", 0.5)

        # Check 2-tag combos
        for pair in combinations(tag_items, 2):
            combo_key = " + ".join(f"{k}={v}" for k, v in pair)
            if combo_key in frequent_combos:
                _accumulate_pattern(patterns, combo_key, trade, pnl, is_win, weight, conf, 2)

        # Check 3-tag combos
        for triple in combinations(tag_items, 3):
            combo_key = " + ".join(f"{k}={v}" for k, v in triple)
            if combo_key in frequent_combos:
                _accumulate_pattern(patterns, combo_key, trade, pnl, is_win, weight, conf, 3)

    # Finalize stats
    for key, p in patterns.items():
        _finalize_stats(p)

    return patterns


def _accumulate_pattern(
    patterns: dict, key: str, trade: dict,
    pnl: float, is_win: bool, weight: float, conf: float,
    combo_size: int,
):
    """Accumulate a trade into a multi-tag pattern bucket."""
    if key not in patterns:
        patterns[key] = {
            "combo_size": combo_size,
            "trades": 0,
            "weighted_trades": 0.0,
            "wins": 0,
            "weighted_wins": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "confidences": [],
            "pnl_values": [],
            "last_trade": None,
        }

    p = patterns[key]
    pnl_pct = (pnl / trade["total_value"] * 100) if trade.get("total_value", 0) > 0 else 0
    p["trades"] += 1
    p["weighted_trades"] += weight
    if is_win:
        p["wins"] += 1
        p["weighted_wins"] += weight
    p["total_pnl"] += pnl
    p["total_pnl_pct"] += pnl_pct
    p["confidences"].append(conf)
    p["pnl_values"].append(pnl)
    p["last_trade"] = trade.get("closed_at", trade.get("timestamp"))


def _build_strategy_stats(weighted_trades: list) -> dict:
    """Aggregate performance by the agent's self-reported strategy field.

    Reads strategy from the journal entry reasoning/strategy fields.
    """
    strategies = {}
    for trade in weighted_trades:
        # Try to get strategy from the journal entry
        strategy = trade.get("strategy", "")
        if not strategy:
            # Fall back to extracting from reasoning
            reasoning = trade.get("reasoning", "")
            strategy = _extract_strategy_from_reasoning(reasoning)

        if not strategy:
            strategy = "unknown"

        strategy = strategy.lower().strip()

        pnl = trade["realized_pnl"]
        is_win = pnl >= 0
        weight = trade["_weight"]
        conf = trade.get("confidence", 0.5)
        pnl_pct = (pnl / trade["total_value"] * 100) if trade.get("total_value", 0) > 0 else 0

        if strategy not in strategies:
            strategies[strategy] = {
                "strategy": strategy,
                "trades": 0,
                "weighted_trades": 0.0,
                "wins": 0,
                "weighted_wins": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "confidences": [],
                "pnl_values": [],
                "last_trade": None,
            }

        s = strategies[strategy]
        s["trades"] += 1
        s["weighted_trades"] += weight
        if is_win:
            s["wins"] += 1
            s["weighted_wins"] += weight
        s["total_pnl"] += pnl
        s["total_pnl_pct"] += pnl_pct
        s["confidences"].append(conf)
        s["pnl_values"].append(pnl)
        s["last_trade"] = trade.get("closed_at", trade.get("timestamp"))

    # Finalize stats
    for key, s in strategies.items():
        _finalize_stats(s)

    return strategies


def _extract_strategy_from_reasoning(reasoning: str) -> str:
    """Try to extract a strategy name from reasoning text."""
    # Look for common strategy naming patterns
    reasoning_lower = reasoning.lower()
    for pattern in [
        "mean_reversion", "breakout", "momentum", "range_bound", "scalp",
        "pre_catalyst", "post_catalyst", "cash_preservation", "trend_following",
        "dip_buy", "oversold_bounce", "resistance_sell",
    ]:
        if pattern in reasoning_lower:
            return pattern
    return ""


def _finalize_stats(stats: dict):
    """Compute final statistics from accumulated raw data and clean up internals."""
    n = stats["trades"]
    wt = stats.get("weighted_trades", 0)
    stats["win_rate"] = round(stats["weighted_wins"] / wt, 3) if wt > 0 else 0
    stats["avg_pnl"] = round(stats["total_pnl"] / n, 2) if n > 0 else 0
    stats["avg_pnl_pct"] = round(stats["total_pnl_pct"] / n, 2) if n > 0 else 0
    stats["avg_confidence"] = round(sum(stats["confidences"]) / n, 2) if n > 0 else 0
    stats["total_pnl"] = round(stats["total_pnl"], 2)

    if stats["pnl_values"]:
        stats["best_pnl"] = round(max(stats["pnl_values"]), 2)
        stats["worst_pnl"] = round(min(stats["pnl_values"]), 2)

    # Clean up internal fields
    for field in ["weighted_trades", "weighted_wins", "total_pnl_pct", "confidences", "pnl_values"]:
        stats.pop(field, None)


def _build_outcome_type_stats(trades: list) -> dict:
    """Aggregate outcome_type stats from closed trades.

    v6.1 Blindspot #1: "60% of losses were timing_wrong — consider holding longer."
    """
    losses = [t for t in trades if t.get("realized_pnl", 0) < 0]
    wins = [t for t in trades if t.get("realized_pnl", 0) >= 0]

    loss_types = {}
    for t in losses:
        ot = t.get("outcome_type", "unclassified")
        loss_types[ot] = loss_types.get(ot, 0) + 1

    win_types = {}
    for t in wins:
        ot = t.get("outcome_type", "unclassified")
        win_types[ot] = win_types.get(ot, 0) + 1

    total_losses = len(losses)
    total_wins = len(wins)

    # Compute percentages for losses
    loss_breakdown = {}
    for ot, count in loss_types.items():
        loss_breakdown[ot] = {
            "count": count,
            "pct": round(count / total_losses * 100, 1) if total_losses > 0 else 0,
        }

    # Generate actionable insights
    insights = []
    if total_losses >= 3:
        dominant_type = max(loss_types, key=loss_types.get) if loss_types else None
        if dominant_type and loss_types[dominant_type] / total_losses > 0.5:
            pct = loss_types[dominant_type] / total_losses * 100
            advice = {
                "timing_wrong": "consider holding longer or adjusting entry timing",
                "thesis_wrong": "re-evaluate thesis construction before entering",
                "execution_wrong": "let winners run longer, size positions more carefully",
                "external_shock": "these are unavoidable — focus on position sizing",
                "spread_cost": "target larger moves or reduce trade frequency",
            }
            insight = advice.get(dominant_type, "review this pattern")
            insights.append(
                f"{pct:.0f}% of losses were {dominant_type} — {insight}"
            )

    # Thesis correctness rate
    thesis_correct_count = sum(1 for t in trades if t.get("thesis_correct") is True)
    thesis_total = sum(1 for t in trades if t.get("thesis_correct") is not None)

    return {
        "loss_breakdown": loss_breakdown,
        "win_breakdown": win_types,
        "total_losses": total_losses,
        "total_wins": total_wins,
        "thesis_correct_rate": round(thesis_correct_count / thesis_total, 3) if thesis_total > 0 else None,
        "insights": insights,
    }


def _build_exit_quality_stats(trades: list) -> dict:
    """Compute exit quality metrics from intra-trade price data.

    v6.1 Blindspot #2: exit_quality = (exit - entry) / (peak - entry).
    1.0 = perfect exit at peak, negative = exited at a loss despite being profitable.
    """
    qualities = []
    for t in trades:
        if t.get("realized_pnl") is None:
            continue

        entry_price = t.get("price", 0)
        close_price = t.get("close_price", 0)
        peak_price = t.get("peak_price")

        if not peak_price or entry_price <= 0 or close_price <= 0:
            continue

        # Only meaningful for BUY trades where peak differs from entry
        if t.get("action") != "BUY":
            continue

        peak_move = peak_price - entry_price
        if abs(peak_move) < 0.01:
            # No meaningful peak movement
            continue

        actual_move = close_price - entry_price
        exit_quality = actual_move / peak_move if peak_move != 0 else 0

        qualities.append({
            "trade_id": t.get("trade_id"),
            "exit_quality": round(exit_quality, 3),
            "entry": entry_price,
            "exit": close_price,
            "peak": peak_price,
            "pnl": t.get("realized_pnl", 0),
        })

    if not qualities:
        return {"avg_exit_quality": None, "trades_measured": 0}

    avg_quality = sum(q["exit_quality"] for q in qualities) / len(qualities)

    # Trades where we were profitable at some point but exited at a loss
    profitable_then_lost = [
        q for q in qualities
        if q["peak"] > q["entry"] and q["exit"] < q["entry"]
    ]

    return {
        "avg_exit_quality": round(avg_quality, 3),
        "trades_measured": len(qualities),
        "profitable_then_lost": len(profitable_then_lost),
        "best_exit": max(q["exit_quality"] for q in qualities),
        "worst_exit": min(q["exit_quality"] for q in qualities),
    }


def _detect_anomalous_trades(
    trades: list,
    theses: dict,
    multi_patterns: dict,
) -> list[dict]:
    """Detect trades that won in bad conditions or lost in good conditions.

    v6.1 Blindspot #7: Right-for-wrong-reasons detection.
    """
    anomalies = []

    for trade in trades:
        if trade.get("realized_pnl") is None or not trade.get("tags"):
            continue

        pnl = trade["realized_pnl"]
        is_win = pnl >= 0
        trade_id = trade.get("trade_id", "unknown")
        tags = trade["tags"]

        # Check each tag against known patterns
        matching_win_rates = []
        for tag_name, tag_value in tags.items():
            key = f"{tag_name}:{tag_value}"
            if key in theses:
                stats = theses[key]
                if stats["trades"] >= 3:
                    matching_win_rates.append(stats["win_rate"])

        if not matching_win_rates:
            continue

        avg_expected_wr = sum(matching_win_rates) / len(matching_win_rates)

        # Won in bad conditions (expected < 30% win rate)
        if is_win and avg_expected_wr < 0.30:
            anomalies.append({
                "trade_id": trade_id,
                "type": "lucky_win",
                "pnl": pnl,
                "avg_expected_win_rate": round(avg_expected_wr, 3),
                "reason": f"Won despite conditions averaging {avg_expected_wr:.0%} win rate",
            })

        # Lost in good conditions (expected > 70% win rate)
        elif not is_win and avg_expected_wr > 0.70:
            anomalies.append({
                "trade_id": trade_id,
                "type": "unlucky_loss",
                "pnl": pnl,
                "avg_expected_win_rate": round(avg_expected_wr, 3),
                "reason": f"Lost despite conditions averaging {avg_expected_wr:.0%} win rate",
            })

    return anomalies


def _build_thesis_consistency_stats(trades: list) -> dict:
    """Compute thesis consistency metrics.

    v6.1 Blindspot #9: What percentage of trades aligned with the MID thesis,
    and do thesis-consistent or contrarian trades perform better?
    """
    consistent = [t for t in trades if t.get("thesis_consistent") is True]
    inconsistent = [t for t in trades if t.get("thesis_consistent") is False]
    unknown = [t for t in trades if t.get("thesis_consistent") is None]

    total_known = len(consistent) + len(inconsistent)

    consistent_wins = sum(1 for t in consistent if t.get("realized_pnl", 0) >= 0)
    inconsistent_wins = sum(1 for t in inconsistent if t.get("realized_pnl", 0) >= 0)

    consistent_pnl = sum(t.get("realized_pnl", 0) for t in consistent)
    inconsistent_pnl = sum(t.get("realized_pnl", 0) for t in inconsistent)

    return {
        "total_known": total_known,
        "consistent_count": len(consistent),
        "inconsistent_count": len(inconsistent),
        "unknown_count": len(unknown),
        "consistency_rate": round(len(consistent) / total_known, 3) if total_known > 0 else None,
        "consistent_win_rate": round(consistent_wins / len(consistent), 3) if consistent else None,
        "inconsistent_win_rate": round(inconsistent_wins / len(inconsistent), 3) if inconsistent else None,
        "consistent_total_pnl": round(consistent_pnl, 2),
        "inconsistent_total_pnl": round(inconsistent_pnl, 2),
        "contrarian_edge": (
            round(
                (inconsistent_wins / len(inconsistent) if inconsistent else 0)
                - (consistent_wins / len(consistent) if consistent else 0),
                3,
            )
            if consistent and inconsistent else None
        ),
    }


def find_matching_patterns(current_tags: dict, ledger: dict = None) -> list[dict]:
    """Find all playbook patterns that match the current market conditions.

    Returns a list of matching patterns sorted by relevance (more specific
    patterns first, then by trade count).

    Each result includes:
        - pattern: the pattern key string
        - combo_size: 1 for single-tag, 2 for 2-tag, 3 for 3-tag
        - trades, wins, win_rate, avg_pnl
        - signal: "STRONG_AVOID", "AVOID", "FAVORABLE", or "NEUTRAL"
    """
    if ledger is None:
        ledger = load_json(LEDGER_PATH, default={})

    matches = []

    # Match single-tag patterns
    theses = ledger.get("theses", {})
    for tag_name, tag_value in current_tags.items():
        key = f"{tag_name}:{tag_value}"
        if key in theses:
            stats = theses[key]
            matches.append({
                "pattern": key.replace(":", "="),
                "combo_size": 1,
                **_extract_match_stats(stats),
            })

    # Match multi-tag patterns
    multi = ledger.get("multi_tag_patterns", {})
    tag_items = sorted(current_tags.items())

    # Check 2-tag combos
    for pair in combinations(tag_items, 2):
        combo_key = " + ".join(f"{k}={v}" for k, v in pair)
        if combo_key in multi:
            stats = multi[combo_key]
            matches.append({
                "pattern": combo_key,
                "combo_size": 2,
                **_extract_match_stats(stats),
            })

    # Check 3-tag combos
    for triple in combinations(tag_items, 3):
        combo_key = " + ".join(f"{k}={v}" for k, v in triple)
        if combo_key in multi:
            stats = multi[combo_key]
            matches.append({
                "pattern": combo_key,
                "combo_size": 3,
                **_extract_match_stats(stats),
            })

    # Sort: more specific patterns first, then by trade count
    matches.sort(key=lambda m: (-m["combo_size"], -m["trades"]))

    return matches


def _extract_match_stats(stats: dict) -> dict:
    """Extract the relevant stats from a pattern for matching output."""
    n = stats["trades"]
    wr = stats["win_rate"]
    avg = stats.get("avg_pnl", 0)

    # Classify signal strength
    if n >= 5 and wr < 0.25:
        signal = "STRONG_AVOID"
    elif n >= 3 and wr < 0.40:
        signal = "AVOID"
    elif n >= 3 and wr > 0.55:
        signal = "FAVORABLE"
    else:
        signal = "NEUTRAL"

    return {
        "trades": n,
        "wins": stats["wins"],
        "win_rate": wr,
        "avg_pnl": avg,
        "signal": signal,
    }


def format_matching_patterns_for_brief(matches: list) -> str:
    """Format matching patterns into text for the agent's brief."""
    if not matches:
        return "No matching patterns in playbook for current conditions."

    parts = ["Current conditions match these playbook patterns:"]

    # Show strong signals first
    strong_avoids = [m for m in matches if m["signal"] == "STRONG_AVOID"]
    avoids = [m for m in matches if m["signal"] == "AVOID"]
    favorables = [m for m in matches if m["signal"] == "FAVORABLE"]
    neutrals = [m for m in matches if m["signal"] == "NEUTRAL"]

    for m in strong_avoids:
        parts.append(
            f"  >>> [{m['pattern']}]: {m['wins']}/{m['trades']} wins "
            f"({m['win_rate']:.0%}), avg ${m['avg_pnl']:+.2f} — STRONG AVOID <<<"
        )

    for m in avoids:
        parts.append(
            f"  [{m['pattern']}]: {m['wins']}/{m['trades']} wins "
            f"({m['win_rate']:.0%}), avg ${m['avg_pnl']:+.2f} — AVOID"
        )

    for m in favorables:
        parts.append(
            f"  [{m['pattern']}]: {m['wins']}/{m['trades']} wins "
            f"({m['win_rate']:.0%}), avg ${m['avg_pnl']:+.2f} — FAVORABLE"
        )

    if neutrals:
        parts.append(f"  ({len(neutrals)} other matching patterns with neutral signal)")

    return "\n".join(parts)


def _compute_calibration(trades: list) -> dict:
    """Check if the agent's confidence predicts win rate."""
    if len(trades) < 5:
        return {"status": "insufficient_data", "trades": len(trades)}

    high_conf = [t for t in trades if t.get("confidence", 0) >= 0.7]
    low_conf = [t for t in trades if t.get("confidence", 0) < 0.5]

    high_wr = (
        sum(1 for t in high_conf if t["realized_pnl"] >= 0) / len(high_conf)
        if high_conf else None
    )
    low_wr = (
        sum(1 for t in low_conf if t["realized_pnl"] >= 0) / len(low_conf)
        if low_conf else None
    )

    result = {
        "high_confidence_trades": len(high_conf),
        "high_confidence_win_rate": round(high_wr, 3) if high_wr is not None else None,
        "low_confidence_trades": len(low_conf),
        "low_confidence_win_rate": round(low_wr, 3) if low_wr is not None else None,
    }

    if high_wr is not None and low_wr is not None:
        result["calibrated"] = high_wr > low_wr
        result["spread"] = round(high_wr - low_wr, 3)
    else:
        result["calibrated"] = None
        result["spread"] = None

    return result


def validate_ledger(theses: dict, total_trades: int) -> list[str]:
    """Flag suspicious patterns in the aggregated data."""
    warnings = []

    for key, stats in theses.items():
        n = stats["trades"]
        # Flag impossibly good results
        if n >= 3 and stats["win_rate"] == 1.0:
            warnings.append(
                f"SUSPICIOUS: {key} has 100% win rate on {n} trades"
            )
        # Flag data errors
        if stats["wins"] > n:
            warnings.append(f"DATA ERROR: {key} has more wins ({stats['wins']}) than trades ({n})")

    if warnings:
        for w in warnings:
            logger.warning(f"Ledger validation: {w}")

    return warnings


def format_playbook_for_brief(ledger: dict) -> str:
    """Format the thesis ledger as text for Claude's trading brief."""
    total = ledger.get("total_trades", 0)
    active = ledger.get("active_trades", total)

    if total == 0:
        return "No closed trades yet. Building your playbook..."

    theses = ledger.get("theses", {})

    # Separate into significant (N>=3) and preliminary
    significant = {k: v for k, v in theses.items() if v["trades"] >= 3}
    preliminary = {k: v for k, v in theses.items() if v["trades"] < 3}

    parts = []

    if not significant:
        parts.append(f"Building playbook... ({total} trades, need more for stats)")
        if preliminary:
            parts.append("Early observations (N<3, treat as preliminary):")
            for key, s in sorted(preliminary.items(), key=lambda x: x[1]["trades"], reverse=True)[:8]:
                tag_label = key.replace(":", "=")
                wr = f"{s['win_rate']:.0%}" if s["trades"] > 0 else "?"
                parts.append(f"  {tag_label}: {s['trades']} trades, {s['wins']} wins ({wr})")
        return "\n".join(parts)

    # Best setups (win rate > 55%)
    best = {k: v for k, v in significant.items() if v["win_rate"] > 0.55}
    if best:
        parts.append("Best setups (win rate > 55%, N>=3):")
        for key, s in sorted(best.items(), key=lambda x: x[1]["win_rate"], reverse=True):
            tag_label = key.replace(":", "=")
            parts.append(
                f"  {tag_label}: {s['wins']}/{s['trades']} wins ({s['win_rate']:.0%}), "
                f"avg {'+' if s['avg_pnl'] >= 0 else ''}${s['avg_pnl']:.2f}"
            )

    # Worst setups (win rate < 40%)
    worst = {k: v for k, v in significant.items() if v["win_rate"] < 0.40}
    if worst:
        parts.append("Worst setups (win rate < 40%, N>=3) — DO NOT TRADE THESE:")
        for key, s in sorted(worst.items(), key=lambda x: x[1]["win_rate"]):
            tag_label = key.replace(":", "=")
            parts.append(
                f"  {tag_label}: {s['wins']}/{s['trades']} wins ({s['win_rate']:.0%}), "
                f"avg {'+' if s['avg_pnl'] >= 0 else ''}${s['avg_pnl']:.2f} ← AVOID"
            )

    # Neutral (between 40-55%)
    neutral_count = len(significant) - len(best) - len(worst)
    if neutral_count > 0:
        parts.append(f"({neutral_count} other setups with 40-55% win rate)")

    # Multi-tag patterns (strongest signals only)
    multi = ledger.get("multi_tag_patterns", {})
    strong_multi_avoids = {
        k: v for k, v in multi.items()
        if v["trades"] >= 3 and v["win_rate"] < 0.30
    }
    strong_multi_favorable = {
        k: v for k, v in multi.items()
        if v["trades"] >= 3 and v["win_rate"] > 0.60
    }

    if strong_multi_favorable:
        parts.append("Multi-factor FAVORABLE patterns (N>=3, win rate > 60%):")
        for key, s in sorted(strong_multi_favorable.items(), key=lambda x: x[1]["win_rate"], reverse=True)[:5]:
            parts.append(
                f"  [{key}]: {s['wins']}/{s['trades']} wins ({s['win_rate']:.0%}), "
                f"avg ${s['avg_pnl']:+.2f}"
            )

    if strong_multi_avoids:
        parts.append("Multi-factor AVOID patterns (N>=3, win rate < 30%):")
        for key, s in sorted(strong_multi_avoids.items(), key=lambda x: x[1]["win_rate"])[:5]:
            parts.append(
                f"  [{key}]: {s['wins']}/{s['trades']} wins ({s['win_rate']:.0%}), "
                f"avg ${s['avg_pnl']:+.2f} ← STRONG AVOID"
            )

    # Strategy stats
    strats = ledger.get("strategy_stats", {})
    if strats:
        sig_strats = {k: v for k, v in strats.items() if v["trades"] >= 2}
        if sig_strats:
            parts.append("Strategy performance:")
            for key, s in sorted(sig_strats.items(), key=lambda x: x[1]["win_rate"], reverse=True):
                parts.append(
                    f"  {key}: {s['wins']}/{s['trades']} wins ({s['win_rate']:.0%}), "
                    f"avg ${s['avg_pnl']:+.2f}"
                )

    # Confidence calibration
    cal = ledger.get("calibration", {})
    if cal.get("calibrated") is not None:
        if cal["calibrated"]:
            parts.append(
                f"Confidence calibration: Your high-confidence trades win "
                f"{cal['high_confidence_win_rate']:.0%} vs low-confidence "
                f"{cal['low_confidence_win_rate']:.0%} — confidence IS predictive."
            )
        else:
            parts.append(
                f"Confidence calibration: High-confidence trades win "
                f"{cal['high_confidence_win_rate']:.0%}, low-confidence win "
                f"{cal['low_confidence_win_rate']:.0%} — confidence is NOT predictive. "
                f"Treat all trades as uncertain."
            )

    # Research metrics
    metrics = compute_research_metrics(ledger)
    if metrics:
        parts.append("")
        parts.append("RESEARCH METRICS:")
        parts.append(f"  Experiment efficiency: {metrics['experiment_efficiency']:.0%} (new conditions explored vs total trades)")
        parts.append(f"  Redundant loss rate: {metrics['redundant_loss_rate']:.0%} (should be 0%)")
        if metrics["pattern_discovery_count"] > 0:
            parts.append(f"  Patterns discovered: {metrics['pattern_discovery_count']} (N>=5 with clear signal)")
        parts.append(f"  Calibration error: {metrics['calibration_error']:.2f} (avg |confidence - actual win rate|)")
        if metrics["rolling_win_rate_10"] is not None:
            parts.append(f"  Rolling win rate (last 10): {metrics['rolling_win_rate_10']:.0%}")

    # v6.1: Outcome type analysis (Blindspot #1)
    outcome_stats = ledger.get("outcome_stats", {})
    if outcome_stats.get("insights"):
        parts.append("")
        parts.append("LOSS DIAGNOSIS:")
        for insight in outcome_stats["insights"]:
            parts.append(f"  >>> {insight}")
    if outcome_stats.get("thesis_correct_rate") is not None:
        parts.append(f"  Thesis correct rate: {outcome_stats['thesis_correct_rate']:.0%}")

    # v6.1: Exit quality (Blindspot #2)
    exit_q = ledger.get("exit_quality", {})
    if exit_q.get("avg_exit_quality") is not None and exit_q.get("trades_measured", 0) >= 3:
        parts.append("")
        parts.append("EXIT QUALITY:")
        parts.append(f"  Avg exit quality: {exit_q['avg_exit_quality']:.1%} of peak move captured (N={exit_q['trades_measured']})")
        if exit_q.get("profitable_then_lost", 0) > 0:
            parts.append(f"  >>> {exit_q['profitable_then_lost']} trade(s) were profitable but exited at a loss — hold longer!")

    # v6.1: Anomaly flags (Blindspot #7)
    anomalies = ledger.get("anomaly_flags", [])
    if anomalies:
        lucky = [a for a in anomalies if a["type"] == "lucky_win"]
        unlucky = [a for a in anomalies if a["type"] == "unlucky_loss"]
        if lucky:
            parts.append(f"  Lucky wins detected: {len(lucky)} trade(s) won in losing conditions — do not over-index on these")
        if unlucky:
            parts.append(f"  Bad luck losses: {len(unlucky)} trade(s) lost in winning conditions — may be noise")

    # v6.1: Thesis consistency (Blindspot #9)
    tc = ledger.get("thesis_consistency", {})
    if tc.get("total_known", 0) >= 3:
        parts.append("")
        parts.append("THESIS CONSISTENCY:")
        parts.append(f"  Consistency rate: {tc['consistency_rate']:.0%} of trades aligned with MID thesis")
        if tc.get("consistent_win_rate") is not None:
            parts.append(f"  MID-aligned win rate: {tc['consistent_win_rate']:.0%}")
        if tc.get("inconsistent_win_rate") is not None:
            parts.append(f"  Contrarian win rate: {tc['inconsistent_win_rate']:.0%}")
        if tc.get("contrarian_edge") is not None:
            edge = tc["contrarian_edge"]
            if edge > 0.1:
                parts.append(f"  >>> Contrarian trades outperform by {edge:.0%} — MID thesis may be wrong!")
            elif edge < -0.1:
                parts.append(f"  >>> MID-aligned trades outperform by {abs(edge):.0%} — stay aligned with thesis")

    parts.append(f"(Based on {active} trades from last 90 days, {total} total)")

    return "\n".join(parts)


def compute_research_metrics(ledger: dict = None) -> dict:
    """Compute v6 research quality metrics.

    Returns:
        Dict with experiment_efficiency, redundant_loss_rate,
        pattern_discovery_count, calibration_error, rolling_win_rate_10.
    """
    if ledger is None:
        ledger = load_json(LEDGER_PATH, default={})

    theses = ledger.get("theses", {})
    total_active = ledger.get("active_trades", 0)

    if total_active == 0:
        return {
            "experiment_efficiency": 0.0,
            "redundant_loss_rate": 0.0,
            "pattern_discovery_count": 0,
            "calibration_error": 0.0,
            "rolling_win_rate_10": None,
        }

    # experiment_efficiency: ratio of trades in conditions with N<3 (exploratory)
    # We approximate this: if a single-tag thesis has N < 3, those trades were exploratory
    # A trade is "new" if at least one of its tags had N<3 at the time — but we only have
    # current counts. Rough proxy: tags with small N represent exploratory conditions.
    exploratory_tag_trades = sum(
        v["trades"] for v in theses.values() if v["trades"] < 3
    )
    total_tag_trades = sum(v["trades"] for v in theses.values())
    experiment_efficiency = (
        exploratory_tag_trades / total_tag_trades if total_tag_trades > 0 else 0
    )

    # redundant_loss_rate: % of losses in conditions where playbook shows N>=5 and win_rate < 0.25
    avoid_theses = {
        k: v for k, v in theses.items()
        if v["trades"] >= 5 and v["win_rate"] < 0.25
    }
    # Count losses in avoid conditions (approximate: losses in those theses)
    redundant_losses = 0
    for v in avoid_theses.values():
        redundant_losses += v["trades"] - v["wins"]  # losses in that thesis
    # Total losses across all theses (approximate from journal)
    journal = load_json(JOURNAL_PATH, default=[])
    closed = [e for e in journal if e.get("realized_pnl") is not None]
    total_losses = sum(1 for e in closed if e["realized_pnl"] < 0)
    redundant_loss_rate = (
        redundant_losses / total_losses if total_losses > 0 else 0
    )

    # pattern_discovery_count: patterns with N>=5 and clear signal (>55% or <30% win rate)
    pattern_discovery_count = sum(
        1 for v in theses.values()
        if v["trades"] >= 5 and (v["win_rate"] > 0.55 or v["win_rate"] < 0.30)
    )
    # Also count multi-tag patterns
    multi = ledger.get("multi_tag_patterns", {})
    pattern_discovery_count += sum(
        1 for v in multi.values()
        if v["trades"] >= 5 and (v["win_rate"] > 0.55 or v["win_rate"] < 0.30)
    )

    # calibration_error: avg |stated confidence - actual win rate| by confidence band
    cal = ledger.get("calibration", {})
    cal_error = 0.0
    if cal.get("high_confidence_win_rate") is not None:
        # High conf band: stated ~0.8, actual is high_confidence_win_rate
        cal_error += abs(0.8 - cal["high_confidence_win_rate"])
        bands = 1
        if cal.get("low_confidence_win_rate") is not None:
            cal_error += abs(0.35 - cal["low_confidence_win_rate"])
            bands += 1
        cal_error /= bands

    # rolling_win_rate_10: win rate of last 10 closed trades
    recent_closed = [e for e in closed if e.get("realized_pnl") is not None]
    recent_closed.sort(key=lambda e: e.get("closed_at", e.get("timestamp", "")))
    last_10 = recent_closed[-10:] if len(recent_closed) >= 10 else recent_closed
    rolling_wr = None
    if last_10:
        wins_10 = sum(1 for e in last_10 if e["realized_pnl"] >= 0)
        rolling_wr = wins_10 / len(last_10)

    return {
        "experiment_efficiency": round(experiment_efficiency, 3),
        "redundant_loss_rate": round(min(redundant_loss_rate, 1.0), 3),
        "pattern_discovery_count": pattern_discovery_count,
        "calibration_error": round(cal_error, 3),
        "rolling_win_rate_10": round(rolling_wr, 3) if rolling_wr is not None else None,
    }


def get_learning_metrics(ledger: dict) -> dict:
    """Compute Mistake Repetition Rate and Thesis Calibration Score.

    Also includes v6 research metrics.
    """
    theses = ledger.get("theses", {})
    total = ledger.get("active_trades", 0)

    # Identify "avoid" theses (win rate < 35%, N>=5)
    avoid_theses = {
        k: v for k, v in theses.items()
        if v["trades"] >= 5 and v["win_rate"] < 0.35
    }

    # Mistake repetition: how many recent trades hit an "avoid" thesis?
    avoid_trades = sum(v["trades"] for v in avoid_theses.values())

    # Include research metrics
    research = compute_research_metrics(ledger)

    return {
        "avoid_theses": list(avoid_theses.keys()),
        "avoid_trade_count": avoid_trades,
        "mistake_rate": round(avoid_trades / total, 3) if total > 0 else 0,
        "calibration": ledger.get("calibration", {}),
        "total_theses": len(theses),
        "significant_theses": len([v for v in theses.values() if v["trades"] >= 3]),
        "research_metrics": research,
    }
