"""Thesis Builder — nightly aggregation of trade tags into a statistical playbook.

Reads all trades from trade_journal.json, groups by single-tag values,
computes win rates and P&L stats, writes thesis_ledger.json.

Runs after market close. No LLM calls. Pure Python math.
"""

from datetime import datetime, timezone, timedelta
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

    Groups trades by individual tag values, computes weighted stats.
    Returns the complete ledger dict.
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
        n = t["trades"]
        t["win_rate"] = round(t["weighted_wins"] / t["weighted_trades"], 3) if t["weighted_trades"] > 0 else 0
        t["avg_pnl"] = round(t["total_pnl"] / n, 2) if n > 0 else 0
        t["avg_pnl_pct"] = round(t["total_pnl_pct"] / n, 2) if n > 0 else 0
        t["avg_confidence"] = round(sum(t["confidences"]) / n, 2) if n > 0 else 0
        t["total_pnl"] = round(t["total_pnl"], 2)

        # Best and worst trade
        if t["pnl_values"]:
            t["best_pnl"] = round(max(t["pnl_values"]), 2)
            t["worst_pnl"] = round(min(t["pnl_values"]), 2)

        # Clean up internal fields
        del t["weighted_trades"]
        del t["weighted_wins"]
        del t["total_pnl_pct"]
        del t["confidences"]
        del t["pnl_values"]

    # Confidence calibration
    calibration = _compute_calibration(weighted_trades)

    # Validation
    warnings = validate_ledger(theses, len(weighted_trades))

    ledger = {
        "last_updated": iso_now(),
        "total_trades": total_all,
        "active_trades": len(weighted_trades),
        "theses": theses,
        "calibration": calibration,
        "validation_warnings": warnings,
    }

    save_json(LEDGER_PATH, ledger)
    logger.info(
        f"Thesis ledger built: {len(weighted_trades)} active trades, "
        f"{len(theses)} theses, {len(warnings)} warnings"
    )
    return ledger


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

    # Check tag coverage — each trade should appear in exactly 8 tags
    total_from_theses = sum(s["trades"] for s in theses.values())
    expected = total_trades * 8
    if total_from_theses != expected and total_trades > 0:
        warnings.append(
            f"TAG COVERAGE: thesis trade count ({total_from_theses}) "
            f"doesn't match expected ({expected}). Some trades may have missing tags."
        )

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
        parts.append("Best setups (win rate > 55%, N≥3):")
        for key, s in sorted(best.items(), key=lambda x: x[1]["win_rate"], reverse=True):
            tag_label = key.replace(":", "=")
            parts.append(
                f"  {tag_label}: {s['wins']}/{s['trades']} wins ({s['win_rate']:.0%}), "
                f"avg {'+' if s['avg_pnl'] >= 0 else ''}${s['avg_pnl']:.2f}"
            )

    # Worst setups (win rate < 40%)
    worst = {k: v for k, v in significant.items() if v["win_rate"] < 0.40}
    if worst:
        parts.append("Worst setups (win rate < 40%, N≥3):")
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

    parts.append(f"(Based on {active} trades from last 90 days, {total} total)")

    return "\n".join(parts)


def get_learning_metrics(ledger: dict) -> dict:
    """Compute Mistake Repetition Rate and Thesis Calibration Score."""
    theses = ledger.get("theses", {})
    total = ledger.get("active_trades", 0)

    # Identify "avoid" theses (win rate < 35%, N>=5)
    avoid_theses = {
        k: v for k, v in theses.items()
        if v["trades"] >= 5 and v["win_rate"] < 0.35
    }

    # Mistake repetition: how many recent trades hit an "avoid" thesis?
    # This is computed from the ledger's per-thesis trade counts
    avoid_trades = sum(v["trades"] for v in avoid_theses.values())

    return {
        "avoid_theses": list(avoid_theses.keys()),
        "avoid_trade_count": avoid_trades,
        "mistake_rate": round(avoid_trades / total, 3) if total > 0 else 0,
        "calibration": ledger.get("calibration", {}),
        "total_theses": len(theses),
        "significant_theses": len([v for v in theses.values() if v["trades"] >= 3]),
    }
