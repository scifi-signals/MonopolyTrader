"""Hold Analyzer — learns from HOLD decisions by comparing outcomes to trades.

Aggregates HOLDs by market conditions (tags). For each condition group,
computes how often holding was the right call vs trading. Compares
HOLD outcomes against trade outcomes from the playbook.

This closes the loop: the researcher doesn't just avoid known-bad patterns,
it learns WHERE holding is better than trading (and vice versa).
"""

from datetime import datetime, timezone, timedelta
from itertools import combinations
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("hold_analyzer")

HOLD_JOURNAL_PATH = DATA_DIR / "hold_journal.json"
LEDGER_PATH = DATA_DIR / "thesis_ledger.json"
HOLD_ANALYSIS_PATH = DATA_DIR / "hold_analysis.json"

# Thresholds
MEANINGFUL_MOVE_PCT = 0.2  # Match shadow_journal.py
MIN_COUNT_FOR_RECOMMENDATION = 5
EDGE_THRESHOLD = 0.15  # Minimum edge to recommend HOLD_BETTER or TRADE_BETTER


def analyze_holds(lookback_days: int = 30) -> dict:
    """Aggregate resolved HOLDs by market conditions and compare to playbook.

    Groups HOLDs by single-tag and 2-tag combos. For each group:
    - Computes hold_win_rate (correctly avoided loss / total meaningful)
    - Compares against trade win_rate from thesis_ledger for same conditions
    - Generates recommendation: HOLD_BETTER / TRADE_BETTER / INSUFFICIENT_DATA

    Returns analysis dict.
    """
    journal = load_json(HOLD_JOURNAL_PATH, default=[])
    ledger = load_json(LEDGER_PATH, default={})

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Filter to resolved HOLDs within lookback window
    resolved = []
    for entry in journal:
        if not entry.get("shadow_resolved"):
            continue
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                resolved.append(entry)
        except (ValueError, KeyError):
            continue

    if not resolved:
        return {
            "last_updated": iso_now(),
            "total_holds_analyzed": 0,
            "total_resolved": 0,
            "by_tag": {},
            "by_multi_tag": {},
            "summary": {
                "overall_hold_quality": 0,
                "conditions_where_hold_better": [],
                "conditions_where_trade_better": [],
                "insufficient_data": [],
            },
        }

    # Classify each HOLD outcome
    for entry in resolved:
        shadow_pnl = entry.get("shadow_pnl_buy")
        entry_price = entry.get("price", 0)
        if shadow_pnl is None or entry_price <= 0:
            entry["_hold_outcome"] = "neutral"
            continue

        final_price = entry.get("shadow_final_price", entry_price)
        move_pct = abs(final_price - entry_price) / entry_price * 100 if entry_price > 0 else 0

        if move_pct < MEANINGFUL_MOVE_PCT:
            entry["_hold_outcome"] = "neutral"
        elif shadow_pnl < 0:
            entry["_hold_outcome"] = "avoided_loss"  # Price went down, good HOLD
        else:
            entry["_hold_outcome"] = "missed_gain"  # Price went up, bad HOLD

    # Group by single tags
    by_tag = _group_by_single_tags(resolved, ledger)

    # Group by 2-tag combos (most common only)
    by_multi_tag = _group_by_multi_tags(resolved, ledger)

    # Summary
    all_meaningful = [e for e in resolved if e.get("_hold_outcome") != "neutral"]
    avoided = sum(1 for e in all_meaningful if e["_hold_outcome"] == "avoided_loss")
    overall_quality = avoided / len(all_meaningful) if all_meaningful else 0

    conditions_hold_better = [
        k for k, v in {**by_tag, **by_multi_tag}.items()
        if v.get("recommendation") == "HOLD_BETTER"
    ]
    conditions_trade_better = [
        k for k, v in {**by_tag, **by_multi_tag}.items()
        if v.get("recommendation") == "TRADE_BETTER"
    ]
    insufficient = [
        k for k, v in {**by_tag, **by_multi_tag}.items()
        if v.get("recommendation") == "INSUFFICIENT_DATA"
    ]

    analysis = {
        "last_updated": iso_now(),
        "total_holds_analyzed": len(journal),
        "total_resolved": len(resolved),
        "by_tag": by_tag,
        "by_multi_tag": by_multi_tag,
        "summary": {
            "overall_hold_quality": round(overall_quality, 3),
            "conditions_where_hold_better": conditions_hold_better,
            "conditions_where_trade_better": conditions_trade_better,
            "insufficient_data": insufficient,
        },
    }

    return analysis


def _group_by_single_tags(resolved: list, ledger: dict) -> dict:
    """Group resolved HOLDs by individual tag values."""
    groups = {}
    theses = ledger.get("theses", {})

    for entry in resolved:
        tags = entry.get("tags", {})
        outcome = entry.get("_hold_outcome", "neutral")

        for tag_name, tag_value in tags.items():
            key = f"{tag_name}={tag_value}"
            if key not in groups:
                groups[key] = {
                    "hold_count": 0,
                    "avoided_loss": 0,
                    "missed_gain": 0,
                    "neutral": 0,
                }
            groups[key]["hold_count"] += 1
            if outcome == "avoided_loss":
                groups[key]["avoided_loss"] += 1
            elif outcome == "missed_gain":
                groups[key]["missed_gain"] += 1
            else:
                groups[key]["neutral"] += 1

    # Compute metrics and compare to playbook
    for key, stats in groups.items():
        meaningful = stats["avoided_loss"] + stats["missed_gain"]
        stats["hold_win_rate"] = round(
            stats["avoided_loss"] / meaningful, 3
        ) if meaningful > 0 else 0

        # Look up trade win rate from playbook
        # Playbook uses "tag_name:tag_value" format
        ledger_key = key.replace("=", ":")
        playbook_stats = theses.get(ledger_key, {})
        stats["playbook_trade_win_rate"] = playbook_stats.get("win_rate", None)
        stats["playbook_trade_count"] = playbook_stats.get("trades", 0)

        # Generate recommendation
        stats["recommendation"] = _recommend(
            stats["hold_win_rate"],
            stats["playbook_trade_win_rate"],
            meaningful,
            stats["playbook_trade_count"],
        )

    return groups


def _group_by_multi_tags(resolved: list, ledger: dict) -> dict:
    """Group resolved HOLDs by 2-tag combinations (top 10 most frequent)."""
    # Count all 2-tag combos
    combo_counts = {}
    for entry in resolved:
        tags = entry.get("tags", {})
        tag_items = sorted(tags.items())
        for pair in combinations(tag_items, 2):
            combo_key = " + ".join(f"{k}={v}" for k, v in pair)
            combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1

    # Keep top 10 most frequent combos with N >= 5
    frequent = sorted(
        [(k, n) for k, n in combo_counts.items() if n >= 5],
        key=lambda x: -x[1],
    )[:10]

    if not frequent:
        return {}

    frequent_keys = {k for k, _ in frequent}

    # Build stats for frequent combos
    groups = {}
    multi_patterns = ledger.get("multi_tag_patterns", {})

    for entry in resolved:
        tags = entry.get("tags", {})
        outcome = entry.get("_hold_outcome", "neutral")
        tag_items = sorted(tags.items())

        for pair in combinations(tag_items, 2):
            combo_key = " + ".join(f"{k}={v}" for k, v in pair)
            if combo_key not in frequent_keys:
                continue

            if combo_key not in groups:
                groups[combo_key] = {
                    "hold_count": 0,
                    "avoided_loss": 0,
                    "missed_gain": 0,
                    "neutral": 0,
                }
            groups[combo_key]["hold_count"] += 1
            if outcome == "avoided_loss":
                groups[combo_key]["avoided_loss"] += 1
            elif outcome == "missed_gain":
                groups[combo_key]["missed_gain"] += 1
            else:
                groups[combo_key]["neutral"] += 1

    # Compute metrics
    for key, stats in groups.items():
        meaningful = stats["avoided_loss"] + stats["missed_gain"]
        stats["hold_win_rate"] = round(
            stats["avoided_loss"] / meaningful, 3
        ) if meaningful > 0 else 0

        playbook_stats = multi_patterns.get(key, {})
        stats["playbook_trade_win_rate"] = playbook_stats.get("win_rate", None)
        stats["playbook_trade_count"] = playbook_stats.get("trades", 0)

        stats["recommendation"] = _recommend(
            stats["hold_win_rate"],
            stats["playbook_trade_win_rate"],
            meaningful,
            stats["playbook_trade_count"],
        )

    return groups


def _recommend(
    hold_wr: float,
    trade_wr: float | None,
    hold_n: int,
    trade_n: int,
) -> str:
    """Generate recommendation based on HOLD vs trade win rates."""
    if hold_n + trade_n < MIN_COUNT_FOR_RECOMMENDATION:
        return "INSUFFICIENT_DATA"

    if trade_wr is None:
        # No trade data — can't compare
        if hold_n >= MIN_COUNT_FOR_RECOMMENDATION and hold_wr > 0.65:
            return "HOLD_BETTER"
        return "INSUFFICIENT_DATA"

    edge = hold_wr - trade_wr
    if edge > EDGE_THRESHOLD:
        return "HOLD_BETTER"
    elif edge < -EDGE_THRESHOLD:
        return "TRADE_BETTER"
    else:
        return "COMPARABLE"


def format_hold_analysis_for_brief(current_tags: dict = None) -> str:
    """Format hold analysis for the agent's brief.

    Shows conditions where TRADE_BETTER (opportunities to break out of holding)
    and strong HOLD_BETTER (reinforcement to keep holding).

    Returns compact text (3-5 lines) or empty string.
    """
    analysis = load_json(HOLD_ANALYSIS_PATH, default={})
    if not analysis or analysis.get("total_resolved", 0) < 5:
        return ""

    parts = []
    summary = analysis.get("summary", {})
    quality = summary.get("overall_hold_quality", 0)

    # Show overall quality
    total = analysis.get("total_resolved", 0)
    parts.append(f"Hold analysis ({total} resolved): {quality:.0%} correctly avoided losses")

    # Highlight TRADE_BETTER conditions matching current tags
    trade_better = []
    hold_better = []
    all_conditions = {**analysis.get("by_tag", {}), **analysis.get("by_multi_tag", {})}

    for key, stats in all_conditions.items():
        rec = stats.get("recommendation")
        if rec == "TRADE_BETTER":
            trade_better.append((key, stats))
        elif rec == "HOLD_BETTER":
            hold_better.append((key, stats))

    # Show TRADE_BETTER conditions (these are opportunities)
    if trade_better:
        # Filter to conditions matching current tags if provided
        matching = trade_better
        if current_tags:
            matching = [
                (k, s) for k, s in trade_better
                if _condition_matches_tags(k, current_tags)
            ]
        if matching:
            best = sorted(matching, key=lambda x: x[1].get("hold_win_rate", 1))[:2]
            for key, stats in best:
                trade_wr = stats.get("playbook_trade_win_rate", 0) or 0
                hold_wr = stats.get("hold_win_rate", 0)
                parts.append(
                    f"  TRADE opportunity: {key} — hold avoids {hold_wr:.0%} but "
                    f"trades win {trade_wr:.0%}"
                )

    # Show HOLD_BETTER conditions matching current tags
    if hold_better and current_tags:
        matching = [
            (k, s) for k, s in hold_better
            if _condition_matches_tags(k, current_tags)
        ]
        if matching:
            best = sorted(matching, key=lambda x: -x[1].get("hold_win_rate", 0))[:2]
            for key, stats in best:
                hold_wr = stats.get("hold_win_rate", 0)
                trade_wr = stats.get("playbook_trade_win_rate", 0) or 0
                parts.append(
                    f"  HOLD confirmed: {key} — hold avoids {hold_wr:.0%} vs "
                    f"trade wins {trade_wr:.0%}"
                )

    return "\n".join(parts) if len(parts) > 1 else ""


def _condition_matches_tags(condition_key: str, current_tags: dict) -> bool:
    """Check if a condition key matches the current tag values."""
    # Handle both single ("rsi_zone=neutral") and multi ("rsi_zone=neutral + regime=range_bound")
    parts = [p.strip() for p in condition_key.split("+")]
    for part in parts:
        if "=" in part:
            tag_name, tag_value = part.split("=", 1)
            tag_name = tag_name.strip()
            tag_value = tag_value.strip()
            if current_tags.get(tag_name) != tag_value:
                return False
    return True


def rebuild_hold_analysis():
    """Rebuild hold analysis and save to disk. Called nightly."""
    analysis = analyze_holds(lookback_days=30)
    save_json(HOLD_ANALYSIS_PATH, analysis)
    logger.info(
        f"Hold analysis rebuilt: {analysis['total_resolved']} resolved, "
        f"{len(analysis.get('summary', {}).get('conditions_where_hold_better', []))} HOLD_BETTER, "
        f"{len(analysis.get('summary', {}).get('conditions_where_trade_better', []))} TRADE_BETTER"
    )
    return analysis
