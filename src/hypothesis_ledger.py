"""Hypothesis Ledger — formal lifecycle tracking for trade hypotheses.

Every trade and HOLD decision includes a hypothesis from Claude. This module
tracks them through their lifecycle: proposed → tested → confirmed/refuted.

Hypotheses are grouped by (category, condition_tags) — not by free text — to
prevent duplicates and enable aggregation. Auto-retires hypotheses that have
been tested 5+ times with <30% confirmation rate.
"""

import re
from datetime import datetime, timezone, timedelta
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("hypothesis_ledger")

HYPOTHESIS_PATH = DATA_DIR / "hypothesis_ledger.json"
JOURNAL_PATH = DATA_DIR / "trade_journal.json"
HOLD_JOURNAL_PATH = DATA_DIR / "hold_journal.json"

# Category keywords for classification
CATEGORY_KEYWORDS = {
    "mean_reversion": ["mean reversion", "revert", "bounce", "oversold", "overbought", "snap back", "reversal"],
    "momentum": ["momentum", "breakout", "break out", "trending", "continuation", "follow-through"],
    "news_catalyst": ["news", "catalyst", "announcement", "headline", "report", "earnings"],
    "range_play": ["range", "consolidat", "support", "resistance", "channel", "bounded"],
    "trend_following": ["trend", "sma", "moving average", "direction", "slope"],
    "volatility": ["volatility", "vix", "squeeze", "expansion", "contraction"],
}

# Tags relevant to hypothesis conditions (subset of all tags)
HYPOTHESIS_TAGS = [
    "rsi_zone", "trend", "volatility", "regime", "macd",
    "market_context", "time_of_day",
]

# Auto-retire thresholds
MIN_TESTS_FOR_RETIREMENT = 5
MAX_CONFIRMATION_RATE_FOR_RETIREMENT = 0.30


def _load_ledger() -> dict:
    """Load the hypothesis ledger."""
    default = {
        "last_updated": iso_now(),
        "hypotheses": {},
        "next_id": 1,
    }
    ledger = load_json(HYPOTHESIS_PATH, default=default)
    if "hypotheses" not in ledger:
        ledger["hypotheses"] = {}
    if "next_id" not in ledger:
        ledger["next_id"] = len(ledger["hypotheses"]) + 1
    return ledger


def _save_ledger(ledger: dict):
    """Save the hypothesis ledger."""
    ledger["last_updated"] = iso_now()
    save_json(HYPOTHESIS_PATH, ledger)


def _classify_category(text: str) -> str:
    """Infer hypothesis category from free text."""
    text_lower = text.lower()
    best_cat = "unknown"
    best_count = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_cat = category

    return best_cat


def _extract_condition_key(tags: dict) -> tuple:
    """Extract the relevant subset of tags as a hashable condition key."""
    relevant = {}
    for tag_name in HYPOTHESIS_TAGS:
        if tag_name in tags:
            relevant[tag_name] = tags[tag_name]
    return tuple(sorted(relevant.items()))


def _condition_key_to_str(condition_key: tuple) -> str:
    """Convert a condition key tuple to a readable string."""
    return " + ".join(f"{k}={v}" for k, v in condition_key)


def _find_matching_hypothesis(ledger: dict, category: str, condition_key: tuple) -> str | None:
    """Find an existing hypothesis matching this category + conditions."""
    condition_set = set(condition_key)
    for hyp_id, hyp in ledger["hypotheses"].items():
        if hyp.get("category") != category:
            continue
        if hyp.get("status") == "retired":
            continue
        # Check if condition keys overlap significantly (>= 60%)
        hyp_conditions = set(tuple(sorted(hyp.get("conditions", {}).items())))
        if not hyp_conditions:
            continue
        overlap = len(condition_set & hyp_conditions)
        max_len = max(len(condition_set), len(hyp_conditions))
        if max_len > 0 and overlap / max_len >= 0.6:
            return hyp_id
    return None


def log_hypothesis(decision: dict, tags: dict, trade_id: str = None):
    """Log a hypothesis from a decision (BUY, SELL, or HOLD).

    Extracts hypothesis text and conditions, finds or creates a matching
    hypothesis entry, increments test count, links to trade_id.

    Args:
        decision: Claude's decision dict (must include 'hypothesis')
        tags: Current market condition tags
        trade_id: The trade ID to link this hypothesis to
    """
    hypothesis_text = decision.get("hypothesis", "")
    if not hypothesis_text or hypothesis_text.lower() in ("n/a", "n/a - observing", ""):
        return

    category = _classify_category(hypothesis_text)
    condition_key = _extract_condition_key(tags)

    ledger = _load_ledger()

    # Find existing matching hypothesis
    existing_id = _find_matching_hypothesis(ledger, category, condition_key)

    if existing_id:
        hyp = ledger["hypotheses"][existing_id]
        hyp["test_count"] = hyp.get("test_count", 0) + 1
        hyp["last_tested"] = iso_now()
        # Update text if this one is more descriptive
        if len(hypothesis_text) > len(hyp.get("text", "")):
            hyp["text"] = hypothesis_text[:300]
        # Link trade_id
        if trade_id:
            source_ids = hyp.get("source_trade_ids", [])
            if trade_id not in source_ids:
                source_ids.append(trade_id)
                hyp["source_trade_ids"] = source_ids[-10:]
    else:
        # Create new hypothesis
        hyp_id = f"hyp_{ledger['next_id']:04d}"
        ledger["next_id"] += 1
        ledger["hypotheses"][hyp_id] = {
            "id": hyp_id,
            "text": hypothesis_text[:300],
            "category": category,
            "conditions": dict(condition_key),
            "status": "active",
            "proposed_at": iso_now(),
            "last_tested": iso_now(),
            "test_count": 1,
            "confirmations": 0,
            "refutations": 0,
            "confirmation_rate": 0.0,
            "source_trade_ids": [trade_id] if trade_id else [],
        }

    _save_ledger(ledger)


def resolve_hypothesis(trade_id: str, pnl: float):
    """Resolve a hypothesis when a trade closes.

    First tries to find the hypothesis linked to this trade_id.
    Falls back to the most recently tested active hypothesis.

    Args:
        trade_id: The trade ID that closed
        pnl: The realized P&L
    """
    ledger = _load_ledger()
    outcome = "confirmed" if pnl >= 0 else "refuted"

    # First: find hypothesis linked to this trade_id
    hyp_id = None
    hyp = None
    for hid, h in ledger["hypotheses"].items():
        if h.get("status") != "active":
            continue
        if trade_id in h.get("source_trade_ids", []):
            hyp_id = hid
            hyp = h
            break

    # Fallback: most recently tested active hypothesis
    if hyp is None:
        active = [
            (hid, h) for hid, h in ledger["hypotheses"].items()
            if h.get("status") == "active"
        ]
        if not active:
            return
        active.sort(key=lambda x: x[1].get("last_tested", ""), reverse=True)
        hyp_id, hyp = active[0]

    if outcome == "confirmed":
        hyp["confirmations"] = hyp.get("confirmations", 0) + 1
    else:
        hyp["refutations"] = hyp.get("refutations", 0) + 1

    total_tests = hyp.get("confirmations", 0) + hyp.get("refutations", 0)
    hyp["confirmation_rate"] = round(
        hyp.get("confirmations", 0) / total_tests, 3
    ) if total_tests > 0 else 0

    # Track trade IDs
    source_ids = hyp.get("source_trade_ids", [])
    if trade_id not in source_ids:
        source_ids.append(trade_id)
        hyp["source_trade_ids"] = source_ids[-10:]  # Keep last 10

    _save_ledger(ledger)
    logger.info(
        f"Hypothesis {hyp_id} {outcome}: {hyp['text'][:60]}... "
        f"({hyp['confirmations']}/{total_tests} confirmed)"
    )


def auto_retire_hypotheses():
    """Retire hypotheses that have been tested enough and consistently fail.

    Retires if: test_count >= 5 and confirmation_rate < 30%.
    """
    ledger = _load_ledger()
    retired_count = 0

    for hyp_id, hyp in ledger["hypotheses"].items():
        if hyp.get("status") != "active":
            continue

        total_tests = hyp.get("confirmations", 0) + hyp.get("refutations", 0)
        if total_tests < MIN_TESTS_FOR_RETIREMENT:
            continue

        conf_rate = hyp.get("confirmation_rate", 0)
        if conf_rate < MAX_CONFIRMATION_RATE_FOR_RETIREMENT:
            hyp["status"] = "retired"
            hyp["retirement_reason"] = (
                f"Tested {total_tests}x with {conf_rate:.0%} confirmation rate "
                f"(below {MAX_CONFIRMATION_RATE_FOR_RETIREMENT:.0%} threshold)"
            )
            hyp["retired_at"] = iso_now()
            retired_count += 1
            logger.info(
                f"Retired hypothesis {hyp_id}: {hyp['text'][:60]}... "
                f"({conf_rate:.0%} confirmation rate)"
            )

    if retired_count > 0:
        _save_ledger(ledger)

    return retired_count


def suggest_next_hypotheses(current_tags: dict = None) -> list:
    """Suggest hypotheses to test based on gaps and category performance.

    Returns list of suggestion dicts.
    """
    ledger = _load_ledger()
    hypotheses = ledger.get("hypotheses", {})

    if not hypotheses:
        return [{
            "description": "Start with any hypothesis — no data yet",
            "reason": "No hypotheses tracked. State a specific, testable hypothesis with your first trade.",
        }]

    # Aggregate by category
    category_stats = {}
    for hyp in hypotheses.values():
        cat = hyp.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0, "active": 0, "retired": 0,
                "total_tests": 0, "total_confirmations": 0,
            }
        category_stats[cat]["total"] += 1
        if hyp.get("status") == "active":
            category_stats[cat]["active"] += 1
        elif hyp.get("status") == "retired":
            category_stats[cat]["retired"] += 1
        category_stats[cat]["total_tests"] += (
            hyp.get("confirmations", 0) + hyp.get("refutations", 0)
        )
        category_stats[cat]["total_confirmations"] += hyp.get("confirmations", 0)

    for cat, stats in category_stats.items():
        stats["avg_confirmation_rate"] = round(
            stats["total_confirmations"] / stats["total_tests"], 3
        ) if stats["total_tests"] > 0 else None

    suggestions = []

    # Suggest categories that are under-tested
    all_categories = set(CATEGORY_KEYWORDS.keys())
    tested_categories = set(category_stats.keys())
    untested = all_categories - tested_categories

    for cat in untested:
        suggestions.append({
            "description": f"Test a {cat} hypothesis",
            "category": cat,
            "reason": f"No {cat} hypotheses tested yet — unknown territory",
        })

    # Suggest the best-performing category for more testing
    best_cat = max(
        [(cat, stats) for cat, stats in category_stats.items()
         if stats.get("avg_confirmation_rate") is not None and stats["total_tests"] >= 3],
        key=lambda x: x[1]["avg_confirmation_rate"],
        default=None,
    )
    if best_cat:
        cat_name, stats = best_cat
        if stats["avg_confirmation_rate"] > 0.4:
            suggestions.append({
                "description": f"More {cat_name} experiments (best category)",
                "category": cat_name,
                "reason": (
                    f"{cat_name} has {stats['avg_confirmation_rate']:.0%} confirmation rate "
                    f"over {stats['total_tests']} tests — most promising"
                ),
            })

    return suggestions[:3]


def format_hypothesis_ledger_for_brief(current_tags: dict = None) -> str:
    """Format hypothesis ledger for the agent's brief.

    Shows: retired hypotheses matching current conditions (NEVER TRADE),
    active hypotheses (what's being tested), suggested next.

    Returns 3-5 lines of compact text, or empty string.
    """
    ledger = _load_ledger()
    hypotheses = ledger.get("hypotheses", {})

    if not hypotheses:
        return ""

    parts = []
    condition_key = _extract_condition_key(current_tags) if current_tags else ()

    # Count stats
    active = [h for h in hypotheses.values() if h.get("status") == "active"]
    retired = [h for h in hypotheses.values() if h.get("status") == "retired"]

    parts.append(f"Hypotheses: {len(active)} active, {len(retired)} retired")

    # Show retired hypotheses matching current conditions
    if retired and current_tags:
        for hyp in retired:
            hyp_conditions = set(tuple(sorted(hyp.get("conditions", {}).items())))
            current_conditions = set(condition_key)
            if not hyp_conditions:
                continue
            overlap = len(current_conditions & hyp_conditions)
            max_len = max(len(current_conditions), len(hyp_conditions))
            if max_len > 0 and overlap / max_len >= 0.5:
                parts.append(
                    f"  RETIRED [{hyp['category']}]: \"{hyp['text'][:80]}\" — "
                    f"{hyp.get('confirmation_rate', 0):.0%} confirmed over "
                    f"{hyp.get('confirmations', 0) + hyp.get('refutations', 0)} tests"
                )

    # Show active hypotheses matching current conditions
    if active and current_tags:
        matching_active = []
        for hyp in active:
            hyp_conditions = set(tuple(sorted(hyp.get("conditions", {}).items())))
            current_conditions = set(condition_key)
            if not hyp_conditions:
                continue
            overlap = len(current_conditions & hyp_conditions)
            max_len = max(len(current_conditions), len(hyp_conditions))
            if max_len > 0 and overlap / max_len >= 0.5:
                matching_active.append(hyp)

        for hyp in matching_active[:2]:
            total = hyp.get("confirmations", 0) + hyp.get("refutations", 0)
            rate = hyp.get("confirmation_rate", 0)
            parts.append(
                f"  ACTIVE [{hyp['category']}]: \"{hyp['text'][:80]}\" — "
                f"{rate:.0%} confirmed ({total} tests)"
            )

    # Suggest next hypothesis
    suggestions = suggest_next_hypotheses(current_tags)
    if suggestions:
        s = suggestions[0]
        parts.append(f"  EXPLORE: {s['description']} — {s['reason']}")

    return "\n".join(parts) if len(parts) > 1 else ""


def rebuild_hypothesis_ledger():
    """Bootstrap hypothesis ledger from existing trade journal if empty.

    Called at first run or nightly to ensure the ledger reflects all
    historical hypotheses.
    """
    ledger = _load_ledger()

    # If ledger already has hypotheses, just auto-retire
    if ledger.get("hypotheses"):
        retired = auto_retire_hypotheses()
        logger.info(f"Hypothesis ledger: {retired} newly retired")
        return

    # Bootstrap from trade journal
    journal = load_json(JOURNAL_PATH, default=[])
    bootstrapped = 0

    for entry in journal:
        hypothesis_text = entry.get("hypothesis", "")
        tags = entry.get("tags", {})
        if not hypothesis_text or not tags:
            continue

        category = _classify_category(hypothesis_text)
        condition_key = _extract_condition_key(tags)

        hyp_id = f"hyp_{ledger['next_id']:04d}"
        ledger["next_id"] += 1

        pnl = entry.get("realized_pnl")
        confirmations = 1 if pnl is not None and pnl >= 0 else 0
        refutations = 1 if pnl is not None and pnl < 0 else 0
        total = confirmations + refutations

        ledger["hypotheses"][hyp_id] = {
            "id": hyp_id,
            "text": hypothesis_text[:300],
            "category": category,
            "conditions": dict(condition_key),
            "status": "active",
            "proposed_at": entry.get("timestamp", iso_now()),
            "last_tested": entry.get("closed_at", entry.get("timestamp", iso_now())),
            "test_count": 1,
            "confirmations": confirmations,
            "refutations": refutations,
            "confirmation_rate": round(confirmations / total, 3) if total > 0 else 0,
            "source_trade_ids": [entry.get("trade_id", "")] if entry.get("trade_id") else [],
        }
        bootstrapped += 1

    if bootstrapped > 0:
        _save_ledger(ledger)
        logger.info(f"Hypothesis ledger bootstrapped with {bootstrapped} entries from trade journal")

    # Auto-retire after bootstrap
    auto_retire_hypotheses()
