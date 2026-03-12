"""Constraint Generator — hard, data-driven constraints from learning modules.

Synthesizes signals from the playbook, prediction diagnosis, hold analysis,
and hypothesis ledger into concrete constraints that the agent MUST follow.

Constraints are not suggestions — they're empirically-derived rules that
prevent the researcher from repeating known-bad experiments.

Also computes advisory position sizing based on condition confidence.
"""

from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("constraint_generator")

CONSTRAINTS_PATH = DATA_DIR / "active_constraints.json"
LEDGER_PATH = DATA_DIR / "thesis_ledger.json"
DIAGNOSIS_PATH = DATA_DIR / "prediction_diagnosis.json"
HOLD_ANALYSIS_PATH = DATA_DIR / "hold_analysis.json"
HYPOTHESIS_PATH = DATA_DIR / "hypothesis_ledger.json"

# Thresholds
NEVER_TRADE_WIN_RATE = 0.25
NEVER_TRADE_MIN_N = 5
LOW_CONFIDENCE_ACCURACY = 0.40
LOW_CONFIDENCE_MIN_N = 10
HOLD_PREFERRED_EDGE = 0.20
HOLD_PREFERRED_MIN_N = 5

# Position sizing tiers
SIZE_TIERS = {
    "minimum": 0.1,   # Exploratory — unknown conditions
    "small": 0.25,    # Weak signal — some data but mixed
    "normal": 0.5,    # Standard — neutral or moderate confidence
    "confident": 0.75, # Strong positive signal from playbook
}


def generate_constraints() -> dict:
    """Generate hard constraints from all learning module outputs.

    Sources:
    - Playbook: N >= 5 and win_rate < 25% → NEVER_TRADE
    - Prediction diagnosis: accuracy < 40% with N >= 10 → LOW_CONFIDENCE
    - Hold analysis: hold_win_rate exceeds trade_win_rate by 20%+ → HOLD_PREFERRED
    - Hypothesis ledger: retired hypotheses → RETIRED_HYPOTHESIS

    Returns structured constraints dict.
    """
    constraints = []

    # === Source 1: Playbook NEVER_TRADE ===
    ledger = load_json(LEDGER_PATH, default={})
    theses = ledger.get("theses", {})
    multi = ledger.get("multi_tag_patterns", {})

    for key, stats in {**theses, **multi}.items():
        n = stats.get("trades", 0)
        wr = stats.get("win_rate", 1.0)
        if n >= NEVER_TRADE_MIN_N and wr < NEVER_TRADE_WIN_RATE:
            # Normalize key format (playbook uses ":" for single, "=" for multi)
            pattern_key = key.replace(":", "=")
            constraints.append({
                "type": "NEVER_TRADE",
                "source": "playbook",
                "pattern": pattern_key,
                "reason": (
                    f"{stats.get('wins', 0)}/{n} wins ({wr:.0%}) — "
                    f"avg P&L ${stats.get('avg_pnl', 0):+.2f}"
                ),
                "trades": n,
                "win_rate": wr,
            })

    # === Source 2: Prediction diagnosis LOW_CONFIDENCE ===
    diagnosis = load_json(DIAGNOSIS_PATH, default={})
    if not diagnosis.get("insufficient_data", True):
        accuracy_by_condition = diagnosis.get("accuracy_by_condition", {})
        for condition, stats in accuracy_by_condition.items():
            n = stats.get("total", 0)
            acc = stats.get("accuracy", 1.0)
            if n >= LOW_CONFIDENCE_MIN_N and acc < LOW_CONFIDENCE_ACCURACY:
                constraints.append({
                    "type": "LOW_CONFIDENCE",
                    "source": "prediction_diagnosis",
                    "pattern": condition,
                    "reason": (
                        f"Prediction accuracy {acc:.0%} over {n} predictions — "
                        f"your reads are unreliable here"
                    ),
                    "predictions": n,
                    "accuracy": acc,
                })

        # Overall direction accuracy constraint
        cal = diagnosis.get("confidence_calibration", {})
        overall_acc = cal.get("actual_direction_accuracy", 1.0)
        total_analyzed = diagnosis.get("total_analyzed", 0)
        if total_analyzed >= 20 and overall_acc < 0.40:
            constraints.append({
                "type": "LOW_CONFIDENCE",
                "source": "prediction_diagnosis",
                "pattern": "overall_direction",
                "reason": (
                    f"Overall direction accuracy {overall_acc:.0%} over "
                    f"{total_analyzed} predictions — below random"
                ),
                "predictions": total_analyzed,
                "accuracy": overall_acc,
            })

    # === Source 3: Hold analysis HOLD_PREFERRED ===
    hold_analysis = load_json(HOLD_ANALYSIS_PATH, default={})
    all_hold_conditions = {
        **hold_analysis.get("by_tag", {}),
        **hold_analysis.get("by_multi_tag", {}),
    }
    for condition, stats in all_hold_conditions.items():
        if stats.get("recommendation") != "HOLD_BETTER":
            continue
        hold_wr = stats.get("hold_win_rate", 0)
        trade_wr = stats.get("playbook_trade_win_rate")
        meaningful = stats.get("avoided_loss", 0) + stats.get("missed_gain", 0)
        if meaningful < HOLD_PREFERRED_MIN_N:
            continue
        constraints.append({
            "type": "HOLD_PREFERRED",
            "source": "hold_analysis",
            "pattern": condition,
            "reason": (
                f"Hold avoids {hold_wr:.0%} of losses vs "
                f"trade wins {(trade_wr or 0):.0%} ({meaningful} observations)"
            ),
            "hold_win_rate": hold_wr,
            "trade_win_rate": trade_wr,
            "observations": meaningful,
        })

    # === Source 4: Hypothesis ledger RETIRED_HYPOTHESIS ===
    hyp_ledger = load_json(HYPOTHESIS_PATH, default={})
    for hyp_id, hyp in hyp_ledger.get("hypotheses", {}).items():
        if hyp.get("status") != "retired":
            continue
        total_tests = hyp.get("confirmations", 0) + hyp.get("refutations", 0)
        conf_rate = hyp.get("confirmation_rate", 0)
        constraints.append({
            "type": "RETIRED_HYPOTHESIS",
            "source": "hypothesis_ledger",
            "pattern": f"{hyp.get('category', 'unknown')} [{_conditions_to_str(hyp.get('conditions', {}))}]",
            "reason": (
                f"\"{hyp.get('text', '')[:80]}\" — "
                f"{conf_rate:.0%} confirmed over {total_tests} tests"
            ),
            "hypothesis_id": hyp_id,
            "confirmation_rate": conf_rate,
            "tests": total_tests,
        })

    result = {
        "last_updated": iso_now(),
        "total_constraints": len(constraints),
        "constraints": constraints,
        "by_type": _group_by_type(constraints),
    }

    save_json(CONSTRAINTS_PATH, result)
    logger.info(
        f"Constraints generated: {len(constraints)} total — "
        f"{sum(1 for c in constraints if c['type'] == 'NEVER_TRADE')} NEVER_TRADE, "
        f"{sum(1 for c in constraints if c['type'] == 'LOW_CONFIDENCE')} LOW_CONFIDENCE, "
        f"{sum(1 for c in constraints if c['type'] == 'HOLD_PREFERRED')} HOLD_PREFERRED, "
        f"{sum(1 for c in constraints if c['type'] == 'RETIRED_HYPOTHESIS')} RETIRED_HYPOTHESIS"
    )
    return result


def format_constraints_for_brief(current_tags: dict = None) -> str:
    """Format active constraints for the agent's brief.

    Shows constraints that match the current market conditions prominently.
    Returns compact text (2-4 lines) or empty string.
    """
    data = load_json(CONSTRAINTS_PATH, default={})
    constraints = data.get("constraints", [])
    if not constraints:
        return ""

    # Find constraints matching current conditions
    matching = []
    non_matching_count = 0

    for c in constraints:
        if current_tags and _constraint_matches_tags(c, current_tags):
            matching.append(c)
        else:
            non_matching_count += 1

    if not matching and non_matching_count == 0:
        return ""

    parts = []

    # Show matching constraints prominently
    never_trade = [c for c in matching if c["type"] == "NEVER_TRADE"]
    hold_preferred = [c for c in matching if c["type"] == "HOLD_PREFERRED"]
    low_confidence = [c for c in matching if c["type"] == "LOW_CONFIDENCE"]
    retired = [c for c in matching if c["type"] == "RETIRED_HYPOTHESIS"]

    if never_trade:
        for c in never_trade[:3]:
            parts.append(
                f"  >>> NEVER TRADE: {c['pattern']} — {c['reason']} <<<"
            )

    if hold_preferred:
        for c in hold_preferred[:2]:
            parts.append(
                f"  HOLD PREFERRED: {c['pattern']} — {c['reason']}"
            )

    if low_confidence:
        for c in low_confidence[:2]:
            parts.append(
                f"  LOW CONFIDENCE: {c['pattern']} — {c['reason']}"
            )

    if retired:
        for c in retired[:2]:
            parts.append(
                f"  RETIRED HYPO: {c['reason']}"
            )

    if parts:
        header = f"Active constraints ({len(constraints)} total, {len(matching)} match current conditions):"
        parts.insert(0, header)

    return "\n".join(parts) if parts else ""


def compute_suggested_position_size(
    current_tags: dict,
    portfolio: dict,
    price: float,
    config: dict,
) -> dict:
    """Compute advisory position size based on condition confidence.

    Uses playbook win rate + prediction accuracy to determine sizing tier.
    Returns dict with tier, shares, reasoning.
    """
    ticker = config.get("ticker", "TSLA")
    cash = portfolio.get("cash", 0)
    holdings = portfolio.get("holdings", {}).get(ticker, {})
    held_shares = holdings.get("shares", 0)
    portfolio_value = cash + held_shares * price

    # Max position = 50% of portfolio value
    max_position_value = portfolio_value * 0.5
    max_shares = max_position_value / price if price > 0 else 0

    # Check constraints first — if any NEVER_TRADE matches, suggest 0
    data = load_json(CONSTRAINTS_PATH, default={})
    constraints = data.get("constraints", [])
    for c in constraints:
        if c["type"] == "NEVER_TRADE" and _constraint_matches_tags(c, current_tags):
            return {
                "tier": "blocked",
                "shares": 0,
                "max_shares": round(max_shares, 2),
                "reasoning": f"Blocked by constraint: {c['pattern']} — {c['reason']}",
            }

    # Compute confidence score from playbook
    ledger = load_json(LEDGER_PATH, default={})
    theses = ledger.get("theses", {})

    # Average win rate of matching single-tag patterns
    matching_wrs = []
    for tag_name, tag_value in current_tags.items():
        key = f"{tag_name}:{tag_value}"
        if key in theses and theses[key].get("trades", 0) >= 3:
            matching_wrs.append(theses[key]["win_rate"])

    avg_wr = sum(matching_wrs) / len(matching_wrs) if matching_wrs else None

    # Determine tier
    if avg_wr is None or len(matching_wrs) < 2:
        tier = "minimum"
        reasoning = "Exploratory — insufficient playbook data for these conditions"
    elif avg_wr < 0.35:
        tier = "minimum"
        reasoning = f"Weak conditions — avg playbook win rate {avg_wr:.0%}"
    elif avg_wr < 0.50:
        tier = "small"
        reasoning = f"Below-average conditions — avg playbook win rate {avg_wr:.0%}"
    elif avg_wr < 0.60:
        tier = "normal"
        reasoning = f"Moderate confidence — avg playbook win rate {avg_wr:.0%}"
    else:
        tier = "confident"
        reasoning = f"Strong conditions — avg playbook win rate {avg_wr:.0%}"

    # Check if prediction accuracy is low — downgrade by one tier
    diagnosis = load_json(DIAGNOSIS_PATH, default={})
    if not diagnosis.get("insufficient_data", True):
        overall_acc = diagnosis.get("confidence_calibration", {}).get(
            "actual_direction_accuracy", 0.5
        )
        if overall_acc < 0.40:
            tier_order = ["minimum", "small", "normal", "confident"]
            current_idx = tier_order.index(tier)
            if current_idx > 0:
                tier = tier_order[current_idx - 1]
                reasoning += f" (downgraded: prediction accuracy only {overall_acc:.0%})"

    shares = round(SIZE_TIERS[tier] * max_shares, 2)
    # Ensure minimum cash reserve ($100)
    if shares * price > cash - 100:
        shares = max(round((cash - 100) / price, 2), 0)

    return {
        "tier": tier,
        "shares": shares,
        "max_shares": round(max_shares, 2),
        "reasoning": reasoning,
    }


def _conditions_to_str(conditions: dict) -> str:
    """Convert conditions dict to readable string."""
    if not conditions:
        return "no conditions"
    return ", ".join(f"{k}={v}" for k, v in sorted(conditions.items()))


def _group_by_type(constraints: list) -> dict:
    """Group constraints by type for quick lookup."""
    groups = {}
    for c in constraints:
        ctype = c["type"]
        if ctype not in groups:
            groups[ctype] = []
        groups[ctype].append(c)
    return groups


def _constraint_matches_tags(constraint: dict, current_tags: dict) -> bool:
    """Check if a constraint's pattern matches the current tag values."""
    pattern = constraint.get("pattern", "")

    # Handle "overall_direction" special case (always matches)
    if pattern == "overall_direction":
        return True

    # Handle hypothesis patterns: "category [conditions]"
    if constraint.get("type") == "RETIRED_HYPOTHESIS":
        # Match if any condition in the hypothesis matches current tags
        hyp_ledger = load_json(HYPOTHESIS_PATH, default={})
        hyp_id = constraint.get("hypothesis_id")
        if hyp_id:
            hyp = hyp_ledger.get("hypotheses", {}).get(hyp_id, {})
            conditions = hyp.get("conditions", {})
            if not conditions:
                return False
            matches = sum(
                1 for k, v in conditions.items()
                if current_tags.get(k) == v
            )
            return matches >= len(conditions) * 0.5
        return False

    # Handle "tag=value" and "tag=value + tag=value" patterns
    parts = [p.strip() for p in pattern.split("+")]
    for part in parts:
        if "=" in part:
            tag_name, tag_value = part.split("=", 1)
            tag_name = tag_name.strip()
            tag_value = tag_value.strip()
            if current_tags.get(tag_name) != tag_value:
                return False
    return True
