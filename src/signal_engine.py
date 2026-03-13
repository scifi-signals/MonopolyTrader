"""Signal Engine — computes quantitative signal weights from cycle outcomes.

Every tag value gets a weight derived from historical price movements.
No AI involved — pure math on observed data. Weights update every cycle
as new outcomes resolve.

v8: Replaces thesis_builder, constraint_generator, hold_analyzer,
prediction_diagnosis, and pattern_explorer with one data-driven system.
"""

from itertools import combinations
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR
from .outcome_tracker import get_resolved_outcomes, get_total_outcomes

logger = setup_logging("signal_engine")

REGISTRY_PATH = DATA_DIR / "signal_registry.json"

# Minimum change threshold — movements below this are "flat" (noise)
FLAT_THRESHOLD = 0.001  # 0.1%

# Confidence tiers by sample size
CONFIDENCE_TIERS = {
    "high": {"min_n": 30, "factor": 1.0},
    "medium": {"min_n": 10, "factor": 0.6},
    "low": {"min_n": 3, "factor": 0.3},
}

# Minimum n for 2-tag combos to be included
COMBO_MIN_N = 10

# Default horizon for signal computation
DEFAULT_HORIZON = "1h"


def _classify_confidence(n: int) -> tuple[str, float]:
    """Return (confidence_label, confidence_factor) based on sample size."""
    if n >= CONFIDENCE_TIERS["high"]["min_n"]:
        return "high", CONFIDENCE_TIERS["high"]["factor"]
    elif n >= CONFIDENCE_TIERS["medium"]["min_n"]:
        return "medium", CONFIDENCE_TIERS["medium"]["factor"]
    elif n >= CONFIDENCE_TIERS["low"]["min_n"]:
        return "low", CONFIDENCE_TIERS["low"]["factor"]
    else:
        return "insufficient", 0.0


def _compute_signal_stats(changes: list[float]) -> dict:
    """Compute signal statistics from a list of price changes.

    Args:
        changes: List of fractional price changes (e.g., 0.005 = +0.5%)

    Returns:
        Dict with up_rate, avg_up, avg_down, expected_edge, etc.
    """
    n = len(changes)
    if n == 0:
        return {"n": 0, "expected_edge": 0.0}

    up = [c for c in changes if c > FLAT_THRESHOLD]
    down = [c for c in changes if c < -FLAT_THRESHOLD]
    flat = [c for c in changes if abs(c) <= FLAT_THRESHOLD]

    up_rate = len(up) / n if n > 0 else 0
    avg_up = sum(up) / len(up) if up else 0
    avg_down = sum(down) / len(down) if down else 0

    # Expected edge: probability-weighted average outcome
    expected_edge = sum(changes) / n

    confidence_label, confidence_factor = _classify_confidence(n)

    return {
        "n": n,
        "up_count": len(up),
        "down_count": len(down),
        "flat_count": len(flat),
        "up_rate": round(up_rate, 4),
        "avg_up": round(avg_up, 6),
        "avg_down": round(avg_down, 6),
        "avg_change": round(expected_edge, 6),
        "expected_edge": round(expected_edge, 6),
        "confidence": confidence_label,
        "confidence_factor": confidence_factor,
        "weight": round(expected_edge * confidence_factor, 6),
    }


def rebuild_signal_registry(horizon: str = DEFAULT_HORIZON,
                            lookback_days: int = 90) -> dict:
    """Recompute all signal weights from resolved cycle outcomes.

    Called every cycle. Fast — just arithmetic on ~300 records.

    Returns the full registry dict (also saved to disk).
    """
    outcomes = get_resolved_outcomes(min_horizon=horizon, lookback_days=lookback_days)

    if not outcomes:
        registry = {
            "last_updated": iso_now(),
            "horizon": horizon,
            "total_outcomes": get_total_outcomes(),
            "resolved_outcomes": 0,
            "signals": {},
            "combos": {},
        }
        save_json(REGISTRY_PATH, registry)
        return registry

    # Group changes by single-tag values
    tag_changes = {}  # "rsi_zone=oversold" -> [change1, change2, ...]
    for o in outcomes:
        change = o.get("changes", {}).get(horizon)
        if change is None:
            continue
        tags = o.get("tags", {})
        for tag_name, tag_value in tags.items():
            key = f"{tag_name}={tag_value}"
            if key not in tag_changes:
                tag_changes[key] = []
            tag_changes[key].append(change)

    # Compute single-tag signals
    signals = {}
    for key, changes in tag_changes.items():
        stats = _compute_signal_stats(changes)
        if stats["n"] >= CONFIDENCE_TIERS["low"]["min_n"]:
            signals[key] = stats

    # Group changes by 2-tag combos
    combo_changes = {}
    for o in outcomes:
        change = o.get("changes", {}).get(horizon)
        if change is None:
            continue
        tags = o.get("tags", {})
        tag_items = sorted(tags.items())
        for (t1, v1), (t2, v2) in combinations(tag_items, 2):
            combo_key = f"{t1}={v1}+{t2}={v2}"
            if combo_key not in combo_changes:
                combo_changes[combo_key] = []
            combo_changes[combo_key].append(change)

    # Compute 2-tag combo signals
    combos = {}
    for key, changes in combo_changes.items():
        if len(changes) < COMBO_MIN_N:
            continue
        stats = _compute_signal_stats(changes)
        combos[key] = stats

    registry = {
        "last_updated": iso_now(),
        "horizon": horizon,
        "total_outcomes": get_total_outcomes(),
        "resolved_outcomes": len(outcomes),
        "signals": signals,
        "combos": combos,
    }

    save_json(REGISTRY_PATH, registry)
    logger.debug(
        f"Signal registry rebuilt: {len(signals)} single-tag, "
        f"{len(combos)} combo signals from {len(outcomes)} outcomes"
    )
    return registry


def load_registry() -> dict:
    """Load the signal registry from disk."""
    return load_json(REGISTRY_PATH, default={"signals": {}, "combos": {}})


def compute_composite_score(current_tags: dict, registry: dict = None) -> dict:
    """Compute the aggregate signal for current market conditions.

    Strategy:
    1. Collect all matching 2-tag combo signals (most specific)
    2. If combos exist, use combo-weighted average
    3. Otherwise, fall back to single-tag weighted average

    Returns dict with score, confidence, contributing signals, etc.
    """
    if registry is None:
        registry = load_registry()

    signals = registry.get("signals", {})
    combos = registry.get("combos", {})

    # Collect matching single-tag signals
    matching_singles = []
    for tag_name, tag_value in current_tags.items():
        key = f"{tag_name}={tag_value}"
        if key in signals:
            sig = signals[key].copy()
            sig["key"] = key
            matching_singles.append(sig)

    # Collect matching 2-tag combo signals
    matching_combos = []
    tag_items = sorted(current_tags.items())
    for (t1, v1), (t2, v2) in combinations(tag_items, 2):
        combo_key = f"{t1}={v1}+{t2}={v2}"
        if combo_key in combos:
            sig = combos[combo_key].copy()
            sig["key"] = combo_key
            matching_combos.append(sig)

    # Use combos if available (more specific), else singles
    if matching_combos:
        primary = matching_combos
        secondary = matching_singles
    else:
        primary = matching_singles
        secondary = []

    if not primary:
        return {
            "score": 0.0,
            "direction": "neutral",
            "confidence": "none",
            "confidence_factor": 0.0,
            "n": 0,
            "contributing_signals": 0,
            "bullish": [],
            "bearish": [],
            "neutral": [],
            "exploration_targets": _find_exploration_targets(current_tags, signals),
        }

    # Weighted average by sample size
    total_n = sum(s["n"] for s in primary)
    if total_n == 0:
        composite = 0.0
        avg_confidence_factor = 0.0
    else:
        composite = sum(s["expected_edge"] * s["n"] for s in primary) / total_n
        avg_confidence_factor = sum(
            s["confidence_factor"] * s["n"] for s in primary
        ) / total_n

    # Classify direction
    if composite > FLAT_THRESHOLD / 2:
        direction = "bullish"
    elif composite < -FLAT_THRESHOLD / 2:
        direction = "bearish"
    else:
        direction = "neutral"

    # Split into bullish/bearish/neutral for display
    bullish = sorted(
        [s for s in primary if s["expected_edge"] > FLAT_THRESHOLD / 2],
        key=lambda s: -s["expected_edge"],
    )
    bearish = sorted(
        [s for s in primary if s["expected_edge"] < -FLAT_THRESHOLD / 2],
        key=lambda s: s["expected_edge"],
    )
    neutral_sigs = [
        s for s in primary
        if abs(s["expected_edge"]) <= FLAT_THRESHOLD / 2
    ]

    conf_label, _ = _classify_confidence(total_n)

    return {
        "score": round(composite, 6),
        "direction": direction,
        "confidence": conf_label,
        "confidence_factor": round(avg_confidence_factor, 3),
        "n": total_n,
        "contributing_signals": len(primary),
        "bullish": bullish[:5],
        "bearish": bearish[:5],
        "neutral": neutral_sigs[:3],
        "exploration_targets": _find_exploration_targets(current_tags, signals),
    }


def _find_exploration_targets(current_tags: dict, signals: dict) -> list:
    """Identify tags in current conditions with insufficient data."""
    targets = []
    for tag_name, tag_value in current_tags.items():
        key = f"{tag_name}={tag_value}"
        if key not in signals:
            targets.append({"key": key, "n": 0, "reason": "never observed"})
        elif signals[key]["n"] < CONFIDENCE_TIERS["medium"]["min_n"]:
            targets.append({
                "key": key,
                "n": signals[key]["n"],
                "reason": f"low data (n={signals[key]['n']})",
            })
    return targets


def compute_position_size(signal: dict, portfolio_value: float,
                          price: float, config: dict) -> dict:
    """Deterministic position sizing from signal strength.

    No AI involved. Size is a direct function of signal score + confidence.

    Returns dict with tier, shares, value, reasoning.
    """
    score = abs(signal.get("score", 0))
    confidence_factor = signal.get("confidence_factor", 0)
    risk_config = config.get("v8_risk", {})
    min_edge = risk_config.get("min_edge_threshold", 0.001)

    max_position_pct = config.get("risk_params", {}).get("max_position_pct", 0.50)
    min_cash = config.get("risk_params", {}).get("min_cash_reserve", 100.0)
    available_cash = portfolio_value - min_cash

    if score < min_edge:
        return {
            "tier": "no_trade",
            "pct": 0,
            "shares": 0,
            "value": 0,
            "reasoning": f"Edge {score:.4f} below minimum {min_edge:.4f}",
        }

    # Base percentage tiers
    if score > 0.006:
        base_pct = 0.35  # 25-40%
        tier = "large"
    elif score > 0.003:
        base_pct = 0.20  # 15-25%
        tier = "medium"
    elif score > 0.001:
        base_pct = 0.12  # 10-15%
        tier = "small"
    else:
        base_pct = 0.05  # 5% exploration
        tier = "minimum"

    # Apply confidence multiplier
    adjusted_pct = base_pct * max(confidence_factor, 0.3)

    # Cap at max position
    adjusted_pct = min(adjusted_pct, max_position_pct)

    # Compute shares
    position_value = min(portfolio_value * adjusted_pct, available_cash)
    if position_value <= 0 or price <= 0:
        return {
            "tier": "no_trade",
            "pct": 0,
            "shares": 0,
            "value": 0,
            "reasoning": "Insufficient available cash",
        }

    shares = round(position_value / price, 4)

    return {
        "tier": tier,
        "pct": round(adjusted_pct, 3),
        "shares": shares,
        "value": round(position_value, 2),
        "reasoning": (
            f"Score {score:.4f} → {tier} tier at {adjusted_pct:.0%} "
            f"(confidence: {signal.get('confidence', '?')})"
        ),
    }


def get_signal_summary(current_tags: dict, signal: dict = None,
                       sizing: dict = None) -> str:
    """Format the signal section for the agent brief.

    Returns compact text for the === SIGNAL === section.
    """
    if signal is None:
        signal = compute_composite_score(current_tags)

    score = signal["score"]
    direction = signal["direction"]
    confidence = signal.get("confidence", "none")
    n = signal.get("n", 0)

    parts = []
    parts.append(
        f"Composite: {score:+.4f} ({direction}) | "
        f"Confidence: {confidence.upper()} (n={n})"
    )

    # Top bullish signals
    for s in signal.get("bullish", [])[:3]:
        parts.append(
            f"  Bullish: {s['key']} {s['expected_edge']:+.4f} "
            f"(n={s['n']}, {s['up_rate']:.0%} up)"
        )

    # Top bearish signals
    for s in signal.get("bearish", [])[:3]:
        parts.append(
            f"  Bearish: {s['key']} {s['expected_edge']:+.4f} "
            f"(n={s['n']}, {s['up_rate']:.0%} up)"
        )

    # Exploration targets
    targets = signal.get("exploration_targets", [])
    if targets:
        top_target = targets[0]
        parts.append(
            f"  Explore: {top_target['key']} — {top_target['reason']}"
        )

    # Position sizing
    if sizing:
        if sizing["tier"] != "no_trade":
            parts.append(
                f"  Size: {sizing['tier'].upper()} "
                f"({sizing['shares']:.2f} shares, ~${sizing['value']:.0f})"
            )
        else:
            parts.append(f"  Size: NO TRADE — {sizing['reasoning']}")

    return "\n".join(parts)
