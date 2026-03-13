"""Outcome Tracker — records every cycle observation and what price did next.

Every 15-minute cycle, we record: current tags + current price. Then as time
passes, we fill in what the price actually did at +15m, +30m, +1h, +2h, +4h.
This raw data powers the signal engine — no AI involved, just facts.

v8: Replaces prediction_tracker, shadow_journal, and most of thesis_builder.
"""

from datetime import datetime, timezone, timedelta
from itertools import islice
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR

logger = setup_logging("outcome_tracker")

OUTCOMES_PATH = DATA_DIR / "cycle_outcomes.json"

# Price horizons we track (key: minutes after observation)
HORIZONS = {
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
}

# Resolution tolerance: accept a price update if within this many minutes
# of the target horizon (accounts for cycle timing drift)
RESOLUTION_TOLERANCE_MINUTES = 10


def _load_outcomes() -> list:
    return load_json(OUTCOMES_PATH, default=[])


def _save_outcomes(outcomes: list):
    save_json(OUTCOMES_PATH, outcomes)


def _next_id(outcomes: list) -> str:
    if not outcomes:
        return "cyc_00001"
    last_num = 0
    for o in outcomes:
        try:
            num = int(o["id"].split("_")[1])
            if num > last_num:
                last_num = num
        except (ValueError, IndexError, KeyError):
            pass
    return f"cyc_{last_num + 1:05d}"


def log_cycle(price: float, tags: dict, action: str = "pending") -> dict:
    """Record a new cycle observation.

    Args:
        price: Current TSLA price
        tags: Market condition tags computed this cycle
        action: What the system did (filled in later if "pending")

    Returns:
        The new cycle record
    """
    outcomes = _load_outcomes()

    record = {
        "id": _next_id(outcomes),
        "timestamp": iso_now(),
        "price": round(price, 2),
        "tags": tags,
        "action_taken": action,
        "prices": {h: None for h in HORIZONS},
        "changes": {h: None for h in HORIZONS},
        "resolved": False,
    }

    outcomes.append(record)
    _save_outcomes(outcomes)
    return record


def update_action(cycle_id: str, action: str):
    """Update the action_taken field for a cycle (called after decision)."""
    outcomes = _load_outcomes()
    for o in reversed(outcomes):
        if o["id"] == cycle_id:
            o["action_taken"] = action
            _save_outcomes(outcomes)
            return
        # Also match "pending" as the most recent
        if cycle_id == "pending" and o.get("action_taken") == "pending":
            o["action_taken"] = action
            _save_outcomes(outcomes)
            return


def resolve_outcomes(current_price: float) -> int:
    """Fill in price outcomes for past cycles based on elapsed time.

    Called every cycle. For each unresolved past cycle, check if enough
    time has passed to fill in any horizon prices.

    Returns count of newly filled horizon slots.
    """
    outcomes = _load_outcomes()
    now = datetime.now(timezone.utc)
    filled_count = 0

    for record in outcomes:
        if record.get("resolved"):
            continue

        try:
            obs_time = datetime.fromisoformat(record["timestamp"])
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=timezone.utc)
        except (ValueError, KeyError):
            continue

        elapsed_minutes = (now - obs_time).total_seconds() / 60
        all_filled = True

        for horizon_key, horizon_minutes in HORIZONS.items():
            # Already filled
            if record["prices"].get(horizon_key) is not None:
                continue

            # Check if enough time has passed
            if elapsed_minutes >= horizon_minutes - RESOLUTION_TOLERANCE_MINUTES:
                record["prices"][horizon_key] = round(current_price, 2)
                # Compute percentage change
                if record["price"] > 0:
                    change = (current_price - record["price"]) / record["price"]
                    record["changes"][horizon_key] = round(change, 6)
                else:
                    record["changes"][horizon_key] = 0.0
                filled_count += 1
            else:
                all_filled = False

        # Mark as fully resolved when all horizons are filled
        if all_filled:
            record["resolved"] = True

    if filled_count > 0:
        _save_outcomes(outcomes)
        logger.debug(f"Resolved {filled_count} outcome slots")

    return filled_count


def get_resolved_outcomes(min_horizon: str = "1h", lookback_days: int = 90) -> list:
    """Return outcomes resolved to at least the given horizon.

    Args:
        min_horizon: Minimum horizon that must be filled (e.g., "1h")
        lookback_days: Only return outcomes from last N days

    Returns:
        List of outcome dicts with at least the specified horizon resolved
    """
    outcomes = _load_outcomes()
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    result = []

    for o in outcomes:
        # Check horizon is resolved
        if o.get("changes", {}).get(min_horizon) is None:
            continue

        # Check age
        try:
            ts = datetime.fromisoformat(o["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                continue
        except (ValueError, KeyError):
            continue

        result.append(o)

    return result


def get_total_outcomes() -> int:
    """Return total number of outcome records."""
    return len(_load_outcomes())


def prune_outcomes(days: int = 90) -> int:
    """Remove resolved outcomes older than N days.

    Keeps unresolved outcomes regardless of age.
    Returns count of pruned records.
    """
    outcomes = _load_outcomes()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    kept = []
    pruned = 0

    for o in outcomes:
        if not o.get("resolved"):
            kept.append(o)
            continue

        try:
            ts = datetime.fromisoformat(o["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                pruned += 1
                continue
        except (ValueError, KeyError):
            pass

        kept.append(o)

    if pruned > 0:
        _save_outcomes(kept)
        logger.info(f"Pruned {pruned} old outcomes, {len(kept)} remaining")

    return pruned


def migrate_v7_data() -> int:
    """One-time migration: convert v7 data files into cycle outcomes.

    Sources:
    - predictions.json: ~260 records with tags + actual price changes
    - hold_journal.json: ~28 records with tags + shadow P&L
    - trade_journal.json: trades with tags + P&L

    Returns count of records migrated.
    """
    outcomes = _load_outcomes()
    if outcomes:
        logger.info(f"Outcomes already has {len(outcomes)} records — skipping migration")
        return 0

    migrated = 0
    existing_timestamps = set()

    # Source 1: predictions.json (largest dataset)
    predictions = load_json(DATA_DIR / "predictions.json", default=[])
    for p in predictions:
        if not p.get("resolved"):
            continue
        result = p.get("result", {})
        if not result or result.get("error"):
            continue

        ts = p.get("timestamp", "")
        if ts in existing_timestamps:
            continue
        existing_timestamps.add(ts)

        price = p.get("price_at_prediction", 0)
        if price <= 0:
            continue

        tags = p.get("tags", {})
        actual_change = result.get("actual_change_pct", 0)
        final_price = result.get("final_price", 0)
        cycles = p.get("cycles", 2)

        # Map prediction cycles to our horizons
        # cycles=1 → 15m, cycles=2 → 30m, cycles=4 → 1h
        horizon_map = {1: "15m", 2: "30m", 3: "30m", 4: "1h"}
        target_horizon = horizon_map.get(cycles, "30m")

        prices = {h: None for h in HORIZONS}
        changes = {h: None for h in HORIZONS}

        if final_price > 0:
            prices[target_horizon] = round(final_price, 2)
            changes[target_horizon] = round(actual_change / 100, 6) if actual_change else 0.0

        record = {
            "id": f"cyc_{migrated + 1:05d}",
            "timestamp": ts,
            "price": round(price, 2),
            "tags": tags,
            "action_taken": p.get("action_this_cycle", "HOLD"),
            "prices": prices,
            "changes": changes,
            "resolved": False,  # Only partially resolved
            "migrated_from": "predictions",
        }

        # Mark resolved if we have at least one horizon
        if any(v is not None for v in changes.values()):
            record["resolved"] = True

        outcomes.append(record)
        migrated += 1

    # Source 2: hold_journal.json
    holds = load_json(DATA_DIR / "hold_journal.json", default=[])
    for h in holds:
        if not h.get("shadow_resolved"):
            continue

        ts = h.get("timestamp", "")
        if ts in existing_timestamps:
            continue
        existing_timestamps.add(ts)

        price = h.get("price", 0)
        if price <= 0:
            continue

        tags = h.get("tags", {})
        final_price = h.get("shadow_final_price", 0)

        prices = {hz: None for hz in HORIZONS}
        changes = {hz: None for hz in HORIZONS}

        if final_price > 0 and price > 0:
            # Shadow journal resolves after ~8 hours, use 4h horizon
            change = (final_price - price) / price
            prices["4h"] = round(final_price, 2)
            changes["4h"] = round(change, 6)

        record = {
            "id": f"cyc_{migrated + 1:05d}",
            "timestamp": ts,
            "price": round(price, 2),
            "tags": tags,
            "action_taken": "HOLD",
            "prices": prices,
            "changes": changes,
            "resolved": any(v is not None for v in changes.values()),
            "migrated_from": "hold_journal",
        }

        outcomes.append(record)
        migrated += 1

    # Source 3: trade_journal.json (trades with close prices)
    journal = load_json(DATA_DIR / "trade_journal.json", default=[])
    for entry in journal:
        if entry.get("action") != "BUY":
            continue
        if entry.get("close_price") is None:
            continue

        ts = entry.get("timestamp", "")
        if ts in existing_timestamps:
            continue
        existing_timestamps.add(ts)

        price = entry.get("price", 0)
        close_price = entry.get("close_price", 0)
        if price <= 0:
            continue

        tags = entry.get("tags", {})
        hold_minutes = entry.get("hold_minutes", 60)

        prices = {hz: None for hz in HORIZONS}
        changes = {hz: None for hz in HORIZONS}

        if close_price > 0:
            change = (close_price - price) / price
            # Map hold_minutes to nearest horizon
            if hold_minutes <= 20:
                hz = "15m"
            elif hold_minutes <= 45:
                hz = "30m"
            elif hold_minutes <= 90:
                hz = "1h"
            elif hold_minutes <= 180:
                hz = "2h"
            else:
                hz = "4h"
            prices[hz] = round(close_price, 2)
            changes[hz] = round(change, 6)

        record = {
            "id": f"cyc_{migrated + 1:05d}",
            "timestamp": ts,
            "price": round(price, 2),
            "tags": tags,
            "action_taken": "BUY",
            "prices": prices,
            "changes": changes,
            "resolved": any(v is not None for v in changes.values()),
            "migrated_from": "trade_journal",
        }

        outcomes.append(record)
        migrated += 1

    if migrated > 0:
        _save_outcomes(outcomes)
        logger.info(f"Migrated {migrated} records from v7 data")

    return migrated
