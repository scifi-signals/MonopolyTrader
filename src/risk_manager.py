"""Risk Manager — code-enforced risk rules that AI cannot override.

Trailing stops, time stops, daily loss limits, and trade validation.
All rules are deterministic and run every cycle. The AI can influence
the trailing stop percentage (within bounds) but cannot disable stops.

v8: Replaces the v7 approach of asking Claude to enforce discipline.
"""

from datetime import datetime, timezone, timedelta
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR, load_config, now_et

logger = setup_logging("risk_manager")

DAILY_PNL_PATH = DATA_DIR / "daily_pnl.json"
ACTIVE_POSITION_PATH = DATA_DIR / "active_position.json"


class RiskManager:
    """Code-enforced risk rules for the trading system."""

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        v8 = config.get("v8_risk", {})
        self.trailing_stop_default = v8.get("trailing_stop_default_pct", 0.015)
        self.trailing_stop_min = v8.get("trailing_stop_min_pct", 0.005)
        self.trailing_stop_max = v8.get("trailing_stop_max_pct", 0.06)
        self.atr_stop_multiplier = v8.get("atr_stop_multiplier", 1.5)
        self.time_stop_minutes = v8.get("time_stop_minutes", 240)
        self.daily_loss_limit = v8.get("daily_loss_limit", 50.0)
        self.min_edge_threshold = v8.get("min_edge_threshold", 0.001)
        self.eod_close_minutes_before = v8.get("eod_close_minutes_before", 10)
        self.reentry_cooldown_minutes = v8.get("reentry_cooldown_minutes", 60)
        self.max_stops_per_day = v8.get("max_stops_per_day", 2)

        # Progressive tightening tiers: (profit_threshold, stop_pct)
        # Evaluated top-down — first matching tier wins.
        tiers = v8.get("tightening_tiers", [
            [0.05, 0.015],  # 5%+ profit: 1.5% stop
            [0.03, 0.020],  # 3%+ profit: 2.0% stop
            [0.02, 0.025],  # 2%+ profit: 2.5% stop
            [0.01, 0.030],  # 1%+ profit: 3.0% stop
        ])
        self.tightening_tiers = [(t[0], t[1]) for t in tiers]

    def check_trailing_stop(self, current_price: float) -> tuple[bool, str]:
        """Check if trailing stop is triggered on active position.

        For longs: triggers when price drops below peak * (1 - stop_pct).
        For shorts: triggers when price rises above trough * (1 + stop_pct).
        Returns (triggered, reason).
        """
        position = load_active_position()
        if not position:
            return False, "no position"

        direction = position.get("direction", "long")
        extreme = position.get("peak_price", position.get("entry_price", 0))
        stop_pct = position.get("trailing_stop_pct", self.trailing_stop_default)

        if extreme <= 0:
            return False, "no peak/trough price"

        if direction == "short":
            # Short: stop when price rises above trough
            stop_price = extreme * (1 + stop_pct)
            if current_price >= stop_price:
                reason = (
                    f"Trailing stop triggered (SHORT): price ${current_price:.2f} "
                    f"above stop ${stop_price:.2f} "
                    f"(trough ${extreme:.2f}, stop {stop_pct:.1%})"
                )
                logger.warning(reason)
                return True, reason
            return False, f"OK (stop at ${stop_price:.2f}, {((stop_price - current_price) / current_price * 100):.1f}% above)"
        else:
            # Long: stop when price drops below peak
            stop_price = extreme * (1 - stop_pct)
            if current_price <= stop_price:
                reason = (
                    f"Trailing stop triggered: price ${current_price:.2f} "
                    f"below stop ${stop_price:.2f} "
                    f"(peak ${extreme:.2f}, stop {stop_pct:.1%})"
                )
                logger.warning(reason)
                return True, reason
            return False, f"OK (stop at ${stop_price:.2f}, {((current_price - stop_price) / stop_price * 100):.1f}% above)"

    def check_time_stop(self, current_price: float = 0) -> tuple[bool, str]:
        """Check if a stale (non-profitable) position has been held too long.

        Winning trades are exempt — the trailing stop protects them instead.
        Only losing/flat trades get time-stopped. Let winners run.
        Returns (triggered, reason).
        """
        position = load_active_position()
        if not position:
            return False, "no position"

        entry_time_str = position.get("entry_time")
        if not entry_time_str:
            return False, "no entry time"

        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
        except ValueError:
            return False, "invalid entry time"

        now = datetime.now(timezone.utc)
        held_minutes = (now - entry_time).total_seconds() / 60

        if held_minutes >= self.time_stop_minutes:
            # Check if the trade is profitable — if so, let it run
            entry_price = position.get("entry_price", 0)
            direction = position.get("direction", "long")

            if current_price > 0 and entry_price > 0:
                if direction == "short":
                    profitable = current_price < entry_price
                else:
                    profitable = current_price > entry_price

                if profitable:
                    return False, (
                        f"Time limit reached ({held_minutes:.0f} min) but trade is "
                        f"profitable — letting winner run (trailing stop protects)"
                    )

            reason = (
                f"Time stop triggered: held {held_minutes:.0f} min "
                f"(limit: {self.time_stop_minutes} min) on losing/flat trade"
            )
            logger.warning(reason)
            return True, reason

        remaining = self.time_stop_minutes - held_minutes
        return False, f"OK ({held_minutes:.0f} min held, {remaining:.0f} min remaining)"

    def check_eod_close(self) -> tuple[bool, str]:
        """Check if we need to close positions before market close.

        Closes all positions before market close to avoid overnight gap risk.
        Returns (triggered, reason).
        """
        position = load_active_position()
        if not position:
            return False, "no position"

        et = now_et()
        close_time = et.replace(hour=16, minute=0, second=0, microsecond=0)
        cutoff = close_time - timedelta(minutes=self.eod_close_minutes_before)

        if et >= cutoff:
            reason = (
                f"End-of-day close: {et.strftime('%H:%M')} ET — "
                f"closing position to avoid overnight gap risk"
            )
            logger.warning(reason)
            return True, reason

        minutes_until = (cutoff - et).total_seconds() / 60
        return False, f"OK ({minutes_until:.0f} min until EOD close)"

    def tighten_trailing_stop(self, current_price: float):
        """Progressively tighten trailing stop as profit grows.

        Locks in gains by reducing the stop percentage when the trade
        is significantly profitable. Winners give back less as they grow.
        """
        position = load_active_position()
        if not position:
            return

        entry_price = position.get("entry_price", 0)
        if entry_price <= 0 or current_price <= 0:
            return

        direction = position.get("direction", "long")
        current_stop = position.get("trailing_stop_pct", self.trailing_stop_default)

        if direction == "short":
            profit_pct = (entry_price - current_price) / entry_price
        else:
            profit_pct = (current_price - entry_price) / entry_price

        # Find matching tightening tier (first match wins, evaluated top-down)
        new_stop = None
        for threshold, stop_pct in self.tightening_tiers:
            if profit_pct >= threshold:
                new_stop = stop_pct
                break

        if new_stop is None:
            return  # Below lowest tier, don't tighten

        new_stop = max(new_stop, self.trailing_stop_min)

        if new_stop < current_stop:
            position["trailing_stop_pct"] = new_stop
            save_json(ACTIVE_POSITION_PATH, position)
            logger.info(
                f"Trailing stop tightened: {current_stop:.1%} → {new_stop:.1%} "
                f"(profit: {profit_pct:.1%})"
            )

    def check_daily_loss_limit(self) -> tuple[bool, str]:
        """Check if today's realized losses exceed the daily limit.

        Returns (triggered, reason).
        """
        daily = load_daily_pnl()
        realized = daily.get("realized_pnl", 0)

        if realized <= -self.daily_loss_limit:
            reason = (
                f"Daily loss limit hit: ${realized:.2f} realized "
                f"(limit: -${self.daily_loss_limit:.2f})"
            )
            logger.warning(reason)
            return True, reason

        remaining = self.daily_loss_limit + realized
        return False, f"OK (${realized:+.2f} today, ${remaining:.2f} remaining)"

    def check_stop_out_limit(self) -> tuple[bool, str]:
        """Check if today's stop-out count has hit the circuit breaker.

        Trailing stops and time stops count. EOD closes do not — those are
        routine risk management, not failed signals.
        Returns (triggered, reason).
        """
        daily = load_daily_pnl()
        stops = daily.get("stops_today", 0)

        if stops >= self.max_stops_per_day:
            reason = (
                f"Stop-out circuit breaker: {stops} stops today "
                f"(limit: {self.max_stops_per_day}) — market is choppy, done for the day"
            )
            logger.warning(reason)
            return True, reason

        remaining = self.max_stops_per_day - stops
        return False, f"OK ({stops} stops today, {remaining} remaining)"

    def validate_trade(self, action: str, signal: dict,
                       portfolio: dict, price: float,
                       contrarian: bool = False) -> tuple[bool, str]:
        """Pre-trade validation. Returns (allowed, reason).

        Checks:
        1. Signal score exceeds minimum edge threshold (skip for contrarian)
        2. Direction matches signal sign (skip for contrarian)
        3. Daily loss limit not hit
        """
        score = signal.get("score", 0)

        if not contrarian:
            # Check minimum edge
            if abs(score) < self.min_edge_threshold:
                return False, f"Edge too small: {score:.4f} < {self.min_edge_threshold}"

            # Check direction match
            if action == "BUY" and score < 0:
                return False, f"Cannot BUY on bearish signal ({score:+.4f})"
            if action == "SHORT" and score > 0:
                return False, f"Cannot SHORT on bullish signal ({score:+.4f})"

        # Check daily loss limit (always enforced, even for contrarian)
        limit_hit, reason = self.check_daily_loss_limit()
        if limit_hit:
            return False, f"Daily limit: {reason}"

        return True, "OK"

    def clamp_stop_pct(self, requested: float) -> float:
        """Clamp a requested trailing stop % to allowed range."""
        return max(self.trailing_stop_min, min(self.trailing_stop_max, requested))

    def compute_atr_stop_pct(self, atr: float, price: float) -> float:
        """Compute trailing stop % from ATR. Adapts to actual volatility.

        Uses ATR × multiplier as the stop distance, then converts to %.
        Falls back to default if ATR is unavailable.
        """
        if atr <= 0 or price <= 0:
            return self.trailing_stop_default

        stop_distance = atr * self.atr_stop_multiplier
        stop_pct = stop_distance / price
        return self.clamp_stop_pct(stop_pct)


# --- Active Position Tracking ---

def load_active_position() -> dict | None:
    """Load active position metadata (trailing stop state)."""
    data = load_json(ACTIVE_POSITION_PATH, default=None)
    if data is None or not data:
        return None
    return data


def save_active_position(entry_price: float, trailing_stop_pct: float,
                         decision: dict = None, direction: str = "long"):
    """Store active position metadata when opening a trade."""
    position = {
        "entry_price": round(entry_price, 2),
        "entry_time": iso_now(),
        "peak_price": round(entry_price, 2),  # High-water for longs, low-water for shorts
        "trailing_stop_pct": trailing_stop_pct,
        "exit_criteria": decision.get("exit_criteria", "") if decision else "",
        "signal_score_at_entry": decision.get("_signal_score", 0) if decision else 0,
        "direction": direction,
    }
    save_json(ACTIVE_POSITION_PATH, position)
    logger.info(
        f"Active {direction} position opened: ${entry_price:.2f}, "
        f"trailing stop {trailing_stop_pct:.1%}"
    )


def update_peak_price(current_price: float):
    """Update high/low-water mark for trailing stop. Called every cycle.

    For longs: tracks peak (highest price seen).
    For shorts: tracks trough (lowest price seen).
    """
    position = load_active_position()
    if not position:
        return

    direction = position.get("direction", "long")

    if direction == "short":
        # Track trough for short positions
        if current_price < position.get("peak_price", float("inf")):
            position["peak_price"] = round(current_price, 2)
            save_json(ACTIVE_POSITION_PATH, position)
    else:
        # Track peak for long positions
        if current_price > position.get("peak_price", 0):
            position["peak_price"] = round(current_price, 2)
            save_json(ACTIVE_POSITION_PATH, position)


def clear_active_position():
    """Remove active position on close."""
    save_json(ACTIVE_POSITION_PATH, {})
    logger.info("Active position cleared")


def has_active_position() -> bool:
    """Check if there's an active position being tracked."""
    pos = load_active_position()
    return pos is not None and pos.get("entry_price", 0) > 0


# --- Daily P&L Tracking ---

def load_daily_pnl() -> dict:
    """Load today's P&L tracking."""
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily = load_json(DAILY_PNL_PATH, default={})

    # Reset if it's a new day
    if daily.get("date") != today:
        daily = {
            "date": today,
            "realized_pnl": 0.0,
            "trades_today": 0,
            "stops_today": 0,
            "trading_halted": False,
        }
        save_json(DAILY_PNL_PATH, daily)

    # Backfill stops_today for existing daily records
    if "stops_today" not in daily:
        daily["stops_today"] = 0
        save_json(DAILY_PNL_PATH, daily)

    return daily


def record_trade_pnl(realized_pnl: float):
    """Record a trade's P&L for daily tracking."""
    daily = load_daily_pnl()
    daily["realized_pnl"] = round(daily.get("realized_pnl", 0) + realized_pnl, 2)
    daily["trades_today"] = daily.get("trades_today", 0) + 1
    save_json(DAILY_PNL_PATH, daily)
    logger.info(
        f"Daily P&L updated: ${daily['realized_pnl']:+.2f} "
        f"({daily['trades_today']} trades today)"
    )


def record_stop_out():
    """Increment today's stop-out count (trailing stop or time stop)."""
    daily = load_daily_pnl()
    daily["stops_today"] = daily.get("stops_today", 0) + 1
    save_json(DAILY_PNL_PATH, daily)
    logger.info(f"Stop-out recorded: {daily['stops_today']} stops today")


def get_daily_pnl_value() -> float:
    """Get today's realized P&L as a float."""
    return load_daily_pnl().get("realized_pnl", 0.0)


# --- Re-entry Cooldown ---

LAST_EXIT_PATH = DATA_DIR / "last_exit.json"


def record_exit_time(reason: str = ""):
    """Record when a position was exited (for cooldown tracking)."""
    save_json(LAST_EXIT_PATH, {
        "exit_time": iso_now(),
        "reason": reason,
    })
    logger.info(f"Exit recorded for cooldown tracking")


def check_reentry_cooldown(cooldown_minutes: int = 60) -> tuple[bool, str]:
    """Check if enough time has passed since the last exit.

    Returns (blocked, reason). blocked=True means we're in cooldown.
    """
    data = load_json(LAST_EXIT_PATH, default={})
    exit_time_str = data.get("exit_time")
    if not exit_time_str:
        return False, "no recent exit"

    try:
        exit_time = datetime.fromisoformat(exit_time_str)
        if exit_time.tzinfo is None:
            exit_time = exit_time.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return False, "invalid exit time"

    now = datetime.now(timezone.utc)
    elapsed = (now - exit_time).total_seconds() / 60

    if elapsed < cooldown_minutes:
        remaining = cooldown_minutes - elapsed
        return True, (
            f"Re-entry cooldown: {elapsed:.0f}min since last exit, "
            f"{remaining:.0f}min remaining (limit: {cooldown_minutes}min)"
        )

    return False, f"cooldown clear ({elapsed:.0f}min since last exit)"
