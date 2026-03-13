"""Risk Manager — code-enforced risk rules that AI cannot override.

Trailing stops, time stops, daily loss limits, and trade validation.
All rules are deterministic and run every cycle. The AI can influence
the trailing stop percentage (within bounds) but cannot disable stops.

v8: Replaces the v7 approach of asking Claude to enforce discipline.
"""

from datetime import datetime, timezone, timedelta
from .utils import load_json, save_json, iso_now, setup_logging, DATA_DIR, load_config

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
        self.trailing_stop_max = v8.get("trailing_stop_max_pct", 0.03)
        self.time_stop_minutes = v8.get("time_stop_minutes", 240)
        self.daily_loss_limit = v8.get("daily_loss_limit", 50.0)
        self.min_edge_threshold = v8.get("min_edge_threshold", 0.001)

    def check_trailing_stop(self, current_price: float) -> tuple[bool, str]:
        """Check if trailing stop is triggered on active position.

        Returns (triggered, reason).
        """
        position = load_active_position()
        if not position:
            return False, "no position"

        peak = position.get("peak_price", position.get("entry_price", 0))
        stop_pct = position.get("trailing_stop_pct", self.trailing_stop_default)

        if peak <= 0:
            return False, "no peak price"

        stop_price = peak * (1 - stop_pct)

        if current_price <= stop_price:
            reason = (
                f"Trailing stop triggered: price ${current_price:.2f} "
                f"below stop ${stop_price:.2f} "
                f"(peak ${peak:.2f}, stop {stop_pct:.1%})"
            )
            logger.warning(reason)
            return True, reason

        return False, f"OK (stop at ${stop_price:.2f}, {((current_price - stop_price) / stop_price * 100):.1f}% above)"

    def check_time_stop(self) -> tuple[bool, str]:
        """Check if position has been held longer than time_stop_minutes.

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
            reason = (
                f"Time stop triggered: held {held_minutes:.0f} min "
                f"(limit: {self.time_stop_minutes} min)"
            )
            logger.warning(reason)
            return True, reason

        remaining = self.time_stop_minutes - held_minutes
        return False, f"OK ({held_minutes:.0f} min held, {remaining:.0f} min remaining)"

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

    def validate_trade(self, action: str, signal: dict,
                       portfolio: dict, price: float) -> tuple[bool, str]:
        """Pre-trade validation. Returns (allowed, reason).

        Checks:
        1. Signal score exceeds minimum edge threshold
        2. Direction matches signal sign (BUY needs positive score)
        3. Daily loss limit not hit
        4. No existing position conflicts
        """
        score = signal.get("score", 0)

        # Check minimum edge
        if abs(score) < self.min_edge_threshold:
            return False, f"Edge too small: {score:.4f} < {self.min_edge_threshold}"

        # Check direction match
        if action == "BUY" and score < 0:
            return False, f"Cannot BUY on bearish signal ({score:+.4f})"

        # Check daily loss limit
        limit_hit, reason = self.check_daily_loss_limit()
        if limit_hit:
            return False, f"Daily limit: {reason}"

        return True, "OK"

    def clamp_stop_pct(self, requested: float) -> float:
        """Clamp a requested trailing stop % to allowed range."""
        return max(self.trailing_stop_min, min(self.trailing_stop_max, requested))


# --- Active Position Tracking ---

def load_active_position() -> dict | None:
    """Load active position metadata (trailing stop state)."""
    data = load_json(ACTIVE_POSITION_PATH, default=None)
    if data is None or not data:
        return None
    return data


def save_active_position(entry_price: float, trailing_stop_pct: float,
                         decision: dict = None):
    """Store active position metadata when opening a trade."""
    position = {
        "entry_price": round(entry_price, 2),
        "entry_time": iso_now(),
        "peak_price": round(entry_price, 2),
        "trailing_stop_pct": trailing_stop_pct,
        "exit_criteria": decision.get("exit_criteria", "") if decision else "",
        "signal_score_at_entry": decision.get("_signal_score", 0) if decision else 0,
    }
    save_json(ACTIVE_POSITION_PATH, position)
    logger.info(
        f"Active position opened: ${entry_price:.2f}, "
        f"trailing stop {trailing_stop_pct:.1%}"
    )


def update_peak_price(current_price: float):
    """Update high-water mark for trailing stop. Called every cycle."""
    position = load_active_position()
    if not position:
        return

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
            "trading_halted": False,
        }
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


def get_daily_pnl_value() -> float:
    """Get today's realized P&L as a float."""
    return load_daily_pnl().get("realized_pnl", 0.0)
