"""Shared helpers — logging, time, formatting, config loading."""

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

# Project root is one level up from src/
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
KNOWLEDGE_DIR = ROOT_DIR / "knowledge"
LOGS_DIR = ROOT_DIR / "logs"
COSTS_DIR = LOGS_DIR / "costs"
DASHBOARD_DIR = ROOT_DIR / "dashboard"
CONFIG_PATH = ROOT_DIR / "config.json"

# Pricing per million tokens (update when models change)
MODEL_PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_json(path: Path, default=None):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default if default is not None else {}


def save_json(path: Path, data, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def iso_now() -> str:
    return now_utc().isoformat()


def is_market_open(config: dict = None) -> bool:
    """Check if US stock market is currently open (weekday, within hours)."""
    if config is None:
        config = load_config()
    et = now_et()
    # Weekday check (0=Mon, 4=Fri)
    if et.weekday() > 4:
        return False
    mh = config["market_hours"]
    open_h, open_m = map(int, mh["open"].split(":"))
    close_h, close_m = map(int, mh["close"].split(":"))
    market_open = et.replace(hour=open_h, minute=open_m, second=0, microsecond=0)
    market_close = et.replace(hour=close_h, minute=close_m, second=0, microsecond=0)
    return market_open <= et <= market_close


def setup_logging(name: str = "monopoly_trader", level=logging.INFO) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # File handler (rotating to prevent disk fill on small servers)
    fh = RotatingFileHandler(LOGS_DIR / "agent.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"


def format_pct(value: float) -> str:
    return f"{value:+.2f}%"


def _log_api_usage(model: str, input_tokens: int, output_tokens: int, caller: str = ""):
    """Log API token usage to daily cost file."""
    COSTS_DIR.mkdir(parents=True, exist_ok=True)
    today = now_utc().strftime("%Y-%m-%d")
    cost_path = COSTS_DIR / f"{today}.json"

    pricing = MODEL_PRICING.get(model, {"input": 3.00, "output": 15.00})
    cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

    daily = {"date": today, "calls": [], "totals": {}}
    if cost_path.exists():
        try:
            with open(cost_path) as f:
                daily = json.load(f)
        except (json.JSONDecodeError, KeyError):
            pass

    daily["calls"].append({
        "timestamp": now_utc().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "caller": caller,
    })

    # Recalculate totals
    total_input = sum(c["input_tokens"] for c in daily["calls"])
    total_output = sum(c["output_tokens"] for c in daily["calls"])
    total_cost = sum(c["cost_usd"] for c in daily["calls"])
    daily["totals"] = {
        "calls": len(daily["calls"]),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": round(total_cost, 4),
    }

    with open(cost_path, "w") as f:
        json.dump(daily, f, indent=2, default=str)


def get_cost_summary(days: int = 30) -> dict:
    """Get API cost summary over recent days."""
    COSTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(COSTS_DIR.glob("*.json"), reverse=True)[:days]

    total_cost = 0.0
    total_calls = 0
    total_input = 0
    total_output = 0
    daily_costs = []

    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            totals = data.get("totals", {})
            day_cost = totals.get("cost_usd", 0)
            total_cost += day_cost
            total_calls += totals.get("calls", 0)
            total_input += totals.get("input_tokens", 0)
            total_output += totals.get("output_tokens", 0)
            daily_costs.append({
                "date": data.get("date", f.stem),
                "cost_usd": day_cost,
                "calls": totals.get("calls", 0),
            })
        except Exception:
            continue

    trading_days = len(daily_costs)
    return {
        "total_cost_usd": round(total_cost, 4),
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "trading_days": trading_days,
        "avg_daily_cost_usd": round(total_cost / trading_days, 4) if trading_days else 0,
        "daily_breakdown": daily_costs,
    }


def call_ai_with_fallback(
    system: str,
    user: str,
    max_tokens: int = 1500,
    config: dict = None,
) -> tuple[str, str]:
    """Call an AI model with automatic fallback between providers.

    Tries Anthropic (direct) first, then OpenRouter as fallback.
    Returns (response_text, model_id).

    Requires ANTHROPIC_API_KEY in env. For fallback, set OPENROUTER_API_KEY.
    """
    if config is None:
        config = load_config()

    model = config.get("anthropic_model", "claude-sonnet-4-20250514")
    logger = logging.getLogger("ai_fallback")

    # Provider 1: Anthropic direct
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        for path in ["anthropic_api_key.txt", "../anthropic_api_key.txt"]:
            try:
                with open(path) as f:
                    anthropic_key = f.read().strip()
                    break
            except FileNotFoundError:
                continue

    if anthropic_key:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=anthropic_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text.strip()
            # Log token usage
            try:
                usage = response.usage
                _log_api_usage(model, usage.input_tokens, usage.output_tokens, caller="anthropic_direct")
            except Exception:
                pass
            return text, model
        except Exception as e:
            logger.warning(f"Anthropic direct failed: {e}")

    # Provider 2: OpenRouter fallback
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        for path in ["openrouter_api_key.txt", "../openrouter_api_key.txt"]:
            try:
                with open(path) as f:
                    openrouter_key = f.read().strip()
                    break
            except FileNotFoundError:
                continue

    if openrouter_key:
        try:
            import httpx
            # Map model name to OpenRouter model ID
            or_model_map = {
                "claude-sonnet-4-20250514": "anthropic/claude-sonnet-4",
                "claude-haiku-4-5-20251001": "anthropic/claude-haiku-4-5",
            }
            or_model = or_model_map.get(model, f"anthropic/{model}")

            resp = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": or_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            # Log token usage from OpenRouter
            try:
                usage = data.get("usage", {})
                _log_api_usage(
                    model,
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    caller="openrouter_fallback",
                )
            except Exception:
                pass
            logger.info(f"OpenRouter fallback succeeded with {or_model}")
            return text, f"openrouter/{or_model}"
        except Exception as e:
            logger.error(f"OpenRouter fallback also failed: {e}")

    raise RuntimeError("All AI providers failed. Check API keys and connectivity.")


def generate_id(prefix: str, existing_ids: list = None) -> str:
    """Generate a sequential ID like txn_001, lesson_012, etc."""
    if not existing_ids:
        return f"{prefix}_001"
    nums = []
    for eid in existing_ids:
        try:
            nums.append(int(eid.split("_")[-1]))
        except (ValueError, IndexError):
            continue
    next_num = max(nums, default=0) + 1
    return f"{prefix}_{next_num:03d}"
