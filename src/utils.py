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
DASHBOARD_DIR = ROOT_DIR / "dashboard"
CONFIG_PATH = ROOT_DIR / "config.json"


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
