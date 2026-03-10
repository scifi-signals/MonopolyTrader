"""Observability — anomaly detection, decision tracing, health checks.

v4: Simplified. Removed kill switches (Claude evaluates everything),
    removed knowledge_base imports, removed strategy-specific checks.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .utils import (
    load_config, load_json, save_json, iso_now,
    DATA_DIR, LOGS_DIR, setup_logging
)

logger = setup_logging("observability")

TRACES_DIR = LOGS_DIR / "traces"
ALERTS_PATH = LOGS_DIR / "alerts.json"
HEALTH_PATH = DATA_DIR / "health.json"


class AnomalyDetector:
    """Detect anomalies in trading behavior."""

    def check_all(self, trace: dict = None, history: list = None) -> list[dict]:
        """Run anomaly checks. Returns list of alerts."""
        alerts = []

        checks = [
            self._check_concentration_drift,
            self._check_overtrading,
        ]

        for check in checks:
            try:
                result = check()
                if result:
                    alerts.append(result)
            except Exception as e:
                logger.warning(f"Anomaly check failed: {e}")

        return alerts

    def _check_concentration_drift(self) -> dict | None:
        from .portfolio import load_portfolio
        portfolio = load_portfolio()
        config = load_config()
        ticker = config["ticker"]
        h = portfolio.get("holdings", {}).get(ticker, {})
        total = portfolio.get("total_value", 1)
        if h.get("shares", 0) > 0 and total > 0:
            position_pct = (h["shares"] * h.get("current_price", 0)) / total
            if position_pct > 0.75:
                return {
                    "type": "concentration_drift",
                    "severity": "WARNING",
                    "message": f"Position concentration at {position_pct:.0%} (>75%)",
                    "timestamp": iso_now(),
                    "status": "active",
                }
        return None

    def _check_overtrading(self) -> dict | None:
        from .portfolio import load_transactions
        txns = load_transactions()
        if not txns:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent = [t for t in txns if datetime.fromisoformat(t["timestamp"]) > cutoff]
        if len(recent) > 10:
            return {
                "type": "overtrading",
                "severity": "WARNING",
                "message": f"{len(recent)} trades in 24h (possible overtrading)",
                "timestamp": iso_now(),
                "status": "active",
            }
        return None


class DecisionTracer:
    """Write full audit trails per decision cycle."""

    def trace(self, inputs: dict, decision: dict, execution: dict = None):
        """Save a complete trace of one decision cycle."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now = datetime.now(timezone.utc).strftime("%H%M%S")

        trace_dir = TRACES_DIR / today
        trace_dir.mkdir(parents=True, exist_ok=True)

        trace = {
            "timestamp": iso_now(),
            "inputs": {
                "price": inputs.get("price"),
                "regime": inputs.get("regime"),
                "vix": inputs.get("vix"),
            },
            "decision": {
                "action": decision.get("action"),
                "shares": decision.get("shares"),
                "confidence": decision.get("confidence"),
                "reasoning": decision.get("reasoning", "")[:500],
            },
            "execution": execution or {},
        }

        trace_path = trace_dir / f"trace_{now}.json"
        save_json(trace_path, trace)
        return trace


class HealthChecker:
    """Check component health status."""

    def check(self) -> dict:
        """Run health checks on all components."""
        components = {}

        # Market data
        try:
            from .market_data import get_current_price
            price = get_current_price("TSLA")
            components["market_data"] = {
                "healthy": True,
                "detail": f"TSLA: ${price['price']}",
            }
        except Exception as e:
            components["market_data"] = {"healthy": False, "detail": str(e)}

        # Portfolio
        try:
            from .portfolio import load_portfolio
            p = load_portfolio()
            components["portfolio"] = {
                "healthy": True,
                "detail": f"Value: ${p['total_value']:.2f}",
            }
        except Exception as e:
            components["portfolio"] = {"healthy": False, "detail": str(e)}

        # Trade journal
        try:
            from .journal import load_journal
            journal = load_journal()
            components["journal"] = {
                "healthy": True,
                "detail": f"{len(journal)} entries",
            }
        except Exception as e:
            components["journal"] = {"healthy": False, "detail": str(e)}

        # Claude API
        try:
            import os
            has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            if not has_key:
                for path in ["anthropic_api_key.txt", "../anthropic_api_key.txt", ".env"]:
                    p = Path(path)
                    if p.exists():
                        content = p.read_text()
                        if "ANTHROPIC_API_KEY" in content:
                            has_key = True
                            break
            components["claude_api"] = {
                "healthy": has_key,
                "detail": "API key found" if has_key else "No API key",
            }
        except Exception as e:
            components["claude_api"] = {"healthy": False, "detail": str(e)}

        health = {
            "timestamp": iso_now(),
            "overall": all(c["healthy"] for c in components.values()),
            "components": components,
        }

        save_json(HEALTH_PATH, health)
        return health


def save_alert(alert: dict):
    """Save an alert to the alerts log."""
    alerts = load_json(ALERTS_PATH, default=[])
    alert.setdefault("timestamp", iso_now())
    alert.setdefault("status", "active")
    alerts.append(alert)
    if len(alerts) > 500:
        alerts = alerts[-500:]
    save_json(ALERTS_PATH, alerts)


def get_active_alerts() -> list:
    """Get all active (unresolved) alerts."""
    alerts = load_json(ALERTS_PATH, default=[])
    return [a for a in alerts if a.get("status") == "active"]
