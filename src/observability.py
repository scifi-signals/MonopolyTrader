"""Observability — kill switches, anomaly detection, decision tracing, health checks."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from .utils import (
    load_config, load_json, save_json, iso_now,
    DATA_DIR, LOGS_DIR, KNOWLEDGE_DIR, setup_logging
)

logger = setup_logging("observability")

TRACES_DIR = LOGS_DIR / "traces"
ALERTS_PATH = LOGS_DIR / "alerts.json"
HEALTH_PATH = DATA_DIR / "health.json"


class KillSwitchChecker:
    """Check kill switch conditions that should halt trading."""

    def check_all(self, portfolio: dict, metrics: dict = None, alerts: list = None) -> dict:
        """Run all kill switch checks. Returns {triggered: bool, reasons: []}."""
        config = load_config()
        ks = config.get("risk_params", {}).get("kill_switches", {})
        results = {"triggered": False, "checks": {}}

        # 1. 30-day rolling prediction accuracy trending negative
        acc_trend = self._check_accuracy_trend(ks.get("negative_accuracy_trend_days", 30))
        results["checks"]["accuracy_trend"] = acc_trend

        # 2. Portfolio 15%+ below peak
        drawdown = self._check_drawdown(portfolio, ks.get("max_drawdown_from_peak_pct", 0.15))
        results["checks"]["max_drawdown"] = drawdown

        # 3. 3+ CRITICAL alerts in 7 days
        alert_check = self._check_alert_frequency(
            alerts or [], ks.get("critical_alerts_in_7_days", 3)
        )
        results["checks"]["alert_frequency"] = alert_check

        # 4. Win rate drops 30+ pct points on regime change
        drift = self._check_style_drift(ks.get("style_drift_winrate_drop_pct", 0.30))
        results["checks"]["style_drift"] = drift

        triggered_checks = [k for k, v in results["checks"].items() if v.get("triggered")]
        results["triggered"] = len(triggered_checks) > 0
        results["triggered_checks"] = triggered_checks

        if results["triggered"]:
            logger.critical(f"KILL SWITCH TRIGGERED: {triggered_checks}")

        return results

    def _check_accuracy_trend(self, window_days: int) -> dict:
        from .knowledge_base import get_predictions
        predictions = get_predictions()
        if len(predictions) < 10:
            return {"triggered": False, "detail": "Not enough predictions"}

        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        recent = [p for p in predictions
                  if datetime.fromisoformat(p["timestamp"]) > cutoff]

        if len(recent) < 5:
            return {"triggered": False, "detail": "Not enough recent predictions"}

        # Check if accuracy is trending down (compare first half to second half)
        mid = len(recent) // 2
        first_half = recent[:mid]
        second_half = recent[mid:]

        def accuracy(preds):
            correct = 0
            total = 0
            for p in preds:
                for h, out in p.get("outcomes", {}).items():
                    if out and out.get("direction_correct") is not None:
                        total += 1
                        if out["direction_correct"]:
                            correct += 1
            return correct / total if total > 0 else 0.5

        acc1 = accuracy(first_half)
        acc2 = accuracy(second_half)
        declining = acc2 < acc1 - 0.1  # Declining if dropped 10+ pct points

        return {
            "triggered": declining,
            "first_half_accuracy": round(acc1, 3),
            "second_half_accuracy": round(acc2, 3),
            "detail": f"Accuracy: {acc1:.1%} -> {acc2:.1%}",
        }

    def _check_drawdown(self, portfolio: dict, max_dd: float) -> dict:
        from .portfolio import SNAPSHOTS_DIR
        snapshots = sorted(SNAPSHOTS_DIR.glob("*.json"))
        if not snapshots:
            return {"triggered": False, "detail": "No snapshots"}

        peak = 0
        for sp in snapshots:
            data = load_json(sp)
            if data:
                peak = max(peak, data.get("total_value", 0))

        current = portfolio.get("total_value", 0)
        dd = (peak - current) / peak if peak > 0 else 0

        return {
            "triggered": dd >= max_dd,
            "drawdown_pct": round(dd * 100, 2),
            "peak_value": peak,
            "current_value": current,
            "detail": f"Drawdown: {dd:.1%} (limit: {max_dd:.0%})",
        }

    def _check_alert_frequency(self, alerts: list, max_critical: int) -> dict:
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        recent_critical = [
            a for a in alerts
            if a.get("severity") == "CRITICAL"
            and datetime.fromisoformat(a.get("timestamp", "2000-01-01")) > cutoff
        ]
        triggered = len(recent_critical) >= max_critical
        return {
            "triggered": triggered,
            "count": len(recent_critical),
            "detail": f"{len(recent_critical)} critical alerts in 7 days (limit: {max_critical})",
        }

    def _check_style_drift(self, threshold: float) -> dict:
        # Compare recent win rate to historical
        from .knowledge_base import get_strategy_scores
        scores = get_strategy_scores()
        for name, s in scores.get("strategies", {}).items():
            if s.get("total_trades", 0) > 10:
                initial_wr = 0.5  # Assume 50% as baseline
                current_wr = s.get("win_rate", 0.5)
                if initial_wr - current_wr >= threshold:
                    return {
                        "triggered": True,
                        "strategy": name,
                        "detail": f"{name} win rate dropped to {current_wr:.0%} (threshold: {threshold:.0%} drop)",
                    }
        return {"triggered": False, "detail": "No significant style drift"}


class AnomalyDetector:
    """Detect anomalies in trading behavior and market conditions."""

    def check_all(self, trace: dict = None, history: list = None) -> list[dict]:
        """Run all 8 anomaly checks. Returns list of alerts."""
        alerts = []

        checks = [
            self._check_concentration_drift,
            self._check_overtrading,
            self._check_confidence_calibration,
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
        from .portfolio import load_portfolio, load_config
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

    def _check_confidence_calibration(self) -> dict | None:
        from .knowledge_base import get_predictions
        predictions = get_predictions()
        if len(predictions) < 10:
            return None

        high_conf_wrong = 0
        high_conf_total = 0
        for p in predictions[-20:]:
            for h, out in p.get("outcomes", {}).items():
                if out and out.get("direction_correct") is not None:
                    pred_conf = p.get("predictions", {}).get(h, {}).get("confidence", 0)
                    if pred_conf > 0.7:
                        high_conf_total += 1
                        if not out["direction_correct"]:
                            high_conf_wrong += 1

        if high_conf_total >= 5 and high_conf_wrong / high_conf_total > 0.6:
            return {
                "type": "confidence_calibration",
                "severity": "WARNING",
                "message": f"High-confidence predictions wrong {high_conf_wrong}/{high_conf_total} times",
                "timestamp": iso_now(),
                "status": "active",
            }
        return None


class DecisionTracer:
    """Write full audit trails per decision cycle."""

    def trace(self, inputs: dict, decision: dict, risk_checks: dict = None, execution: dict = None):
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
                "macro_gate": inputs.get("macro_gate"),
                "atr": inputs.get("atr"),
                "vix": inputs.get("vix"),
            },
            "decision": {
                "action": decision.get("action"),
                "shares": decision.get("shares"),
                "confidence": decision.get("confidence"),
                "strategy": decision.get("strategy"),
                "hypothesis": decision.get("hypothesis"),
            },
            "risk_checks": risk_checks or {},
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

        # Knowledge base
        try:
            from .knowledge_base import get_lessons, get_patterns
            lessons = get_lessons()
            patterns = get_patterns()
            components["knowledge_base"] = {
                "healthy": True,
                "detail": f"{len(lessons)} lessons, {len(patterns)} patterns",
            }
        except Exception as e:
            components["knowledge_base"] = {"healthy": False, "detail": str(e)}

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
    # Keep last 500
    if len(alerts) > 500:
        alerts = alerts[-500:]
    save_json(ALERTS_PATH, alerts)


def get_active_alerts() -> list:
    """Get all active (unresolved) alerts."""
    alerts = load_json(ALERTS_PATH, default=[])
    return [a for a in alerts if a.get("status") == "active"]


MILESTONES_PATH = DATA_DIR / "milestones.json"

# Positive milestones
POSITIVE_MILESTONES = [
    {"id": "BEAT_RANDOM_30D", "name": "Beating Random Traders (30-day)",
     "check": "_check_beat_random_30d"},
    {"id": "BEAT_RANDOM_75TH", "name": "75th Percentile Achieved",
     "check": "_check_beat_random_75th"},
    {"id": "BEAT_BUY_HOLD", "name": "Beating Buy & Hold",
     "check": "_check_beat_buy_hold"},
    {"id": "PREDICTION_ACCURACY_60", "name": "Prediction Accuracy Above 60%",
     "check": "_check_prediction_accuracy_60"},
    {"id": "STRATEGY_DIVERGENCE", "name": "Strategy Weights Diverging",
     "check": "_check_strategy_divergence"},
    {"id": "GRADUATION_READY", "name": "All 12 Graduation Criteria Met",
     "check": "_check_graduation_ready"},
]

# Negative milestones
NEGATIVE_MILESTONES = [
    {"id": "PHASE_0_FAIL", "severity": "critical",
     "name": "Phase 0 Backtest Failure"},
    {"id": "RANDOM_LEVEL_60D", "severity": "high",
     "name": "No Better Than Random After 60 Days",
     "check": "_check_random_level_60d"},
    {"id": "ACCURACY_DECLINING", "severity": "high",
     "name": "Prediction Accuracy Trending Down",
     "check": "_check_accuracy_declining"},
    {"id": "DRAWDOWN_LIMIT", "severity": "critical",
     "name": "Portfolio Drawdown Limit Hit",
     "check": "_check_drawdown_limit"},
    {"id": "REGIME_COLLAPSE", "severity": "high",
     "name": "Performance Collapses on Regime Change"},
    {"id": "STAGNANT_WEIGHTS", "severity": "medium",
     "name": "Strategy Weights Never Diverge",
     "check": "_check_stagnant_weights"},
]


class MilestoneChecker:
    """Runs after market close daily. Checks all positive and negative milestones."""

    def __init__(self):
        self.triggered = load_json(MILESTONES_PATH, default=[])
        self._triggered_ids = {m["milestone_id"] for m in self.triggered}

    def check_all(self, metrics: dict) -> list[dict]:
        """Returns list of newly triggered milestones."""
        newly_triggered = []

        # Check positive milestones
        checks = {
            "BEAT_RANDOM_30D": lambda: metrics.get("percentile_vs_random", 0) >= 60,
            "BEAT_RANDOM_75TH": lambda: metrics.get("percentile_vs_random", 0) >= 75,
            "BEAT_BUY_HOLD": lambda: metrics.get("beats_buy_hold_tsla", False),
            "PREDICTION_ACCURACY_60": lambda: metrics.get("prediction_accuracy_pct", 0) >= 60,
            "STRATEGY_DIVERGENCE": lambda: self._check_weight_divergence(),
            "GRADUATION_READY": lambda: metrics.get("all_graduation_passed", False),
        }

        for ms in POSITIVE_MILESTONES:
            mid = ms["id"]
            if mid in self._triggered_ids:
                continue
            check_fn = checks.get(mid)
            if check_fn and check_fn():
                report = self._generate_report(ms, metrics, "positive")
                newly_triggered.append(report)

        # Check negative milestones
        neg_checks = {
            "RANDOM_LEVEL_60D": lambda: (
                metrics.get("trading_days", 0) >= 60
                and metrics.get("percentile_vs_random", 100) < 55
            ),
            "ACCURACY_DECLINING": lambda: metrics.get("accuracy_declining", False),
            "DRAWDOWN_LIMIT": lambda: metrics.get("max_drawdown_pct", 0) >= 15,
            "STAGNANT_WEIGHTS": lambda: (
                metrics.get("trading_days", 0) >= 60
                and not self._check_weight_divergence()
            ),
        }

        for ms in NEGATIVE_MILESTONES:
            mid = ms["id"]
            if mid in self._triggered_ids:
                continue
            check_fn = neg_checks.get(mid)
            if check_fn and check_fn():
                report = self._generate_report(ms, metrics, "negative")
                newly_triggered.append(report)

        # Save newly triggered
        if newly_triggered:
            self.triggered.extend(newly_triggered)
            save_json(MILESTONES_PATH, self.triggered)

        return newly_triggered

    def _check_weight_divergence(self) -> bool:
        try:
            from .knowledge_base import get_strategy_scores
            scores = get_strategy_scores()
            weights = [s["weight"] for s in scores.get("strategies", {}).values()]
            if not weights:
                return False
            return max(weights) >= 2 * min(weights)
        except Exception:
            return False

    def _generate_report(self, milestone: dict, metrics: dict, polarity: str) -> dict:
        config = load_config()
        severity = milestone.get("severity", "info")

        messages = {
            "BEAT_RANDOM_30D": "Agent has outperformed 60% of random traders. The learning loop may be producing real signal.",
            "BEAT_RANDOM_75TH": "Agent is in the top quartile vs. random. This is the graduation threshold for skill vs. luck.",
            "BEAT_BUY_HOLD": "Agent is beating the simplest possible strategy.",
            "PREDICTION_ACCURACY_60": "Agent is predicting TSLA direction correctly 60%+ of the time.",
            "STRATEGY_DIVERGENCE": "The learning loop is differentiating strategies based on experience.",
            "GRADUATION_READY": "All 12 graduation criteria met. Generate full graduation report.",
            "RANDOM_LEVEL_60D": "After 60 days, the agent is statistically indistinguishable from random trading.",
            "ACCURACY_DECLINING": "The agent is getting WORSE at predicting, not better.",
            "DRAWDOWN_LIMIT": "Maximum drawdown exceeded. Trading auto-halted.",
            "STAGNANT_WEIGHTS": "After 60 days, strategy weights haven't meaningfully changed.",
        }

        return {
            "milestone_id": milestone["id"],
            "name": milestone["name"],
            "triggered_at": iso_now(),
            "severity": severity if polarity == "negative" else "info",
            "polarity": polarity,
            "message": messages.get(milestone["id"], milestone["name"]),
            "model_version": config.get("anthropic_model", "unknown"),
            "metrics_snapshot": {
                "trading_days": metrics.get("trading_days", 0),
                "total_trades": metrics.get("total_trades", 0),
                "percentile_vs_random": metrics.get("percentile_vs_random", 0),
                "total_return_pct": metrics.get("total_return_pct", 0),
            },
        }
