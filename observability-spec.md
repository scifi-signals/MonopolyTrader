# Observability & Debugging System — Shared Spec for MonopolyTrader + BSM

## Why This Matters

When an AI agent is making autonomous trading decisions every 5 minutes, things will go wrong. The agent might overweight a single noisy signal, the BSM might misinterpret sarcasm as bullish sentiment, or a strategy might quietly bleed money for days before anyone notices. Without proper observability, you're flying blind.

This spec defines the monitoring, alerting, tracing, and debugging systems that both MonopolyTrader and BSM should implement.

---

## 1. Decision Tracing — The Audit Trail

Every decision the agent makes should be fully traceable back to its inputs. If the agent buys TSLA at 10:35 AM and the price drops 3%, you should be able to reconstruct *exactly* why it made that call in under 60 seconds.

### MonopolyTrader: Decision Trace Record

Every decision cycle (including HOLDs) produces a trace:

```json
{
  "trace_id": "trace_2026-02-18_103500",
  "timestamp": "2026-02-18T10:35:00Z",
  "cycle_number": 47,
  "duration_ms": 2340,
  
  "inputs": {
    "market_data": {
      "price": 350.25,
      "change_pct": 1.2,
      "rsi": 62.3,
      "sma_20": 342.10,
      "sma_50": 335.80,
      "macd": "bullish_crossover",
      "volume_vs_avg": 1.3,
      "bollinger_position": 0.72
    },
    "strategy_signals": [
      {"strategy": "momentum", "action": "BUY", "raw_confidence": 0.75, "weight": 0.22, "weighted_score": 0.165},
      {"strategy": "mean_reversion", "action": "HOLD", "raw_confidence": 0.40, "weight": 0.28, "weighted_score": 0.0},
      {"strategy": "sentiment", "action": "BUY", "raw_confidence": 0.68, "weight": 0.20, "weighted_score": 0.136},
      {"strategy": "technical", "action": "BUY", "raw_confidence": 0.55, "weight": 0.18, "weighted_score": 0.099},
      {"strategy": "dca", "action": "BUY", "raw_confidence": 0.30, "weight": 0.12, "weighted_score": 0.036}
    ],
    "aggregate_signal": {"action": "BUY", "total_weighted_score": 0.436},
    "bsm_signals": [
      {"source": "elon_musk", "type": "podcast", "direction": "bullish", "strength": 0.75, "age_hours": 18.5}
    ],
    "knowledge_applied": {
      "lessons": ["lesson_012: Pre-market news often priced in by 10:30"],
      "patterns": ["pattern_005: Post-earnings drift"],
      "research": ["FSD v13 timeline acceleration"]
    },
    "portfolio_state": {
      "cash": 650.00,
      "tsla_shares": 1.0,
      "tsla_value": 350.25,
      "total_value": 1000.25,
      "pnl_pct": 0.025
    }
  },
  
  "agent_raw_response": "{full JSON response from Claude API}",
  "agent_tokens_used": {"input": 3200, "output": 450},
  
  "decision": {
    "action": "BUY",
    "shares": 0.5,
    "price": 350.25,
    "total_cost": 175.13,
    "confidence": 0.72,
    "strategy_primary": "momentum",
    "hypothesis": "TSLA will rise 1-2% over next 2 hours..."
  },
  
  "risk_checks": {
    "max_position_check": {"passed": true, "current_pct": 0.52, "limit": 0.90},
    "max_trade_check": {"passed": true, "trade_pct": 0.175, "limit": 0.25},
    "cash_reserve_check": {"passed": true, "cash_after": 474.87, "min_reserve_pct": 0.10},
    "cooldown_check": {"passed": true, "minutes_since_last_trade": 45, "min_required": 15},
    "daily_loss_check": {"passed": true, "daily_loss_pct": 0.0, "limit": 0.08},
    "stop_loss_check": {"triggered": false}
  },
  
  "execution": {
    "executed": true,
    "slippage_applied": 0.001,
    "final_price": 350.60,
    "status": "success"
  },
  
  "anomalies_detected": []
}
```

### BSM: Analysis Trace Record

Every transcript analysis produces a trace:

```json
{
  "trace_id": "bsm_trace_2026-02-15_221000",
  "timestamp": "2026-02-15T22:10:00Z",
  "content_analyzed": {
    "person": "elon_musk",
    "source": "Lex Fridman Podcast #452",
    "source_type": "podcast",
    "transcript_length_words": 28450,
    "transcription_method": "youtube_captions",
    "transcription_confidence": 0.94
  },
  "analysis": {
    "claude_model": "claude-sonnet-4-20250514",
    "tokens_used": {"input": 18500, "output": 2200},
    "duration_ms": 8400,
    "signals_generated": 2,
    "sentiment_scores": {
      "tsla_specific": 0.82,
      "previous_tsla_specific": 0.78,
      "shift_magnitude": 0.04,
      "shift_significance": "minor"
    }
  },
  "quality_flags": {
    "transcript_quality": "good",
    "speaker_identification_confident": true,
    "sarcasm_or_humor_detected": false,
    "interviewer_leading_questions": true,
    "person_seemed_evasive": false
  },
  "anomalies_detected": []
}
```

---

## 2. Anomaly Detection — Catching Problems Automatically

Both systems should run automatic checks and flag anomalies. These aren't just errors — they're situations that *might* be problems and need human review.

### MonopolyTrader Anomaly Checks

Run after every decision cycle:

```python
class AnomalyDetector:
    
    def check_all(self, trace: dict, history: list[dict]) -> list[dict]:
        """Run all anomaly checks. Return list of flagged issues."""
        anomalies = []
        anomalies += self.check_concentration_drift(trace)
        anomalies += self.check_strategy_dominance(trace, history)
        anomalies += self.check_overtrading(history)
        anomalies += self.check_confidence_calibration(history)
        anomalies += self.check_signal_staleness(trace)
        anomalies += self.check_drawdown_velocity(history)
        anomalies += self.check_agent_consistency(trace, history)
        anomalies += self.check_api_degradation(trace)
        return anomalies
    
    def check_concentration_drift(self, trace):
        """Is the portfolio drifting toward extreme concentration?
        Flag if position > 80% of portfolio."""
    
    def check_strategy_dominance(self, trace, history):
        """Is one strategy dominating all decisions?
        Flag if >70% of recent trades use the same strategy.
        This could mean the weighting system is broken or 
        the agent is in a rut."""
    
    def check_overtrading(self, history):
        """Is the agent trading too frequently?
        Flag if >10 trades in a single day.
        Excessive trading usually means the agent is 
        reacting to noise, not signal."""
    
    def check_confidence_calibration(self, history):
        """Is the agent's confidence score well-calibrated?
        If it says 0.8 confidence and only wins 40% of the time,
        confidence is meaningless. Flag if confidence-to-accuracy
        ratio is off by more than 20%."""
    
    def check_signal_staleness(self, trace):
        """Are any input signals stale?
        Flag if market data is >10 minutes old.
        Flag if BSM signals are >48 hours old and still 
        being weighted heavily."""
    
    def check_drawdown_velocity(self, history):
        """Is the portfolio losing money too fast?
        Flag if portfolio dropped >3% in last 2 hours 
        or >5% in a single day.
        Different from stop-loss — this catches gradual bleeding."""
    
    def check_agent_consistency(self, trace, history):
        """Is the agent contradicting itself?
        Flag if it bought something it said was bearish,
        or if it's flip-flopping (buy → sell → buy within 
        30 minutes on same ticker)."""
    
    def check_api_degradation(self, trace):
        """Is the Claude API response degrading?
        Flag if response time >10s, if response fails to parse,
        or if response is unusually short/generic."""
```

### BSM Anomaly Checks

```python
class BSMAnomalyDetector:
    
    def check_all(self, trace: dict, history: list[dict]) -> list[dict]:
        anomalies = []
        anomalies += self.check_transcript_quality(trace)
        anomalies += self.check_sentiment_spike(trace, history)
        anomalies += self.check_signal_flood(history)
        anomalies += self.check_source_reliability(trace)
        anomalies += self.check_stale_profiles(history)
        return anomalies
    
    def check_transcript_quality(self, trace):
        """Is the transcript reliable?
        Flag if transcription confidence < 80%.
        Flag if speaker identification failed (we might
        attribute interviewer's words to the tracked person).
        Flag if transcript is suspiciously short for the 
        audio duration."""
    
    def check_sentiment_spike(self, trace, history):
        """Did sentiment change too dramatically?
        Flag if a person's sentiment shifted >0.4 in a single 
        appearance. This usually means:
        a) The analysis is wrong (sarcasm, hypothetical, etc.)
        b) Something genuinely huge happened (rare)
        Either way, needs human review."""
    
    def check_signal_flood(self, history):
        """Are we generating too many signals?
        Flag if >10 signals generated in a single day.
        Signal inflation dilutes quality."""
    
    def check_source_reliability(self, trace):
        """Is the source trustworthy?
        Flag if content source is unfamiliar or low-authority.
        Flag if transcript came from an unverified YouTube 
        reupload rather than the official channel."""
    
    def check_stale_profiles(self, history):
        """Are any person profiles outdated?
        Flag if a person hasn't had a new analysis in >30 days.
        Stale profiles produce bad shift detection."""
```

---

## 3. Weight & Influence Tracking — What's Driving Decisions?

This is critical for diagnosing when something is given too much weight. Every decision should show a breakdown of what influenced it and by how much.

### Influence Breakdown

```json
{
  "trace_id": "trace_2026-02-18_103500",
  "influence_breakdown": {
    "technical_indicators": {
      "influence_pct": 35,
      "details": {
        "rsi": {"value": 62.3, "contribution": "bullish"},
        "macd": {"value": "bullish_crossover", "contribution": "strong_bullish"},
        "sma_trend": {"value": "price > sma20 > sma50", "contribution": "bullish"},
        "volume": {"value": "1.3x average", "contribution": "confirming"}
      }
    },
    "strategy_weights": {
      "influence_pct": 30,
      "details": {
        "momentum": {"weight": 0.22, "signal": "BUY", "contribution": 0.165},
        "mean_reversion": {"weight": 0.28, "signal": "HOLD", "contribution": 0.0},
        "sentiment": {"weight": 0.20, "signal": "BUY", "contribution": 0.136},
        "technical": {"weight": 0.18, "signal": "BUY", "contribution": 0.099},
        "dca": {"weight": 0.12, "signal": "BUY", "contribution": 0.036}
      }
    },
    "bsm_signals": {
      "influence_pct": 20,
      "details": {
        "elon_musk_podcast": {"strength": 0.75, "age_hours": 18.5, "decay_adjusted": 0.62}
      }
    },
    "knowledge_base": {
      "influence_pct": 10,
      "details": {
        "lessons_applied": 1,
        "patterns_matched": 1,
        "research_referenced": 1
      }
    },
    "portfolio_state": {
      "influence_pct": 5,
      "details": {
        "cash_available": "sufficient",
        "existing_position": "small",
        "recent_pnl": "neutral"
      }
    }
  },
  "dominant_factor": "technical_indicators",
  "flags": {
    "single_factor_dominance": false,
    "bsm_signal_overweight": false,
    "stale_signal_influence": false
  }
}
```

### Weight Drift Dashboard

Track how influence sources change over time:

```python
def calculate_influence_trends(traces: list[dict], window_days: int = 7) -> dict:
    """Over the last N days, what has been driving decisions?
    
    Returns:
    - Average influence breakdown (what % of decisions driven by 
      technicals vs BSM vs knowledge vs strategies)
    - Drift alerts: Has any single influence source grown >50%?
    - Correlation: Which influence sources correlate with winning trades?
    """
```

---

## 4. Health Dashboard — System Status at a Glance

Both projects should expose a simple health status that can be checked quickly.

### MonopolyTrader Health Check

```json
{
  "status": "healthy",
  "checked_at": "2026-02-18T10:40:00Z",
  "components": {
    "market_data_feed": {
      "status": "ok",
      "last_fetch": "2026-02-18T10:35:00Z",
      "latency_ms": 450
    },
    "claude_api": {
      "status": "ok",
      "last_call": "2026-02-18T10:35:02Z",
      "avg_latency_ms": 2100,
      "error_rate_1h": 0.0
    },
    "bsm_feed": {
      "status": "ok",
      "latest_signal_age_hours": 18.5,
      "signals_available": 2
    },
    "portfolio": {
      "status": "ok",
      "total_value": 1000.25,
      "daily_pnl_pct": 0.025,
      "trades_today": 3,
      "stop_losses_triggered_today": 0
    },
    "knowledge_base": {
      "status": "ok",
      "lessons_count": 24,
      "patterns_count": 7,
      "last_research": "2026-02-17T16:10:00Z"
    },
    "anomalies": {
      "active_count": 0,
      "last_anomaly": "2026-02-16T14:22:00Z",
      "unresolved": []
    }
  },
  "recent_performance": {
    "prediction_accuracy_7d": 0.58,
    "prediction_accuracy_trend": "improving",
    "win_rate_7d": 0.55,
    "sharpe_ratio_7d": 0.42
  }
}
```

### BSM Health Check

```json
{
  "status": "healthy",
  "checked_at": "2026-02-18T10:40:00Z",
  "components": {
    "podcast_monitor": {
      "status": "ok",
      "feeds_tracked": 6,
      "last_check": "2026-02-18T10:00:00Z",
      "new_episodes_24h": 3
    },
    "youtube_monitor": {
      "status": "ok",
      "channels_tracked": 4,
      "last_check": "2026-02-18T09:00:00Z"
    },
    "twitter_monitor": {
      "status": "degraded",
      "accounts_tracked": 4,
      "last_check": "2026-02-18T10:30:00Z",
      "issue": "Rate limited — next check in 12 minutes"
    },
    "transcription": {
      "status": "ok",
      "transcriptions_24h": 2,
      "avg_quality_score": 0.92,
      "cost_24h": "$0.48"
    },
    "analysis": {
      "status": "ok",
      "analyses_24h": 3,
      "signals_generated_24h": 4,
      "avg_signal_strength": 0.52
    },
    "profiles": {
      "persons_tracked": 6,
      "freshest_profile": {"person": "elon_musk", "age_hours": 18},
      "stalest_profile": {"person": "warren_buffett", "age_days": 12}
    },
    "signal_output": {
      "status": "ok",
      "last_write": "2026-02-18T06:30:00Z",
      "active_signals": 2,
      "expired_signals_cleaned": 1
    }
  }
}
```

---

## 5. Alerting — When to Notify the Human

Not every anomaly needs immediate attention. Use a severity system:

### Severity Levels

| Level | Action | Examples |
|-------|--------|---------|
| **INFO** | Log only. Review in daily digest. | New pattern discovered, strategy weight shifted, profile updated |
| **WARN** | Highlight in dashboard. Include in daily report. | Confidence miscalibration >15%, strategy dominance >60%, BSM signal spike, stale profile |
| **ALERT** | Needs attention within hours. | Drawdown >5% in a day, API error rate >10%, agent contradicting itself, transcript quality <70% |
| **CRITICAL** | Stop trading. Needs immediate human review. | Daily loss limit hit, portfolio value dropped >10% from peak, agent stuck in buy-sell loop, all API calls failing |
| **MILESTONE** | Strategic decision point. Prominent dashboard display. Persists until acknowledged. | See MonopolyTrader CLAUDE.md Milestone Alert & Decision System for full positive/negative milestone definitions. Milestones are different from operational alerts — they represent project-level decision points (graduation readiness, retool triggers, kill criteria). |

### Alert Output

Alerts should be written to `logs/alerts.json` and optionally displayed prominently on the dashboard:

```json
{
  "id": "alert_2026-02-18_001",
  "timestamp": "2026-02-18T11:15:00Z",
  "severity": "ALERT",
  "source": "monopoly_trader",
  "check": "check_agent_consistency",
  "title": "Agent flip-flopping on TSLA",
  "detail": "Agent bought 0.5 shares at 10:35, sold at 10:50, bought again at 11:05. Three opposing trades in 30 minutes suggests the agent is reacting to noise or conflicting signals.",
  "affected_trades": ["txn_047", "txn_048", "txn_049"],
  "recommended_action": "Review agent decision traces for these three trades. Check if BSM signal and technical indicators are giving contradictory signals. Consider increasing cooldown period.",
  "resolved": false,
  "resolution_notes": null
}
```

---

## 6. Debugging Tools — CLI Commands

Both projects should include CLI tools for quick debugging:

### MonopolyTrader Debug Commands

```bash
# Show the last N decision traces with full inputs
python src/main.py --debug-traces 5

# Show why a specific trade was made
python src/main.py --explain-trade txn_047

# Show current influence breakdown (what's driving decisions right now)
python src/main.py --influence-report

# Show strategy weight history (are weights evolving sensibly?)
python src/main.py --strategy-history

# Show prediction accuracy breakdown
python src/main.py --accuracy-report

# Show all active anomalies and alerts
python src/main.py --alerts

# Show full health check
python src/main.py --health

# Replay a decision with different inputs (what-if analysis)
python src/main.py --replay-trace trace_2026-02-18_103500 --override '{"rsi": 75}'

# Show the agent's knowledge base utilization
# (is it actually using lessons, or ignoring them?)
python src/main.py --knowledge-usage-report

# Compare agent decisions against a simple baseline
# (would buy-and-hold have been better? by how much?)
python src/main.py --vs-baseline

# Dump the full state of everything for debugging
python src/main.py --dump-state
```

### BSM Debug Commands

```bash
# Show recent analysis traces
python src/main.py --debug-traces 5

# Explain why a specific signal was generated
python src/main.py --explain-signal sig_2026-02-15_001

# Show a person's sentiment timeline
python src/main.py --person-timeline elon_musk

# Reanalyze a transcript (useful if you suspect bad analysis)
python src/main.py --reanalyze 2026-02-15_elon_musk_lex_fridman_452

# Show signal quality report (are signals accurate?)
python src/main.py --signal-accuracy

# Show all active alerts
python src/main.py --alerts

# Show full health check
python src/main.py --health

# Test transcript quality on a specific URL
python src/main.py --test-transcribe "https://youtube.com/watch?v=..."

# Show cost report (how much are we spending on APIs?)
python src/main.py --cost-report
```

---

## 7. Performance & Cost Monitoring

Track API costs and performance to catch runaway spending or degradation.

```json
{
  "period": "2026-02-18",
  "monopoly_trader": {
    "claude_api": {
      "calls": 78,
      "total_input_tokens": 249600,
      "total_output_tokens": 35100,
      "estimated_cost": "$1.82",
      "avg_latency_ms": 2100,
      "errors": 0
    },
    "yfinance": {
      "calls": 156,
      "errors": 2,
      "error_details": ["timeout at 14:22", "rate limited at 15:45"]
    },
    "decision_cycles_completed": 78,
    "decision_cycles_failed": 0
  },
  "bsm": {
    "claude_api": {
      "calls": 5,
      "total_input_tokens": 92000,
      "total_output_tokens": 11000,
      "estimated_cost": "$0.65"
    },
    "whisper_api": {
      "minutes_transcribed": 47,
      "cost": "$0.28"
    },
    "youtube_captions": {
      "transcripts_fetched": 3,
      "cost": "$0.00"
    },
    "total_daily_cost": "$0.93"
  },
  "combined_daily_cost": "$2.75",
  "cost_trend_7d": "stable",
  "budget_alert": false
}
```

---

## 8. Log Structure

Both projects should use structured JSON logging so logs are searchable and parseable.

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """JSON-structured logger for both projects."""
    
    def __init__(self, name: str, log_file: str):
        self.logger = logging.getLogger(name)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
    
    def log(self, level: str, event: str, **kwargs):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "event": event,
            **kwargs
        }
        self.logger.log(
            getattr(logging, level.upper()),
            json.dumps(entry)
        )

# Usage:
log = StructuredLogger("monopoly_trader", "logs/agent.log")

log.log("INFO", "decision_cycle_start", cycle=47)
log.log("INFO", "market_data_fetched", price=350.25, latency_ms=450)
log.log("INFO", "agent_decision", action="BUY", shares=0.5, confidence=0.72)
log.log("WARN", "anomaly_detected", check="overtrading", trades_today=11)
log.log("ERROR", "api_call_failed", service="claude", error="timeout", retry=True)
```

---

## 9. Implementation Notes

### Where This Lives in Each Project

**MonopolyTrader** — Add to `src/`:
- `observability.py` — AnomalyDetector, InfluenceTracker, HealthCheck, DecisionTracer
- Update `main.py` to run anomaly checks after each cycle, write traces, expose health endpoint
- Update `reporter.py` dashboard to include health status, anomaly alerts, and influence breakdown

**BSM** — Add to `src/`:
- `observability.py` — BSMAnomalyDetector, HealthCheck, AnalysisTracer
- Update `main.py` to run anomaly checks after each analysis, write traces
- Update `digest.py` to include health status and quality flags

### Storage

- Decision traces: `logs/traces/YYYY-MM-DD/trace_HHMMSS.json` (one file per cycle)
- Alerts: `logs/alerts.json` (append-only, with resolved/unresolved status)
- Health: `data/health.json` (overwritten each check)
- Influence history: `data/influence_history.json` (daily summaries appended)
- Cost tracking: `logs/costs/YYYY-MM-DD.json` (daily cost reports)

### Retention

- Keep traces for 30 days, then archive to compressed files
- Keep alerts indefinitely (they're small and valuable for pattern analysis)
- Keep health snapshots for 7 days
- Keep cost reports indefinitely

---

## 10. The "What Happened?" Workflow

When something goes wrong, here's the diagnostic workflow:

```
1. CHECK HEALTH
   python src/main.py --health
   → Is everything connected? Any components down?

2. CHECK ALERTS
   python src/main.py --alerts
   → Any active anomalies flagged?

3. FIND THE BAD DECISION
   python src/main.py --debug-traces 10
   → Look at recent decisions. Which one went wrong?

4. EXPLAIN THE DECISION
   python src/main.py --explain-trade txn_047
   → Full input breakdown. What data did the agent see?

5. CHECK INFLUENCE
   python src/main.py --influence-report
   → Was one signal source dominating? Was BSM overweighted?

6. CHECK BSM (if BSM signal was involved)
   cd ../billionaire-signal-monitor
   python src/main.py --explain-signal sig_2026-02-15_001
   → Was the original analysis accurate? Was sentiment misread?

7. CHECK KNOWLEDGE BASE
   python src/main.py --knowledge-usage-report
   → Is the agent applying lessons, or ignoring them?

8. COMPARE TO BASELINE
   python src/main.py --vs-baseline
   → How much worse (or better) are we doing vs. simple buy-and-hold?
```

This workflow should take <5 minutes to diagnose any issue.
