# MonopolyTrader v7 — Research-First Trading with Active Learning

## Overview

Autonomous AI paper trading agent. Manages $1,000 virtual portfolio trading TSLA. Claude is a RESEARCHER studying TSLA patterns — trades are experiments, not bets. The playbook is the primary output, not P&L.

**Dashboard:** https://scifi-signals.github.io/MonopolyTrader/dashboard/
**Server:** DigitalOcean (`science-intel`), systemd service `monopoly-trader` at `/root/monopoly-trader`

## Design Philosophy

- **Researcher, not trader.** The agent studies patterns. Trades are experiments that generate data.
- **HOLD is the default.** Every trade needs a specific, testable hypothesis with expected new learning.
- **Don't repeat failed experiments.** If the playbook shows a pattern loses, trading it again wastes budget.
- **Losses are fine if they teach something new.** Redundant losses are waste.
- **Confidence = playbook win rate,** not narrative conviction.
- Paper money = learning budget, not capital to preserve
- No stop losses — Claude evaluates underwater positions with context every cycle

## Three Rules (only 3)

1. Max 50% of portfolio value in any position
2. Keep $100 cash minimum
3. Fractional shares OK

## v6 Architecture

### Daily Cycle

**9:00 AM — Pre-Market Research** (Sonnet)
- Reads Market Intelligence Document + pre-market price + overnight news
- Produces daily briefing with scenarios, thesis status, recommended posture

**Every 15 minutes — Research Cycle** (Sonnet)
1. Gather data — TSLA price/indicators + options + analyst consensus + institutional data + world + news + events
2. Compute tags — 14 market condition tags (including time_of_day, RSI divergence, news_catalyst, regime_age)
3. Build brief — Market data + constraints + hold analysis + playbook + hypothesis tracker + prediction diagnosis + exploration + journal
4. Claude decides — System prompt includes Market Intelligence Document + researcher identity
5. Log prediction + log hypothesis (if BUY/SELL)
6. Execute trade — Simulate with slippage (if BUY/SELL)
7. Record in journal — Tags, strategy, hypothesis, expected_learning logged
8. Log HOLD to shadow journal — Tracks what would have happened
9. Update shadow prices — Every cycle, updates recent HOLDs with current price
10. On trade close — Haiku lesson + hypothesis resolution + immediate playbook rebuild
11. Monitor portfolio health
12. Update dashboard

**4:15 PM — Nightly Analysis** (Sonnet)
- Archives current MID to mid_history.json (thesis versioning)
- Rebuilds playbook from all closed trades (single-tag + multi-tag + strategy stats)
- Rebuilds hold analysis, prediction diagnosis, hypothesis ledger (v7)
- Generates constraints and exploration map (v7)
- Synthesizes trades, lessons, playbook, hold analysis, prediction diagnosis, hypothesis tracker, exploration map, thesis history into updated MID
- Recommends experiment priorities based on exploration map

### Key v6 Additions

**Prediction Tracking** (`prediction_tracker.py`, `data/predictions.json`)
- Every cycle (BUY/SELL/HOLD), Claude predicts TSLA direction + magnitude over 1-4 cycles
- Predictions scored against reality: direction correct? magnitude correct?
- Prediction scorecard in brief shows accuracy by market condition
- Turns 26 daily cycles into 26 learning opportunities (vs ~1 from trades)
- Feeds into nightly analyst (where reads are strong/weak) and dashboard

**Shadow Journal** (`shadow_journal.py`, `data/hold_journal.json`)
- Every HOLD decision logged with full market context
- Shadow P&L computed: "if I'd bought at $X, price is now $Y"
- Summary shows: HOLDs that avoided losses vs missed gains
- Feeds into agent brief and nightly analyst

**Multi-Tag Pattern Analysis** (`thesis_builder.py`)
- 2-tag and 3-tag combination performance stats
- Strategy-level aggregation (performance by named strategy)
- `find_matching_patterns()` — returns all matching patterns for current conditions
- Strong signals: "STRONG_AVOID" for N>=5 at <25% win rate

**Thesis Versioning** (`analyst.py`, `data/mid_history.json`)
- MID archived before every nightly update
- Thesis history summary: direction, duration, flips
- Thesis accuracy tracking (direction maintained rate)

**Research Metrics** (`thesis_builder.py`)
- Experiment efficiency: ratio of exploratory vs repeated conditions
- Redundant loss rate: should be 0%
- Pattern discovery count: confirmed patterns (N>=5, clear signal)
- Calibration error: confidence vs actual win rate
- Rolling win rate (last 10 trades)

**Expanded Tags** (`tags.py`)
- time_of_day: morning_open, midday, afternoon, power_hour (Eastern time)
- intraday_daily_divergence: aligned, mild, strong divergence between RSI timeframes
- news_catalyst: highest-relevance catalyst type from news feed
- regime_age: new (0-2d), established (3-7d), mature (8+d) — from MID thesis history

### Key v7 Additions (Learning System Upgrade)

**Prediction Diagnosis** (`prediction_diagnosis.py`, `data/prediction_diagnosis.json`)
- Diagnoses WHY predictions are right/wrong (replaces scorecard in brief)
- Detects systematic biases: direction bias, magnitude over-prediction, overconfidence
- Computes accuracy by tag condition and time of day
- Auto-generates prescriptive insights ("your afternoon predictions are 30% accurate")
- Rebuilt nightly

**Hold Analyzer** (`hold_analyzer.py`, `data/hold_analysis.json`)
- Aggregates HOLDs by market conditions, compares outcomes vs trade outcomes
- Per-condition recommendation: HOLD_BETTER / TRADE_BETTER / COMPARABLE
- Shows in brief: "TRADE opportunity: rsi=oversold — hold avoids 40% but trades win 60%"
- Rebuilt nightly

**Hypothesis Ledger** (`hypothesis_ledger.py`, `data/hypothesis_ledger.json`)
- Formal hypothesis lifecycle: proposed → active → retired
- Grouped by (category, condition_tags) not free text — prevents duplicates
- Auto-retires hypotheses tested 5+ times with <30% confirmation rate
- Shows retired hypotheses in brief: "NEVER test this again"
- Bootstraps from existing trade_journal if empty

**Constraint Generator** (`constraint_generator.py`, `data/active_constraints.json`)
- Hard, data-driven constraints from playbook + prediction diagnosis + hold analysis + hypothesis ledger
- Types: NEVER_TRADE, LOW_CONFIDENCE, HOLD_PREFERRED, RETIRED_HYPOTHESIS
- Advisory position sizing: minimum/small/normal/confident tiers based on condition confidence
- Matching constraints shown prominently in brief

**Pattern Explorer** (`pattern_explorer.py`, `data/exploration_map.json`)
- Maps tag space: which conditions have been traded, held-only, or never observed
- Identifies exploration gaps: conditions frequent in HOLDs but never traded
- Ranks candidates by frequency + prediction accuracy + shadow P&L signal
- Designs micro-experiments with hypotheses, small position sizes, success criteria
- Shows matching opportunities in brief

**Data Enrichments**
- QQQ and XLK added to world tickers (tech sector correlation)
- power_hour time bucket (15:00-16:00 ET) — split from afternoon
- regime_age tag: new/established/mature based on MID thesis duration
- Cross-day learning summary in pre-market briefing

### Market Intelligence Document (data/market_intelligence.json)

Persistent JSON in the SYSTEM PROMPT. Updated nightly. Contains:
- **Thesis**: direction (bull/bear/neutral), confidence (0.3-0.9), evidence, invalidation criteria
- **Key levels**: support/resistance from recent price action
- **Active catalysts**: what's driving TSLA right now
- **Sector context**: EV sector and macro positioning
- **Lessons synthesis**: patterns from recent trade lessons
- **What's working/not working**: from playbook stats
- **Experiment priorities**: what to test tomorrow (v6)

## File Structure

```
src/
├── main.py              — Scheduler: 15min cycles + 9AM pre-market + 4:15PM nightly
├── analyst.py           — Nightly MID update + pre-market briefing + thesis versioning
├── agent.py             — SYSTEM_PROMPT (researcher identity) + build_market_brief() + make_decision()
├── prediction_tracker.py — Prediction logging, scoring, scorecard
├── prediction_diagnosis.py — v7: WHY predictions are right/wrong, bias detection, prescriptive insights
├── shadow_journal.py    — HOLD decision tracking + shadow P&L
├── hold_analyzer.py     — v7: HOLD vs TRADE comparison by market conditions
├── hypothesis_ledger.py — v7: Formal hypothesis lifecycle (proposed → active → retired)
├── constraint_generator.py — v7: Hard data-driven constraints from all learning modules
├── pattern_explorer.py  — v7: Active experiment design, tag space mapping, gap identification
├── tags.py              — Mechanical trade tagging (14 market condition tags)
├── thesis_builder.py    — Playbook: single-tag + multi-tag + strategy stats + research metrics
├── market_data.py       — Prices, indicators, regime, world, options, analyst, institutional
├── portfolio.py         — 2-rule validation, execute_trade(), cooldown
├── journal.py           — Trade journal with tags, strategy, hypothesis, Haiku lessons
├── events.py            — FOMC/CPI/NFP calendar + TSLA earnings
├── news_feed.py         — yfinance + RSS news
├── web_search.py        — Web search (disabled)
├── reporter.py          — Dashboard data generation (includes research metrics + shadow)
├── observability.py     — Health checks
└── utils.py             — Config, logging, AI call helpers (supports model override)

data/
├── market_intelligence.json  — Persistent MID (updated nightly)
├── mid_history.json          — MID version history (thesis changes over time)
├── daily_briefing.json       — Today's pre-market analysis
├── predictions.json          — Prediction log (every cycle, scored against reality)
├── prediction_diagnosis.json — v7: Prediction bias analysis + prescriptive insights
├── hold_journal.json         — Shadow journal (HOLD decisions + shadow P&L)
├── hold_analysis.json        — v7: HOLD vs TRADE comparison stats
├── hypothesis_ledger.json    — v7: Hypothesis lifecycle tracking
├── active_constraints.json   — v7: Hard constraints from learning modules
├── exploration_map.json      — v7: Tag space coverage + experiment candidates
├── portfolio.json            — Current portfolio state
├── transactions.json         — Full trade history
├── trade_journal.json        — Journal entries with tags, strategy, hypothesis, lessons
├── thesis_ledger.json        — Playbook stats (single + multi-tag + strategies)
├── snapshots/                — Daily portfolio snapshots
└── latest_cycle.json         — Last decision for dashboard
```

## What Claude Sees Each Cycle

**System Prompt (persistent context):**
- Researcher identity and principles (HOLD default, no repeat experiments, etc.)
- Market Intelligence Document (thesis, levels, catalysts, lessons, experiment priorities)

**User Prompt (per cycle):**
- Today's daily briefing (scenarios, thesis status, posture)
- TSLA price + 18 daily indicators + 5 intraday indicators
- Options flow (put/call ratio, max pain, unusual volume)
- Analyst consensus (buy/hold/sell counts, price targets)
- Institutional data (ownership %, short interest)
- World macro (9 tickers incl QQQ, XLK) + EV peers (5 tickers)
- Regime classification
- News (yfinance + RSS, classified by catalyst type)
- Upcoming events (FOMC, CPI, NFP, TSLA earnings)
- **Hard constraints** (NEVER_TRADE, LOW_CONFIDENCE, HOLD_PREFERRED, RETIRED_HYPOTHESIS)
- **Suggested position size** (tier + reasoning based on playbook + prediction accuracy)
- Drawdown status + last cycle recap
- Pattern matches + **HOLD vs TRADE analysis**
- Playbook (single-tag + multi-tag patterns + strategy stats + research metrics)
- **Hypothesis tracker** (active, retired, suggestions)
- Shadow journal summary (HOLD outcomes: avoided losses vs missed gains)
- **Prediction diagnosis** (bias verdicts + prescriptive insights, replaces scorecard)
- **Exploration opportunity** (matching micro-experiment if conditions align)
- Portfolio state + position limits
- Last 5 trades with hypotheses and lessons

## JSON Response Format

```json
{
  "action": "BUY" | "SELL" | "HOLD",
  "shares": 0.5,
  "confidence": 0.35,
  "strategy": "mean_reversion_oversold",
  "hypothesis": "RSI divergence in range-bound market predicts reversal within 2 hours",
  "expected_learning": "Tests whether intraday RSI < 30 with daily RSI > 50 signals a bounce",
  "reasoning": "Playbook shows 0/3 wins in similar conditions but only 3 trades — need more data...",
  "risk_note": "If price breaks below support at $340, hypothesis is refuted",
  "prediction": {
    "direction": "up",
    "magnitude": "small",
    "cycles": 2,
    "basis": "RSI divergence reverting toward daily mean"
  }
}
```

## Key Commands

```bash
python -m src.main             # Start scheduler
python -m src.main --once      # Single cycle
python -m src.main --report    # Generate dashboard
python -m src.main --analyst   # Run nightly analyst
python -m src.main --premarket # Run pre-market briefing
```

## Server

```bash
systemctl status monopoly-trader
systemctl restart monopoly-trader
tail -f /root/monopoly-trader/logs/agent.log
```

## Success Benchmark

v7 success = active learning rate, not P&L.
- Patterns discovered (N>=5 with clear signal)
- Experiment efficiency (% of trades exploring new conditions)
- Redundant loss rate (should be 0%)
- Calibration error (confidence vs actual outcomes)
- Hold quality (% of HOLDs that correctly avoided losses)
- Hypothesis lifecycle health (active → confirmed/retired, not stale)
- Prediction diagnosis coverage (systematic biases identified and corrected)
- Exploration progress (% of tag space with trade data)
- Also: total return, max drawdown, win rate, Sharpe ratio

## Cost

- ~$0.30/day Sonnet (26 cycles x ~3000 token brief)
- ~$0.07/day Haiku (trade lessons)
- ~$8.50/month total
