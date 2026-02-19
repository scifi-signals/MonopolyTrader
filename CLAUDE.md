# MonopolyTrader — An AI Stock Trading Agent That Learns

## Project Overview

MonopolyTrader is an autonomous AI-powered paper trading agent that manages a virtual portfolio of $1,000 ("Monopoly dollars") by analyzing real market data for **TSLA** and executing simulated trades. Unlike typical trading bots that follow fixed rules, MonopolyTrader is designed to **learn from its mistakes, build institutional knowledge over time, and evolve its strategies** based on accumulated experience.

**Goal**: Grow the $1,000 starting balance — not by being right on day one, but by developing increasingly accurate predictive ability through research, reflection, and adaptation. The agent will be wrong often at first. That's expected. What matters is that it gets *less wrong* over time.

## Core Philosophy: Learn, Don't Just Trade

The agent operates on four principles:

1. **Every trade is a hypothesis.** When the agent buys, it's saying "I believe X will happen because of Y." When the outcome is known, it compares prediction vs. reality and logs what it learned.

2. **Memory compounds into wisdom.** The agent maintains a growing "knowledge base" about TSLA — how it reacts to earnings, Elon's tweets, macro events, sector rotations, and seasonal patterns. This knowledge is built through both active research and post-trade reflection.

3. **Strategies earn trust.** Each strategy starts with equal weight. Over time, the agent adjusts confidence in each strategy based on real performance data. Strategies that consistently fail get deprioritized. New hybrid approaches can emerge from patterns the agent notices.

4. **Be a scientific instrument, not a storyteller.** LLMs are very good at inventing confident causal explanations after the fact. The agent must resist narrative lock-in. Every lesson must be structured, falsifiable, and challenged by a built-in Skeptic layer. If the agent can't prove a lesson with data, it's a story, not a signal. Let data overrule good stories.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  DASHBOARD (HTML/JS)             │
│  Portfolio · Trade log · Strategy evolution ·    │
│  Learning journal · Prediction accuracy          │
├─────────────────────────────────────────────────┤
│                  REPORTER (Python)                │
│  Generates HTML dashboard + JSON API for live UI │
├─────────────────────────────────────────────────┤
│              ┌──────────────────────┐            │
│              │   LEARNING ENGINE    │            │
│              │  Post-trade reviews  │            │
│              │  Strategy evolution  │            │
│              │  Pattern discovery   │            │
│              └──────────┬───────────┘            │
│                         │                        │
│              ┌──────────▼───────────┐            │
│              │    KNOWLEDGE BASE    │            │
│              │  TSLA behavior log   │            │
│              │  Lessons learned     │            │
│              │  Strategy scores     │            │
│              │  Research findings   │            │
│              └──────────┬───────────┘            │
│                         │                        │
├─────────────────────────┼───────────────────────┤
│               AGENT BRAIN (Claude API)           │
│  Reads knowledge base · Makes predictions ·     │
│  States hypothesis before every trade            │
├─────────────────────────────────────────────────┤
│               EXECUTION ENGINE (Python)          │
│  Simulates trades · Enforces rules · Logs txns   │
├─────────────────────────────────────────────────┤
│         ┌───────────────┬───────────────┐       │
│    MARKET DATA     RESEARCH ENGINE   NEWS/SENT  │
│    Prices/OHLCV    Historical study  Headlines   │
│    Indicators      Event analysis    Sentiment   │
│    Volume          Correlation       Social      │
│         └───────────────┴───────────────┘       │
├─────────────────────────────────────────────────┤
│              PORTFOLIO STATE (JSON)               │
│  Cash · Holdings · History · Snapshots           │
└─────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Backend**: Python 3.11+
- **Market Data**: `yfinance` for price data, `ta` (Technical Analysis library) for indicators
- **AI Agent**: Anthropic Claude API (`claude-sonnet-4-20250514`) for decision-making
- **Dashboard**: HTML5 + Vanilla JS + Chart.js for interactive charts
- **Data Storage**: JSON files (portfolio state, transaction log, snapshots)
- **Scheduler**: Python `schedule` library for real-time polling (every 5 minutes during market hours)

---

## File Structure

```
monopoly-trader/
├── CLAUDE.md                 # This file — project spec and instructions
├── config.json               # Global configuration
├── agents/                   # Per-agent configurations (see multi-agent-ensemble-spec.md)
│   ├── alpha.json            # The Technician — pure technical analysis
│   ├── bravo.json            # The Insider — sentiment + BSM focused
│   ├── echo.json             # The Generalist — balanced learner
│   └── foxtrot.json          # The Contrarian — fades consensus
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── portfolio.json        # Current portfolio state
│   ├── transactions.json     # Full trade history
│   ├── strategy_scores.json  # Evolving strategy confidence weights
│   └── snapshots/            # Daily portfolio value snapshots
│       └── YYYY-MM-DD.json
│
├── knowledge/                # THE AGENT'S GROWING BRAIN
│   ├── lessons.json          # Post-trade reflections and lessons learned
│   ├── tsla_profile.json     # Accumulated knowledge about TSLA behavior
│   ├── predictions.json      # Every prediction + outcome for accuracy tracking
│   ├── patterns.json         # Discovered patterns (e.g., "TSLA drops after...")
│   ├── research/             # Deep research reports the agent generates
│   │   ├── earnings_history.json
│   │   ├── catalyst_events.json
│   │   ├── correlation_notes.json
│   │   └── sector_context.json
│   └── journal.md            # Free-form agent journal — reflections in prose
│
├── src/
│   ├── __init__.py
│   ├── main.py               # Entry point — scheduler + main loop
│   ├── ensemble.py           # Multi-agent orchestrator (runs all agents in parallel)
│   ├── comparison.py         # Agent comparison: leaderboard, correlation, harmony
│   ├── meta_learner.py       # Cross-agent analysis, regime detection, suggestions
│   ├── agent.py              # Claude-powered decision engine (per-agent instance)
│   ├── market_data.py        # Price fetching + technical indicators
│   ├── portfolio.py          # Portfolio management + trade execution
│   ├── benchmarks.py         # Benchmark tracking, graduation criteria, random trader simulation
│   ├── strategies.py         # Strategy definitions + signal generators
│   ├── researcher.py         # Deep research engine — studies TSLA history
│   ├── learner.py            # Post-trade review + knowledge extraction
│   ├── knowledge_base.py     # Read/write to the knowledge/ directory
│   ├── reporter.py           # Dashboard HTML generation
│   ├── observability.py      # Anomaly detection, decision tracing, health checks
│   └── utils.py              # Shared helpers (logging, time, formatting)
│
├── dashboard/
│   ├── index.html            # Main dashboard page
│   ├── style.css             # Dashboard styling
│   ├── app.js                # Dashboard interactivity + chart rendering
│   └── data.json             # Latest portfolio data for dashboard
│
└── logs/
    ├── agent.log             # Agent decision log
    ├── trades.log            # Trade execution log
    ├── alerts.json           # Active and resolved anomaly alerts
    ├── traces/               # Full decision traces per cycle
    │   └── YYYY-MM-DD/
    │       └── trace_HHMMSS.json
    └── costs/                # Daily API cost reports
        └── YYYY-MM-DD.json
```

---

## Data Schemas

### config.json
```json
{
  "ticker": "TSLA",
  "starting_balance": 1000.00,
  "currency": "Monopoly Dollars",
  "poll_interval_minutes": 15,
  "market_hours": {
    "open": "09:30",
    "close": "16:00",
    "timezone": "America/New_York"
  },
  "risk_params": {
    "max_position_pct": 0.65,
    "max_single_trade_pct": 0.20,
    "stop_loss_method": "dynamic_atr",
    "stop_loss_atr_multipliers": {
      "low_vix": 2.0,
      "normal_vix": 2.5,
      "high_vix": 3.0,
      "vix_thresholds": [20, 30]
    },
    "position_sizing_method": "inverse_atr",
    "max_risk_per_trade_pct": 0.02,
    "min_cash_reserve_pct": 0.15,
    "enable_fractional_shares": true,
    "slippage_per_side_pct": 0.0005,
    "slippage_volatile_per_side_pct": 0.0015,
    "slippage_note": "0.05% baseline, but real bid-ask on 15-min TSLA bars can hit 0.15-0.30% round-trip on volatile days. Use volatile rate when VIX > 25.",
    "earnings_blackout_hours": 48,
    "gap_risk_size_reduction_pct": 0.50,
    "bsm_conviction_cap_pct": 0.10,
    "macro_gate": {
      "spy_daily_drop_threshold": -0.02,
      "vix_threshold": 30,
      "elevated_confidence_required": 0.80
    },
    "kill_switches": {
      "negative_accuracy_trend_days": 30,
      "max_drawdown_from_peak_pct": 0.15,
      "critical_alerts_in_7_days": 3,
      "style_drift_winrate_drop_pct": 0.30
    },
    "milestone_alerts": {
      "enabled": true,
      "notification_method": "dashboard_prominent + log",
      "note": "See Milestone Alert & Decision System section for full spec"
    }
  },
  "anthropic_model": "claude-sonnet-4-20250514",
  "strategies_enabled": [
    "momentum",
    "mean_reversion",
    "sentiment",
    "technical_signals",
    "dca"
  ],
  "learning": {
    "research_on_startup": true,
    "review_after_every_trade": true,
    "deep_research_interval_hours": 24,
    "journal_entry_interval_hours": 12,
    "prediction_horizon_minutes": [30, 120, 1440]
  }
}
```

### portfolio.json
```json
{
  "cash": 1000.00,
  "holdings": {
    "TSLA": {
      "shares": 0,
      "avg_cost_basis": 0,
      "current_price": 0,
      "unrealized_pnl": 0
    }
  },
  "total_value": 1000.00,
  "total_pnl": 0,
  "total_pnl_pct": 0,
  "total_trades": 0,
  "winning_trades": 0,
  "losing_trades": 0,
  "created_at": "2026-02-18T00:00:00Z",
  "last_updated": "2026-02-18T00:00:00Z"
}
```

### Transaction Record (with hypothesis)
```json
{
  "id": "txn_001",
  "timestamp": "2026-02-18T10:35:00Z",
  "action": "BUY",
  "ticker": "TSLA",
  "shares": 1.5,
  "price": 350.25,
  "total_cost": 525.38,
  "cash_after": 474.62,
  "strategy": "momentum",
  "confidence": 0.78,
  "hypothesis": "TSLA will rise 2-4% over the next 2 hours based on strong upward momentum, positive delivery report sentiment, and RSI indicating room to run before overbought territory.",
  "reasoning": "RSI at 62 (bullish, not overbought), price above both SMA20 and SMA50 with widening gap, volume 1.3x average suggesting conviction. Recent delivery numbers beat expectations. Similar setups in my knowledge base have led to continued upside 65% of the time.",
  "signals": {
    "rsi": 62.3,
    "sma_20": 342.10,
    "sma_50": 335.80,
    "macd_signal": "bullish_crossover",
    "sentiment_score": 0.72,
    "volume_trend": "above_average"
  },
  "knowledge_applied": ["lesson_012", "pattern_005"],
  "review": null
}
```

### Prediction Record (predictions.json)
```json
{
  "id": "pred_001",
  "timestamp": "2026-02-18T10:35:00Z",
  "price_at_prediction": 350.25,
  "predictions": {
    "30min": {"direction": "up", "target": 353.50, "confidence": 0.72},
    "2hr": {"direction": "up", "target": 358.00, "confidence": 0.65},
    "1day": {"direction": "up", "target": 362.00, "confidence": 0.55}
  },
  "reasoning": "Momentum continuation after positive delivery report...",
  "outcomes": {
    "30min": {"actual": 351.80, "direction_correct": true, "error_pct": 0.48},
    "2hr": {"actual": 347.20, "direction_correct": false, "error_pct": 3.10},
    "1day": null
  },
  "linked_trade": "txn_001"
}
```

### Lesson Record (lessons.json)
```json
{
  "id": "lesson_012",
  "timestamp": "2026-02-18T14:45:00Z",
  "linked_trade": "txn_001",
  "category": "correlated_market_move",
  "what_i_predicted": "2-4% rise over 2 hours",
  "what_actually_happened": "Rose 0.4% in 30min then reversed, ending 2hr window down 0.9%",
  "initial_explanation": "Overweighted delivery report sentiment. The positive news was already priced in by market open.",
  "skeptic_review": {
    "simpler_explanation": "SPY dropped 1.2% during the same 2-hour window. TSLA's decline was consistent with broad market selloff, not TSLA-specific.",
    "sample_size": 3,
    "validated": false,
    "regime_dependent": true,
    "falsifiable_test": "If TSLA declines when SPY is flat after a pre-market delivery beat, this lesson holds. If TSLA declines only when SPY also declines, the lesson is about market correlation, not delivery report timing."
  },
  "lesson": "Pre-market delivery beats may be priced in by 10:30 AM, but this instance was confounded by a broad market selloff. Need more data points where SPY is flat to isolate the delivery-beat effect.",
  "confidence_adjustment": {
    "sentiment": -0.04,
    "momentum": -0.02
  },
  "weight": 1.0,
  "decay_rate": 0.95,
  "times_validated": 0,
  "times_contradicted": 1,
  "last_validated": null
}
```

### Pattern Record (patterns.json)
```json
{
  "id": "pattern_005",
  "discovered": "2026-02-25T16:10:00Z",
  "name": "TSLA Post-Earnings Drift",
  "description": "After TSLA earnings beats, the stock tends to drift upward for 3-5 trading days as analysts revise targets, then consolidate. After earnings misses, the drop is usually front-loaded (day 1) with partial recovery by day 3.",
  "evidence": ["txn_003", "txn_008", "research_earnings_q4_2025"],
  "reliability": 0.62,
  "sample_size": 8,
  "last_tested": "2026-02-25T00:00:00Z",
  "tags": ["earnings", "drift", "multi-day"]
}
```

### Strategy Scores (strategy_scores.json)
```json
{
  "last_updated": "2026-02-25T16:15:00Z",
  "strategies": {
    "momentum": {
      "weight": 0.22,
      "initial_weight": 0.20,
      "total_trades": 12,
      "winning_trades": 5,
      "win_rate": 0.42,
      "avg_return_pct": -0.3,
      "total_pnl": -14.50,
      "trend": "declining",
      "notes": "Performing poorly in current choppy market. Consider deprioritizing until clear trend emerges."
    },
    "mean_reversion": {
      "weight": 0.28,
      "initial_weight": 0.20,
      "total_trades": 8,
      "winning_trades": 6,
      "win_rate": 0.75,
      "avg_return_pct": 1.2,
      "total_pnl": 45.20,
      "trend": "improving",
      "notes": "Strong performer. TSLA's high volatility creates frequent reversion opportunities."
    }
  },
  "rebalance_history": [
    {
      "timestamp": "2026-02-22T16:10:00Z",
      "changes": {"momentum": -0.03, "mean_reversion": +0.05},
      "reason": "Weekly review: momentum underperforming in range-bound market"
    }
  ]
}
```

---

## Module Specifications

### 1. market_data.py — Market Data Service

**Responsibilities:**
- Fetch real-time and historical OHLCV data via `yfinance`
- Calculate technical indicators using `ta` library
- Provide a clean data bundle for the agent

**Key Functions:**
```python
get_current_price(ticker: str) -> dict
# Returns: {price, change, change_pct, volume, timestamp}

get_price_history(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame
# Returns OHLCV DataFrame

get_intraday(ticker: str, interval: str = "5m") -> pd.DataFrame
# Returns intraday candles

calculate_indicators(df: pd.DataFrame) -> dict
# Returns: {rsi_14, sma_20, sma_50, ema_12, ema_26, macd, macd_signal,
#           bollinger_upper, bollinger_lower, atr_14, volume_sma_20, obv}
# NOTE: atr_14 (Average True Range) is critical for volatility-adjusted stop losses

get_macro_regime() -> dict:
# Returns: {spy_daily_change, vix_level, regime: "normal"|"elevated"|"crisis",
#           confidence_threshold_override: float or None}
# This sits ABOVE all agents — if SPY is melting down or VIX is spiking,
# all agents must require higher conviction to trade.

get_market_summary(ticker: str) -> dict
# Bundles current price + indicators + macro regime into one payload
```

### 2. strategies.py — Strategy Engine

**Responsibilities:**
- Define each trading strategy as a signal generator
- Each strategy returns a signal: BUY / SELL / HOLD with confidence (0-1)
- **Confidence is weighted by the strategy's historical performance score**
- Combine signals from multiple strategies using evolved weights

**Strategies:**

| Strategy | Signal Logic |
|----------|-------------|
| **Momentum** | Buy when price > SMA20 > SMA50, RSI 50-70; Sell when price < SMA20, RSI > 75 |
| **Mean Reversion** | Buy when price < lower Bollinger, RSI < 30; Sell when price > upper Bollinger, RSI > 70 |
| **Sentiment** | Analyze recent news headlines via Claude; score -1 to +1 |
| **Technical Signals** | MACD crossovers, volume spikes, support/resistance levels |
| **DCA (Dollar Cost Averaging)** | Small periodic buys regardless of price, with tactical adjustments |

```python
class StrategySignal:
    action: str        # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0 (raw signal strength)
    weight: float      # from strategy_scores.json (earned trust)
    strategy: str      # strategy name
    reasoning: str     # human-readable explanation

def evaluate_all_strategies(market_data: dict, portfolio: dict, scores: dict) -> list[StrategySignal]
# Returns weighted signals from all enabled strategies

def aggregate_signals(signals: list[StrategySignal]) -> StrategySignal
# Weighted combination using evolved trust scores
```

### 3. researcher.py — Deep Research Engine (NEW)

**Responsibilities:**
- Study TSLA's historical behavior to build predictive knowledge
- Research runs on a schedule (daily) and on-demand before major events
- Uses Claude API + web search to gather and synthesize findings
- Writes findings to the knowledge base

**Research Tasks (run periodically):**

```python
async def research_earnings_history(ticker: str) -> dict:
    """Study how TSLA has reacted to past earnings reports.
    Analyze: beat/miss, pre-report drift, post-report move, 
    recovery timeline, options activity patterns."""

async def research_catalyst_events(ticker: str) -> dict:
    """Study impact of Elon tweets, product announcements, 
    delivery reports, regulatory news, analyst upgrades/downgrades.
    Build a catalog of event types → typical market reactions."""

async def research_correlations(ticker: str) -> dict:
    """Study TSLA's correlation with: NASDAQ, interest rates, 
    oil prices, BTC, other EV stocks. Identify leading indicators."""

async def research_seasonal_patterns(ticker: str) -> dict:
    """Study day-of-week effects, monthly patterns, 
    options expiration effects, quarter-end behavior."""

async def research_current_context(ticker: str) -> dict:
    """What's happening RIGHT NOW that matters? 
    Upcoming earnings, pending regulations, macro environment,
    competitor news, supply chain developments."""

async def research_on_demand(ticker: str, topic: str) -> dict:
    """Agent can request targeted research on a specific question
    it encounters during trading, e.g., 'How does TSLA react 
    when 10-year yield crosses 4.5%?'"""
```

**Output**: Each research task produces a structured report that gets stored in `knowledge/research/` and summarized into `knowledge/tsla_profile.json`.

### 4. learner.py — Learning Engine with Skeptic Layer (NEW)

**Responsibilities:**
- Conduct post-trade reviews comparing predictions to outcomes
- Extract lessons and update the knowledge base
- **Challenge every lesson through a built-in Skeptic that demands structured, falsifiable categorization**
- Adjust strategy confidence weights based on performance
- Discover new patterns from trade history
- Apply lesson decay — old lessons lose weight unless re-validated
- Write journal entries reflecting on overall progress

**CRITICAL: The Skeptic Layer**

LLMs are excellent at inventing confident causal stories after the fact. "TSLA dropped because Elon tweeted about Mars" sounds wise but might be nonsense — maybe the whole market dropped. The Skeptic layer prevents narrative lock-in by forcing every lesson through structured categorization.

**Lesson Categories (every lesson MUST be assigned one):**
```python
LESSON_CATEGORIES = {
    "signal_correct": "The signal was right and the trade worked as predicted",
    "signal_early": "Right direction but wrong timing — signal fired too soon",
    "signal_late": "Right direction but entered too late — most of the move was over",
    "signal_wrong": "The signal was simply incorrect — predicted direction was wrong",
    "risk_sizing_error": "Direction was right but position size was wrong (too big/small)",
    "regime_mismatch": "Signal would have worked in a different market regime",
    "external_shock": "Unpredictable external event overwhelmed the signal",
    "stop_loss_whipsaw": "Stop loss triggered but price recovered — stop was too tight",
    "correlated_market_move": "TSLA moved with the broader market, not on its own signal",
    "noise_trade": "There was no real signal — this was a low-conviction trade that shouldn't have been taken"
}
```

**Skeptic Prompt (runs on every lesson before it's saved):**
```
You are the Skeptic. Your job is to challenge the lesson the agent just 
extracted from a trade. Ask:

1. SIMPLER EXPLANATION: Was there a simpler explanation? Did the whole 
   market move the same direction? Was this just correlation with SPY?
   Check SPY's movement during the same period.

2. SAMPLE SIZE: How many times has this pattern occurred? If fewer than 
   5 occurrences, label this lesson as "unvalidated" — it might be noise.

3. SURVIVORSHIP: Is the agent only remembering the times this pattern 
   worked and forgetting the times it didn't?

4. REGIME DEPENDENCY: Would this lesson have worked 6 months ago? In a 
   different rate environment? In a bear market?

5. FALSIFIABILITY: What would DISPROVE this lesson? If nothing could 
   disprove it, it's not a lesson — it's a belief.

Categorize the lesson into exactly one of the structured categories.
If the lesson doesn't survive scrutiny, downgrade it to "unvalidated" 
and reduce its weight in the knowledge base.
```

**Model Version Tracking [v3 — Gemini r3 insight]:**
Every stored lesson, prediction, and analysis must include the LLM model version that generated it. If you upgrade models (e.g., from Sonnet 4.5 to a future Opus 4.7), treat this as a regime change — trigger a review of the entire knowledge base. A knowledge base built on one model's reasoning patterns may be misinterpreted by a different model.

```json
{
  "lesson_id": "lesson_042",
  "model_version": "claude-sonnet-4-5-20250929",
  "skeptic_model_version": "claude-haiku-4-5-20251001",
  "generated_at": "2026-03-15T14:30:00Z"
}
```

**Lesson Decay:**
```python
def apply_lesson_decay(lessons: list, decay_rate: float = 0.95) -> list:
    """Lessons lose weight over time unless re-validated.
    
    - Each lesson has a 'weight' starting at 1.0
    - Every week, weight *= decay_rate (default 0.95)
    - If a lesson is re-validated by a new trade outcome, weight resets to 1.0
    - Lessons with weight < 0.3 are archived (not deleted, but excluded 
      from agent prompts)
    - A lesson from 3 months ago in a different rate regime might be 
      actively harmful today
    """
```

**The Learning Loop:**

```
TRADE HAPPENS → Wait for outcome → REVIEW
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
           Extract Lesson      Adjust Strategy     Discover Pattern
           (what went wrong    Scores (reward/     (do I see this
            or right, why)      punish strategy)    pattern repeating?)
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      ▼
                              UPDATE KNOWLEDGE BASE
                              (lessons, patterns, scores, journal)
```

**Key Functions:**

```python
async def review_trade(trade: dict, outcome: dict) -> dict:
    """After a trade closes (or after prediction horizon passes):
    - Compare hypothesis vs actual outcome
    - Call Claude to analyze what went right/wrong
    - Generate a lesson record
    - Return confidence adjustments for strategies used"""

async def review_predictions(predictions: list) -> dict:
    """Check all pending predictions whose time horizon has passed.
    Score accuracy. Update running accuracy metrics."""

async def discover_patterns(trades: list, lessons: list) -> list:
    """Periodically analyze trade history + lessons looking for
    recurring patterns. E.g., 'I keep losing on momentum trades
    during the first hour — maybe TSLA's opening volatility 
    makes momentum unreliable early in the session.'"""

async def evolve_strategy_weights(scores: dict, recent_trades: list) -> dict:
    """Recalculate strategy weights based on recent performance.
    Uses exponential decay — recent trades matter more than old ones.
    Ensures no strategy drops below 0.05 weight (always give it a chance)."""

async def write_journal_entry(portfolio: dict, knowledge: dict) -> str:
    """Periodic reflection: How am I doing? What's working? 
    What should I try differently? Written in natural language 
    and appended to knowledge/journal.md"""
```

### 5. knowledge_base.py — Knowledge Manager (NEW)

**Responsibilities:**
- CRUD operations for all knowledge files
- Query knowledge relevant to current trading decisions
- Summarize knowledge for inclusion in agent prompts
- Manage knowledge size (prune old, low-value entries)

**Key Functions:**

```python
def get_relevant_knowledge(market_context: dict) -> dict:
    """Given current market conditions, retrieve the most relevant:
    - Lessons (matching current signals/conditions)
    - Patterns (matching current setup)
    - Research findings (relevant to current catalysts)
    Returns a condensed knowledge bundle for the agent prompt."""

def add_lesson(lesson: dict) -> None
def add_pattern(pattern: dict) -> None  
def add_research(topic: str, findings: dict) -> None
def update_strategy_scores(scores: dict) -> None

def get_prediction_accuracy() -> dict:
    """Calculate rolling accuracy stats:
    - Overall direction accuracy (30min, 2hr, 1day)
    - Accuracy by strategy
    - Accuracy trend over time (improving?)
    - Biggest blind spots"""

def get_knowledge_summary() -> str:
    """One-paragraph summary of what the agent has learned so far.
    Used in dashboard and journal entries."""
```

### 6. agent.py — AI Decision Engine

**Responsibilities:**
- Receive market data + strategy signals + portfolio state + **relevant knowledge**
- Make predictions with explicit hypotheses
- Use Claude API to make final BUY/SELL/HOLD decision
- Every decision references what it's learned

**Agent Prompt Structure:**
```
SYSTEM: You are MonopolyTrader, an AI trading agent managing a virtual 
portfolio. You are LEARNING — you started with $1,000 and your goal is to 
grow it by getting better at predicting TSLA's movements over time.

You have a knowledge base of lessons from past trades, patterns you've 
discovered, and research you've conducted. USE THIS KNOWLEDGE. Reference 
specific lessons and patterns when making decisions. If you're uncertain, 
say so — it's better to HOLD than to make a low-conviction trade.

Every trade is a hypothesis. State clearly what you predict will happen 
and why. You will be reviewed on this prediction later.

USER: 
<market_data>{current price, indicators, recent candles}</market_data>
<strategy_signals>{weighted signals from all strategies}</strategy_signals>
<portfolio>{current holdings, cash, P&L, recent trades}</portfolio>
<relevant_knowledge>
  <lessons>{lessons matching current conditions}</lessons>
  <patterns>{patterns that may apply}</patterns>
  <research>{relevant research findings}</research>
  <prediction_accuracy>{your recent accuracy stats}</prediction_accuracy>
</relevant_knowledge>

Analyze the data and decide: BUY, SELL, or HOLD.
Respond in JSON: {
  action, shares, confidence, strategy,
  hypothesis: "I predict TSLA will [direction] by [amount] over [timeframe] because [reasoning]",
  reasoning,
  knowledge_applied: [list of lesson/pattern IDs used],
  risk_note,
  research_request: "optional — if you want to research something before next decision"
}
```

### 7. portfolio.py — Portfolio Manager

**Responsibilities:**
- Execute simulated trades (update cash, holdings)
- Enforce risk rules (position limits, stop losses)
- Calculate P&L (realized + unrealized)
- Save/load portfolio state
- Create daily snapshots

**Key Functions:**
```python
def execute_trade(action: str, shares: float, price: float, decision: dict) -> dict
# Validates trade against risk rules, executes, logs, returns result

def check_stop_losses(current_price: float) -> Optional[dict]
# Returns forced sell signal if stop loss triggered

def get_portfolio_summary() -> dict
# Current state with all P&L calculations

def save_snapshot() -> None
# Save current state to snapshots/YYYY-MM-DD.json

def get_performance_vs_benchmark() -> dict
# Compare portfolio returns vs buy-and-hold TSLA + SPY
```

### 8. reporter.py — Dashboard Generator

**Responsibilities:**
- Generate the HTML dashboard with embedded data
- Create JSON data files for the JS frontend

**Dashboard Sections:**
- 📊 **"Am I Beating the Market?" (PRIMARY VIEW)** — Agent return vs all 4 benchmarks (Buy&Hold TSLA, SPY, DCA, Random median). This is the first thing you see. Includes verdict (outperforming/underperforming/inconclusive), alpha, agent percentile vs random traders, and statistical significance. See benchmarking-graduation-spec.md.
- 🎓 **Graduation Progress** — Which of the 12 graduation criteria are met (green) vs unmet (red). Progress toward real money.
- 📈 **Portfolio Value Chart** — Line chart of total value over time with all benchmark lines overlaid
- 💰 **Current Position** — Holdings, cash, total value, P&L
- 🧠 **Learning Progress** — Prediction accuracy over time (is the agent improving?)
- 📊 **Strategy Evolution** — How strategy weights have shifted over time (area chart)
- 🎯 **Prediction Scoreboard** — Recent predictions with outcomes (correct/incorrect)
- 📋 **Trade Log** — Sortable table with hypothesis, outcome, and lesson for each trade
- 📓 **Agent Journal** — The agent's own reflections on its progress
- 🔬 **Knowledge Base Browser** — Lessons, patterns, research findings
- ⚡ **Live Status** — Current price, last decision, next poll time, system health

### 9. main.py — Orchestrator

**The Decision Cycle (every 15 minutes during market hours):**
```python
def run_cycle():
    1. Fetch market data + calculate indicators (including ATR)
    2. Classify current regime (50-day SPY slope + VIX terciles)
    3. Check macro gate (SPY, VIX) — apply elevated conviction thresholds if needed
    4. Check earnings blackout — block new positions if within 48hrs of earnings
    5. Check gap risk — reduce max position size before known events
    6. Run strategy signals (weighted by evolved, regime-tagged scores)
    7. Check stop losses using dynamic ATR levels (auto-sell if triggered)
    8. Query knowledge base for relevant context (decay-weighted, regime-matched)
    9. Check BSM conviction caps — if BSM conflicts with signal, cap position size
    10. Call agent for decision (with knowledge context + macro regime)
    11. If action == HOLD: log decision with counterfactual tracking
    12. If action != HOLD: execute trade with slippage simulation (0.05% per side), record hypothesis
    13. Size position inversely to stop distance (wider ATR stop = smaller position)
    14. Check pending predictions — score any whose horizon has passed
    15. Score any pending HOLD counterfactuals
    16. If a recent trade closed: run post-trade review → Skeptic challenge (separate model) → extract lesson → regime-tag
    17. Run counterfactual sampling on validated lessons periodically
    18. Apply weekly lesson decay
    19. Check kill switches (30-day accuracy trend, drawdown, style drift)
    20. Write full decision trace (inputs, outputs, influence breakdown, regime tag)
    21. Run anomaly detection on this cycle's trace
    22. Update dashboard data + health status
    23. Log everything (structured JSON)
```

**The Research Cycle (daily, after market close):**
```python
def run_daily_research():
    1. Run all research tasks (earnings, catalysts, correlations, context)
    2. Discover new patterns from recent trade history
    3. Evolve strategy weights based on recent performance
    4. Write journal entry
    5. Generate full dashboard report
    6. Save daily snapshot
```

**The Bootstrap (first run):**
```python
def bootstrap():
    """On very first run, before any trading:
    1. Initialize portfolio with $1,000
    2. Run ALL research tasks to build initial knowledge base
    3. Study last 2 years of TSLA price action
    4. Identify current market regime (trending, range-bound, volatile)
    5. Write initial journal entry: 'Day 1 — here's what I know so far'
    6. Start trading with informed (but humble) initial positions"""
```

---

## Risk Management Rules

1. **Max Position Size**: Never invest more than 65% of portfolio in a single stock (TSLA is too volatile for 90%)
2. **Max Single Trade**: No single trade larger than 20% of portfolio value
3. **Volatility-Adjusted Stop Loss**: Use ATR (Average True Range) based stops instead of fixed percentage. Default: 2x ATR below entry price. Fixed 5% stops on TSLA cause whipsaw death — the stock can swing 5% intraday on a normal Tuesday and finish green.
4. **Fractional Shares**: Enabled — allows meaningful positions even with small portfolio
5. **Cash Reserve**: Always maintain at least 15% cash for opportunities
6. **Cool-down**: Minimum 15 minutes between trades on same ticker
7. **Daily Loss Limit**: If portfolio drops 8% in a day, stop trading until next day
8. **Earnings Blackout**: No new positions 48 hours before known TSLA earnings dates unless the agent is explicitly running an earnings strategy. Earnings volatility is a different beast.
9. **Global Macro Gate**: Before any trade, check the macro regime. If SPY is down >2% on the day or VIX is above 30, require higher conviction thresholds (confidence > 0.80) for any BUY. A perfect TSLA setup fails if the whole market is melting down.

---

## Dependencies (requirements.txt)

```
yfinance>=0.2.31
ta>=0.11.0
anthropic>=0.40.0
schedule>=1.2.0
pandas>=2.0.0
numpy>=1.24.0
python-dateutil>=2.8.0
```

---

## Build Order

Build and test in this order:

### Phase 0 — Walk-Forward Backtest Sanity Check (BUILD THIS FIRST)
1. Download 2022-2025 TSLA + SPY OHLCV daily data via yfinance
2. Implement the 5 technical strategies (momentum, mean reversion, sentiment proxy via volume/price patterns, technical signals, DCA) against historical data — no LLM needed
3. Implement dynamic ATR stop losses (2x/2.5x/3x by VIX regime) with inverse position sizing
4. Implement slippage simulation (0.05% per side baseline, 0.15% when VIX > 25)
5. Implement the regime classifier (50-day SPY slope + VIX terciles)
6. Run 100 random trader simulations on the same data with same risk rules
7. Calculate all benchmark comparisons (buy-and-hold TSLA, SPY, DCA $50/week)
8. Output results report: do any strategies beat random traders on risk-adjusted metrics?
9. **If no strategy beats random on 3 years of historical data, STOP — the learning loop has nothing to learn from. Trigger PHASE_0_FAIL milestone.**

### Phase 1 — Foundation
10. `config.json` — Full configuration including risk params, macro gate, kill switches, milestone alerts, slippage, ATR multipliers
11. `market_data.py` — Price data, indicators, ATR, SPY, VIX, regime classifier (`classify_regime()`), macro gate check (`get_macro_regime()`)
12. `portfolio.py` — Portfolio state management, trade simulation WITH slippage, dynamic ATR stop losses, inverse position sizing, gap risk sizing, earnings blackout check
13. `benchmarks.py` — All 4 benchmarks (Buy&Hold TSLA, SPY, DCA, Random x100), slippage-adjusted performance, graduation criteria checker. See benchmarking-graduation-spec.md.
14. `knowledge_base.py` — Read/write knowledge files with decay, regime tags, model version tracking (start with empty scaffolding)
15. `utils.py` — Structured JSON logging, time helpers, formatting
16. **Test**: Fetch TSLA + SPY + VIX data, calculate indicators + ATR + regime, execute a mock trade with slippage, verify benchmarks track correctly

### Phase 2 — Intelligence
17. `strategies.py` — All 5 strategy signal generators with dynamic weighting, regime-aware signals
18. `agent.py` — Wire up Claude API (Sonnet for agent brain) for decision-making with knowledge context, macro regime, and BSM conviction caps. Include HOLD tracking with counterfactual logging.
19. **Test**: Feed real data through strategies → agent → mock trade. Verify hypothesis recorded, HOLD counterfactuals logged, benchmarks updating.

### Phase 3 — Learning (the heart of the project)
20. `learner.py` — Post-trade review with Skeptic layer (separate cheaper model — Haiku), structured lesson categories, hard auto-reject rules, counterfactual sampling, lesson decay, regime tagging, model version tracking
21. `researcher.py` — Deep research tasks that build TSLA knowledge (earnings history, tweet impact, correlations, seasonal patterns). Daily + on-demand.
22. **Test**: Execute a trade, wait for outcome, run review through Skeptic, verify lesson is created with correct category/regime tag/model version, verify strategy scores update, verify Skeptic rejects a bogus "correlated market move" when SPY was flat

### Phase 4 — Automation + Ensemble
23. `main.py` — Scheduler with decision cycle (15-min) and research cycle (daily). Implement bootstrap sequence for first run.
24. `ensemble.py` — Multi-agent orchestrator for 3 agents (Alpha, Bravo, Echo). Each gets own portfolio, own config, same market data. Start ultra-lean with just Echo if preferred.
25. `comparison.py` — Agent leaderboard, correlation analysis, harmony detection
26. `milestones.py` — Milestone checker (positive + negative milestones), Decision Point Report generator, model upgrade awareness
27. **Test**: Run for a full market day. Verify: predictions scored, lessons extracted with Skeptic review, journal written, milestones checked, benchmarks updated, ensemble comparison calculated.

### Phase 5 — Dashboard + Observability
28. `reporter.py` — Generate data JSON for dashboard
29. `observability.py` — Decision tracing, anomaly detection, health checks, influence tracking, kill switches. See observability-spec.md.
30. `dashboard/index.html` + `style.css` + `app.js` — Build interactive dashboard with:
    - **"Am I Beating the Market?"** as primary view (agent vs 4 benchmarks, verdict, agent percentile vs random)
    - **Graduation Progress** — 12 criteria checklist, green/red status
    - **Milestone Alerts** — prominent banners for positive/negative milestones with action buttons
    - **Ensemble Leaderboard** (if multi-agent)
    - Portfolio value chart with all benchmark lines overlaid
    - Prediction accuracy trend over time
    - Strategy weight evolution chart
    - Trade log with hypothesis, outcome, lesson, Skeptic review
    - HOLD log with counterfactual outcomes
    - Agent journal
    - Knowledge base browser (lessons with regime tags, decay weights)
    - Health status + anomaly alerts
    - Human audit button (1-click "this was dumb because..." on each trade)
    - Regime timeline
31. Add CLI debug commands: `--explain-trade`, `--influence-report`, `--alerts`, `--health`, `--debug-traces`, `--milestones`, `--graduation-report`
32. **Test**: Run for several days. Verify dashboard tells the full story. Trigger a test milestone and verify it displays prominently.

### Phase 6 — BSM Integration (SEPARATE PROJECT, build after MonopolyTrader is running)
33. Build Billionaire Signal Monitor as separate project per billionaire-signal-monitor-CLAUDE.md
34. Wire up signal output to MonopolyTrader's `data/bsm_signals/` directory
35. Implement BSM conviction caps in agent decision logic (BSM bearish + tech bullish → cap at 10%)
36. Verify Agent Bravo correctly reads and weights BSM signals as regime/context (not trade triggers)
37. **Test**: Generate a mock BSM signal, verify it modifies conviction/sizing without triggering a trade on its own

---

## Running the Agent

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY=your-key-here

# Initialize portfolio and start trading
python src/main.py

# Or run a single decision cycle (for testing)
python src/main.py --once

# Generate dashboard report
python src/main.py --report

# View dashboard
open dashboard/index.html
```

---

## Success Metrics

The agent's success is measured not just by P&L, but by **learning trajectory** and **benchmark outperformance**:

1. **Benchmark Outperformance** — Is the agent beating Buy&Hold TSLA, SPY, DCA, and random trading? This is the single most important metric. If the answer is no, nothing else matters. See benchmarking-graduation-spec.md for the full benchmark framework and graduation criteria.

2. **Agent vs. Random Percentile** — Does the agent rank above the 75th percentile among 100 simulated random traders with the same rules? This separates skill from luck.

3. **Prediction Accuracy Trend** — Is the agent getting better at predicting direction over time? Track 7-day rolling accuracy for 30min, 2hr, and 1-day predictions. Week 1 might be 50% (coin flip). If it's 60%+ by week 4, that's real learning.

4. **Strategy Evolution** — Are strategy weights shifting in response to performance? Are the winning strategies getting more weight?

5. **Sharpe Ratio** — Risk-adjusted return must be > 0.5 annualized. High returns with wild volatility aren't skill.

6. **Graduation Progress** — How many of the 12 graduation criteria are met? The goal is all 12 sustained over 30 days, which earns the agent a shot at real money (starting with just $100 at Stage 1).

---

## Milestone Alert & Decision System

The agent must proactively notify the operator when it reaches significant thresholds — both good and bad. This is not buried in logs. Milestone alerts are **prominently displayed on the dashboard** with a distinct visual treatment and persist until acknowledged.

### How Alerts Work

The system checks milestone conditions after every daily close. When a milestone is hit, it:
1. Generates a **Milestone Report** (HTML + JSON) with full context
2. Displays it prominently on the dashboard (different from anomaly alerts — these are strategic, not operational)
3. Logs it to `data/milestones.json`
4. Includes a **recommended action** and the data supporting it

### Positive Milestones (Things Are Working)

```json
{
  "milestones_positive": [
    {
      "id": "BEAT_RANDOM_30D",
      "name": "Beating Random Traders (30-day)",
      "condition": "Agent above 60th percentile of random traders for 30 consecutive days",
      "message": "🟢 Echo has outperformed 60% of random traders for a full month. The learning loop may be producing real signal. Consider expanding to 3-agent ensemble.",
      "recommended_action": "Review agent journal and top lessons. If learning quality is high, add Alpha and Bravo agents."
    },
    {
      "id": "BEAT_RANDOM_75TH",
      "name": "75th Percentile Achieved",
      "condition": "Agent above 75th percentile of random traders for 14 consecutive days",
      "message": "🟢 Agent is in the top quartile vs. random. This is the graduation threshold for skill vs. luck.",
      "recommended_action": "Begin tracking all 12 graduation criteria simultaneously."
    },
    {
      "id": "BEAT_BUY_HOLD",
      "name": "Beating Buy & Hold",
      "condition": "Agent total return exceeds TSLA buy-and-hold for 30 consecutive days",
      "message": "🟢 Agent is beating the simplest possible strategy. This is the core value proposition.",
      "recommended_action": "Verify with statistical significance tests. If p < 0.05, prepare graduation evaluation."
    },
    {
      "id": "PREDICTION_ACCURACY_60",
      "name": "Prediction Accuracy Above 60%",
      "condition": "2-hour directional prediction accuracy > 60% over 30-day rolling window",
      "message": "🟢 Agent is predicting TSLA direction correctly 60%+ of the time. This is well above coin-flip and suggests real learning.",
      "recommended_action": "Examine which lesson categories and regime conditions produce the highest accuracy."
    },
    {
      "id": "STRATEGY_DIVERGENCE",
      "name": "Strategy Weights Diverging",
      "condition": "Largest strategy weight is 2x or more the smallest weight",
      "message": "🟢 The learning loop is differentiating strategies. The agent is developing preferences based on experience.",
      "recommended_action": "Review which strategies earned trust and verify with per-strategy win rates."
    },
    {
      "id": "GRADUATION_READY",
      "name": "All 12 Graduation Criteria Met",
      "condition": "All 12 criteria pass simultaneously over 30-day window + regime diversity met",
      "message": "🎓 GRADUATION CANDIDATE. The agent has met all criteria for real money. Generate full graduation report.",
      "recommended_action": "Generate graduation report. Review carefully. If convinced, begin Stage 1 with $100 real money."
    }
  ]
}
```

### Negative Milestones (Decision Points — Not Automatic Kills)

**Important design choice**: Hitting a negative milestone does NOT automatically kill the project. It triggers a **structured decision point** where the system presents the data and the operator decides: retool, pause, or kill. This is because:
- Models improve over time (Opus 4.7, 4.8 may be available)
- Market regimes change (a bad 90 days might be followed by conditions where the agent thrives)
- Some failures are fixable (bad prompt, wrong ATR multiplier) vs. fundamental (the approach doesn't work)

```json
{
  "milestones_negative": [
    {
      "id": "PHASE_0_FAIL",
      "name": "Phase 0 Backtest Failure",
      "condition": "No technical strategy beats random traders on 2022-2025 historical data",
      "severity": "critical",
      "message": "🔴 PHASE 0 FAILED. No strategy shows edge on historical data. The learning loop has nothing to build on.",
      "options": [
        {"action": "kill", "reasoning": "If math can't find signal in 3 years of data, LLMs won't either."},
        {"action": "retool", "reasoning": "Try different indicators, different timeframes, or a different stock entirely."},
        {"action": "investigate", "reasoning": "Check if the backtest implementation has bugs before concluding no signal exists."}
      ]
    },
    {
      "id": "RANDOM_LEVEL_60D",
      "name": "No Better Than Random After 60 Days",
      "condition": "Agent below 55th percentile of random traders after 60 trading days",
      "severity": "high",
      "message": "🟡 After 60 days, the agent is statistically indistinguishable from random trading. The learning loop is not producing edge.",
      "options": [
        {"action": "continue_30_more", "reasoning": "Learning may need more time to compound. Extend to 90 days before deciding."},
        {"action": "retool", "reasoning": "Review lessons and skeptic rejections. Is the agent learning anything real? Try different model, different prompts, or different strategy weights."},
        {"action": "upgrade_model", "reasoning": "A newer, more capable model may extract signal that the current model cannot. Check if a model upgrade is available."},
        {"action": "kill", "reasoning": "If 60 days of trading + hundreds of lessons hasn't produced any edge, the approach may be fundamentally limited."}
      ]
    },
    {
      "id": "STATIC_BEATS_LEARNER",
      "name": "Static Agent Outperforming Learning Agent",
      "condition": "Alpha (no learning) outperforms Echo (with learning) for 30 consecutive days",
      "severity": "high",
      "message": "🟡 The static technical agent is beating the learning agent. The LLM learning loop may be adding noise, not wisdom.",
      "options": [
        {"action": "investigate", "reasoning": "Check if Echo's learned lessons are actually hurting performance. Review skeptic rejection rate — is it too lenient?"},
        {"action": "retool_skeptic", "reasoning": "Tighten the skeptic layer. Increase minimum sample size for validated lessons. Add more aggressive counterfactual testing."},
        {"action": "upgrade_model", "reasoning": "The learning quality depends on the model's reasoning ability. A more capable model may learn better."},
        {"action": "accept", "reasoning": "Maybe simple technical trading IS better for TSLA and the learning loop is over-engineering. Pivot to Alpha as primary agent."}
      ]
    },
    {
      "id": "ACCURACY_DECLINING",
      "name": "Prediction Accuracy Trending Down",
      "condition": "30-day rolling prediction accuracy has declined for 30 consecutive days",
      "severity": "high",
      "message": "🟡 The agent is getting WORSE at predicting, not better. The knowledge base may be accumulating bad lessons.",
      "options": [
        {"action": "purge_lessons", "reasoning": "Wipe lessons with weight < 0.5 and unvalidated lessons. Let the agent learn fresh."},
        {"action": "regime_check", "reasoning": "Has the market regime changed? Old lessons may be poisoning new decisions."},
        {"action": "model_upgrade", "reasoning": "Try a newer model version. Reasoning improvements could fix declining accuracy."},
        {"action": "pause_30_days", "reasoning": "Stop trading, let lessons decay, then restart with a cleaner knowledge base."}
      ]
    },
    {
      "id": "DRAWDOWN_LIMIT",
      "name": "Portfolio Drawdown Limit Hit",
      "condition": "Portfolio drops 15%+ from peak value",
      "severity": "critical",
      "message": "🔴 Maximum drawdown exceeded. Trading auto-halted.",
      "auto_action": "halt_trading",
      "options": [
        {"action": "regress_stage", "reasoning": "Drop back one stage. If at Stage 0, continue paper trading with tighter risk rules."},
        {"action": "investigate", "reasoning": "Was this a single bad event or systematic failure? Check if stop losses and macro gate were working."},
        {"action": "retool_risk", "reasoning": "Tighten ATR multipliers, reduce max position, increase cash reserve."}
      ]
    },
    {
      "id": "REGIME_COLLAPSE",
      "name": "Performance Collapses on Regime Change",
      "condition": "Win rate drops 30+ percentage points when market regime changes",
      "severity": "high",
      "message": "🟡 Agent's performance collapsed with the regime change. Lessons learned in the previous regime are not transferring.",
      "options": [
        {"action": "force_regime_review", "reasoning": "Make the agent re-read all regime-tagged lessons from similar past regimes."},
        {"action": "increase_decay", "reasoning": "Accelerate lesson decay rate to flush out regime-specific lessons faster."},
        {"action": "wait", "reasoning": "The agent may need 2-3 weeks to adapt to the new regime. Monitor but don't panic."}
      ]
    },
    {
      "id": "STAGNANT_WEIGHTS",
      "name": "Strategy Weights Never Diverge",
      "condition": "After 60 days, no strategy weight has moved more than 5% from its starting value",
      "severity": "medium",
      "message": "🟡 The learning loop isn't differentiating strategies. Weights are essentially unchanged from day one.",
      "options": [
        {"action": "investigate", "reasoning": "Is the weight adjustment mechanism too conservative? Are lessons too evenly distributed?"},
        {"action": "retool", "reasoning": "Increase weight adjustment sensitivity or change the weight update formula."},
        {"action": "accept", "reasoning": "Maybe equal weighting IS optimal for TSLA. Not all markets reward specialization."}
      ]
    }
  ]
}
```

### The Retool vs. Kill Framework

When a negative milestone is hit, the system generates a **Decision Point Report** that includes:

1. **What happened** — the milestone data
2. **Context** — current regime, recent model updates, known market events
3. **Agent's own analysis** — what the agent thinks went wrong (take with skepticism)
4. **Skeptic's analysis** — the separate model's assessment
5. **Options with tradeoffs** — as shown above
6. **Model landscape check** — are newer/better models available that might change the outcome?
7. **Cost-to-date** — how much has been spent, what has been learned
8. **Research value assessment** — even if the trading fails, is the experiment producing publishable insights?

**Kill criteria (hard stops — these are genuine "the approach doesn't work" signals):**
- Phase 0 backtest failure AND retool attempt also fails → the stock doesn't have tradeable patterns
- After 120+ trading days across multiple regimes AND model upgrades attempted: still below 60th percentile of random → fundamental limitation
- Monthly API cost > 50% of best-case monthly portfolio return for 3 consecutive months at current stage → economics don't work

**Retool triggers (fixable problems):**
- Model upgrade available → re-run with better model before killing
- Skeptic too lenient/strict → adjust thresholds
- Specific lesson categories dominating failures → adjust the learning loop
- Risk rules causing position sizes too small → loosen one layer at a time

**The Model Upgrade Consideration:**
You're right that Opus 4.7, 4.8, or 4.9 may be available within the experiment's timeframe. The system should:
- Track which model version is running
- When a new model is released, flag it as an available upgrade
- If the agent is underperforming, "upgrade model" should always be an option before "kill"
- After a model upgrade, treat the first 14 days as a recalibration period (don't judge performance)
- Include model version in the milestone report so you can see: "This result was with Sonnet 4.5. Opus 4.7 is now available."

### Milestone Check Implementation

```python
class MilestoneChecker:
    """Runs after market close daily. Checks all positive and negative milestones."""
    
    def check_all(self, metrics: dict, history: dict) -> list[dict]:
        """Returns list of newly triggered milestones."""
        triggered = []
        for milestone in ALL_MILESTONES:
            if self.evaluate_condition(milestone, metrics, history):
                if not self.already_triggered(milestone):
                    triggered.append(self.generate_milestone_report(milestone, metrics))
        return triggered
    
    def generate_milestone_report(self, milestone: dict, metrics: dict) -> dict:
        """Generate a full Decision Point Report for negative milestones
        or a Celebration Report for positive ones."""
        report = {
            "milestone_id": milestone["id"],
            "triggered_at": datetime.utcnow().isoformat(),
            "severity": milestone.get("severity", "info"),
            "message": milestone["message"],
            "current_metrics": metrics,
            "model_version": current_model_version,
            "newer_model_available": check_for_model_upgrades(),
            "cost_to_date": calculate_total_spend(),
            "days_running": calculate_trading_days(),
            "regime_history": get_regime_history(),
        }
        if milestone["severity"] in ["high", "critical"]:
            report["options"] = milestone["options"]
            report["auto_action"] = milestone.get("auto_action", None)
        return report
    
    def display_on_dashboard(self, milestone_report: dict):
        """Show prominently on dashboard. Positive milestones get a green banner.
        Negative milestones get a yellow/red banner with action buttons."""
```

---

## Integration: Billionaire Signal Monitor (BSM)

MonopolyTrader receives external intelligence from the companion Billionaire Signal Monitor project. BSM tracks podcast appearances, interviews, tweets, and SEC filings from influential financial figures (Elon Musk, Cathie Wood, Ray Dalio, etc.) and writes structured trading signals to `data/bsm_signals/`.

**Reading BSM signals**: During each decision cycle, `market_data.py` or `researcher.py` should check for fresh BSM signals:
```python
def get_bsm_signals(signal_path: str = "data/bsm_signals/latest_signals.json") -> list[dict]:
    """Read latest BSM signals. Filter expired. Return relevant signals."""
```

**Signal format**: Each BSM signal includes `source_person`, `ticker`, `direction` (bullish/bearish/caution), `strength` (0-1), `confidence` (0-1), `summary`, `time_horizon`, and `decay` info.

**How the agent should use BSM signals**: BSM signals feed into the agent prompt alongside strategy signals and market data. The agent should weigh them based on the source person's prediction track record (included in the signal) and how recently the signal was generated.

BSM is a separate project with its own CLAUDE.md. It runs independently and writes to the shared `data/bsm_signals/` directory. MonopolyTrader only reads from this directory — it never writes to it.

---

## Notes

- The agent uses **Monopoly dollars** — this is paper trading only, no real money involved
- All trades execute at real market prices fetched from Yahoo Finance
- A small slippage model (0.1%) is applied to simulate real execution
- The agent's reasoning is fully logged for post-analysis
- Dashboard auto-refreshes every 60 seconds when open
- **The agent is expected to lose money early.** The design prioritizes learning rate over initial returns. A smart agent that loses $50 in week 1 but develops accurate TSLA intuition is better than one that gets lucky on a few trades.
- The knowledge base is the agent's most valuable asset — protect it, back it up, and watch it grow
